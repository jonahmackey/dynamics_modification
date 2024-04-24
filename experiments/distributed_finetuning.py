import os
import csv
import sys
import argparse
from datetime import timedelta
import open_clip

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp

from src.eval import evaluate 
from src.finetune import finetune_epoch
from src.utils import cosine_lr, AverageMeter, AccuracyMeter
from src.modeling import ImageClassifier, get_classification_head
from src.datasets.registry import get_dataset


def run_experiment(rank, args):
    
    # init process group
    print(f'From Rank: {rank}, ==> Initializing Process Group...')
    setup_processes(world_size=args['world_size'], rank=rank)
    print('Process group ready!')
    sys.stdout.flush()
    
    # setup model 
    model, preprocess = setup_model(args)
    model.init_ls()
    model.freeze_param_subset()
    model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, 
                                                      device_ids=[rank],
                                                      find_unused_parameters=True)
    
    # setup dataset
    dataset = get_dataset(args['dataset_name'], 
                          preprocess=preprocess, 
                          location=args['data_path'], 
                          batch_size=args['batch_size'],
                          distributed=True)
    num_epochs = 1 + args['num_iters'] // len(dataset.train_loader)
    
    # setup optimization 
    loss_func = nn.CrossEntropyLoss()
        
    params = [p for p in model.module.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args['lr'])
        
    num_batches = len(dataset.train_loader)
    scheduler = cosine_lr(optimizer=optimizer, 
                          base_lrs=args['lr'], 
                          warmup_length=args['warmup_steps'], 
                          steps=num_epochs * num_batches)
    
    ### Fine-tuning Loop ###
    print('\n'+'='*30 + f' Fine-tuning LayerScale | Model: {args["model_name"]} | Dataset: {args["dataset_name"]}' + '='*30) 
            
    # zeroshot accuracy
    if rank == 0:
        print(f'\nEvaluating Zeroshot Accuracy...')
        zeroshot_accuracy = evaluate(model=model, 
                                     data_loader=dataset.test_loader)
        print(f'Zeroshot Accuracy: {100 * zeroshot_accuracy:.2f}%')
        sys.stdout.flush()
    dist.barrier()
        
    # finetuning
    meters = {
        'loss': AverageMeter(),
        'accuracy': AccuracyMeter(),
    }
        
    print(f'\nFine-tuning LayerScale...')
    for epoch in range(1, num_epochs + 1):
        
        dataset.train_loader.sampler.set_epoch(epoch)
    
        # finetuning epoch
        epoch_loss, epoch_accuracy = finetune_epoch(model=model,
                                                    data_loader=dataset.train_loader,
                                                    loss_fn=loss_func,
                                                    optimizer=optimizer,
                                                    scheduler=scheduler,
                                                    meters=meters,
                                                    epoch=epoch,
                                                    num_epochs=num_epochs,
                                                    clip_grad_norm=args['clip_grad_norm'],
                                                    print_every=args['print_every'])
        
    
        print(f'Fine-tuning Epoch Loss: {epoch_loss:.6f} | Epoch: {epoch}/{num_epochs}')
        print(f'Fine-tuning Epoch Accuracy: {epoch_accuracy:.2f} | Epoch: {epoch}/{num_epochs}')
            
            # evaluation 
        if rank == 0:
            print(f'\nEvaluating...')
            test_accuracy = evaluate(model=model, 
                                     data_loader=dataset.test_loader)
                
            print(f'Test Accuracy: {100 * test_accuracy:.2f}% | Epoch: {epoch}/{num_epochs}\n')
            sys.stdout.flush()
            
        dist.barrier()
        
    # saving results
    if rank == 0:
        # save gammas 
        model.module.save_gamma(args['results_path'] + '/gammas.pt')
        
        # save results to CSV
        stats = {'zeroshot_accuracy': zeroshot_accuracy,
                 'final_test_accuracy': test_accuracy}
        
        stats = dict(stats, **args)
        
        with open(f'{args["results_path"]}/results.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        print(f'\nSaved Results to {args["results_path"]}/results.csv')
        
    dist.barrier()
    dist.destroy_process_group()
    
    
def setup_processes(world_size, rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend='nccl', 
                            world_size=world_size, 
                            rank=rank,
                            timeout=timedelta(minutes=10))
    
    
def setup_model(args):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(args['model_name'], 
                                                                      pretrained='openai', 
                                                                      cache_dir='/home/jmackey/.cache/clip')
    classification_head = get_classification_head(model=clip_model, 
                                                  dataset_name=args['dataset_name'], 
                                                  heads_path=args['heads_path'])
    model = ImageClassifier(image_encoder=clip_model.visual,
                            classification_head=classification_head,
                            preprocess=preprocess)
    
    return model, preprocess
            
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', default='ViT-B-32', type=str)
    
    parser.add_argument('--dataset_name', default='MNIST', type=str)
    parser.add_argument('--data_path', default='./datasets', type=str)
    
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--clip_grad_norm', action='store_true')
    parser.add_argument('--print_every', default=100, type=int)
    
    parser.add_argument('--exp_type', default='finetune-layerscale', type=str)
    parser.add_argument('--heads_path', default='./heads', type=str)
    parser.add_argument('--results_path', default='./results', type=str)
    
    parser.add_argument("--job_id", default='', type=str)
    parser.add_argument("--world_size", default=1, type=int)
    
    parsed_args = parser.parse_args()
    
    # experiment inputs
    args = {
        'model_name': parsed_args.model_name,
        'dataset_name': parsed_args.dataset_name,
        'data_path': parsed_args.data_path,
        'lr': parsed_args.lr,
        'batch_size': parsed_args.batch_size,
        'num_iters': parsed_args.num_iters,
        'warmup_steps': parsed_args.warmup_steps,
        'clip_grad_norm': parsed_args.clip_grad_norm,
        'print_every': parsed_args.print_every,
        'exp_type': parsed_args.exp_type,
        'heads_path': parsed_args.heads_path + f'/{parsed_args.model_name}',
        'results_path': parsed_args.results_path,
        'job_id': parsed_args.job_id,
        'world_size': parsed_args.world_size,
    }
    
    mp.spawn(run_experiment, args=(args,), world_size=args['world_size'],)
        