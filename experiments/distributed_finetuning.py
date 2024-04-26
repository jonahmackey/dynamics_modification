import os
import csv
import sys
from datetime import timedelta
import open_clip

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data.distributed

from src.eval import evaluate 
from src.finetune import finetune_epoch
from src.utils import cosine_lr, AverageMeter, AccuracyMeter
from src.modeling import ImageClassifier, get_classification_head
from src.datasets.registry import get_dataset
from src.neural_collapse import compute_neural_collapse


def run_experiment(rank, args):
    torch.cuda.set_device(rank)
    
    # init process group
    print(f'From Rank {rank} ==> Initializing Process Group...')
    setup_processes(world_size=args['world_size'], rank=rank)
    print('Process group ready!')
    sys.stdout.flush()
    
    # setup model 
    model, preprocess = setup_model(args)
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
    optimizer = optim.AdamW(params, lr=args['lr'], weight_decay=args['weight_decay'])
        
    num_batches = len(dataset.train_loader)
    scheduler = cosine_lr(optimizer=optimizer, 
                          base_lrs=args['lr'], 
                          warmup_length=args['warmup_steps'], 
                          steps=num_epochs * num_batches)
    
    ### Fine-tuning Loop ###
    print('\n'+'='*30 + f' Fine-tuning LayerScale (Rank {rank}) | Model: {args["model_name"]} | Dataset: {args["dataset_name"]} | FT Method: {args["ft_method"]} ' + '='*30) 
            
    # zeroshot accuracy
    if rank == 0:
        print(f'\nEvaluating Zeroshot Accuracy... (Rank {rank})')
        zeroshot_accuracy = evaluate(model=model, 
                                     data_loader=dataset.test_loader)
        print(f'Zeroshot Accuracy: {100 * zeroshot_accuracy:.2f}% (Rank {rank})')
        sys.stdout.flush()
    dist.barrier()
        
    # finetuning
    meters = {
        'loss': AverageMeter(),
        'accuracy': AccuracyMeter(),
    }
        
    print(f'\nFine-tuning LayerScale... (Rank {rank})')
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
        
    
        print(f'Fine-tuning Epoch Loss: {epoch_loss:.6f} | Epoch: {epoch}/{num_epochs} (Rank {rank})')
        print(f'Fine-tuning Epoch Accuracy: {epoch_accuracy:.2f} | Epoch: {epoch}/{num_epochs} (Rank {rank})')
            
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
        # compute finetuned NC statistics
        print(f'Computing Fine-tuned NC Statistics...')
        Sw_invSb = compute_neural_collapse(image_encoder=model.module.image_encoder, 
                                           data_loader=dataset.test_loader, 
                                           num_classes=len(dataset.classnames))
        print(f'Done Computing Fine-tuned NC Statistics\n')
        sys.stdout.flush()
        
        nc_dict = {f'Sw_invSb {i + 1}': Sw_invSb[i] for i in range(len(Sw_invSb))}
        
        # save gammas 
        model.module.save_params(args['results_path'] + '/model_params.pt')
        
        # save results to CSV
        stats = {'zeroshot_accuracy': zeroshot_accuracy,
                 'final_test_accuracy': test_accuracy}
        
        stats = dict(stats, **args)
        stats = dict(stats, **nc_dict)
        
        print(f'\nSaving results to {args["results_path"]}/results.csv')
        with open(f'{args["results_path"]}/results.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        
    # dist.barrier()
    dist.destroy_process_group()
    
    
def setup_processes(world_size, rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend='nccl', 
                            world_size=world_size, 
                            rank=rank)
    
    
def setup_model(args):
    clip_model, _, preprocess = open_clip.create_model_and_transforms(args['model_name'], 
                                                                      pretrained='openai', 
                                                                      cache_dir='/home/jmackey/.cache/clip')
    classification_head = get_classification_head(model=clip_model, 
                                                  dataset_name=args['dataset_name'], 
                                                  heads_path=args['heads_path'])
    model = ImageClassifier(image_encoder=clip_model.visual,
                            classification_head=classification_head,
                            preprocess=preprocess,
                            ft_method=args['ft_method'])
    
    return model, preprocess
        