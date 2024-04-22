import os
import csv
import sys
import open_clip

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data.distributed

from src.eval import evaluate 
from src.finetune import finetune_epoch
from src.utils import cosine_lr, AverageMeter, AccuracyMeter

from experiments.experiment_base import Experiment


class GammaFT(Experiment):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def setup_experiment(self):
        print("Setting up experiment...")
        
        ngpus_per_node = torch.cuda.device_count()
        self.local_rank = int(os.environ.get("SLURM_LOCALID")) 
        self.rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + self.local_rank
        torch.cuda.set_device(self.local_rank)
        
        print(f'From Rank: {self.rank}, ==> Initializing Process Group...')
        dist.init_process_group(backend=self.dist_backend, init_method=self.init_method, world_size=self.world_size, rank=self.rank)
        print("Process group ready!")
        
        print(f'From Rank: {self.rank}, ==> Initializing model..')
        self.setup_model()
        self.model.init_gamma()
        self.model.freeze_params_no_gamma()
        self.model.cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])
        
        print(f'From Rank: {self.rank}, ==> Preparing data..')
        self.setup_dataset()
        self.setup_optimization()
        
    def setup_optimization(self):
        # loss
        self.loss_func = nn.CrossEntropyLoss()
        
        # optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=self.lr)
        
        # scheduler
        num_batches = len(self.dataset.train_loader)
        self.scheduler = cosine_lr(optimizer=self.optimizer, 
                                   base_lrs=self.lr, 
                                   warmup_length=self.warmup_steps, 
                                   steps=self.num_epochs * num_batches)
    
    def run(self):
        print('\n'+'='*30 + f' Fine-tuning LayerScale | Model: {self.model_name} | Dataset: {self.dataset_name}' + '='*30) 
            
        # zeroshot accuracy
        if self.rank == 0:
            print(f'\nEvaluating Zeroshot Accuracy...')
            zeroshot_accuracy = evaluate(model=self.model, 
                                        data_loader=self.dataset.test_loader)
            print(f'Zeroshot Accuracy: {100 * zeroshot_accuracy:.2f}%')
            sys.stdout.flush()
            
        dist.barrier()
        
        # finetuning layerscale
        meters = {
            'loss': AverageMeter(),
            'accuracy': AccuracyMeter(),
        }
        
        print(f'\nFine-tuning LayerScale...')
        for epoch in range(1, self.num_epochs + 1):
            
            if self.world_size > 1:
                self.dataset.train_sampler.set_epoch(epoch)
        
            # finetuning epoch
            epoch_loss, epoch_accuracy = finetune_epoch(model=self.model,
                                                        data_loader=self.dataset.train_loader,
                                                        loss_fn=self.loss_func,
                                                        optimizer=self.optimizer,
                                                        scheduler=self.scheduler,
                                                        meters=meters,
                                                        epoch=epoch,
                                                        num_epochs=self.num_epochs,
                                                        clip_grad_norm=self.clip_grad_norm,
                                                        print_every=self.print_every)
            
        
            print(f'Fine-tuning Epoch Loss: {epoch_loss:.6f} | Epoch: {epoch}/{self.num_epochs}')
            print(f'Fine-tuning Epoch Accuracy: {epoch_accuracy:.2f} | Epoch: {epoch}/{self.num_epochs}')
            
            # evaluation 
            if self.rank == 0:
                print(f'\nEvaluating...')
                test_accuracy = evaluate(model=self.model, 
                                    data_loader=self.dataset.test_loader)
                    
                print(f'Test Accuracy: {100 * test_accuracy:.2f}% | Epoch: {epoch}/{self.num_epochs}\n')
                sys.stdout.flush()
                
            dist.barrier()
        
        # saving results
        if self.rank == 0:
            # save gammas 
            self.model.save_gamma(self.results_path + '/gammas.pt')
            
            # save results to CSV
            stats = {'zeroshot_accuracy': zeroshot_accuracy,
                        'final_test_accuracy': test_accuracy}
            
            stats = dict(stats, **self.__getstate__())
            
            with open(f'{self.results_path}/results.csv', 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=stats.keys())
                writer.writeheader()
                writer.writerow(stats)
            print(f'\nSaved Results to {self.results_path}/results.csv')
        