import csv
import sys

import torch.nn as nn
import torch.optim as optim

from src.eval import evaluate 
from src.finetune import finetune_epoch
from src.utils import cosine_lr, AverageMeter, AccuracyMeter
from src.neural_collapse import compute_neural_collapse

from experiments.experiment_base import Experiment


class FinetuningExperiment(Experiment):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def setup_experiment(self):
        super().setup_experiment()
        
        self.model.cuda()
        
        if not (self.ft_method == 'zeroshot'):
            self.setup_optimization()
        
    def setup_optimization(self):
        # loss
        self.loss_func = nn.CrossEntropyLoss()
        
        # optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        
        # scheduler
        num_batches = len(self.dataset.train_loader)
        self.scheduler = cosine_lr(optimizer=self.optimizer, 
                                   base_lrs=self.lr, 
                                   warmup_length=self.warmup_steps, 
                                   steps=self.num_epochs * num_batches)
    
    def run(self):
        print('\n'+'='*30 + f' Running Experiment | Model: {self.model_name} | Dataset: {self.dataset_name} | FT Method: {self.ft_method} ' + '='*30) 
        
        if not (self.ft_method == 'zeroshot'):
            ### Fine-tuning Loop ###
            meters = {
                'loss': AverageMeter(),
                'accuracy': AccuracyMeter(),
            }
            
            print(f'\nFine-tuning LayerScale...')
            for epoch in range(1, self.num_epochs + 1):
            
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
                print(f'\nEvaluating...')
                test_accuracy = evaluate(model=self.model, data_loader=self.dataset.test_loader)
                    
                print(f'Test Accuracy: {100 * test_accuracy:.2f}% | Epoch: {epoch}/{self.num_epochs}\n')
                sys.stdout.flush()
        else:
            print(f'\nEvaluating Zeroshot...')
            test_accuracy = evaluate(model=self.model, data_loader=self.dataset.test_loader)
            print(f'Zeroshot Accuracy: {100 * test_accuracy:.2f}%\n')
            sys.stdout.flush()
            
            # manually setting attributes to None 
            self.lr = None
            self.num_iters = None
            self.warmup_steps = None
            self.num_epochs = None 
            self.weight_decay = None
            self.clip_grad_norm = None
            self.print_every = None
        
        # compute NC statistics
        print(f'Computing NC Statistics...')
        Sw_invSb = compute_neural_collapse(image_encoder=self.model.image_encoder, 
                                           data_loader=self.dataset.test_loader, 
                                           num_classes=len(self.dataset.classnames))
        print(f'Done Computing NC Statistics\n')
        sys.stdout.flush()
        
        nc_dict = {f'Sw_invSb {i + 1}': Sw_invSb[i] for i in range(len(Sw_invSb))}
        
        # save model params
        if not (self.ft_method == 'zeroshot'):
            self.model.save_params(self.results_path + '/model_params.pt')
        
        # save results to CSV
        stats = {'accuracy': test_accuracy}
        stats = dict(stats, **self.__getstate__())
        stats = dict(stats, **nc_dict)
        
        print(f'\nSaving results to {self.results_path}/results.csv')
        with open(f'{self.results_path}/results.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        
