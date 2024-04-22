import csv
import sys
import torch.nn as nn
import torch.optim as optim

from src.eval import evaluate 
from src.finetune import finetune_epoch
from src.utils import cosine_lr, AverageMeter
from src.neural_collapse import compute_neural_collapse

from experiments.experiment_base import Experiment


class GammaFT(Experiment):
    def __init__(self, args) -> None:
        super().__init__(args)
        
    def setup_experiment(self):
        super().setup_experiment()
        
        self.model.init_gamma(device=self.device)
        self.model.freeze_params_no_gamma()
        
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
        print(f'\nEvaluating Zeroshot Accuracy...')
        zeroshot_accuracy = evaluate(model=self.model, 
                                     data_loader=self.dataset.test_loader, 
                                     device=self.device)
        print(f'Zeroshot Accuracy: {100 * zeroshot_accuracy:.2f}%')
        sys.stdout.flush()
        
        # zeroshot NC statistics 
        print(f'\nComputing Zeroshot NC Statistics...')
        Sw_invSb_zs = compute_neural_collapse(image_encoder=self.model.image_encoder, 
                                              data_loader=self.dataset.test_loader, 
                                              num_classes=len(self.dataset.classnames), 
                                              device=self.device)
        print(f'Done Computing Zeroshot NC Statistics')
        sys.stdout.flush()
        
        # finetuning layerscale
        meter = AverageMeter()
        
        print(f'\nFine-tuning LayerScale...')
        for epoch in range(1, self.num_epochs + 1):
            # finetuning epoch
            epoch_loss = finetune_epoch(model=self.model,
                                        data_loader=self.dataset.train_loader,
                                        loss_fn=self.loss_func,
                                        optimizer=self.optimizer,
                                        scheduler=self.scheduler,
                                        meter=meter,
                                        device=self.device,
                                        epoch=epoch,
                                        num_epochs=self.num_epochs,
                                        clip_grad_norm=self.clip_grad_norm,
                                        print_every=self.print_every)
            
        
            print(f'Fine-tuning Epoch Loss: {epoch_loss:.6f} | Epoch: {epoch}/{self.num_epochs}')
            
            # evaluation 
            print(f'\nEvaluating...')
            test_accuracy = evaluate(model=self.model, 
                                data_loader=self.dataset.test_loader, 
                                device=self.device)
                 
            print(f'Test Accuracy: {100 * test_accuracy:.2f}% | Epoch: {epoch}/{self.num_epochs}\n')
            sys.stdout.flush()
            
        # compute finetuned NC statistics
        print(f'Computing Fine-tuned NC Statistics...')
        Sw_invSb_ft = compute_neural_collapse(image_encoder=self.model.image_encoder, 
                                              data_loader=self.dataset.test_loader, 
                                              num_classes=len(self.dataset.classnames), 
                                              device=self.device)
        print(f'Done Computing Fine-tuned NC Statistics\n')
        sys.stdout.flush()
        
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
        
        # save NC statistics to CSV
        with open(f'{self.results_path}/NC_stats.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Sw_invSb_zs", "Sw_invSb_ft"])
            writer.writerows(zip(Sw_invSb_zs, Sw_invSb_ft))
        print(f'Saved NC Statistics to {self.results_path}/NC_stats.csv')