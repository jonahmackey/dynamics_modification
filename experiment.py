import csv

import torch.nn as nn
import torch.optim as optim

import open_clip

from src.eval import evaluate 
from src.finetune import finetune_epoch
from src.utils import cosine_lr, AverageMeter
from src.neural_collapse import compute_NC_stats
from src.modeling import ImageClassifier, get_classification_head
from src.datasets.registry import get_dataset


class Experiment():
    def __init__(self, args) -> None:
        
        # set attributes
        for key, value in args.items():
            setattr(self, key, value)
            
            if type(value) == dict:
                for k, v in value.items():
                    setattr(self, k, v)
                                
        # get model 
        clip_model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained='openai')
        classification_head = get_classification_head(model=clip_model, 
                                                      model_name=self.model_name, 
                                                      dataset_name=self.dataset_name, 
                                                      data_path=self.data_path, 
                                                      device=self.device, 
                                                      save_path=self.heads_path)
        
        self.model = ImageClassifier(image_encoder=clip_model.visual,
                                     classification_head=classification_head,
                                     preprocess=preprocess)
        
        # modify model 
        self.model.add_gamma_layer()
        self.model.freeze_params_no_gamma()
        self.model.to(self.device)
        
        # get dataset
        self.dataset = get_dataset(self.dataset_name, 
                                   preprocess=preprocess, 
                                   location=self.data_path, 
                                   batch_size=self.batch_size)
        
        # loss func
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        
        # optimizer and scheduler 
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=self.lr)
        
        num_batches = len(self.dataset.train_loader)
        self.scheduler = cosine_lr(optimizer=self.optimizer, 
                                   base_lrs=self.lr, 
                                   warmup_length=self.warmup_steps, 
                                   steps=self.num_epochs * num_batches)
        
    def finetune_experiment(self):        
        print('='*30 + f' Fine-tuning Gamma | Model: {self.model_name} | Dataset: {self.dataset_name}' + '='*30) 
        
        if self.save_models:
            final_model_path = f'{self.save_path}/gamma-ft_final.pth'
            best_model_path = f'{self.save_path}/gamma-ft_best.pth'
        
        # zeroshot accuracy
        zeroshot_accuracy = evaluate(model=self.model, 
                                     data_loader=self.dataset.test_loader, 
                                     device=self.device)
        print(f'\nZeroshot Accuracy: {100 * zeroshot_accuracy:.2f}%\n')
        
        # zeroshot NC statistics 
        print(f'\nComputing Zeroshot NC Statistics...\n')
        Sw_invSb_zs, norm_M_CoV_zs, cos_M_zs, W_M_dist_zs = compute_NC_stats(image_encoder=self.model.image_encoder, 
                                                                             data_loader=self.dataset.test_loader, 
                                                                             num_classes=len(self.dataset.classnames), 
                                                                             device=self.device)
        
        # finetuning gamma
        meter = AverageMeter() 
        best_test_accuracy = 0.0
        best_test_epoch = 0
        
        for epoch in range(self.num_epochs):
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
            test_accuracy = evaluate(model=self.model, 
                                data_loader=self.dataset.test_loader, 
                                device=self.device)
            
            # track best model performance
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_test_epoch = epoch
                
                # save best model 
                if self.save_models:
                    self.model.save(best_model_path)
                
            print(f'\nTest Accuracy: {100 * test_accuracy:.2f}% (Best: {100 * best_test_accuracy:.2f}%) | Epoch: {epoch}/{self.num_epochs} (Best: {best_test_epoch}/{self.num_epochs})\n')
        
        # save results to CSV
        stats = {'zeroshot accuracy': zeroshot_accuracy,
                 'final_test_accuracy': test_accuracy,
                 'best_test_accuracy': best_test_accuracy,
                 'best_test_epoch': best_test_epoch,}
        
        # save final model 
        if self.save_models:
            self.model.save(best_model_path)
            stats = dict(stats, **{'final_model_path': final_model_path, 'best_model_path': best_model_path})
        
        stats = dict(stats, **self.__getstate__())
        
        with open(f'{self.save_path}/results.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        
        # save NC statistics to CSV
        print(f'\nComputing Fine-tuned NC Statistics...\n')
        Sw_invSb_ft, norm_M_CoV_ft, cos_M_ft, W_M_dist_ft = compute_NC_stats(image_encoder=self.model.image_encoder, 
                                                                             data_loader=self.dataset.test_loader, 
                                                                             num_classes=len(self.dataset.classnames), 
                                                                             device=self.device)
        
        with open(f'{self.save_path}/NC_stats.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Sw_invSb_zs", "norm_M_CoV_zs", "cos_M_zs", "W_M_dist_zs", "Sw_invSb_ft", "norm_M_CoV_ft", "cos_M_ft", "W_M_dist_ft"])
            writer.writerows(zip(Sw_invSb_zs, norm_M_CoV_zs, cos_M_zs, W_M_dist_zs, Sw_invSb_ft, norm_M_CoV_ft, cos_M_ft, W_M_dist_ft))

    
    def bb_experiment(self):
        # get zeroshot accuracy and NC statistics before modifying the model
        # modify the forward pass to include BB steps, save the model
        # evaluate and get NC statistics
        # save these things to a CSV
        pass  
    
    
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # remove fields that should not be saved
        attributes = ['model',
                      'dataset',
                      'loss_func',
                      'optimizer',
                      'scheduler']
        
        for attr in attributes:
            try:
                del state[attr]
            except:
                pass
        
        return state
            