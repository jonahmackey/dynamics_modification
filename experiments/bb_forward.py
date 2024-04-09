import csv

from src.eval import evaluate 
from src.neural_collapse import compute_neural_collapse

from experiments.experiment_base import Experiment


class BBForward(Experiment):
    def __init__(self, args) -> None:
        super().__init__(args)
    
    def run(self):
        print('='*30 + f' Evaluating BB Forward Pass | Model: {self.model_name} | Dataset: {self.dataset_name}' + '='*30) 
        
        if self.save_model:
            bb_model_path = f'{self.save_path}/bb_forward.pt'
        
        # zeroshot accuracy
        print(f'\nEvaluating Zeroshot Accuracy...')
        zeroshot_accuracy = evaluate(model=self.model, 
                                     data_loader=self.dataset.test_loader, 
                                     device=self.device)
        print(f'Zeroshot Accuracy: {100 * zeroshot_accuracy:.2f}%')
        
        # zeroshot NC statistics 
        print(f'\nComputing Zeroshot NC Statistics...')
        Sw_invSb_zs = compute_neural_collapse(image_encoder=self.model.image_encoder, 
                                              data_loader=self.dataset.test_loader, 
                                              num_classes=len(self.dataset.classnames), 
                                              device=self.device)
        print(f'Done Computing Zeroshot NC Statistics')
        
        # modify forward pass of model 
        self.model.add_bb_forward(self.device)
        
        # bb forward accuracy
        print(f'\nEvaluating BB Forward Accuracy...')
        bb_accuracy = evaluate(model=self.model, 
                               data_loader=self.dataset.test_loader, 
                               device=self.device)
        print(f'BB Forward Accuracy: {100 * bb_accuracy:.2f}%')
            
        # compute bb NC statistics
        print(f'\nComputing BB Forward NC Statistics...')
        Sw_invSb_g = compute_neural_collapse(image_encoder=self.model.image_encoder, 
                                             data_loader=self.dataset.test_loader, 
                                             num_classes=len(self.dataset.classnames), 
                                             device=self.device)
        print(f'Done Computing BB Forward NC Statistics')
        
        stats = {'zeroshot accuracy': zeroshot_accuracy,
                 'bb_accuracy': bb_accuracy}
        
        # save final model 
        if self.save_model:
            self.model.save(bb_model_path)
            stats = dict(stats, **{'bb_model_path': bb_model_path})
        
        stats = dict(stats, **self.__getstate__())
        
        # save results to CSV
        with open(f'{self.save_path}/results.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        print(f'\nSaved Results to {self.save_path}/results.csv')
        
        # save NC statistics to CSV 
        with open(f'{self.save_path}/NC_stats.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Sw_invSb_zs", "Sw_invSb_g"])
            writer.writerows(zip(Sw_invSb_zs, Sw_invSb_g))
        print(f'Saved NC Statistics to {self.save_path}/NC_stats.csv')