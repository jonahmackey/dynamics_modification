from abc import abstractmethod, ABC

import open_clip
from src.modeling import ImageClassifier, get_classification_head
from src.datasets.registry import get_dataset

class Experiment(ABC):
    def __init__(self, args) -> None:
        # set basic attributes
        for key, value in args.items():
            setattr(self, key, value) 
            
            
    def setup_experiment(self):
        self.setup_model()
        self.setup_dataset()                
    
    def setup_model(self):
        clip_model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, 
                                                                          pretrained='openai', 
                                                                          cache_dir='/home/jmackey/.cache/clip')
        classification_head = get_classification_head(model=clip_model, 
                                                      dataset_name=self.dataset_name,
                                                      heads_path=self.heads_path)
        
        self.model = ImageClassifier(image_encoder=clip_model.visual,
                                     classification_head=classification_head,
                                     preprocess=preprocess,
                                     ft_method=self.ft_method)
        
        self.preprocess = preprocess
        
    def setup_dataset(self):
        self.dataset = get_dataset(self.dataset_name, 
                                   preprocess=self.preprocess, 
                                   location=self.data_path, 
                                   batch_size=self.batch_size)
        self.num_epochs = 1 + self.num_iters // len(self.dataset.train_loader)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # remove fields that should not be saved
        attributes = ['model',
                      'dataset',
                      'loss_func',
                      'optimizer',
                      'scheduler',
                      'preprocess']
        
        for attr in attributes:
            try:
                del state[attr]
            except:
                pass
        
        return state
    
    @abstractmethod
    def run(self):
        raise NotImplementedError
    
    