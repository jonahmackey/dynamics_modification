import os

import torch
import torch.nn as nn
import open_clip

from src.datasets.templates import get_templates
from src.datasets.classnames import get_classnames


ft_method_to_key = {
    'ln-ls': ['ls_1.gamma', 'ls_2.gamma', 'ls_1.beta', 'ls_2.beta', 'ln_pre.weight', 'ln_1.weight', 'ln_2.weight', 'ln_post.weight', 'ln_pre.bias', 'ln_1.bias', 'ln_2.bias', 'ln_post.bias'],
    'ls': ['ls_1.gamma', 'ls_2.gamma', 'ls_1.beta', 'ls_2.beta'],
    'ls-gamma': ['ls_1.gamma', 'ls_2.gamma'],
    'ls-beta': ['ls_1.beta', 'ls_2.beta'],
    'ln': ['ln_pre.weight', 'ln_1.weight', 'ln_2.weight', 'ln_post.weight', 'ln_pre.bias', 'ln_1.bias', 'ln_2.bias', 'ln_post.bias'],
    'ln-weight': ['ln_pre.weight', 'ln_1.weight', 'ln_2.weight', 'ln_post.weight'],
    'ln-bias': ['ln_pre.bias', 'ln_1.bias', 'ln_2.bias', 'ln_post.bias'],
    'bitfit': ['bias']
}


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.gamma + self.beta


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, preprocess, ft_method='full'):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.preprocess = preprocess
        self.ft_method = ft_method
        
        if not (ft_method == 'full'):
            self.init_ls()
        self.freeze_param_subset()
        
    def init_ls(self):
        embed_dim = self.image_encoder.transformer.width
        
        for resblock in self.image_encoder.transformer.resblocks:
            resblock.ls_1 = LayerScale(embed_dim)
            resblock.ls_2 = LayerScale(embed_dim)
            
    def freeze_param_subset(self): 
        if not (self.ft_method == 'full'):
            for name, param in self.image_encoder.named_parameters():
                param.requires_grad = False
                
                for id in ft_method_to_key[self.ft_method]:
                    if name.endswith(id):
                        param.requires_grad = True
                        break
        
        self.classification_head.weight.requires_grad = False
        self.classification_head.bias.requires_grad = False
        
    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch.save(self, filename)
        
    def save_params(self, filename):
        print(f'Saving model parameters to {filename}') 
        params_state_dict = {}
        
        if self.ft_method == 'full':
            params_state_dict = self.image_encoder.state_dict()
        else:
            for name in self.image_encoder.state_dict():
                for id in ft_method_to_key[self.ft_method]:
                    if name.endswith(id):
                        params_state_dict[name] = self.image_encoder.state_dict()[name]
                        break
                    
        torch.save(params_state_dict, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return torch.load(filename)
    

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        
        self.normalize = normalize
        
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
            
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        torch.save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch.load(filename)


def build_classification_head(model, dataset_name):    
    templates = get_templates(dataset_name)
    classnames = get_classnames(dataset_name)
    
    logit_scale = model.logit_scale

    model.cuda()
    model.eval()

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        
        for classname in classnames:
            texts = []
            
            for t in templates:
                texts.append(t(classname))
                
            texts = open_clip.tokenize(texts).cuda() # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).cuda()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(model, dataset_name, heads_path):
    head_path = heads_path + f'/head_{dataset_name}.pt'
    
    if os.path.exists(head_path):
        print(f'Classification head exists at {head_path}')
        return ClassificationHead.load(head_path)
    
    print(f'Did not find classification head at {head_path}, building one from scratch.')
    classification_head = build_classification_head(model, dataset_name)
    classification_head.save(head_path)
    
    return classification_head