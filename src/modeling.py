import os
import torch

import open_clip
from open_clip.transformer import LayerScale

from src.datasets.templates import get_templates
from src.datasets.classnames import get_classnames


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, preprocess):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.preprocess = preprocess
        
    def add_gamma_layer(self):
        embed_dim = self.image_encoder.transformer.width
        
        for resblock in self.image_encoder.transformer.resblocks:
            resblock.ls_1 = LayerScale(embed_dim, init_values=1.0)
            resblock.ls_2 = LayerScale(embed_dim, init_values=1.0)

    def freeze_params_no_gamma(self):
        for name, param in self.image_encoder.named_parameters():
            if ('ls_1' in name) or ('ls_2' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        torch.save(self, filename)

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


def build_classification_head(model, dataset_name, device):    
    templates = get_templates(dataset_name)
    classnames = get_classnames(dataset_name)
    
    logit_scale = model.logit_scale

    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        
        for classname in classnames:
            texts = []
            
            for t in templates:
                texts.append(t(classname))
                
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(model, model_name, dataset_name, device, save_path):
    head_path = save_path + f'/head_{dataset_name}.pt'
    
    if os.path.exists(head_path):
        print(f'Classification head for {model_name} on {dataset_name} exists at {head_path}')
        return ClassificationHead.load(head_path)
    
    print(f'Did not find classification head for {model_name} on {dataset_name} at {head_path}, building one from scratch.')
    classification_head = build_classification_head(model, dataset_name, device)
    classification_head.save(head_path)
    
    return classification_head