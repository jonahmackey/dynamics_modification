import os
import torch
import argparse
from datetime import datetime

from experiment import Experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', default='ViT-B-32', type=str)
    
    parser.add_argument('--dataset_name', default='MNIST', type=str)
    parser.add_argument('--data_path', default='./datasets', type=str)
    
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=1, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--clip_grad_norm', action='store_true')
    parser.add_argument('--print_every', default=100, type=int)
    
    parser.add_argument('--exp_type', default='gamma-ft', type=str)
    parser.add_argument('--save_models', action='store_true')
    parser.add_argument('--heads_path', default='./models/heads', type=str)
    parser.add_argument('--results_path', default='./results', type=str)
    
    parsed_args = parser.parse_args()
    
    # make directories
    save_path = f'{parser.results_path}/{parser.exp_type}/{parser.model_name}_{parser.dataset_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(parser.heads_path, exist_ok=True)
    
    # experiment inputs
    args = {
        'model_name': parsed_args.model_name,
        'dataset_name': parsed_args.dataset_name,
        'data_path': parsed_args.data_path,
        'lr': parsed_args.lr,
        'batch_size': parsed_args.batch_size,
        'num_epochs': parsed_args.num_epochs,
        'warmup_steps': parsed_args.warmup_steps,
        'clip_grad_norm': parsed_args.clip_grad_norm,
        'print_every': parsed_args.print_every,
        'exp_type': parsed_args.exp_type,
        'save_models': parsed_args.save_models,
        'heads_path': parser.heads_path,
        'save_path': save_path,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # run experiment
    exp = Experiment(args)
    
    if parsed_args.experiment_type == 'gamma-ft':
        exp.finetune_experiment()
    