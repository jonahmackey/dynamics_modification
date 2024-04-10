import os
import torch
import argparse

from experiments.bb_forward import BBForward
from experiments.finetune_layerscale import FinetuneLayerScale


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
    
    parser.add_argument('--exp_type', default='finetune-layerscale', type=str)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--heads_path', default='./models/heads', type=str)
    parser.add_argument('--results_path', default='./results', type=str)
    parser.add_argument("--job_id", default='', type=str)
    
    parsed_args = parser.parse_args()
    
    # make directories
    save_path = f'{parsed_args.results_path}/{parsed_args.exp_type}/{parsed_args.model_name}_{parsed_args.dataset_name}_{parsed_args.job_id}'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(parsed_args.heads_path, exist_ok=True)
    
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
        'save_model': parsed_args.save_model,
        'heads_path': parsed_args.heads_path,
        'save_path': save_path,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    # run experiment
    if parsed_args.exp_type == 'finetune-layerscale':
        exp = FinetuneLayerScale(args)
        exp.setup_experiment()
        exp.run()
    elif parsed_args.exp_type == 'bb-forward':
        exp = BBForward(args)
        exp.setup_experiment() 
        exp.run()