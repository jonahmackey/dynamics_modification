import argparse

import torch.multiprocessing as mp

from experiments.distributed_finetuning import run_experiment


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', default='ViT-B-32', type=str)
    
    parser.add_argument('--dataset_name', default='MNIST', type=str)
    parser.add_argument('--data_path', default='./datasets', type=str)
    
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_iters', default=1000, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--clip_grad_norm', action='store_true')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--print_every', default=100, type=int)
    
    parser.add_argument('--ft_method', default='ls_gamma', type=str)
    parser.add_argument('--heads_path', default='./heads', type=str)
    parser.add_argument('--results_path', default='./results', type=str)
    
    parser.add_argument("--job_id", default='', type=str)
    parser.add_argument("--world_size", default=1, type=int)
    
    parsed_args = parser.parse_args()
    
    # experiment inputs
    args = {
        'model_name': parsed_args.model_name,
        'dataset_name': parsed_args.dataset_name,
        'data_path': parsed_args.data_path,
        'lr': parsed_args.lr,
        'batch_size': parsed_args.batch_size // parsed_args.world_size,
        'num_iters': parsed_args.num_iters,
        'warmup_steps': parsed_args.warmup_steps,
        'weight_decay': parsed_args.weight_decay,
        'clip_grad_norm': parsed_args.clip_grad_norm,
        'print_every': parsed_args.print_every,
        'ft_method': parsed_args.ft_method,
        'heads_path': parsed_args.heads_path + f'/{parsed_args.model_name}',
        'results_path': parsed_args.results_path,
        'job_id': parsed_args.job_id,
        'world_size': parsed_args.world_size,
    }
    
    mp.spawn(run_experiment, args=(args,), nprocs=args['world_size'])