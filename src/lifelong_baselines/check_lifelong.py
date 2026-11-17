from lifelong_learning import LifelongBaselines
import argparse
import random
import numpy as np
import torch
import os


random.seed(511)
np.random.seed(511)
torch.manual_seed(511)

MNIST_PATH = '/dataset/mnist_test_samples.pkl'
CIFAR10_PATH = '/dataset/cifar10_test_samples.pkl'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_epochs', type=int, default=10)
    parser.add_argument('-dataset_type', type=str, default='mnist')
    parser.add_argument('-method', type=str, default='naive')
    parser.add_argument('-agent_num', type=int, default=6)
    parser.add_argument('-total_time', type=int, default=10)
    parser.add_argument('-is_backbone', type=int, default=0)
    parser.add_argument('-ewc_lambda', type=float, default=1.0)
    parser.add_argument('-lwf_alpha', type=float, default=0.1)
    parser.add_argument('-lwf_temp', type=float, default=0.1)
    parser.add_argument('-gem_pnum', type=int, default=5)
    parser.add_argument('-mem_size', type=int, default=5)
    parser.add_argument('-device', type=str, default='cpu')

    args = parser.parse_args()
    is_backbone = False if args.is_backbone == 0 else True
    print(f'Experiment started, dataset type: {args.dataset_type}, number of agents: {args.agent_num},\ntotal time: {args.total_time}, using backbone: {is_backbone}')

    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + args.device)

    print('Running on device: %s' % device)

    solver = LifelongBaselines(agent_num=args.agent_num,
                               dataset_type=args.dataset_type,
                               total_time=args.total_time,
                               is_backbone=is_backbone,
                               device=device)
    

    method = args.method
    dataset_path = solver.BASE_PATH + MNIST_PATH if args.dataset_type == 'mnist' else solver.BASE_PATH + CIFAR10_PATH

    print(f'Using method: {method}')

    if method == 'ewc':
        
        result = solver.benchmark(method, dataset_path, args.num_epochs, args.ewc_lambda)

    elif method == 'lwf':

        result = solver.benchmark(method, dataset_path, args.num_epochs, args.lwf_alpha, args.lwf_temp)

    elif method in ['gem', 'online_gem']:
        result = solver.benchmark(method, dataset_path, args.num_epochs, args.gem_pnum)

    elif method in ['icarl', 'bic', 'mir', 'mer']:
        result = solver.benchmark(method, dataset_path, args.num_epochs, args.mem_size)

    else:
        # naive, mas
        result = solver.benchmark(method, dataset_path, args.num_epochs)


    file_name = f"""{args.dataset_type}_agentnum={args.agent_num}_time={args.total_time}_backbone={is_backbone}_method={method}_epoch={args.num_epochs}.npy"""

    if not os.path.exists('lifelong_baseline_result'):
        os.mkdir('lifelong_baseline_result')

    file_path = os.path.join('lifelong_baseline_result', file_name)
    print(result)

    np.save(file_path, result)

