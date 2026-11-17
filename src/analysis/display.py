import os
import numpy as np

dataset = 'cifar10'

graph_methods = [
        'dual_ascend', 'glasso', 'oracle',
        'mtrl', 'pds', 'admm',
        'gl_sigrep', 'gl_logdet', 'neighbor_select'
    ]

oracle_adj = np.load(os.path.join(dataset, 'adj_collab_mem_oracle_0.1.npy'))
print(oracle_adj.shape)
print(oracle_adj[0][0])

for graph_method in graph_methods:
    if graph_method not in ['dual_ascend', 'oracle', 'mtrl', 'pds', 'admm']:
        for graph_param in [0.01, 0.1, 1, 10]:
            file_name = f'acc_collab_mem_{graph_method}_{graph_param}.npy'
            adj_file_name = f'adj_collab_mem_{graph_method}_{graph_param}.npy'
            path = os.path.join(dataset, file_name)
            adj_path = os.path.join(dataset, adj_file_name)
            acc = np.load(path).mean(axis=-1)
            adj = np.load(adj_path)
            diff = (adj - oracle_adj)**2
            tmp1 = np.sum(np.sum(diff, axis=-1), axis=-1)
            tmp2 = np.sum(np.sum(oracle_adj**2, axis=-1), axis=-1)
            gmse = np.mean(tmp1/tmp2, axis=0)

            print('------------------------')
            print(graph_method, graph_param)
            print(np.around(acc, 4))
            print(np.around(gmse, 4))
    else:
        graph_param = 0.1
        file_name = f'acc_collab_mem_{graph_method}_{graph_param}.npy'
        adj_file_name = f'adj_collab_mem_{graph_method}_{graph_param}.npy'
        path = os.path.join(dataset, file_name)
        adj_path = os.path.join(dataset, adj_file_name)
        acc = np.load(path).mean(axis=-1)
        adj = np.load(adj_path)
        diff = (adj - oracle_adj) ** 2
        tmp1 = np.sum(np.sum(diff, axis=-1), axis=-1)
        tmp2 = np.sum(np.sum(oracle_adj ** 2, axis=-1), axis=-1)
        gmse = np.mean(tmp1 / tmp2, axis=0)

        print('------------------------')
        print(graph_method, graph_param)
        print(np.around(acc, 4))
        print(np.around(gmse, 4))
