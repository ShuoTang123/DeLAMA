import numpy as np
import torch
import pickle
import random
import networkx as nx
import matplotlib.pyplot as plt
import os


def visualize(G, sparsity, file_path):
    plt.figure()
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightgreen', font_size=8, width=0.5)
    plt.title("Generated graph, sparsity: %.2f" % sparsity)
    plt.savefig(file_path, bbox_inches='tight')


def calculate_sparsity(G):
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    max_possible_edges = num_nodes * (num_nodes - 1) / 2
    
    sparsity = num_edges / max_possible_edges
    return sparsity


def generate_er_graph(N, p):    
    G = nx.erdos_renyi_graph(N, p)

    sparsity = calculate_sparsity(G)

    return G, sparsity


def generate_ws_graph(N, k, p):
    G = nx.watts_strogatz_graph(N, k, p)
    sparsity = calculate_sparsity(G)
    return G, sparsity


def generate_ba_graph(N, m):
    G = nx.barabasi_albert_graph(N, m)
    
    sparsity = calculate_sparsity(G)

    return G, sparsity



def generate_communication_graph(node_num, graph_type, dataset_size, path, sparsity_level):

    cnt = 0
    assert graph_type in ['WS', 'ER', 'BA'] 
    assert sparsity_level in ['low', 'medium', 'high']

    cnt = 0

    sparsity_dict = {'low':0.3, 'medium':0.3, 'high':0.5}

    er_p = {'low':0.5, 'medium':0.3, 'high':0.5}
    k = {'low':2, 'medium':4, 'high':6}
    m = {'low':1, 'medium':2, 'high':2}
    
    result = []
    print('Start generate...')
    while True:
        if graph_type == 'ER':
            graph, sparsity = generate_er_graph(node_num, er_p[sparsity_level])

        elif graph_type == 'WS':
            graph, sparsity = generate_ws_graph(node_num, k[sparsity_level], 0.2)

        elif graph_type == 'BA':
            graph, sparsity = generate_ba_graph(node_num, m[sparsity_level])

        print(sparsity)

        if graph_type!='ER' or (abs(sparsity - sparsity_dict[sparsity_level]) < 0.05 and nx.is_connected(graph)):
            cnt += 1
            adj = nx.to_numpy_matrix(graph)
            adj = np.asarray(adj)
            result.append(adj)
            print(cnt, sparsity)

        if cnt == dataset_size:
            break
    
    result = np.stack(result)

    np.save(path, result)
        


        

    
if __name__ == '__main__':
    N = 6
    G_TYPE = ['ER', 'BA']
    SIZE = 200
    LEVEL = ['low', 'high']

    for g_type in G_TYPE:
        for level in LEVEL:
            FILE_NAME = "communication_graph_%s_%s.npy" % (g_type, level)
            data_path = 'dataset/' + FILE_NAME
            print('Generating %s graph, with sparsity level %s' % (g_type, level))
            generate_communication_graph(N, g_type, SIZE, data_path, level)
            print('Data saved to %s' % data_path)


# cnt = 0
# directory = 'test_generated_graph'
# N = 6  # Number of nodes

# # ER graph
# er_p = 0.5  # Probability of edge creation



# # WS graph
# k = 4
# ws_p = 0.2


# # BA graph
# m = 2


# graph_type = 'ER'

# if not os.path.exists(directory):
#     os.mkdir(directory)

# target_sparsity = [0.35, 0.55]
# while True:
#     if graph_type == 'ER':

#         graph, sparsity_level = generate_er_graph(N, er_p)

#     elif graph_type == 'WS':
#         graph, sparsity_level = generate_ws_graph(N, k, ws_p)

#     elif graph_type == 'BA':
#         graph, sparsity_level = generate_ba_graph(N, m)

#     # print(sparsity_level)

#     if graph_type!='ER' or (abs(sparsity_level - er_p)<0.05 and nx.is_connected(graph)):
#         cnt += 1
#         print(sparsity_level)
#         base_path = os.path.join(directory, graph_type)
#         if not os.path.exists(base_path):
#             os.mkdir(base_path)
#         save_path = os.path.join(base_path, 'graph_%s.png' % cnt)
#         visualize(graph, sparsity_level, save_path)
#     if cnt == 10:
#         break

# # # visualize(er_graph, sparsity_level)
