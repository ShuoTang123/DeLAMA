import torch
import torch.nn as nn
import torch.nn.functional as f
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    import networkx as nx
except ImportError:
    nx = None


class NonLinearTransform(nn.Module):
    def __init__(self, intermediate_size, output_dim):
        super(NonLinearTransform, self).__init__()
        self.layer1 = nn.Linear(1, intermediate_size)
        self.layer2 = nn.Linear(intermediate_size, intermediate_size)
        self.layer3 = nn.Linear(intermediate_size, intermediate_size)
        self.layer4 = nn.Linear(intermediate_size, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        return x


def plot_groups(gt, groups):
    total_gt, total_label, group_num = [], [], len(groups)
    for i in range(group_num):
        agent_id = groups[i][0]

        total_gt.append(gt[agent_id])
        total_label.append(i * np.ones(1000, dtype=int))

    total_label, total_gt = np.concatenate(total_label), np.concatenate(total_gt)

    scatter = plt.scatter(total_gt[:, 0], total_gt[:, 1], c=[total_label], cmap='viridis', s=0.5)
    plt.legend(*scatter.legend_elements(), loc='lower left', title='Group ID')
    plt.show()


def plot_agents(agents_data, t):
    label_set, data_list = [], []
    agent_num = len(agents_data)
    for i in range(agent_num):
        data_list.append(agents_data[i][t])
        label_set.append(i * np.ones(len(agents_data[i][0])))
    label_set, data_list = np.concatenate(label_set), np.concatenate(data_list)
    scatter = plt.scatter(data_list[:, 0], data_list[:, 1], c=[label_set], cmap='viridis', s=0.5)
    plt.legend(*scatter.legend_elements(), loc='lower right', title='Agent ID')
    plt.show()

def draw_function(function_plots):
    T, agent_num = len(function_plots), len(function_plots[0])

    plt.figure(figsize=(25, 10))
    for t in range(T):
        plt.subplot(2, 5, t + 1)
        for j in range(agent_num):
            plt.plot(function_plots[t][j][0], function_plots[t][j][1])
    plt.show()


def draw_heatmap(matrix_set):
    plt.figure(figsize=(35, 10))
    T = len(matrix_set)
    for t in range(T):
        plt.subplot(2, 5, t + 1)
        matrix = matrix_set[t]
        sns_plot = sns.heatmap(matrix, annot=True)
    plt.show()


def draw_graph(adj_set):
    plt.figure(figsize=(25, 10))
    T, agent_num = len(adj_set), len(adj_set[0])
    for t in range(T):
        plt.subplot(2, 5, t + 1)
        adj_matrix = adj_set[t]
        G = nx.DiGraph()
        for j in range(agent_num):
            G.add_node(j)
        for j in range(agent_num):
            for k in range(agent_num):
                if adj_matrix[j][k] > 0:
                    G.add_edge(j, k, weight=adj_matrix[j][k])

        pos = nx.circular_layout(G)
        label = {}
        for j in range(agent_num):
            label[j] = j

        nx.draw_networkx_edges(G, pos, width=[float(d['weight']) for (u, v, d) in G.edges(data=True)],
                               edge_color='#4169E1')
        nx.draw_networkx_nodes(G, pos, node_color='#000000', label=label)
        nx.draw_networkx_labels(G, pos=pos, labels=label, font_color='w')

    plt.show()


def visualize(func1, func2, graphs, testloss1, testloss2):
    draw_function(func1)

    draw_function(func2)

    draw_graph(graphs)

    draw_heatmap(graphs)

    plt.subplots(figsize=(10, 8))
    sns.heatmap(np.around(testloss1 - testloss2, 1), annot=True)
    plt.show()