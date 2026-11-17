import os
import pickle
import time

import numpy as np
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import os
from models import init_model
from graph_algorithms import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# torch.set_default_dtype(torch.float64)

from classification_dataset import Lifelong_Classification_Task, Classification_Scheduler


def create_subset(datapath, is_train, transform, datasettype):
    if datasettype == 'mnist':
        img_data = torchvision.datasets.MNIST(datapath, download=True, train=is_train, transform=transform)
    else:
        img_data = torchvision.datasets.CIFAR10(datapath, download=True, train=is_train, transform=transform)
    dataset = {}
    for i in range(len(img_data)):
        img, label = img_data[i][0], img_data[i][1]
        if label in dataset.keys():
            dataset[label].append(img)
        else:
            dataset[label] = [img]
    for key in dataset.keys():
        dataset[key] = torch.stack(dataset[key][:100])
    return dataset


def create_dataset(sample_size, train_size, test_size, dataset_type):
    train_path, supervise_path, test_train_path, test_path = 'dataset/%s_train_samples.npy' % dataset_type, \
                                                             'dataset/%s_supervise_samples.npy' % dataset_type, \
                                                             'dataset/%s_testtrain_samples.npy' % dataset_type, \
                                                             'dataset/%s_test_samples.npy' % dataset_type

    if os.path.exists(train_path) and os.path.exists(supervise_path) \
            and os.path.exists(test_path) and os.path.exists(test_train_path):
        train = torch.tensor(np.load(train_path), dtype=torch.int64)
        supervise = torch.tensor(np.load(supervise_path), dtype=torch.int64)
        test_train = torch.tensor(np.load(test_train_path), dtype=torch.int64)
        test = torch.tensor(np.load(test_path), dtype=torch.int64)
        assert train_size == train.shape[0]
        assert train_size == supervise.shape[0]
        assert test_size == test.shape[0]
        assert test_size == test_train.shape[0]

        print('Dataset loaded...')

        return train, supervise, test_train, test

    train, supervise, test_train, test = [], [], [], []
    if dataset_type == 'mnist':
        train_range, test_train_range, test_range = 3000, 2000, 500
    else:
        train_range, test_train_range, test_range = 3000, 2000, 1000
    for i in range(train_size):
        sample_train, sample_supervise = [], []
        for j in range(10):
            choice_train = np.random.choice(train_range, sample_size, replace=False)
            choice_supervise = np.random.choice(train_range, sample_size, replace=False)
            sample_train.append(choice_train)
            sample_supervise.append(choice_supervise)
        sample_train = np.stack(sample_train)
        sample_supervise = np.stack(sample_supervise)

        train.append(sample_train)
        supervise.append(sample_supervise)

    for i in range(test_size):
        sample_testtrain, sample_test = [], []
        for j in range(10):
            choice_test = np.random.choice(test_range, sample_size, replace=False)
            choice_sample_testtrain = np.random.choice(test_train_range, sample_size, replace=False) + train_range

            sample_testtrain.append(choice_sample_testtrain)
            sample_test.append(choice_test)

        sample_test = np.stack(sample_test)
        sample_testtrain = np.stack(sample_testtrain)

        test.append(sample_test)
        test_train.append(sample_testtrain)

    train, supervise, test_train, test = np.stack(train), np.stack(supervise), np.stack(test_train), np.stack(test)

    np.save(train_path, train)
    np.save(supervise_path, supervise)
    np.save(test_train_path, test_train)
    np.save(test_path, test)
    print('Dataset saved...')
    return train, supervise, test_train, test


class LifelongLearning:
    def __init__(self, n_classes, feature_dim, T, float_type=torch.float32):
        self.T = T
        self.gama = 0.1

        self.classes = n_classes
        self.feature_dim = feature_dim + 1
        self.param_dim = int(n_classes * (feature_dim + 1))
        self.float_type = float_type

        # self.trainset = torch.randint(0, 50, [10, 10])
        # self.testset = torch.randint(0, 5, [7, 10])

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Running on cuda...')
        else:
            self.device = torch.device('cpu')
            print('Running on cpu...')
        # self.device = torch.device('cpu')

        self.phi = None

        self.criterion = nn.CrossEntropyLoss()

    def transform(self, x):
        x_trans = self.phi(x.to(self.float_type))
        X = torch.concat([x_trans, torch.ones(len(x_trans), 1, device=self.device)], dim=1)
        return X

    def first_order_derivative(self, X, Y, y_hat):
        tmp = X.T @ (y_hat - Y) / len(X)
        return torch.flatten(tmp.T)

    def second_order_derivative(self, X, y_hat):
        total = []
        for p in range(self.classes):
            tmp = []
            for q in range(self.classes):
                if q != p:
                    sub_matrix = X.T @ torch.diag(-y_hat[:, p] * y_hat[:, q]) @ X / len(X)
                    tmp.append(sub_matrix)
                else:
                    sub_matrix = X.T @ torch.diag(
                        -y_hat[:, p] * y_hat[:, p] + y_hat[:, p]) @ X / len(X)
                    tmp.append(sub_matrix)
            row_matrix = torch.concat(tmp, dim=1)
            total.append(row_matrix)
        hessian = torch.concat(total, dim=0) + 2 * self.gama * torch.eye(self.param_dim, device=self.device)
        # print(np.around(hessian.cpu().numpy(), 4))
        return hessian

    def inside_iteration(self, theta, X, Y):
        assert theta.shape[0] == self.param_dim
        M = theta.reshape(self.classes, self.feature_dim).T

        y_hat = torch.softmax(X @ M, dim=1)
        gradient = self.first_order_derivative(X, Y, y_hat)
        hessian = self.second_order_derivative(X, y_hat)
        theta_new = -torch.linalg.inv(hessian) @ gradient + theta

        return theta_new

    def local_learning(self, X, Y, iternum):
        assert X.shape[1] == self.feature_dim
        assert Y.shape[1] == self.classes
        theta = torch.zeros(self.param_dim, device=self.device)

        for k in range(iternum):
            theta = self.inside_iteration(theta, X, Y)

        M = theta.reshape(self.classes, self.feature_dim).T
        y_hat = torch.softmax(X @ M, dim=1)
        hessian = self.second_order_derivative(X, y_hat)
        grad = self.first_order_derivative(X, Y, y_hat)

        return theta, hessian, grad

    def train(self, lr, batch_size, dataset_type='mnist', epochs=100):
        optimizer = torch.optim.Adam(self.phi.parameters(), lr=lr)
        # train_base = Lifelong_Classification_Task(mode=True, time=self.T, dataset_type=dataset_type)
        # test_base = Lifelong_Classification_Task(mode=False, time=self.T, dataset_type=dataset_type)
        mode = 0 if dataset_type == 'mnist' else 1
        self.phi = init_model(mode=mode, feature_dim=self.feature_dim - 1, device=self.device)

        train_base = Lifelong_Classification_Task(mode=True, time=self.T, dataset_type=dataset_type)
        test_base = Lifelong_Classification_Task(mode=False, time=self.T, dataset_type=dataset_type)

        sample_size = 10
        train_size = 500
        test_size = 100
        train, supervise, test_train, test = create_dataset(sample_size, train_size, test_size,
                                                            dataset_type=dataset_type)

        for epoch in range(epochs):
            curr_index, batch_num = torch.randperm(train_size), int((train_size - 1) / batch_size) + 1
            loss = 0
            for batch_id in range(batch_num):
                # print('Current batch: %s' % (batch_id + 1))
                # Current batch
                batch_index = curr_index[batch_id * batch_size: batch_id * batch_size + batch_size]
                t1 = time.time()
                batch_loss = 0
                optimizer.zero_grad()
                for index in batch_index:
                    # data = train_base.query(trainset[index], dataset_size=query_size, divide_size=divide_size)
                    data, supervise_data = train_base.sample(train[index]), train_base.sample(supervise[index])
                    # data = train_base.single_agent_data(agent_0_dis, class_id_to_index, trainset[index])

                    out_loss, theta, out_acc = self.forward_for_one_sample(data, supervise_data)
                    # print(out_loss, index)
                    batch_loss = batch_loss + out_loss
                batch_loss.backward()
                t2 = time.time()
                if batch_id + 1 == batch_num:
                    print('Current batch: %s, batch loss: %.3f, Cost %.3f seconds' %
                          (batch_id, batch_loss.detach().item(), t2 - t1), end='\n')
                else:
                    print('Current batch: %s, batch loss: %.3f, Cost %.3f seconds' %
                          (batch_id, batch_loss.detach().item(), t2 - t1), end='\r')
                optimizer.step()
                loss = loss + batch_loss.detach()
            test_acc, loss = self.test(test_train, test, train_base, test_base), loss / train_size
            print('Epoch: %s, Cross entropy loss: %.3f' % (epoch + 1, loss.detach().item()))
            print('Test acc: ')
            print(np.around(test_acc.numpy(), 3))

        model_name = 'phi_%s_%s_%s_%s.pt' % (lr, batch_size, dataset_type, epochs)
        torch.save(self.phi.state_dict(), os.path.join('model/', model_name))
        print('Training finished, model saved at %s' % (os.path.join('model', model_name)))

    def test(self, test_train=None, test=None, train_base=None, test_base=None):
        total_acc = torch.zeros(self.classes, self.classes)
        test_size = len(test_train)
        assert len(test_train) == len(test)
        with torch.no_grad():
            for index in range(test_size):
                test_data = test_base.sample(test[index])
                test_train_data = train_base.sample(test_train[index])
                _, theta_record, acc_record = self.forward_for_one_sample(test_train_data, test_data)
                total_acc += acc_record
        return total_acc / test_size

    def forward_for_one_sample(self, data, supervise_data):
        theta, h, h_a = torch.zeros(self.param_dim, device=self.device, dtype=self.float_type), \
            torch.zeros(self.param_dim, self.param_dim, device=self.device, dtype=self.float_type), \
            torch.zeros(self.param_dim, device=self.device, dtype=self.float_type)
        grad_t = torch.zeros(self.param_dim, device=self.device, dtype=self.float_type)
        loss_record, theta_record, acc_record = torch.zeros(self.T, self.T), [], []
        for t in range(self.T):
            current_train, labels = data[t]

            train_x, Y = current_train.to(self.device), \
                F.one_hot(labels, num_classes=self.classes).to(self.device)

            X = self.transform(train_x)
            alpha, hessian, grad = self.local_learning(X, Y, iternum=0)

            h, h_a, grad_t = (t * h + hessian) / (t + 1), \
                             (t * h_a + hessian @ alpha) / (t + 1), (t * grad_t + grad) / (t + 1)

            theta = torch.linalg.inv(h) @ (h_a - grad_t)
            theta_record.append(theta)
            M = theta.reshape(self.classes, self.feature_dim).T

            acc = torch.zeros(self.classes)

            for i in range(t + 1):
                supervise_x, Y = supervise_data[i][0].to(self.device), \
                    F.one_hot(supervise_data[i][1], num_classes=self.classes).to(self.device)

                X = self.transform(supervise_x.to(torch.float64))
                # y_hat = torch.softmax(X @ M, dim=1)
                y_hat = X @ M

                p_hat = torch.softmax(X @ M, dim=1)
                predicted, true = torch.max(p_hat, 1)[1], torch.max(Y, 1)[1]
                acc_value = torch.sum(predicted == true) / len(predicted)
                acc[i] = acc_value

                # loss_t = loss_t + self.criterion(y_hat, Y.to(torch.float64))
                loss_record[t][i] = self.criterion(y_hat, Y.to(torch.float64))

            acc_record.append(acc)

            # loss = loss + loss_t / self.T
        theta_record = torch.stack(theta_record)
        acc_record = torch.stack(acc_record)
        loss = 0
        for t in range(self.T):
            tmp = torch.sum(loss_record[:, t]) / (self.T - t)
            loss = loss + tmp
        return loss / self.T, theta_record, acc_record

    def display(self, dataset_type, model_path, plot=False):
        test_base = Lifelong_Classification_Task(mode=False, time=self.T, dataset_type=dataset_type)
        train_base = Lifelong_Classification_Task(mode=True, time=self.T, dataset_type=dataset_type)
        query_sample = torch.randint(0, 1, [10])
        train_data = train_base.query(query_sample, dataset_size=2000, divide_size=2000)
        test_data = test_base.full_set()

        if dataset_type == 'mnist':
            self.phi = init_model(mode=0, feature_dim=self.feature_dim - 1, device=self.device)
            state_dic = torch.load(model_path, weights_only=False)
        else:
            self.phi = init_model(mode=1, feature_dim=self.feature_dim - 1, device=self.device)
            state_dic = torch.load(model_path, weights_only=False)

        self.phi.load_state_dict(state_dic)

        # test_data = test_base.single_agent_data(agent_0_dis, class_id_to_index, testset[index])
        self.phi.eval()
        with torch.no_grad():
            print('Forwarding..')
            train_loss, theta_record, acc_record = self.forward_for_one_sample(train_data, test_data)
            print(train_loss)

        if plot:
            sns.heatmap(acc_record, annot=True)
            plt.title(dataset_type)
            plt.ylabel('T')
            plt.xlabel('Class Acc')
            plt.show()
        else:
            acc_record = acc_record.numpy()
            print(np.around(acc_record, 3))

    def check(self, dataset_type, model_path):
        test_base = Lifelong_Classification_Task(mode=False, time=self.T, dataset_type=dataset_type)
        train_base = Lifelong_Classification_Task(mode=True, time=self.T, dataset_type=dataset_type)
        query_sample = torch.randint(0, 1, [10])
        train_data = train_base.query(query_sample, dataset_size=100, divide_size=5000)
        test_data = test_base.full_set()

        if dataset_type == 'mnist':
            self.phi = init_model(mode=0, feature_dim=self.feature_dim - 1, device=self.device)
            state_dic = torch.load(model_path, weights_only=False)
        else:
            self.phi = init_model(mode=1, feature_dim=self.feature_dim - 1, device=self.device)
            state_dic = torch.load(model_path, weights_only=False)

        self.phi.load_state_dict(state_dic)

        self.phi.eval()

        with torch.no_grad():
            print('Forwarding..')
            total_train_data, total_train_label = [], []
            for t in range(self.T):
                current_train, labels = train_data[t]

                total_train_data.append(current_train)
                total_train_label.append(labels)

            total_train_data, total_train_label = torch.concat(total_train_data).to(self.device), \
                torch.concat(total_train_label)

            total_X = self.transform(total_train_data)
            total_Y = F.one_hot(total_train_label, num_classes=self.classes).to(self.device)

            alpha, hessian, grad = self.local_learning(total_X, total_Y)

            h, h_a, grad_t = hessian, hessian @ alpha, grad

            theta = torch.linalg.inv(h) @ (h_a - grad_t)

            M = theta.reshape(self.classes, self.feature_dim).T
            acc = torch.zeros(self.classes)
            for i in range(self.classes):
                test_x, Y = test_data[i][0].to(self.device), \
                    F.one_hot(test_data[i][1], num_classes=self.classes).to(self.device)

                X = self.transform(test_x)
                y_hat = torch.softmax(X @ M, dim=1)
                predicted, true = torch.max(y_hat, 1)[1], torch.max(Y, 1)[1]
                acc_value = torch.sum(predicted == true) / len(predicted)
                acc[i] = acc_value
            print('Avg acc: %.4f' % (torch.mean(acc)))
            print(np.around(acc, 3))


class CollaborativeLearning(LifelongLearning):
    
    def __init__(self, agent_num, n_classes, feature_dim, T, float_type):
        super(CollaborativeLearning, self).__init__(n_classes, feature_dim, T, float_type)
        self.agent_num = agent_num
        self.hyper_param = None
        self.gama = 0.1
        self.graph_hyper = 0.1


    def _graph_learning(self, method, **kwargs):
        assert method in ['dual_ascend', 'dual_ascend_distributed', 'glasso', 'oracle',
                          'mtrl', 'pds', 'admm',
                          'gl_sigrep', 'gl_logdet', 'neighbor_select']
        
        if method == 'oracle':
            assert len(kwargs.keys()) == 1
            datainfo = kwargs['datainfo']
            gt_graph = torch.zeros(self.agent_num, self.agent_num, dtype=self.float_type, device=self.device)
            for i in range(self.agent_num):
                for j in range(self.agent_num):
                    id_1, id_2 = datainfo[i]['group_id'], datainfo[j]['group_id']
                    if id_1 == id_2:
                        gt_graph[i][j] = 1
            adj_matrix = normalize(gt_graph)
            return adj_matrix
        else:
            assert len(kwargs) >= 2
            thetas, param = kwargs['thetas'], kwargs['param']

            if method == 'dual_ascend':
                return dual_ascent(thetas, param)[0]

            elif method == 'dual_ascend_distributed':
                return distributed_dual_ascent(thetas, 
                                               param, 
                                               torch.ones(self.agent_num, device=self.device)
                                            )[0]
            
            elif method == 'mtrl':
                return DMTL(thetas)[0]

            elif method == 'glasso':
                return glasso(thetas, alpha=self.graph_hyper)[0]

            elif method == 'pds':
                return pds(thetas)[0]

            elif method == 'admm':
                return admm(thetas)[0]

            elif method == 'gl_sigrep':
                return gl_sigrep(thetas, lamda=self.graph_hyper)
            
            elif method == 'gl_logdet':
                return gl_logdet(thetas, lamda=self.graph_hyper)
            
            elif method == 'neighbor_select':
                return neighbor_select(thetas, lamda=self.graph_hyper)

    def _update_beta_with_w(self, curr_record, adj_matrix, h, h_a, param):
        """Almost 10 iterations to converge"""
        agent_num, feature_dim = len(curr_record), len(curr_record[0])
        assert agent_num == self.agent_num
        lambda1, lambda2 = param[0], param[1]

        for k in range(10):
            # print(adj_matrix.dtype, curr_record.dtype, h_a.dtype, param.dtype)
            b = lambda2 * adj_matrix @ curr_record + h_a

            identity = torch.repeat_interleave(torch.eye(feature_dim, device=self.device).unsqueeze(0),
                                               agent_num, dim=0)

            degree_vec = torch.sum(adj_matrix, dim=1).unsqueeze(-1).unsqueeze(-1)

            diffusion_matrix = h + identity * (lambda2 * degree_vec + lambda1)

            curr_record = torch.bmm(torch.linalg.inv(diffusion_matrix), b.unsqueeze(-1)).squeeze(-1)

        return curr_record

    def _loss(self, theta, supervise_img, supervise_label):
        M = theta.reshape(self.classes, self.feature_dim).T

        supervise_x, Y = supervise_img.to(self.device), \
            F.one_hot(supervise_label, num_classes=self.classes).to(self.device)

        X = self.transform(supervise_x.to(torch.float64))
        y_hat = X @ M

        p_hat = torch.softmax(X @ M, dim=1)
        predicted, true = torch.max(p_hat, 1)[1], torch.max(Y, 1)[1]

        acc_value = torch.sum(predicted == true) / len(predicted)

        loss = self.criterion(y_hat, Y.to(torch.float64))
        return loss, acc_value

    def _save_model(self, hyper_param_path, lambdas, model_path):

        torch.save(self.phi.state_dict(), model_path)
        np.save(hyper_param_path, lambdas.detach().cpu().numpy())

    def _load_parameters(self, dataset_type, hyper_param_path=None, model_path=None):
        if hyper_param_path:
            self.hyper_param = torch.tensor(np.load(hyper_param_path)).to(self.device)
            print('Init hyper param loaded..')
            print(self.hyper_param)
        else:
            self.hyper_param = torch.tensor([0, 0.1, 0.1], dtype=self.float_type, device=self.device)
            print('Hyper param initialized..')

        mode = 0 if dataset_type == 'mnist' else 1
        self.phi = init_model(mode=mode, feature_dim=self.feature_dim - 1,
                              device=self.device, float_type=self.float_type)
        flag = 1
        if model_path:
            state_dic = torch.load(model_path, weights_only=False)
            self.phi.load_state_dict(state_dic)
            flag = 0
        print('model initialized..') if flag == 1 else print('model loaded..')

    def _prepare_model_name(self, with_collaboration: bool, 
                            lr, 
                            batch_size: int,
                            dataset_type: str,
                            epochs: int):
        if with_collaboration:
            assert len(lr) == 2
            graph_lr, nn_lr = lr[0], lr[1]
            model_name = 'collaboration_phi_%s_%s_%s_%s.pt' % (lr, batch_size, dataset_type, epochs)
            hyper_param_name = 'collaboration_param_%s_%s_%s_%s.npy' % (lr, batch_size, dataset_type, epochs)
            return model_name, hyper_param_name, [graph_lr, nn_lr]
        else:
            nn_lr = lr
            model_name = 'nocollab_phi_%s_%s_%s_%s.pt' % (lr, batch_size, dataset_type, epochs)
            hyper_param_name = 'nocollab_param_%s_%s_%s_%s.npy' % (lr, batch_size, dataset_type, epochs)

            return model_name, hyper_param_name, [nn_lr]
    
    def _load_datasets(self, dataset_type:str):
        train_file_name = 'dataset/%s_train_samples.pkl' % dataset_type
        test_file_name = 'dataset/%s_test_samples.pkl' % dataset_type

        with open(train_file_name, 'rb') as f:
            train_set = pickle.load(f)
        with open(test_file_name, 'rb') as f:
            test_set = pickle.load(f)

        return train_set, test_set

    def _init_train_modules(self, flag:int, with_collaboration:bool, lr):
        if flag % 2 == 0 and with_collaboration:
            assert len(lr) == 2
            params = nn.Parameter(self.hyper_param.to(self.device), requires_grad=True)
            graph_lr = lr[0]
            optimizer = torch.optim.Adam([params], lr=graph_lr)
            lambdas = torch.relu(params) + 1e-5
        else:
            params = self.hyper_param
            lambdas = torch.relu(self.hyper_param) + 1e-5
            nn_lr = lr[0]
            optimizer = torch.optim.Adam(self.phi.parameters(), lr=nn_lr)

        return lambdas, params, optimizer

    def train(self, lr, batch_size, **kwargs):
        
        assert len(kwargs) == 5
        param_path = kwargs['param_path']
        model_path = kwargs['model_path']
        dataset_type = kwargs['dataset_type']
        epochs = kwargs['epochs']
        with_collaboration = kwargs['with_collaboration']
        print(f'Training phase, init param path:{param_path}, init model path:{model_path}\nDataset: {dataset_type}, epochs: {epochs}, Collaboration: {with_collaboration}')

        data_base = Classification_Scheduler(time=self.T, dataset_type=dataset_type)
        print('Database initialized...\n')

        train_set, test_set = self._load_datasets(dataset_type=dataset_type)
        train_size, test_size = len(train_set), len(test_set)
        print('Dataset loaded...\n')


        model_name, hyper_param, lr = self._prepare_model_name(with_collaboration,
                                                           lr,
                                                           batch_size,
                                                           dataset_type,
                                                           epochs)

        new_param_path = os.path.join('param/', hyper_param)
        new_model_path = os.path.join('model/', model_name)
        alternative_times = 1

        # perform alternative iterations on model and hyper_parameters
        print(f'Start to alternative iterations for {2 * alternative_times} times')
        for flag in range(int(2 * alternative_times)):
            
            # test first
            self._load_parameters(dataset_type, param_path, model_path)
            test_acc, test_loss = self.test(test_set, data_base, self.hyper_param, with_collaboration)
            print(test_acc.numpy().mean(axis=-1))


            best_test_loss = test_loss
            last_test_loss = test_loss
            patience = 5
            trigger_times = 0
            
            # flag = 1

            optim_target = 'hyper_params' if flag % 2 == 0 and with_collaboration else 'backbone'

            print(f'{flag}th iteration, iteration target: {optim_target}')

            param_load_path = param_path if flag == 0 else new_param_path
            model_load_path = model_path if flag == 0 else new_model_path

            self._load_parameters(dataset_type, param_load_path, model_load_path)

            # set learning modules
            lambdas, params, optimizer = self._init_train_modules(flag, with_collaboration, lr)
            print('Optimizer prepared...')

            for epoch in range(epochs):
                curr_index, batch_num = torch.randperm(train_size), int((train_size - 1) / batch_size) + 1
                loss = 0
                for batch_id in range(1):

                    batch_index = curr_index[batch_id * batch_size: batch_id * batch_size + batch_size]
                    t1 = time.time()
                    batch_loss = 0

                    optimizer.zero_grad()
                    for index in batch_index:
                        data = data_base.get_data(train_set[index], input_mode=True, gt_mode=True)

                        out_loss, out_acc, _ = self._forward_for_one_sample(data, train_set[index], lambdas, with_collaboration)

                        batch_loss = batch_loss + out_loss
                    batch_loss.backward(retain_graph=True)
                    t2 = time.time()
                    if batch_id + 1 == batch_num:
                        print('Mode: %s, Current batch: %s, batch loss: %.3f, Cost %.3f seconds' %
                              (flag % 2, batch_id, batch_loss.detach().item(), t2 - t1), end='\n')
                    else:
                        print('Mode: %s, Current batch: %s, batch loss: %.3f, Cost %.3f seconds' %
                              (flag % 2, batch_id, batch_loss.detach().item(), t2 - t1), end='\r')
                    optimizer.step()
                    loss = loss + batch_loss.detach()
                loss = loss / train_size
                test_acc, test_loss = self.test(test_set, data_base, params, with_collaboration)
                print('Mode: %s, Epoch: %s, Cross entropy loss: %.3f' % (flag % 2, epoch + 1, loss.detach().item()))
                print(np.around(params.detach().cpu(), 3))
                print('Test acc: ')
                print(np.around(test_acc.numpy().mean(axis=-1), 3))

                if test_loss > last_test_loss:
                    trigger_times += 1
                    print('Trigger times: ', trigger_times)

                    if trigger_times >= patience:
                        print('Early stopping!')
                        break
                else:
                    print('Trigger times: 0')
                    trigger_times = 0
                    if test_loss <= best_test_loss:
                        lambdas = torch.relu(params) + 1e-5
                        self.save_model(new_param_path, lambdas, new_model_path)
                        best_test_loss = test_loss
                        print('Best model saved, test loss: %.4f' % best_test_loss)

                last_test_loss = test_loss

            if not with_collaboration:
                break
            
            break
        print('Training finished...')

    def test(self, test_set, data_base, params, with_collaboration):
        total_acc = torch.zeros(self.T, self.agent_num)
        total_loss = 0
        test_size = len(test_set)

        with torch.no_grad():
            lambdas = torch.relu(params) + 1e-5
            for index in range(test_size):
                # print(index)
                test_data = data_base.get_data(test_set[index], input_mode=True, gt_mode=False)

                out_loss, out_acc, _ = self._forward_for_one_sample(test_data, test_set[index], lambdas, with_collaboration)
                total_acc += out_acc
                total_loss += out_loss
                # print(out_acc)
        return total_acc / test_size, total_loss / test_size

    def _forward_for_one_sample(self, data, datainfo, param, with_collaboration=True, graph_alg='dual_ascend'):

        agent_num = len(data)
        assert agent_num == self.agent_num

        hessian_sum = [torch.zeros(self.param_dim, self.param_dim, device=self.device, dtype=self.float_type)
                       for _ in range(agent_num)]
        hessian_alpha_sum = [torch.zeros(self.param_dim, device=self.device, dtype=self.float_type)
                             for _ in range(agent_num)]
        gradient = [torch.zeros(self.param_dim, device=self.device, dtype=self.float_type) for _ in range(agent_num)]

        loss = 0
        acc_record = torch.zeros(self.T, self.agent_num)
        adj_matrix_record = torch.zeros(self.T, self.agent_num, self.agent_num, dtype=self.float_type)
        for t in range(self.T):

            param_t = param

            for j in range(agent_num):

                train_data, gt_data = data[j]
                current_train, labels = train_data
                # gt_train, gt_labels = gt_data

                train_x, Y = current_train[t].to(self.device), \
                    F.one_hot(labels[t], num_classes=self.classes).to(self.device)

                X = self.transform(train_x)
                alpha, hessian, grad = self.local_learning(X, Y, iternum=0)

                hessian_sum[j] = (t * hessian_sum[j] + hessian) / (t + 1)
                hessian_alpha_sum[j] = (t * hessian_alpha_sum[j] + hessian @ alpha) / (t + 1)
                gradient[j] = (t * gradient[j] + grad) / (t + 1)

            h, h_a,  grad_t = torch.stack(hessian_sum, dim=0), torch.stack(hessian_alpha_sum, dim=0), torch.stack(
                gradient, dim=0)

            theta_record = torch.bmm(torch.linalg.inv(h), (- grad_t).unsqueeze(-1)).squeeze(-1)

            if with_collaboration:

                if graph_alg == 'oracle':
                    adj_matrix = self._graph_learning(graph_alg, datainfo=datainfo)
                else:
                    adj_matrix = self._graph_learning(graph_alg, thetas=theta_record, param=param_t)
                                
                adj_matrix = adj_matrix.to(self.float_type)

                theta_record = self._update_beta_with_w(theta_record, adj_matrix, h, h_a - grad_t, param_t)

                adj_matrix_record[t] = adj_matrix

            for j in range(self.agent_num):
                gt_data = data[j][1]
                gt_train, gt_labels = gt_data
                current_img, current_label = gt_train[t], gt_labels[t]
                current_loss, current_acc = self._loss(theta_record[j], current_img, current_label)
                acc_record[t][j] = current_acc
                loss = loss + current_loss

        loss = loss / self.T / self.agent_num

        return loss, acc_record, adj_matrix_record


class ClassificationResultDisplay(CollaborativeLearning):
    def __init__(self, agent_num, n_classes, feature_dim, T, float_type):
        super(ClassificationResultDisplay, self).__init__(agent_num, n_classes, feature_dim, T, float_type)

    def generate_local_model(self, mode):
        backbone_list, fc_list = [], []
        for i in range(self.agent_num):
            backbone = init_model(mode, self.feature_dim-1, self.device, self.float_type)
            fc = nn.Linear(in_features=self.feature_dim-1, out_features=self.classes, device=self.device)
            backbone_list.append(backbone)
            fc_list.append(fc)
        return backbone_list, fc_list

    def no_collab_no_memory(self, data, dataset_type):
        agent_num = len(data)
        acc_record = torch.zeros(self.T, self.agent_num)
        mode = 0 if dataset_type == 'mnist' else 1
        total_loss = 0
        adj_matrix = torch.zeros(self.T, self.agent_num, self.agent_num)
        for t in range(self.T):
            backbone_list, fc_list = self.generate_local_model(mode=mode)
            for j in range(agent_num):
                optimizer = torch.optim.Adam(list(backbone_list[j].parameters()) +
                                             list(fc_list[j].parameters()), lr=1e-3)
                train_data, gt_data = data[j]
                agent_train, agent_label = train_data
                current_train, labels = agent_train[t].to(self.device), agent_label[t].to(self.device)
                # start to train
                MAX_EPOCH = 10
                backbone_list[j].train()
                fc_list[j].train()
                for epoch in range(MAX_EPOCH):
                    optimizer.zero_grad()
                    feature = backbone_list[j](current_train)
                    logits = fc_list[j](feature)

                    loss = self.criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                # evaluation
                backbone_list[j].eval()
                fc_list[j].eval()
                gt_train, gt_labels = gt_data
                current_img, current_label = gt_train[t].to(self.device), gt_labels[t].to(self.device)

                with torch.no_grad():
                    output = fc_list[j](backbone_list[j](current_img))
                    pred_label = torch.max(torch.softmax(output, dim=1), 1)[1]
                    current_acc = torch.sum(pred_label == current_label) / len(current_label)
                    test_loss = self.criterion(output, current_label)
                    total_loss += test_loss.cpu()

                acc_record[t][j] = current_acc.cpu()

        return total_loss/self.T/self.agent_num, acc_record, adj_matrix

    def collab_no_memory(self, data, param, data_info, method='dual_ascend'):
        # makesure the model parameters are loaded
        self.phi.eval()
        agent_num = len(data)

        assert agent_num == self.agent_num

        hessian_sum = [torch.zeros(self.param_dim, self.param_dim, device=self.device, dtype=self.float_type)
                       for _ in range(agent_num)]
        hessian_alpha_sum = [torch.zeros(self.param_dim, device=self.device, dtype=self.float_type)
                             for _ in range(agent_num)]
        gradient = [torch.zeros(self.param_dim, device=self.device, dtype=self.float_type) for _ in range(agent_num)]

        loss = 0
        acc_record = torch.zeros(self.T, self.agent_num)
        adj_matrix_record = []
        for t in range(self.T):

            param_t = param

            for j in range(agent_num):
                train_data, gt_data = data[j]
                current_train, labels = train_data
                # gt_train, gt_labels = gt_data

                train_x, Y = current_train[t].to(self.device), \
                             F.one_hot(labels[t], num_classes=self.classes).to(self.device)

                X = self.transform(train_x)
                alpha, hessian, grad = self.local_learning(X, Y, iternum=0)

                hessian_sum[j], hessian_alpha_sum[j], gradient[j] = hessian, hessian @ alpha, grad

            h, h_a, grad_t = torch.stack(hessian_sum, dim=0), torch.stack(hessian_alpha_sum, dim=0), torch.stack(
                gradient, dim=0)

            theta_record = torch.bmm(torch.linalg.inv(h), (- grad_t).unsqueeze(-1)).squeeze(-1)

            if method == 'oracle':
                adj_matrix = self._graph_learning(method, datainfo=data_info)
            else:
                adj_matrix = self._graph_learning(method, thetas=theta_record, 
                                                 param=param_t, 
                                                 )

            theta_record = self._update_beta_with_w(theta_record, adj_matrix, h, h_a - grad_t, param_t)

            adj_matrix_record.append(adj_matrix)

            for j in range(self.agent_num):
                gt_data = data[j][1]
                gt_train, gt_labels = gt_data
                current_img, current_label = gt_train[t], gt_labels[t]
                current_loss, current_acc = self._loss(theta_record[j], current_img, current_label)
                acc_record[t][j] = current_acc
                loss = loss + current_loss

        loss = loss / self.T / self.agent_num
        adj_matrix_record = torch.stack(adj_matrix_record)
        return loss, acc_record, adj_matrix_record

    def no_collab_memory(self, data, datainfo, param):
        # makesure model parameters are loaded
        self.phi.eval()
        return self._forward_for_one_sample(data, datainfo, param, with_collaboration=False)

    def collab_memory(self, data, param, datainfo, method):
        # makesure model parameters are loaded
        self.phi.eval()
        return self._forward_for_one_sample(data, datainfo, param, with_collaboration=True, graph_alg=method)

    def forward(self, data_set, data_base, params, with_collaboration, with_memory, dataset_type, method):
        # makesure model parameters are loaded

        total_acc = torch.zeros(self.T, self.agent_num)
        total_loss = 0
        set_size = len(data_set)
        adj_res = []

        for index in range(set_size):
            data = data_base.get_data(data_set[index], input_mode=True, gt_mode=False)

            if not with_collaboration and not with_memory:
                out_loss, out_acc, out_adj = self.no_collab_no_memory(data, dataset_type)
            else:
                with torch.no_grad():
                    lambdas = torch.relu(params) + 1e-5
                    if with_collaboration and with_memory:
                        out_loss, out_acc, out_adj = self.collab_memory(data, lambdas,  data_set[index], method)
                    elif with_collaboration and not with_memory:
                        out_loss, out_acc, out_adj = self.collab_no_memory(data, lambdas, data_set[index], method)
                    elif not with_collaboration and with_memory:
                        out_loss, out_acc, out_adj = self.no_collab_memory(data, lambdas, data_set[index])

            total_acc += out_acc
            total_loss += out_loss
            adj_res.append(out_adj.cpu())

        return total_acc / set_size, total_loss / set_size, torch.stack(adj_res)

    def result(self, dataset_type, hyper_param_path, model_path, 
               with_collaboration, with_memory, 
               method, graph_hyper=0.1):

        test_path = 'dataset/%s_test_samples.pkl' % dataset_type

        with open(test_path, 'rb') as f:
            test_set = pickle.load(f)

        data_base = Classification_Scheduler(time=self.T, agent_num=self.agent_num, dataset_type=dataset_type)

        self._load_parameters(dataset_type, hyper_param_path, model_path)

        self.graph_hyper = graph_hyper

        acc, loss, adj = self.forward(test_set, 
                                      data_base, 
                                      torch.relu(self.hyper_param),
                                      with_collaboration, 
                                      with_memory, 
                                      dataset_type, 
                                      method
                                      )

        return acc, loss, adj


