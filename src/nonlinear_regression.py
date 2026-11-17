import pickle

import numpy as np
import time
import torch
from utils import NonLinearTransform
import torch.nn as nn
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from tqdm import tqdm
try:
    import seaborn
except ImportError:
    seaborn = None
from graph_algorithms import *

torch.manual_seed(511)
torch.set_default_dtype(torch.float64)


class NonLinearUnroll:
    def __init__(self, 
                 unroll_type=0, 
                 device=0,
                 trainset_path='dataset/train_nonlinear.pkl',
                 testset_path='dataset/test_nonlinear.pkl',
                 devset_path='dataset/dev_nonlinear.pkl'):
        
        # initialize model parameters (lambdas and phi)
        self.inter_dim, self.outdim, self.lambda0 = 100, 10, 1e-4
        self.hyper_param = np.random.randn(3)
        assert device == 'cpu' or device in [i for i in range(8)]
        self.device = torch.device(device) if device == 'cpu' else torch.device(f'cuda:{device}')
        self.phi = NonLinearTransform(self.inter_dim, self.outdim).to(self.device).double()

        # set the number of agents, time and type of unrolling        
        self.agent_num, self.T = 6, 10
        self.unroll_type = unroll_type
        
        # identify the dataset path
        self.train_path = trainset_path
        self.test_path = testset_path
        self.dev_path = devset_path

    def _load_model(self, model_path, param_path):
        state_dic = torch.load(model_path, map_location=self.device, weights_only=False)
        self.phi.load_state_dict(state_dic)
        param = np.load(param_path)
        self.hyper_param = param

    def _transform(self, x):
        try:
            x_trans = self.phi(x.to(torch.float64).unsqueeze(-1))
        except:
            x_trans = self.phi(torch.tensor(x, dtype=torch.float64).unsqueeze(-1))
        X = torch.concat([x_trans, torch.ones(len(x_trans), 1, device=self.device)], dim=1)
        return X
    
    def _loss(self, data_samples, theta_s):
        assert len(data_samples) == self.agent_num
        assert len(theta_s) == self.agent_num

        loss = 0
        for j in range(self.agent_num):
            x_input, y_gt = torch.tensor(data_samples[j][:, 0], dtype=torch.float64, device=self.device), \
                        torch.tensor(data_samples[j][:, 1], dtype=torch.float64, device=self.device)

            X = self._transform(x_input)
            y_hat = X @ theta_s[j]

            loss = loss + torch.sum((y_hat - y_gt) ** 2) / len(y_gt)

        return loss
    
    def _model_update_with_collab(self, curr_record, adj_matrix, h, h_a, param):
        """Almost 10 iterations to converge"""
        agent_num, feature_dim = len(curr_record), len(curr_record[0])
        assert agent_num == self.agent_num
        lambda1, lambda2 = param[0], param[1]

        # perform message passing for 10 iterations
        for k in range(10):
            # prev = curr_record.detach().clone()

            b = lambda2 * adj_matrix @ curr_record + h_a

            identity = torch.repeat_interleave(torch.eye(feature_dim, device=self.device).unsqueeze(0),
                                               agent_num, dim=0)

            degree_vec = torch.sum(adj_matrix, dim=1).unsqueeze(-1).unsqueeze(-1)

            diffusion_matrix = h + identity * (lambda2 * degree_vec + lambda1)

            curr_record = torch.bmm(torch.linalg.inv(diffusion_matrix), b.unsqueeze(-1)).squeeze(-1)

        return curr_record
    

    def _local_learning(self, x, y):
        X = self._transform(x)

        alpha = torch.linalg.inv(
            X.T @ X + self.lambda0 * len(x) * torch.eye(self.outdim + 1, device=self.device)) \
                @ X.T @ y
        hessian = 2 / len(x) * X.T @ X + 2 * self.lambda0 * torch.eye(self.outdim + 1, device=self.device)

        return alpha, hessian


    def _forward_for_one_sample(self, sample, param):

        data_sample = sample['data']
        ground_truth = sample['gt']

        agent_num, T = len(data_sample), data_sample[0].shape[0]
        # type 0: recurrent unrolling, type 1: use different param at each time
        if self.unroll_type == 1:
            assert param.shape[0] == T
        assert agent_num == self.agent_num
        assert T == self.T
        assert self.unroll_type in [0, 1]

        hessian_sum = [torch.zeros(self.outdim + 1, self.outdim + 1, device=self.device, dtype=torch.float64)
                       for _ in range(agent_num)]
        hessian_alpha_sum = [torch.zeros(self.outdim + 1, device=self.device, dtype=torch.float64)
                             for _ in range(agent_num)]
        
        loss = 0

        for t in range(T):

            param_t = param if self.unroll_type == 0 else param[t]

            for j in range(agent_num):
                x, y = torch.tensor(data_sample[j][t][:, 0], dtype=torch.float64, device=self.device), \
                    torch.tensor(data_sample[j][t][:, 1], dtype=torch.float64, device=self.device)
                
                alpha, hessian = self._local_learning(x, y)

                hessian_sum[j] = hessian_sum[j] + hessian
                hessian_alpha_sum[j] = hessian_alpha_sum[j] + hessian @ alpha.to(torch.double)

            h, h_a = torch.stack(hessian_sum, dim=0) / (t + 1), torch.stack(hessian_alpha_sum, dim=0) / (t + 1)

            theta_record = torch.bmm(torch.linalg.inv(h), h_a.unsqueeze(-1)).squeeze(-1)

            for k in range(1):
                adj_matrix = dual_ascent(beta_s=theta_record, param=param_t)

                theta_record = self._model_update_with_collab(theta_record, adj_matrix, h, h_a, param_t)

            loss = loss + self._loss(ground_truth, theta_record)

        loss = loss / T / agent_num
        return loss

    def test(self, param):
        with open(self.test_path, 'rb') as f:
            test_set = pickle.load(f)
        with torch.no_grad():
            loss = 0
            for i in range(len(test_set)):
                out = self._forward_for_one_sample(test_set[i], param)
                loss = loss + out
        return loss / len(test_set)

    def train(self, 
              param_save_path, 
              phi_save_path, 
              loss_record_path, 
              param_record_path,
              phi_lr, 
              param_lr, 
              epochs=1000, 
              batch_size=16,
              param_init_path=None,
              model_init_path=None
              ):
        # initialize parameters 
        if param_init_path and model_init_path:
            self._load_model(model_init_path, param_init_path)

        # Get learnable parameters
        if self.unroll_type == 0:
            params = nn.Parameter(torch.tensor(self.hyper_param.copy()).to(self.device), requires_grad=True)
        elif self.unroll_type == 1:
            tmp = np.ones((self.T, 1)) @ self.hyper_param.copy()[None]
            params = nn.Parameter(torch.tensor(tmp).to(self.device), requires_grad=True)

        best_test_loss = 100
        last_test_loss = 100
        patience = 5
        trigger_times = 0
        loss_record = []

        optimizer = torch.optim.Adam([
            {'params': params, 'lr': param_lr},
            {'params': self.phi.parameters(), 'lr': phi_lr}
        ])

        with open(self.train_path, 'rb') as f:
            trainset = pickle.load(f)

        # record parameters in training
        assert self.unroll_type in [0, 1]
        param_record = np.zeros((3, epochs)) if self.unroll_type == 0 else np.zeros((self.T, 3, epochs))

        for epoch in range(epochs):
            loss = 0
            t1 = time.time()
            # Shuffle
            curr_index, batch_num = torch.randperm(len(trainset)), int((len(trainset) - 1) / batch_size) + 1

            for batch_id in range(batch_num):
                t_t0 = time.time()
                # Current batch
                batch_index = curr_index[batch_id * batch_size: batch_id * batch_size + batch_size]

                batch_loss = 0
                optimizer.zero_grad()

                # Forward
                lambdas = torch.relu(params) + 1e-5
                for index in batch_index:
                    out = self._forward_for_one_sample(trainset[index], lambdas)
                    batch_loss = batch_loss + out
                batch_loss.backward()

                # Clip gradient
                nn.utils.clip_grad_norm_([params], max_norm=20, norm_type=2)

                optimizer.step()

                # print('Batch Loss: %.4f, Batch ID: %s' % (batch_loss.detach(), batch_id))

                t_t1 = time.time()
                # print('Grad:\t', np.around(params.grad.to('cpu').numpy(), 3))
                # print('Time: %.3f' % (t_t1 - t_t0))

                loss = loss + batch_loss.detach()
            test_loss, loss = self.test(torch.relu(params)), loss / len(trainset)
            t2 = time.time()
            print('Epoch: %s, Loss: %.4f, Test Loss: %.4f, Time: %.3f\n' % (epoch + 1, loss, test_loss, t2 - t1))
            lambdas = torch.relu(params).detach() + 1e-5
            if self.unroll_type == 0:
                print('Lambda1: %.4f, Lambda2: %.4f, Lambda3: %.4f\n' % (lambdas[0], lambdas[1], lambdas[2]))
            else:
                print(np.around(lambdas.to('cpu').numpy(), 3))

            # record model parameters
            if self.unroll_type == 0:
                param_record[:, epoch] = lambdas.to('cpu').numpy()
            elif self.unroll_type == 1:
                param_record[:, :, epoch] = lambdas.to('cpu').numpy()

            # decide early stopping and save check points
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
                    self.save_model(param_save_path, lambdas, phi_save_path)
                    best_test_loss = test_loss
                    print('Best model saved, test loss: %.4f' % best_test_loss)

            last_test_loss = test_loss

            loss_record.append(np.asarray([loss.detach().to('cpu'), test_loss.to('cpu')]))

        record = np.stack(loss_record)
        np.save(loss_record_path, record)
        np.save(param_record_path, param_record)
        print('Training finished...')

    def search_params(self, n):
        """Search the best initialization of hyper parameters"""
        param_matrix = torch.rand((n, 3)) * 0.1
        best_index = -1
        best_loss = 1e10

        with open(self.test_path, 'rb') as f:
            test_set = pickle.load(f)

        for i in range(param_matrix.shape[0]):
            with torch.no_grad():
                loss = 0
                for j in range(len(test_set)):
                    out = self._forward_for_one_sample(test_set[j], param_matrix[i])
                    loss = loss + out
            print('params:')
            print(param_matrix[i])
            print('loss: %.3f' % (loss / len(test_set)))
            if loss < best_loss:
                best_loss = loss
                best_index = i

        np.save('init_param.npy', param_matrix[best_index].numpy())

    def save_model(self, hyper_param_path, param, model_path=None):
        np.save(hyper_param_path, param.to('cpu').detach().numpy())
        torch.save(self.phi.state_dict(), model_path)


class RegressionResultDisplay(NonLinearUnroll):
    METHODS = [
            'admm', 'pds', 'dual_ascend', 'dual_ascend_distributed',
            'mtrl', 'glasso', 'gl_sigrep', 'gl_logdet', 'neighbor_select'
        ]
    SETTINGS = ['colla_mem', 'nocolla_mem', 'colla_nomem']
    GL_FUNC = {
            'admm': admm, 'pds':pds, 'dual_ascend': dual_ascent, 'dual_ascend_distributed': distributed_dual_ascent,
            'mtrl': DMTL, 'glasso': glasso, 'gl_sigrep': gl_sigrep, 'gl_logdet':gl_logdet, 'neighbor_select':neighbor_select
        }

    def __init__(self, setting='colla_mem', method='dual_ascend', unroll_type=0, device=0):
        super(RegressionResultDisplay, self).__init__(
            unroll_type=unroll_type,
            device=device,
            trainset_path='dataset/train_nonlinear.pkl',
            testset_path='dataset/test_nonlinear.pkl',
            devset_path='dataset/dev_nonlinear.pkl'
            )
        self.method = method
        self.setting = setting
        
        assert self.method in RegressionResultDisplay.METHODS
        assert self.setting in RegressionResultDisplay.SETTINGS

        self.with_collaboration = True if self.setting in ['colla_mem', 'colla_nomem'] else False
        self.with_memory = True if self.setting in ['colla_mem', 'nocolla_mem'] else False
        
        self.gl_methods = RegressionResultDisplay.GL_FUNC

        self.graph_param = 0.1


    # def load_model(self, model_path, param_path):
    #     state_dic = torch.load(model_path, map_location=self.device)
    #     self.phi.load_state_dict(state_dic)
    #     param = np.load(param_path)
    #     self.hyper_param = param

    def dataset_visualization(self, sample_id, agent_id, file_path):

        with open(self.test_path, 'rb') as f:
            test_set = pickle.load(f)

        sample = test_set[sample_id]

        data_sample = sample['data']
        ground_truth = sample['gt']
        x_min, x_max = min(ground_truth[agent_id][:, 0]), max(ground_truth[agent_id][:, 0])
        y_min, y_max = min(ground_truth[agent_id][:, 1]), max(ground_truth[agent_id][:, 1])

        agent_num, T = len(data_sample), data_sample[0].shape[0]
        T = 5

        # type_agent = torch.zeros(agent_num, dtype=torch.int)
        # group_id = torch.zeros(agent_num, dtype=torch.int)
        # for j in range(len(sample['partition'])):
        #     for k, agent in enumerate(sample['partition'][j]):
        #         group_id[agent] = j
        #         type_agent[agent] = sample['agent_type'][j][k]
        seaborn.set_style('darkgrid')
        fig, ax = plt.subplots(1, 6, figsize=(60, 10))
        for t in range(T):
            for j in range(t + 1):
                x, y = data_sample[agent_id][j][:, 0], data_sample[agent_id][j][:, 1]
                ax[t].scatter(x, y, s=3)
            ax[t].set_xlim((x_min - 1, x_max + 1))
            ax[t].set_ylim((y_min - 1, y_max + 1))
            if t > 0:
                ax[t].yaxis.set_ticklabels([])
            ax[t].tick_params(axis='both', which='major', labelsize=40)
            ax[t].set_title('T=%s' % (t + 1), fontsize=50)

        ax[T].set_xlim((x_min - 1, x_max + 1))
        ax[T].set_ylim((y_min - 1, y_max + 1))
        ax[T].yaxis.set_ticklabels([])
        ax[T].scatter(ground_truth[agent_id][:, 0], ground_truth[agent_id][:, 1], s=0.5)
        ax[T].tick_params(axis='both', which='major', labelsize=40)
        ax[T].set_title('Ground truth function', fontsize=50)
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight')

    def graph_learning(self, beta_s, params=None, *args):
        assert self.method in ['admm', 'pds', 'dual_ascend',
                               'dual_ascend_distributed', 'mtrl', 'glasso',
                               'gl_sigrep', 'gl_logdet', 'neighbor_select']
        
        no_hyperparam_methods = ['admm', 'pds', 'mtrl']
        one_hyperparam_methods = ['gl_sigrep', 'gl_logdet', 'neighbor_select', 'glasso']

        if self.method in no_hyperparam_methods:
            return torch.relu(self.gl_methods[self.method](beta_s)[0])
        elif self.method in one_hyperparam_methods:
            return torch.relu(self.gl_methods[self.method](beta_s, self.graph_param)[0])

        elif self.method == 'dual_ascend':
            if len(args) >= 1:
                communication_graph = args[0]
                return dual_ascent(beta_s, params, communication_graph)[0]
            else:
                adj = dual_ascent(beta_s, params)[0]
                # print(np.around(adj.to('cpu').numpy(), 1))
                return adj
        
        elif self.method == 'dual_ascend_distributed':
            if len(args) >= 1:
                communication_graph = args[0]
                return distributed_dual_ascent(beta_s, 
                                               params, 
                                               torch.ones(self.agent_num, device=self.device), 
                                               communication_graph
                                            )[0]
            else:
                return distributed_dual_ascent(beta_s, 
                                               params, 
                                               torch.ones(self.agent_num, device=self.device)
                                            )[0]


    def _forward_for_one_sample(self, sample, param, *args):

        data_sample = sample['data']
        ground_truth = sample['gt']

        agent_num, T = len(data_sample), data_sample[0].shape[0]
        if self.unroll_type == 1:
            assert param.shape[0] == T
        assert agent_num == self.agent_num
        assert T == self.T

        hessian_sum = [torch.zeros(self.outdim + 1, self.outdim + 1, device=self.device, dtype=torch.float64)
                       for _ in range(agent_num)]
        hessian_alpha_sum = [torch.zeros(self.outdim + 1, device=self.device, dtype=torch.float64)
                             for _ in range(agent_num)]

        loss_record, adj_matrix_record = torch.zeros(T), torch.zeros(T, self.agent_num, self.agent_num)
        func_plot = torch.zeros(T, self.agent_num, ground_truth[0].shape[0], 2)

        for t in range(T):

            if self.unroll_type == 0:
                param_t = param
            elif self.unroll_type == 1:
                param_t = param[t]

            
            for j in range(agent_num):
                x, y = torch.tensor(data_sample[j][t][:, 0], dtype=torch.float64, device=self.device), \
                    torch.tensor(data_sample[j][t][:, 1], dtype=torch.float64, device=self.device)

                alpha, hessian = self._local_learning(x, y)
                if self.with_memory:
                    hessian_sum[j] = hessian_sum[j] + hessian
                    hessian_alpha_sum[j] = hessian_alpha_sum[j] + hessian @ alpha.to(torch.double)
                else:
                    hessian_sum[j] = hessian
                    hessian_alpha_sum[j] = hessian @ alpha.to(torch.double)

            h, h_a = torch.stack(hessian_sum, dim=0) / (t + 1), torch.stack(hessian_alpha_sum, dim=0) / (t + 1)
            
            theta_record = torch.bmm(torch.linalg.inv(h), h_a.unsqueeze(-1)).squeeze(-1)
            
            if self.with_collaboration:
                for k in range(1):
                    if len(args)>=1:
                        communication_graph = args[0]
                        adj_matrix = self.graph_learning(theta_record, param_t, communication_graph)
                    else:
                        adj_matrix = self.graph_learning(beta_s=theta_record, params=param_t)
                    theta_record = self._model_update_with_collab(theta_record, adj_matrix, h, h_a, param_t)
                adj_matrix_record[t] = adj_matrix

            for j in range(agent_num):
                x_input, y_gt = torch.tensor(ground_truth[j][:, 0], dtype=torch.float64, device=self.device), \
                    torch.tensor(ground_truth[j][:, 1], dtype=torch.float64, device=self.device)

                X = self._transform(x_input)
                y_hat = X @ theta_record[j]

                loss_record[t] = loss_record[t] + torch.sum((y_hat - y_gt) ** 2) / len(y_gt)
                func_plot[t][j][:, 0] = x_input
                func_plot[t][j][:, 1] = y_hat

            

        loss_record = loss_record / agent_num
        return loss_record, adj_matrix_record, func_plot

    def regression_res_display(self, model_path, hyper_param_path, graph_param=0.1, *args):
        self._load_model(model_path, hyper_param_path)
        with open(self.test_path, 'rb') as f:
            test_set = pickle.load(f)

        print(len(test_set))
        self.graph_param = graph_param

        param = torch.tensor(self.hyper_param, dtype=torch.float64)

        if len(args) >= 1:
            communication_graph_path = args[0]
            communication_graph_set = torch.tensor(np.load(communication_graph_path), device=self.device)

        with torch.no_grad():
            loss = torch.zeros(self.T)
            GMSE = torch.zeros(self.T)
            for i in tqdm(range(len(test_set))):
                if len(args)>=1:
                    communication_graph = communication_graph_set[i]
                    out, adj, _ = self._forward_for_one_sample(test_set[i], param, communication_graph)
                else:
                    out, adj, _ = self._forward_for_one_sample(test_set[i], param)
                partition = test_set[i]['partition']
                gt_adj = torch.zeros(self.agent_num, self.agent_num)
                for group in partition:
                    for k in range(len(group)):
                        for j in range(k+1, len(group)):
                            gt_adj[group[k]][group[j]] = 1
                gt_adj = (gt_adj + gt_adj.T)
                gt_adj = 30 * gt_adj / torch.sum(gt_adj)

                diff = (adj - gt_adj)
                gmse = torch.sum(torch.sum(diff**2, dim=-1), dim=-1)/torch.norm(gt_adj)**2

                GMSE = GMSE + gmse
                loss = loss + out
        return loss / len(test_set), GMSE / len(test_set)

    def regression_colla_graph_display(self, model_path, hyper_param_path, sample_index, save_name):
        self._load_model(model_path, hyper_param_path)
        with open(self.test_path, 'rb') as f:
            test_set = pickle.load(f)

        param = torch.tensor(self.hyper_param, dtype=torch.float64)

        with torch.no_grad():
            _, adj_matrix, _ = self._forward_for_one_sample(test_set[sample_index], param)

        adj_matrix = adj_matrix.cpu().numpy()

        return adj_matrix

    def regression_res_visualization(self, model_path, hyper_param_path, sample_index, file_path):
        self._load_model(model_path, hyper_param_path)
        with open(self.test_path, 'rb') as f:
            test_set = pickle.load(f)

        param = torch.tensor(self.hyper_param, dtype=torch.float64)

        with torch.no_grad():
            _, _, func_data = self._forward_for_one_sample(test_set[sample_index], param)

        T = 5
        seaborn.set_style('darkgrid')
        fig, ax = plt.subplots(1, 5, figsize=(50, 10))
        for t in range(T):
            for j in range(self.agent_num):
                x, y = func_data[t][j][:, 0], func_data[t][j][:, 1]
                ax[t].plot(x, y, linewidth=3)

            if t > 0:
                ax[t].yaxis.set_ticklabels([])
            ax[t].tick_params(axis='both', which='major', labelsize=40)
            ax[t].set_title('T=%s' % (t + 1), fontsize=50)

        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight')


if __name__ == '__main__':



    # solver = NonLinearUnroll(hyper_param=np.asarray([0.0, 0.01, 1.0]), unroll_type=1, is_pretrain=True,
    #                          model_path='model/learned_phi_type1.pt')
    # solver.train('param/hyper_param_type1.npy', 'model/learned_phi_type1.pt', 'test.npy', 'param_record_test.npy',
    #              phi_lr=1e-3, param_lr=1e-3, epochs=100, batch_size=16)
    # solver.search_params()

    # hyper_param = np.load('param/init_param.npy')
    # hyper_param = np.load('param/Param_params_lr=0.0001_phi_lr=0.001_type=0_Fri_Nov_18_02_44_13_2022.npy')
    # print(hyper_param)
    diplayer = RegressionResultDisplay(setting='colla_nomem', method='dual_ascend', device=6)

    # diplayer.dataset_visualization(2, 3, file_path='figure/regression_dataset2.pdf')
    # param_path = 'param/init_param.npy'
    # model_path = 'model/learned_phi_type1.pt'
    # param_path = 'param/Param_params_lr=0.0001_phi_lr=0.001_type=0_Fri_Nov_18_02_44_13_2022.npy'
    # model_path = 'model/Phi_params_lr=0.0001_phi_lr=0.001_type=0_Fri_Nov_18_02_44_13_2022.pt'
    param_path = 'param/Param_params_lr=0.0001_phi_lr=0.001_type=0_Mon_Nov_14_22_20_59_2022.npy'
    model_path = 'model/Phi_params_lr=0.0001_phi_lr=0.001_type=0_Mon_Nov_14_22_20_59_2022.pt'

    communicate_path = "dataset/communication_graph_ER_high.npy"

    # diplayer.dataset_visualization(2, 1, file_path='figure/regression_dataset1.pdf')

    res, gmse = diplayer.regression_res_display(model_path, param_path)

#     # res, gmse = diplayer.regression_res_display(model_path, param_path, communicate_path)
    print(res)
    print(gmse)
    # np.save('result/regression/mse_mtrl_0.1.npy', res.numpy())
    # np.save('result/regression/gmse_mtrl_0.1.npy', gmse.numpy())
#     print(torch.mean(res), torch.mean(gmse))
#     # save_name = 'result_visualization/figs/regression_adj7.pdf'
#     # adj = diplayer.regression_colla_graph_display(model_path, param_path, sample_index=7, save_name=save_name)
#     # print(np.around(adj, 1))
#     # diplayer.regression_res_visualization(model_path, param_path, sample_index=7, file_path='figure/unroll_function7.pdf')

#sigrep 0.1
# tensor([5.0359, 2.1474, 0.8160, 0.3650, 0.2110, 0.1298, 0.1070, 0.0910, 0.0880, 0.0826])
# tensor([0.9714, 0.8921, 0.7932, 0.7402, 0.7101, 0.6963, 0.6859, 0.6787, 0.6753, 0.6727])

#sigrep 1
# tensor([3.9828, 1.4738, 0.6196, 0.2884, 0.1645, 0.1102, 0.0955, 0.0829, 0.0821, 0.0768])
# tensor([1.9323, 1.1623, 0.5145, 0.2510, 0.1318, 0.0742, 0.0484, 0.0299, 0.0253, 0.0178])

#sigrep 10
# tensor([2.4932, 0.6851, 0.2263, 0.1406, 0.1163, 0.0874, 0.0825, 0.0797, 0.0784, 0.0768])
# tensor([0.4166, 0.2410, 0.1034, 0.0522, 0.0350, 0.0255, 0.0239, 0.0232, 0.0230, 0.0228])

#sigrep 100
# tensor([2.5827, 0.9274, 0.4323, 0.2591, 0.1732, 0.1406, 0.1226, 0.1062, 0.0987, 0.0936])
# tensor([0.2232, 0.1795, 0.1591, 0.1516, 0.1480, 0.1468, 0.1464, 0.1462, 0.1461, 0.1458])

#sigrep 1000
# tensor([3.0294, 1.2622, 0.6884, 0.4478, 0.2985, 0.2258, 0.1911, 0.1528, 0.1380, 0.1287])
# tensor([0.2871, 0.2813, 0.2784, 0.2772, 0.2767, 0.2763, 0.2762, 0.2761, 0.2761, 0.2761])

#glasso alpha=1e-3
# tensor([3.0270, 1.0787, 0.4884, 0.2570, 0.1802, 0.1184, 0.1031, 0.0874, 0.0843, 0.0813])
# tensor([1.6159, 0.6569, 0.4328, 0.3761, 0.3388, 0.3188, 0.3100, 0.3020, 0.2981, 0.2954])

#glasso alpha=1e-4
# tensor([3.0318, 1.0794, 0.4891, 0.2574, 0.1805, 0.1185, 0.1032, 0.0874, 0.0844, 0.0813])
# tensor([1.8628, 0.6562, 0.4333, 0.3765, 0.3390, 0.3190, 0.3101, 0.3022, 0.2982, 0.2955])

#glasso alpha=1e-2
# tensor([2.9889, 1.0778, 0.4818, 0.2530, 0.1770, 0.1173, 0.1026, 0.0872, 0.0842, 0.0812])
# tensor([1.7994, 0.6231, 0.4291, 0.3732, 0.3366, 0.3169, 0.3083, 0.3006, 0.2969, 0.2940])

#glasso alpha=1e-1
# tensor([2.8378, 0.9339, 0.4070, 0.2086, 0.1430, 0.1082, 0.0965, 0.0847, 0.0826, 0.0804])
# tensor([1.1432, 0.7042, 0.4769, 0.3979, 0.3545, 0.3352, 0.3267, 0.3191, 0.3149, 0.3106])


#neighbor_select lambda=1
# tensor([5.7560e+22, 6.0421e+41, 3.8241e+07, 3.3705e-01, 1.7924e-01, 1.1083e-01, 9.3959e-02, 8.4427e-02, 8.2338e-02, 8.2650e-02])
# tensor([1.2173, 1.0900, 1.0045, 0.8675, 0.8066, 0.7440, 0.7163, 0.7022, 0.7071, 0.6870])

#neighbor_select lambda=0.1
# tensor([9.3941e+23, 3.8885e+19, 1.4229e+13, 4.0156e-01, 1.9729e-01, 1.2332e-01, 1.0322e-01, 8.9094e-02, 8.5620e-02, 8.2606e-02])
# tensor([1.3422, 1.2699, 1.3114, 1.2167, 1.0756, 1.0398, 1.0042, 0.9357, 0.9297, 0.8669])

#neighbor_select lambda=10
# tensor([7.5538e+25, 2.4181e+11, 3.3280e+07, 2.5205e-01, 8.2294e-01, 1.4414e-01, 1.5474e-01, 2.4983e-01, 1.0410e-01, 1.0792e-01])
# tensor([1.1137, 0.8925, 0.8339, 0.7856, 0.7711, 0.7686, 0.7699, 0.7570, 0.7675, 0.7603])

#neighbor_select lambda=100
# tensor([1.5248e+12, 6.8202e+12, 4.1394e+03, 1.0525e+00, 1.8278e+00, 2.0942e+00, 1.0172e-01, 9.2866e-02, 8.2494e-02, 8.7585e-02])
# tensor([1.4852, 1.1444, 1.0778, 0.9632, 0.9486, 0.9643, 0.9590, 0.9602, 0.9599, 0.9611])

#neighbor_select lambda=1000
# tensor([5.4457e+00, 1.7799e+14, 2.8424e+06, 3.6105e-01, 1.7256e-01, 2.4529e-01, 3.9433e-01, 1.2780e-01, 2.8656e-01, 2.1381e-01])
# tensor([1.0296, 1.0466, 1.0142, 1.0152, 1.0165, 1.0153, 1.0159, 1.0161, 1.0153, 1.0170])


#gl_logdet 0.1
# tensor([3.3207, 1.2730, 0.4477, 0.2093, 0.1387, 0.0952, 0.0831, 0.0764, 0.0741, 0.0716])
# tensor([ 23.6758,  25.7606,  36.4274,  56.5637,  68.0672,  79.5645,  88.8389, 99.8197, 103.4469, 107.8043])

#gl_logdet 1
# tensor([3.2395, 1.2609, 0.4537, 0.2139, 0.1416, 0.0974, 0.0847, 0.0776, 0.0751, 0.0726])
# tensor([1.0391, 0.9327, 0.8719, 0.9005, 0.8916, 0.8878, 0.8849, 0.9056, 0.8499, 0.8331])

#gl_logdet 10
# tensor([3.4031, 1.3876, 0.5151, 0.2431, 0.1560, 0.1060, 0.0901, 0.0808, 0.0777, 0.0748])
# tensor([0.8701, 0.8327, 0.7976, 0.7765, 0.7661, 0.7604, 0.7567, 0.7520, 0.7489, 0.7461])