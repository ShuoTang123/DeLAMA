import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import *
try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None
import shutil
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib.colors import ListedColormap

torch.set_default_dtype(torch.float64)


def around(x, y, map):
    feature = torch.ones(8)
    for k in range(9):
        x_axis, y_axis = int(k / 3) - 1, k % 3 - 1

        if x_axis == 0 and y_axis == 0:
            continue
        x_axis, y_axis = x + x_axis, y + y_axis
        if x_axis < 0 or x_axis >= map.shape[0] or y_axis < 0 or y_axis >= map.shape[1]:
            continue
        else:
            if k >= 4:
                index = k - 1
            else:
                index = k
            feature[index] = map[x_axis][y_axis]
    return feature


class MappingUnroll:
    def __init__(self, hyper_param, feature_dim, T, agent_num):
        self.hyper_param = hyper_param
        self.interdim, self.gama = 100, 0.1

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.agent_num, self.T = agent_num, T

        self.feature_dim = feature_dim + 1
        self.param_dim = int(self.feature_dim)
        self.criterion = nn.BCELoss()

        self.phi = UnLinearNetwork(10, self.interdim, feature_dim).to(self.device).double()

        self.train_set_path = 'dataset/sample_check_train.pkl'
        self.test_set_path = 'dataset/sample_check_test.pkl'

        self.show = False

    def transform(self, x):
        X = self.phi(x.to(torch.float64))
        X = torch.concat([X, torch.ones(len(X), 1, device=self.device)], dim=1)
        return X

    def first_order_derivative(self, X, Y, y_hat):
        tmp = X.T @ (y_hat - Y) / len(X)
        return torch.flatten(tmp.T)

    def second_order_derivative(self, X, y_hat):
        sub_matrix = X.T @ torch.diag(
            -y_hat[:, 0] * y_hat[:, 0] + y_hat[:, 0]) @ X / len(X)

        hessian = sub_matrix + 2 * self.gama * torch.eye(self.param_dim, device=self.device)
        return hessian

    def local_learning(self, X, Y):
        assert X.shape[1] == self.feature_dim
        assert Y.shape[1] == 1
        theta = torch.zeros(self.param_dim, device=self.device)

        y_hat = torch.sigmoid(X @ theta).unsqueeze(-1)
        # y_hat = torch.softmax(X @ M, dim=1)
        hessian = self.second_order_derivative(X, y_hat)
        grad = self.first_order_derivative(X, Y, y_hat)

        return theta, hessian, grad

    def forward_for_one_sample(self, data, lambdas, with_collaboration=True, is_train=True):
        hessian_sum = [torch.zeros(self.param_dim, self.param_dim, device=self.device, dtype=torch.float64)
                       for _ in range(self.agent_num)]
        hessian_alpha_sum = [torch.zeros(self.param_dim, device=self.device, dtype=torch.float64)
                             for _ in range(self.agent_num)]

        gradient = [torch.zeros(self.param_dim, device=self.device, dtype=torch.float64) for _ in range(self.agent_num)]

        loss = 0
        acc_record = torch.zeros(self.T, self.agent_num)
        adj_matrix_record = []
        height, width = data[0].room.shape[0], data[0].room.shape[1]
        mapping_result = torch.ones(self.T, self.agent_num, height, width)
        if with_collaboration:
            visited = torch.ones(height, width, 8)
        else:
            visited = torch.ones(self.agent_num, height, width, 8)

        for t in range(self.T):
            for j in range(self.agent_num):
                robot = data[j]

                pos_train, label = robot.data[t][:, 0:10], robot.data[t][:, 10]

                if self.show:
                    for point in pos_train[:, :2]:
                        coord_x, coord_y = int(point[0]), int(point[1])
                        try:
                            if with_collaboration:
                                visited[coord_x][coord_y] = around(coord_x, coord_y, robot.room)
                            else:
                                visited[j][coord_x][coord_y] = around(coord_x, coord_y, robot.room)
                        except:
                            pass
                pos_train = torch.tensor(pos_train, dtype=torch.float64)
                label = torch.tensor(label, dtype=torch.float64).unsqueeze(-1)

                train_x, Y = pos_train.to(self.device), \
                    label.to(self.device)

                X = self.transform(train_x)
                alpha, hessian, grad = self.local_learning(X, Y)

                hessian_sum[j] = (t * hessian_sum[j] + hessian) / (t + 1)
                hessian_alpha_sum[j] = (t * hessian_alpha_sum[j] + hessian @ alpha) / (t + 1)
                gradient[j] = (t * gradient[j] + grad) / (t + 1)

            h, h_a, grad_t = torch.stack(hessian_sum, dim=0), torch.stack(hessian_alpha_sum, dim=0), torch.stack(
                gradient, dim=0)

            theta_record = torch.bmm(torch.linalg.inv(h), (- grad_t).unsqueeze(-1)).squeeze(-1)

            if with_collaboration:
                adj_matrix = self.dual_ascent(beta_s=theta_record, param=lambdas)

                theta_record = self.update_beta_with_w(theta_record, adj_matrix, h, h_a - grad_t, lambdas)

                adj_matrix_record.append(adj_matrix)

            for j in range(self.agent_num):
                gt_pos, gt_label = data[j].gt[:, 0:10], data[j].gt[:, 10]

                gt_pos = torch.tensor(gt_pos, dtype=torch.float64)
                gt_label = torch.tensor(gt_label, dtype=torch.float64)

                current_loss, current_acc = self.loss(theta_record[j], gt_pos, gt_label)

                acc_record[t][j] = current_acc
                loss = loss + current_loss
                if self.show:
                    # print(theta_record[j])
                    x_range, y_range = torch.arange(0, width), torch.arange(0, height)
                    x_m, y_m = torch.meshgrid(x_range, y_range, indexing='xy')
                    index_feature = torch.concat([x_m.unsqueeze(-1), y_m.unsqueeze(-1)], dim=-1)

                    if with_collaboration:
                        total_feature = torch.concat([index_feature, visited], dim=-1)
                    else:
                        total_feature = torch.concat([index_feature, visited[j]], dim=-1)

                    for i in range(height):
                        input_feature = total_feature[i].to(self.device)

                        X = self.transform(input_feature)

                        y_pred = X @ theta_record[j]
                        p_hat = torch.sigmoid(y_pred)
                        mapping_result[t, j, i, :] = p_hat.cpu().detach()

        loss = loss / self.T / self.agent_num
        return loss, acc_record, adj_matrix_record, mapping_result

    def update_beta_with_w(self, curr_record, adj_matrix, h, h_a, param):
        """Almost 10 iterations to converge"""
        agent_num, feature_dim = len(curr_record), len(curr_record[0])
        assert agent_num == self.agent_num
        lambda1, lambda2 = param[0], param[1]

        for k in range(10):
            # prev = curr_record.detach().clone()

            b = lambda2 * adj_matrix @ curr_record + h_a

            identity = torch.repeat_interleave(torch.eye(feature_dim, device=self.device).unsqueeze(0),
                                               agent_num, dim=0)

            degree_vec = torch.sum(adj_matrix, dim=1).unsqueeze(-1).unsqueeze(-1)

            diffusion_matrix = h + identity * (lambda2 * degree_vec + lambda1)

            curr_record = torch.bmm(torch.linalg.inv(diffusion_matrix), b.unsqueeze(-1)).squeeze(-1)

            # curr = torch.stack(curr_record, dim=0).clone().detach()
            # curr = curr_record.detach().clone()
            # diff = torch.sum((prev - curr) ** 2)
            # print(diff, k)
        return curr_record

    def distance_vec(self, beta_s):
        agent_num = len(beta_s)

        beta_tmp = torch.repeat_interleave(beta_s.unsqueeze(-1), agent_num, dim=2)
        permuted_beta = torch.permute(beta_tmp, (2, 1, 0))
        diff_map = torch.sum((permuted_beta - beta_tmp) ** 2, dim=1) + 1e10 * torch.eye(agent_num, device=self.device)

        return diff_map

    def dual_ascent(self, beta_s, param):
        """
        Optimize graph structure by Newton method.
        :param beta_s: model parameters
        :param param: hyper parameter
        """
        agent_num = len(beta_s)
        assert agent_num == self.agent_num
        lambda2, lambda3 = param[1], param[2]
        w = torch.ones((agent_num, agent_num), device=self.device) - torch.eye(agent_num, device=self.device)
        edge_num = int(agent_num * (agent_num - 1) / 2)
        b = 1e-4
        y = self.distance_vec(beta_s)

        z = - lambda2 * torch.min(y)
        for k in range(10):
            u = (-1 / lambda3) * (lambda2 * y + z * torch.ones(agent_num, device=self.device))
            sp = (u + (u ** 2 + b) ** 0.5) / 2
            sp_p = 0.5 * (1 + u / (u ** 2 + b) ** 0.5)
            sp_pp = b / (2 * (u ** 2 + b) ** 1.5)
            w = sp

            f = torch.sum(u * sp_p) + torch.sum(sp * (1 - sp_p))
            f_hat = torch.sum(2 * sp_p - sp_p ** 2) + torch.sum((u - sp) * sp_pp)

            f = f - 2 * edge_num
            f_hat = (-1 / lambda3) * f_hat

            z = z - f / f_hat

        adj_matrix = w / (torch.sum(w) + 1e-5) * 2 * edge_num

        return adj_matrix

    def loss(self, theta, supervise_pos, supervise_label):
        # M = theta.reshape(self.classes, self.feature_dim).T

        supervise_x, Y = supervise_pos.to(self.device), \
            supervise_label.unsqueeze(-1).to(torch.float64).to(self.device)

        X = self.transform(supervise_x.to(torch.float64))
        # y_hat = torch.softmax(X @ M, dim=1)
        y_hat = torch.sigmoid(X @ theta)

        predicted, true = (y_hat - 0.5) > 0, Y.squeeze()
        # print(predicted.shape)
        # print(true.shape)
        acc_value = torch.sum(predicted == true) / len(predicted)

        loss = self.criterion(y_hat, Y.squeeze())
        # print(loss)
        return loss, acc_value

    def save_model(self, hyper_param_path, lambdas, model_path):

        torch.save(self.phi.state_dict(), model_path)
        np.save(hyper_param_path, lambdas.detach().cpu().numpy())

    def train(self, lr, batch_size, init_file=None, epochs=100, with_collaboration=True):
        with open(self.train_set_path, 'rb') as f:
            train_set = pickle.load(f)
        with open(self.test_set_path, 'rb') as f:
            test_set = pickle.load(f)

        print('dataset loaded...')

        directions = 90

        train_size, test_size = len(train_set), len(test_set)

        if with_collaboration:
            assert len(lr) == 2
            params = nn.Parameter(self.hyper_param.to(self.device), requires_grad=True)
            graph_optimizer = torch.optim.Adam([params], lr=lr[0])
            lambdas = torch.relu(params) + 1e-5
            network_optimizer = torch.optim.Adam(self.phi.parameters(), lr=lr[1])
            model_name = 'collaboration_phi_%s_%s_%s_direction%s.pt' % (lr, batch_size, epochs, directions)
            hyper_param = 'collaboration_param_%s_%s_%s_directions%s.npy' % (lr, batch_size, epochs, directions)
        else:
            params = self.hyper_param
            model_name = 'nocollab_phi_%s_%s_%s_directions%s.pt' % (lr, batch_size, epochs, directions)
            hyper_param = 'nocollab_param_%s_%s_%s_directions%s.npy' % (lr, batch_size, epochs, directions)
            lambdas = torch.relu(self.hyper_param) + 1e-5
            network_optimizer = torch.optim.Adam(self.phi.parameters(), lr=lr)

        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('param'):
            os.mkdir('param')
        hyper_param_path = os.path.join('param/', hyper_param)
        model_path = os.path.join('model/', model_name)

        try:
            hyper_parameters = np.load(hyper_param_path)
            lambdas = torch.tensor(hyper_parameters, dtype=torch.float64) + 1e-5
            print("Loading lambdas successfully.")
            state_dic = torch.load(model_path, map_location=self.device, weights_only=False)
            self.phi.load_state_dict(state_dic)
            print("Loading model successfully!")
        except:
            pass

        for flag in range(10):
            if flag == 0:
                continue

            best_test_loss = 100
            last_test_loss = 100
            patience = 5
            trigger_times = 0

            optimizer = network_optimizer
            if flag % 2 == 0 and with_collaboration:
                optimizer = graph_optimizer
            for epoch in range(epochs):
                curr_index, batch_num = torch.randperm(train_size), int((train_size - 1) / batch_size) + 1
                loss = 0
                # for batch_id in tqdm(range(batch_num), desc='Epoch: %s' % (epoch + 1)):
                with tqdm(range(batch_num), desc='Epoch: %s' % (epoch + 1)) as tbar:
                    for batch_id in tbar:
                        batch_index = curr_index[batch_id * batch_size: batch_id * batch_size + batch_size]
                        batch_loss = 0

                        optimizer.zero_grad()
                        for index in batch_index:
                            data = train_set[index]

                            out_loss, out_acc, _, _ = self.forward_for_one_sample(data, lambdas, with_collaboration,
                                                                                  is_train=True)
                            batch_loss = batch_loss + out_loss
                        batch_loss.backward(retain_graph=True)
                        # batch_loss.backward()

                        optimizer.step()
                        loss = loss + batch_loss.detach()

                        # tqdm.write('Mode: %s, Batch loss: %.3f' % (flag % 2, batch_loss.detach().item()))
                        tbar.set_postfix(mode=flag % 2, Batch_loss=batch_loss.detach().item())

                loss = loss / train_size
                test_acc, test_loss = self.test(test_set, params, with_collaboration)
                print('Mode: %s, Epoch: %s, Cross entropy loss: %.3f' % (flag % 2, epoch + 1, loss.detach().item()))
                print(np.around(params.detach().cpu(), 3))
                print('Test acc: ', end='')
                avg_test_acc = torch.sum(test_acc, dim=1) / self.agent_num
                print(np.around(avg_test_acc.numpy(), 3))

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
                        self.save_model(hyper_param_path, lambdas, model_path)
                        best_test_loss = test_loss
                        print('Best model saved, test loss: %.4f' % best_test_loss)

                last_test_loss = test_loss

            if not with_collaboration:
                break
        print('Training finished...')

    def test(self, test_set, params, with_collaboration):
        total_acc = torch.zeros(self.T, self.agent_num)
        total_loss = 0
        test_size = len(test_set)

        with torch.no_grad():
            lambdas = torch.relu(params) + 1e-5
            # for index in range(test_size):
            with tqdm(range(test_size), desc='Testing...') as tbar:
                for index in tbar:
                    test_data = test_set[index]

                    out_loss, out_acc, _, _ = self.forward_for_one_sample(test_data, lambdas, with_collaboration,
                                                                          is_train=False)
                    total_acc += out_acc
                    total_loss += out_loss
        return total_acc / test_size, total_loss / test_size

    def display(self, sample_index, hyper_param_path, model_path, axes, with_collaboration=False, mode='test'):
        directions = 90

        # testset = np.load('dataset/mapping_check_test.npy')
        # testset = torch.tensor(testset, dtype=torch.float64, device=self.device)
        if mode == 'test':
            with open(self.test_set_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            with open(self.train_set_path, 'rb') as f:
                dataset = pickle.load(f)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.phi.load_state_dict(state_dict)

        hyper_param = np.load(hyper_param_path)

        lambdas = torch.tensor(hyper_param, dtype=torch.float64) + 1e-5

        robots = dataset[sample_index]

        self.show = True
        with torch.no_grad():
            out_loss, out_acc, adj_record, mapping_res = self.forward_for_one_sample(robots, lambdas,
                                                                                     with_collaboration)
        self.show = False
        # if not os.path.exists('tmp'):
        #     os.mkdir('tmp')
        # gif = []
        # if not os.path.exists('gifs'):
        #     os.mkdir('gifs')
        # gif_name = 'gifs/%smapping_%s_%s_%s.gif' % (with_collaboration, mode, directions, sample_index)
        gt_map = robots[0].room

        res, recall, precision = [], [], []

        num_subplot = self.agent_num + 1
        for t in range(self.T):
            # file_name = 'tmp/%s.png' % t

            needPlot = (t + 1) % 3 == 0 or t == 0
            index = 0 if t == 0  else t // 3 + 1
            print(index, needPlot)
            robot = robots[0]
            curr_map = (mapping_res[t] > 0.5).to(float)
            robot_map = curr_map[0]

            if needPlot:
                axes[index].set_title('T=%s' % (t + 1), fontsize=20)

                custom_cmap = ListedColormap(['black', '#e5e5e5'])

                axes[index].imshow(robot_map, cmap=custom_cmap)
                axes[index].set_xticks([])
                axes[index].set_yticks([])

            samples = robot.data[t]
            for i in range(samples.shape[0]):
                samp_pos = np.array(samples[i, 0:2], dtype=int).tolist()
                if needPlot:
                    axes[index].plot(samp_pos[1], samp_pos[0], marker='.', color='white', markersize=0.5)



            # curr_map = (mapping_res[t] - 0.5 > 0).to(float)

            total_f1, total_precision, total_recall = 0, 0, 0

            for k in range(self.agent_num):
                robot_map = curr_map[k]
                # if t == self.T - 1:
                #     axes[k].set_title('agent %s' % (k + 1))

                #     custom_cmap = ListedColormap(['black', '#e5e5e5'])

                #     axes[k].imshow(robot_map, cmap=custom_cmap)
                #     axes[k].set_xticks([])
                #     axes[k].set_yticks([])

                #     samples = robots[k].data[t]
                #     for i in range(samples.shape[0]):
                #         samp_pos = np.array(samples[i, 0:2], dtype=int).tolist()
                #         axes[k].plot(samp_pos[1], samp_pos[0], marker='.', color='white', markersize=0.5)

                predicted_label, gt_label = robot_map.flatten(), gt_map.flatten()
                total_f1 += f1_score(gt_label, predicted_label)
                total_recall += recall_score(gt_label, predicted_label)
                total_precision += precision_score(gt_label, predicted_label)

            # plt.savefig(file_name, bbox_inches='tight')
            # gif.append(imageio.imread('tmp/%s.png' % t))

            res.append(total_f1 / self.agent_num)
            recall.append(total_recall / self.agent_num)
            precision.append(total_precision / self.agent_num)

        # axes[4].set_title('truth image')
        # axes[4].imshow(gt_map, cmap=custom_cmap)
        # axes[4].set_xticks([])
        # axes[4].set_yticks([])
        # imageio.mimsave(gif_name, gif, fps=1)
        # shutil.rmtree('tmp')
        return res, axes

    def displayWithTime(self, sample_index, hyper_param_path, model_path, axes, with_collaboration=True, mode='test'):
        if mode == 'test':
            with open(self.test_set_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            with open(self.train_set_path, 'rb') as f:
                dataset = pickle.load(f)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.phi.load_state_dict(state_dict)

        hyper_param = np.load(hyper_param_path)

        lambdas = torch.tensor(hyper_param, dtype=torch.float64) + 1e-5

        robots = dataset[sample_index]

        self.show = True
        with torch.no_grad():
            out_loss, out_acc, adj_record, mapping_res = self.forward_for_one_sample(robots, lambdas,
                                                                                     with_collaboration)
        self.show = False
        if not os.path.exists('gifs'):
            os.mkdir('gifs')
        gt_map = robots[0].room
        res = []

        for t in range(self.T-1):
            needPlot = (t + 1) % 3 == 0 or t == 0
            index = 0 if t == 0  else t // 3 + 1
            robot = robots[0]
            curr_map = (mapping_res[t] > 0.5).to(float)
            robot_map = curr_map[0]

            if needPlot:
                axes[index].set_title('T=%s' % (t + 1), fontsize=20)

                custom_cmap = ListedColormap(['black', '#e5e5e5'])

                axes[index].imshow(robot_map, cmap=custom_cmap)
                axes[index].set_xticks([])
                axes[index].set_yticks([])

            samples = robot.data[t]
            for i in range(samples.shape[0]):
                samp_pos = np.array(samples[i, 0:2], dtype=int).tolist()
                if needPlot:
                    axes[index].plot(samp_pos[1], samp_pos[0], marker='.', color='white', markersize=0.5)

            predicted_label, gt_label = robot_map.flatten(), gt_map.flatten()
            res.append(f1_score(gt_label, predicted_label))
        axes[4].set_title('Truth Map', fontsize=20)
        axes[4].imshow(gt_map, cmap=custom_cmap)
        axes[4].set_xticks([])
        axes[4].set_yticks([])
        # plt.tight_layout()
        # plt.axis('off')
        return res, axes


if __name__ == '__main__':
    solver = MappingUnroll(hyper_param=torch.tensor([0, 0.1, 0.1]), feature_dim=50, T=10, agent_num=4)
    # solver.train(lr=1e-3, batch_size=2, epochs=100, with_collaboration=False)
    solver.display(sample_index=5, model_path='model/nocollab_phi_0.001_2_100_directions90.pt',
                   hyper_param_path='param/nocollab_param_0.001_2_100_directions90.npy',
                   with_collaboration=False, mode='test')
