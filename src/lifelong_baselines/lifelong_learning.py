from avalanche.benchmarks.generators import dataset_benchmark, nc_benchmark 
from avalanche.models.base_model import BaseModel
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ''))
from classification_dataset import Classification_Scheduler
from torch.utils.data import Dataset, TensorDataset
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from models import init_model
import torch
import numpy as np
import pickle
from lifelong_utils import *


class LifelongBaselines:
    BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
    def __init__(self, agent_num, dataset_type, total_time, is_backbone, device) -> None:
        assert dataset_type in ['mnist', 'cifar10']
        self.dataset_type = dataset_type
        self.total_time = total_time
        self.is_backbone = is_backbone
        self.agent_num = agent_num
        self.backbone_path = self.BASE_PATH + f'model/collaboration_phi_[0.0001, 0.001]_2_{dataset_type}_100.pt'
        self.device = device

        """
        possible methods:
        'naive', 'ewc', 'lwf', 'gem',
        'online_naive', 'online_gem', 'mas',
        'icarl', 'bic', 'mir', 'mer'
        """
        self.strategy_dict = {
            'naive': naive_strategy, 'ewc': ewc_strategy, 'lwf': lwf_strategy, 'gem': gem_strategy,
            'online_naive': online_naive_strategy, 'online_gem': online_gem_strategy, 'mas': mas_strategy,
            'icarl': icarl_strategy, 'bic': bic_strategy, 'mir': mir_strategy, 'mer': mer_strategy
        }

    def _create_datasets(self, data):
        img_set, label_set = data
        assert len(img_set) == len(label_set)
        time = len(img_set)
        if self.is_backbone:
            result = []
            state_dic = torch.load(self.backbone_path, weights_only=False)
            mode = 0 if self.dataset_type == 'mnist' else 1

            backbone = init_model(mode, feature_dim=50, device='cpu', float_type=torch.float32)

            backbone.load_state_dict(state_dic)
            for t in range(self.total_time):
                with torch.no_grad():
                    features = backbone(img_set[t])
                result.append(TensorDataset(features, label_set[t]))
        else:
            result = [TensorDataset(img_set[t], label_set[t]) for t in range(time)]
        
        return result
    
    def _cl_strategy(self, 
                     method: str, 
                     model: BaseModel, 
                     optimizer, 
                     criterion,
                     num_epochs: int, 
                     *args
                     ):
        assert method in self.strategy_dict.keys()
        cl_strategy = self.strategy_dict[method](model, optimizer, criterion, num_epochs, self.device, *args)

        return cl_strategy
    
    def _close_std(self):
        original_stdout_fd = os.dup(1)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, 1)
        os.close(null_fd)

        return original_stdout_fd
    
    def _open_std(self, original_stdout_fd):
        os.dup2(original_stdout_fd, 1)
        os.close(original_stdout_fd)

    def benchmark(self, method, dataset_path, num_epochs, *args):
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)

        data_base = Classification_Scheduler(
            time=self.total_time, 
            agent_num=self.agent_num, 
            dataset_type=self.dataset_type
        )
        
        total_acc = np.zeros(self.total_time)

        for index in range(len(dataset)):
            data = data_base.get_data(dataset[index], input_mode=True, gt_mode=False)

            for agent_id in range(self.agent_num):
                train_data, gt_data = data[agent_id]

                train_datasets, test_datasets = self._create_datasets(train_data), self._create_datasets(gt_data)
                if method == 'icarl':
                    scenario = nc_benchmark(
                        train_dataset=train_datasets,
                        test_dataset=test_datasets,
                        n_experiences=10,
                        task_labels=False
                    )
                else:
                    scenario = dataset_benchmark(
                        train_datasets=train_datasets,
                        test_datasets=test_datasets
                    )

                if self.is_backbone:
                    model = LinearClassifier(feature_dim=50, n_classes=5)
                else:
                    model = ImageClassifier(self.dataset_type, n_classes=5)

                optimizer = Adam(model.parameters(), lr=1e-3)
                criterion = CrossEntropyLoss()

                cl_strategy = self._cl_strategy(method, model, optimizer, criterion, num_epochs, *args)

                results = []

                for experience in scenario.train_stream:
                    # print("Start of experience: ", experience.current_experience)

                    original_stdout_fd = self._close_std()

                    cl_strategy.train(experience)

                    self._open_std(original_stdout_fd)

                    original_stdout_fd = self._close_std()

                    eval_res = cl_strategy.eval(scenario.test_stream)

                    self._open_std(original_stdout_fd)

                    key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{experience.current_experience}'
                    eval_acc = eval_res[key]
                    results.append(eval_acc)
                print(results)
                total_acc += np.asarray(results)
        
        total_acc = total_acc / self.agent_num / len(dataset)

        return total_acc

                
# solver = LifelongBaselines(
#     agent_num=6,
#     dataset_type='mnist',
#     total_time=10,
#     is_backbone=True
# )

# print(solver.benchmark('online_naive', solver.BASE_PATH + '/dataset/mnist_test_samples.pkl', 20, 1))






