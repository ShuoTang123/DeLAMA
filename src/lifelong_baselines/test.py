from avalanche.training.supervised import EWC, GEM, LwF, Naive
from avalanche.benchmarks.generators import dataset_benchmark 
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models.base_model import BaseModel
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ''))
from avalanche.models import SimpleMLP, SimpleCNN

from classification_dataset import Classification_Scheduler
from torch.utils.data import Dataset, TensorDataset
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from models import Phi
import torch
import pickle

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')
BACKBONE_PATH = BASE_PATH + 'model/collaboration_phi_[0.0001, 0.001]_2_mnist_100.pt'


class LinearClassifier(nn.Module, BaseModel):
    def __init__(self, feature_dim, n_classes):
        super().__init__()
        self.classifier = nn.Linear(in_features=feature_dim, out_features=n_classes)

    def forward(self, feature):
        out = self.classifier(feature)
        return out
    
    def get_features(self, feature):
        return feature


def create_datasets(data, is_backbone=False):
    img_set, label_set = data
    assert len(img_set) == len(label_set)
    time = len(img_set)
    if is_backbone:
        result = []
        state_dic = torch.load(BACKBONE_PATH, weights_only=False)
        backbone = Phi(n_classes=50)
        backbone.load_state_dict(state_dic)
        for t in range(T):
            with torch.no_grad():
                features = backbone(img_set[t])
            result.append(TensorDataset(features, label_set[t]))
    else:
        result = [TensorDataset(img_set[t], label_set[t]) for t in range(time)]
    
    return result

# two dataset_types, mnist or cifar10
dataset_type = 'mnist'
T = 10
agent_num = 6
is_backbone = False


dataset_path = BASE_PATH + '/dataset/%s_test_samples.pkl' % dataset_type
index = 1

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

data_base = Classification_Scheduler(time=T, agent_num=agent_num, dataset_type=dataset_type)


# trainset: input_mode=True, gt_mode=True
# testset: input_mode=True, gt_mode=False
data = data_base.get_data(dataset[index], input_mode=True, gt_mode=False)
# agent 0's data
train_data, gt_data = data[0]

train_datasets, test_datasets = create_datasets(train_data, is_backbone), create_datasets(gt_data, is_backbone)

scenario = dataset_benchmark(
    train_datasets=train_datasets,
    test_datasets=test_datasets
)

if is_backbone:
    model = LinearClassifier(feature_dim=50, n_classes=5)
else:
    model = SimpleMLP(num_classes=5, input_size=int(32 * 32))


# optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()
cl_strategy = Naive(
    model, optimizer, criterion,
    train_mb_size=100, train_epochs=100, eval_mb_size=100
)

# cl_strategy = EWC(
#     model=model,
#     optimizer=optimizer,
#     criterion=criterion,
#     ewc_lambda=1
# )

# cl_strategy = LwF(
#     model=model,
#     optimizer=optimizer,
#     criterion=criterion,
#     alpha=0.1,
#     temperature=0.1
# )


results = []

for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)

    original_stdout_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)


    os.dup2(null_fd, 1)
    os.close(null_fd)

    cl_strategy.train(experience)

    os.dup2(original_stdout_fd, 1)
    os.close(original_stdout_fd)


    original_stdout_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.close(null_fd)

    eval_res = cl_strategy.eval(scenario.test_stream)

    os.dup2(original_stdout_fd, 1)
    os.close(original_stdout_fd)

    key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{experience.current_experience}'
    eval_acc = eval_res[key]
    results.append(eval_acc)


print(results)










