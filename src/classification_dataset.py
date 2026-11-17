import math

import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.transforms.functional import resize, InterpolationMode, normalize
from PIL import Image
import numpy as np
import pickle
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from typing import Any


def get_data(dataset_type, if_train):
    if dataset_type == 'mnist':
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Resize(32)
        # ])
        # img_set = torchvision.datasets.MNIST('dataset/mnist', download=True, train=True, transform=transform)
        raw_folder = 'dataset/mnist/MNIST/raw/'
        image_file = f"{'train' if if_train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(raw_folder, image_file))

        label_file = f"{'train' if if_train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(raw_folder, label_file))

        return data, targets
    elif dataset_type == 'cifar10':
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        # img_set = torchvision.datasets.CIFAR10('dataset/cifar10', download=True, train=True, transform=transform)
        base_folder = "cifar-10-batches-py"
        root = "dataset/cifar10"
        train_list = [
            ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
            ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
            ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
            ["data_batch_4", "634d18415352ddfa80567beed471001a"],
            ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
        ]

        test_list = [
            ["test_batch", "40351d587109b95175f43aff81a1287e"],
        ]

        downloaded_list = train_list if if_train else test_list
        data: Any = []
        targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                data.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        # data = data.transpose((0, 2, 3, 1))  # convert to HWC
        targets = torch.tensor(targets, dtype=torch.int64)
        return data, targets

    elif dataset_type == 'cifar100':
        base_folder = "cifar-100-python"
        root = "dataset/cifar100"
        if not os.path.exists(os.path.join(root, base_folder)):
            alt_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "core",
                "cifar100",
            )
            if os.path.exists(os.path.join(alt_root, base_folder)):
                root = alt_root
        file_name = "train" if if_train else "test"
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data = entry["data"].reshape(-1, 3, 32, 32)
        targets = torch.tensor(entry["fine_labels"], dtype=torch.int64)
        return data, targets

    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def transform(subset, dataset_type):
    if dataset_type == 'mnist':
        img = resize(subset, size=[32, 32], interpolation=InterpolationMode.BILINEAR,
                     max_size=None, antialias=None) / 255
        img = img.unsqueeze(1)
        # print(img.shape)
    elif dataset_type == 'cifar10':
        subset = torch.tensor(subset, dtype=torch.float)
        img = normalize(subset / 255, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif dataset_type == 'cifar100':
        subset = torch.tensor(np.asarray(subset), dtype=torch.float)
        img = normalize(
            subset / 255.0,
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
    return img


class Lifelong_Classification_Task:
    def __init__(self, mode=True, time=10, dataset_type='mnist'):
        self.dataset_type = dataset_type
        self.T = time
        img_data, labels = get_data(dataset_type, if_train=mode)
        self.newlabel, index = torch.sort(labels)
        self.img_data_sorted = img_data[index]
        prev = -1
        self.start_index = []
        for i in range(len(self.newlabel)-1):
            if self.newlabel[i] != self.newlabel[i+1]:
                self.start_index.append(prev+1)
                prev = i
        self.start_index.append(prev+1)
        print(self.start_index)

    def query(self, query_index, dataset_size=10, divide_size=100):
        # query_index = torch.randint(0, 50, [10])
        # print(query_index)
        assert len(query_index) == self.T
        # assert dataset_size <= 100
        # assert max(query_index) <= int(1000/divide_size)
        data = []
        for i in range(self.T):
            start = int(query_index[i] * divide_size) + self.start_index[i]
            choice = torch.tensor(np.random.choice(divide_size, dataset_size, replace=False), dtype=torch.int64)
            # lower_bound, upper_bound = self.start_index[i]+start, self.start_index[i]+start+dataset_size
            # print(lower_bound, upper_bound)
            # data_ids = torch.arange(lower_bound, upper_bound, step=1)
            data_ids = start + choice
            subset = self.img_data_sorted[data_ids]
            img = transform(subset, dataset_type=self.dataset_type)
            labels = (i * torch.ones(dataset_size)).to(torch.int64)
            data.append([img, labels])
        return data

    def sample(self, sample_indexes):
        assert len(sample_indexes) == self.T

        data = []
        for i in range(self.T):
            start = self.start_index[i]
            choice = sample_indexes[i]
            data_ids = start + choice
            subset = self.img_data_sorted[data_ids]
            img = transform(subset, dataset_type=self.dataset_type)
            labels = (i * torch.ones(len(choice))).to(torch.int64)
            data.append([img, labels])
        return data

    def full_set(self):
        data = []
        if self.dataset_type == 'mnist':
            choice = torch.arange(0, 800, step=1)
        elif self.dataset_type == 'cifar100':
            choice = torch.arange(0, 100, step=1)
        else:
            choice = torch.arange(0, 1000, step=1)
        for i in range(self.T):
            start = self.start_index[i]
            data_ids = start + choice
            subset = self.img_data_sorted[data_ids]
            img = transform(subset, dataset_type=self.dataset_type)
            labels = (i * torch.ones(len(choice))).to(torch.int64)
            data.append([img, labels])
        return data


def map_group_id_2_index(group_num, class_num_eachgroup, agentnum_eachgroup):
    class_map_table = [torch.zeros(class_num_eachgroup[i], dtype=torch.int64) for i in range(group_num)]
    agent_map_table = [torch.zeros(agentnum_eachgroup[i], dtype=torch.int64) for i in range(group_num)]

    total_class, total_agents = 0, 0
    for i in range(group_num):
        class_num_curr_group = class_num_eachgroup[i]
        agent_num_curr_group = agentnum_eachgroup[i]

        class_index_map = torch.arange(total_class, total_class + class_num_curr_group, step=1)
        total_class += class_num_curr_group

        agent_index_map = torch.arange(total_agents, total_agents + agent_num_curr_group, step=1)
        total_agents += agent_num_curr_group

        class_map_table[i] = class_index_map
        agent_map_table[i] = agent_index_map

    return class_map_table, agent_map_table


class Classification_Scheduler:
    def __init__(self, time=10, agent_num=6, dataset_type='mnist'):
        self.dataset_type = dataset_type
        self.T = time
        img_data_train, labels_train = get_data(dataset_type, if_train=True)
        newlabel_train, index_train = torch.sort(labels_train)
        self.img_data_sorted_train = img_data_train[index_train]
        prev = -1
        self.start_index_train = []
        for i in range(len(newlabel_train) - 1):
            if newlabel_train[i] != newlabel_train[i + 1]:
                self.start_index_train.append(prev + 1)
                prev = i
        self.start_index_train.append(prev + 1)
        print(self.start_index_train)

        img_data_test, labels_test = get_data(dataset_type, if_train=False)
        newlabel_test, index_test = torch.sort(labels_test)
        self.img_data_sorted_test = img_data_test[index_test]
        prev = -1
        self.start_index_test = []
        for i in range(len(newlabel_test) - 1):
            if newlabel_test[i] != newlabel_test[i + 1]:
                self.start_index_test.append(prev + 1)
                prev = i
        self.start_index_test.append(prev + 1)
        print(self.start_index_test)

        self.agent_num = agent_num

    def non_iid_split(self, agent_num, class_num, split_type='dirichlet'):

        if split_type == 'dirichlet':
            label_distribution = np.random.dirichlet([0.5 for i in range(agent_num)], [self.T, class_num])
            label_distribution = torch.tensor(10 * np.around(label_distribution, 1), dtype=torch.int64)
        elif split_type == 'random_select':
            select_label = np.random.choice(class_num, [self.T, agent_num])
            label_distribution = np.eye(class_num)[select_label]
            label_distribution = torch.tensor(10 * label_distribution, dtype=torch.int64)
            label_distribution = torch.permute(label_distribution, [0, 2, 1])

        # sns.heatmap(label_distribution[0], annot=True)
        # plt.ylabel('Class id')
        # plt.xlabel('Distribution')
        # plt.show()
        return label_distribution

    def schedule(self, group_num, split_type='dirichlet'):
        total_classes = 20 if self.dataset_type == 'cifar100' else 10
        class_num = int(total_classes/group_num)
        class_num_each_group = [class_num for i in range(group_num)]
        group_agent_num = int(self.agent_num/group_num)
        agent_num_each_group = [group_agent_num for i in range(group_num)]

        class_map_table, agent_map_table = map_group_id_2_index(group_num, class_num_each_group, agent_num_each_group)

        schedule_result = []
        for group_id in range(group_num):
            table = self.non_iid_split(group_agent_num, class_num, split_type=split_type)
            schedule_result.append([table, class_map_table[group_id], agent_map_table[group_id]])

        return schedule_result

    def agent_schedule(self, schedule_result):
        res = [{} for i in range(self.agent_num)]
        for group_id in range(len(schedule_result)):
            group_agents = schedule_result[group_id][2]
            class_table = schedule_result[group_id][1]
            data_schedule_res = schedule_result[group_id][0]
            for i, agent_id in enumerate(group_agents):
                res[agent_id]['group_id'] = group_id
                res[agent_id]['class_table'] = class_table
                res[agent_id]['schedule_data'] = data_schedule_res[:, :, i]
        return res

    def generate_indexes(self, schedule_data, class_table, start_index, sample_range, sample_start):
        res_index, res_label = [], []
        for t in range(self.T):
            current_distribution = schedule_data[t]
            class_num = len(current_distribution)
            indexes, labels = [], []
            for i in range(class_num):
                number = int(current_distribution[i])
                if number == 0:
                    continue

                tmp_label = (i * torch.ones(number)).to(torch.int64)
                current_class_id = int(class_table[i])
                tmp_index = start_index[current_class_id] + sample_start + \
                            np.random.choice(sample_range, number, replace=False)
                tmp_index = torch.tensor(tmp_index, dtype=torch.int64)

                indexes.append(tmp_index)
                labels.append(tmp_label)

            indexes = torch.concat(indexes)
            labels = torch.concat(labels)
            res_index.append(indexes)
            res_label.append(labels)

        return res_index, res_label

    def create_one_datasample(self, sample_range, sample_start, input_start_index, gt_start_index,
                              supervise_range, supervise_start):
        schedule_res = self.schedule(group_num=2, split_type='random_select')
        agents_sched_res = self.agent_schedule(schedule_res)
        for j in range(self.agent_num):
            agent_schedule_data, agent_class_table = agents_sched_res[j]['schedule_data'], \
                agents_sched_res[j]['class_table']
            tmp_index, tmp_label = self.generate_indexes(agent_schedule_data, agent_class_table,
                                                         input_start_index,
                                                         sample_range, sample_start=sample_start)
            agents_sched_res[j]['index'] = tmp_index
            agents_sched_res[j]['label'] = tmp_label

            gt_schedule_data = (torch.ones(agent_schedule_data.shape) * 10).to(torch.int64)
            gt_index, gt_label = self.generate_indexes(gt_schedule_data, agent_class_table,
                                                       gt_start_index,
                                                       supervise_range, sample_start=supervise_start)

            agents_sched_res[j]['gt_index'] = gt_index
            agents_sched_res[j]['gt_label'] = gt_label

        return agents_sched_res

    def dataset_create(self, train_size, test_size):
        train, test = [], []
        train_path = 'dataset/%s_train_samples.pkl' % self.dataset_type
        test_path = 'dataset/%s_test_samples.pkl' % self.dataset_type
        if self.dataset_type == 'mnist':
            train_range, test_train_range, test_range = 3000, 2000, 500
        elif self.dataset_type == 'cifar100':
            train_range, test_train_range, test_range = 300, 200, 100
        else:
            train_range, test_train_range, test_range = 3000, 2000, 1000
        for i in range(train_size):
            one_datasample = self.create_one_datasample(train_range, 0, self.start_index_train, self.start_index_train,
                                                        train_range, 0)
            train.append(one_datasample)

        for i in range(test_size):
            one_datasample = self.create_one_datasample(test_train_range, train_range, self.start_index_train,
                                                        self.start_index_test, test_range, 0)
            test.append(one_datasample)

        with open(train_path, 'wb') as f:
            pickle.dump(train, f)

        with open(test_path, 'wb') as f:
            pickle.dump(test, f)
        print('Dataset create finished...')
        return train, test

    def sample(self, sample_indexes, mode):
        img_set = []
        for t in range(self.T):
            if mode:
                subset = self.img_data_sorted_train[sample_indexes[t]]
            else:
                subset = self.img_data_sorted_test[sample_indexes[t]]

            img = transform(subset, dataset_type=self.dataset_type)
            img_set.append(img)
        return img_set

    def get_data(self, data_sample, input_mode, gt_mode):
        assert self.agent_num == len(data_sample)
        result = []
        for i in range(self.agent_num):
            agent_dic = data_sample[i]
            input_index, gt_index = agent_dic['index'], agent_dic['gt_index']
            input_label, gt_label = agent_dic['label'], agent_dic['gt_label']
            img, gt_img = self.sample(input_index, input_mode), self.sample(gt_index, gt_mode)
            result.append([[img, input_label], [gt_img, gt_label]])

        return result


# dataset = Classification_Scheduler(time=10, agent_num=6, dataset_type='cifar10')
# train, test = dataset.dataset_create(train_size=100, test_size=100)
# with open('dataset/mnist_test_samples.pkl', 'rb') as f:
#     test = pickle.load(f)
# print(len(dataset.get_data(train[0], input_mode=True, gt_mode=True)[0][0][0]))
# print(test[0])


