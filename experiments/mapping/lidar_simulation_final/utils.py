import numpy as np
import pickle
import torch
from PIL import Image
from torch import nn, Tensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import image

import Robot


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')


class UnLinearNetwork(nn.Module):
    def __init__(self, input_dim, mediate_dim, output_dim):
        super().__init__()
        self.linearNet = nn.Sequential(
            nn.Linear(input_dim, mediate_dim),
            Sine(),
            # nn.ReLU(),
            nn.Linear(mediate_dim, mediate_dim),
            nn.ReLU(),
            # Sine(),
            nn.Linear(mediate_dim, mediate_dim),
            nn.ReLU(),
            # nn.Linear(mediate_dim, mediate_dim),
            # nn.ReLU(),
            # nn.Linear(mediate_dim, mediate_dim),
            # nn.ReLU(),
            nn.Linear(mediate_dim, output_dim)
        )

    def forward(self, x):
        x = self.linearNet(x)
        x = torch.flatten(x, 1)
        return x


def img2binary(img_path):
    """image file to binary img array: Blocked(1) Free(0)"""
    room = Image.open(img_path).convert('L')  # room_type = image
    room_array = np.asarray(room)  # room_type = np_array
    boolean = np.zeros(room_array.shape) == room_array
    room_binary = boolean.astype(int)
    return room_binary


def save_datasets(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def getTestData(img_path, step=1):
    img = img2binary(img_path)
    height, width = img.shape
    res = []
    for i in range(0, height, step):
        for j in range(0, width, step):
            res.append([[i, j], img[i, j]])
    return res


def testData2Dataset(test_path: str, trans=None):
    class TestDataset(Dataset):
        def __init__(self, test_list, transform):
            self.data = test_list
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            pos, val = self.data[idx]
            pos = np.array(pos, dtype='int')
            res = {'pos': pos, 'val': val}

            if self.transform:
                res = self.transform(res)

            return res

    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    return TestDataset(test_data, trans)


class TotalDataset(object):
    """Keep all agents' datasets.

    datasets[t, i] returns a dataset of samples at time-t, agent-i."""

    class SamplesDataset(Dataset):
        def __init__(self, data, is_train, transform):
            self.samples = data
            self.transform = transform
            self.train = is_train

        def __len__(self):
            return self.samples.shape[0]

        def __getitem__(self, idx):
            x, y, val, gt = self.samples[idx, :]
            pos = np.array([x, y], dtype='int')
            if self.train:
                res = {'pos': pos, 'val': val}
            else:
                res = {'pos': pos, 'val': gt}

            if self.transform:
                res = self.transform(res)

            return res

    def __init__(self, data_list, is_train=True, transform=None):
        """Retain datasets sampled from robot, either training or testing."""
        self.data = np.array(data_list, dtype=object)
        self.data = np.transpose(self.data)
        self.T = self.data.shape[0]
        self.agents = self.data.shape[1]
        self.train = is_train
        self.transform = transform

    def __getitem__(self, tup):
        """
        Return dataset of time t, agent i
        :param tup: (t, i), t in range(T) and i in range(agents_num)
        :return: dataset {"pos": np.array[x, y], "val": detect values, "gt": ground truth}
        """
        t, i = tup
        assert t in range(self.T) and i in range(self.agents)
        samples = TotalDataset.SamplesDataset(self.data[t, i], self.train, self.transform)
        return samples


class ToTensor(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        pos, val = sample['pos'], sample['val']
        pos = torch.tensor(pos, device=self.device)
        return {'pos': pos.float(), 'val': torch.tensor(val, device=self.device).float()}


def normalize(pos: Tensor, h, w):
    x = torch.reshape(pos[:, 0] / h * 2 - 1, [-1, 1])
    y = torch.reshape(pos[:, 1] / w * 2 - 1, [-1, 1])
    return torch.cat((x, y), dim=1)


def makeDataset(robots_file: str, is_train=True, transform=None):
    """
    Obtain data from saved robots_file to a TotalDataset instance.

    :param transform: transform data
    :param is_train: return train dataset or test dataset.
    :param robots_file: file.pkl saving all the agent robots.
    :return: TotalDataset, use TotalDataset[T, agents] to get a dataset.
    """
    with open(robots_file, 'rb') as inp:
        robots = pickle.load(inp)

    data_all = [r.data for r in robots]

    td = TotalDataset(data_all, is_train, transform)
    return td


def plot_sampling(robot: Robot.Robot):
    rand = np.random.randint(len(robot.data))
    data = robot.data[rand]

    plt.imshow((robot.room ^ 1) * 255.9, cmap='gray', vmin=0, vmax=255)
    for point in data:
        x, y, label = point
        color = 'blue' if label == 0 else 'red'
        plt.plot([y], [x], marker='.', markersize=2, color=color)
    plt.show()


def plot_graph(w: Tensor):
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        ax.matshow(w.numpy(), cmap='gist_gray')
    except:
        ax.matshow(w.cpu().detach().numpy(), cmap='gist_gray')
    plt.show()


def te_points(img_path, cp, colors):
    assert len(cp) == len(colors)
    img = image.imread(img_path)

    for i in range(len(cp)):
        for j in range(len(cp[i]) - 1):
            x, y = [cp[i][j][1], cp[i][j + 1][1]], [cp[i][j][0], cp[i][j + 1][0]]
            # plt.plot(x, y, color=colors[i], linewidth=2)
            plt.arrow(x=x[0], y=y[0], dx=x[1] - x[0], dy=y[1] - y[0], width=1, color=colors[i],
                      head_width=5, head_length=3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()
