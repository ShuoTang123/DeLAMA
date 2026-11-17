import matplotlib.pyplot as plt
import numpy as np
from typing import List
from utils import *
from Robot import Robot
from tqdm import tqdm


class DataSampler:
    def __init__(self, moving_step=2, sampling_step=2, agent_num=5,
                 time=10):

        self.mov_step, self.samp_step = moving_step, sampling_step
        self.agent_num, self.T = agent_num, time

    def sample(self, maps_set, critical_path, save_path, is_debug=False):
        all_robots = []
        for i in tqdm(range(len(maps_set)), desc='Sampling and creating set...'):
            current_map = maps_set[i]
            robots = []
            for j in range(self.agent_num):
                robot = Robot(current_map, self.mov_step, self.samp_step)
                try:
                    robot.setPath(critical_path[j])
                except:
                    raise ValueError(f"T={self.T} exceeds the maximum of sampling time")
                robot.sample(self.T)
                robots.append(robot)
            all_robots.append(robots)
        with open(save_path, 'wb') as f:
            pickle.dump(all_robots, f)

    def sampleAtPoints(self, curr_map: np.ndarray, coordinate: List[int], part: int):
        """The coordinate divide the bitmap into 4 parts.
        0: left upp, 1: right upp, 2: left down, 3: right down"""
        robot = Robot(curr_map, self.mov_step, self.samp_step, sampling_var=0.1)
        m, n = curr_map.shape

        if part <= 1:
            x_upp, x_down = 0, coordinate[0]
        else:
            x_upp, x_down = coordinate[0] + 1, m
        if part % 2 == 0:
            y_left, y_right = 0, coordinate[1]
        else:
            y_left, y_right = coordinate[1] + 1, n

        dp = []
        while len(dp) < self.T:
            ptr = [np.random.randint(x_upp, x_down), np.random.randint(y_left, y_right)]
            if not robot.isBlocked(ptr):
                dp.append(ptr)
            dp = np.unique(np.array(dp), axis=0).tolist()
        robot.path = [tuple(ptr) for ptr in dp]
        robot.sample(self.T)
        return robot

    def sampleRoom(self, curr_map: np.ndarray, coord: List[int]):
        part_index = [np.sum(curr_map[:coord[0], :coord[1]] == 0),
                      np.sum(curr_map[:coord[0], (coord[1] + 1):] == 0),
                      np.sum(curr_map[(coord[0] + 1):, :coord[1]] == 0),
                      np.sum(curr_map[(coord[0] + 1):, (coord[1] + 1):] == 0)]
        part_index = [i for i in range(4) if part_index[i] > 300]
        robots = []
        for j in range(self.agent_num):
            if j < len(part_index):
                robot = self.sampleAtPoints(curr_map, coord, part_index[j])
            else:
                robot = self.sampleAtPoints(curr_map, coord, part_index[np.random.randint(len(part_index))])
            # plot_sampling(robot)
            robots.append(robot)
        return robots


if __name__ == '__main__':
    train_set = np.load('dataset/train_mapset.npy', allow_pickle=True)
    test_set = np.load('dataset/test_mapset.npy', allow_pickle=True)
    train_coords = np.load('dataset/train_coords.npy')
    test_coords = np.load('dataset/test_coords.npy')
    sampler = DataSampler(agent_num=4, time=12)

    all_robots = []
    for i in tqdm(range(len(train_set)), desc='Collecting training data...'):
        robots = sampler.sampleRoom(curr_map=train_set[i], coord=train_coords[i])
        all_robots.append(robots)
    with open('dataset/sample_check_train_11_16.pkl', 'wb') as f:
        pickle.dump(all_robots, f)

    all_robots = []
    for i in tqdm(range(len(test_set)), desc='Collecting testing data...'):
        robots = sampler.sampleRoom(curr_map=test_set[i], coord=test_coords[i])
        all_robots.append(robots)
    with open('dataset/sample_check_test_11_16.pkl', 'wb') as f:
        pickle.dump(all_robots, f)
