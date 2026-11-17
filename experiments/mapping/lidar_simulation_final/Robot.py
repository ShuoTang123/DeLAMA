import random
import numpy as np
from math import floor, cos, sin, pi, sqrt, ceil
from CriticalPath import CriticalPath


def updatePos(start, dstep):
    return start[0] + dstep[0], start[1] + dstep[1]


direct_num = 90
detection_max = 20


class Robot(object):
    directions = [(cos(2 * pi * k / direct_num), sin(2 * pi * k / direct_num))
                  for k in range(direct_num)]

    def __init__(self, map, step_length, samp_step, sampling_var=0):
        """
        Create a Robot instance from a room image and a given agent type.

        :param map: np array
        :param step_length: length of moving step.
        :param samp_step: length of sampling step
        :param sampling_var: gaussian variance when sampling.
        """
        self.room = map
        self.gt = []
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                feature = [0 for k in range(8)]
                for k in range(9):
                    x_axis, y_axis = int(k / 3) - 1, k % 3 - 1
                    if x_axis == 0 and y_axis == 0:
                        continue
                    x_axis, y_axis = i + x_axis, j + y_axis
                    if x_axis < 0 or x_axis >= self.room.shape[0] or y_axis < 0 or y_axis >= self.room.shape[1]:
                        continue
                    else:
                        if k >= 4:
                            index = k - 1
                        else:
                            index = k
                        feature[index] = self.room[x_axis][y_axis]
                feature.append(map[i][j])
                feature.insert(0, j)
                feature.insert(0, i)
                self.gt.append(feature)
        self.gt = np.asarray(self.gt)

        self.samp_step = samp_step
        self.samp_var = sampling_var
        self.cp = CriticalPath(self.room, step_length)
        self.path = []  # update in setPath method
        self.data = []  # update in sample method

    def sample(self, T):
        if len(self.path) == 0:
            raise Exception("The path contains no coordinates!")
        tmp_data = [self.positionSample(pos) for pos in self.path]

        length = ceil(len(tmp_data) / T)
        if length > 1:
            tmp_data = [tmp_data[i: i + length] for i in range(0, len(tmp_data), length)]
        for i in range(len(tmp_data)):
            if length > 1:
                current_data = np.concatenate(tmp_data[i], axis=0)
            else:
                current_data = tmp_data[i]
            dp = []
            real_data = []
            for point in current_data:
                x, y, label = point
                if label == 1:
                    real_data.append(point)
                else:
                    dp.append(point)
            dp = np.unique(np.array(dp), axis=0)
            real_data.extend(list(dp))
            np.random.shuffle(real_data)
            dataset = []
            for point in real_data:
                x, y, label = point
                feature = [0 for k in range(8)]
                for k in range(9):
                    x_axis, y_axis = int(k / 3) - 1, k % 3 - 1

                    if x_axis == 1 and y_axis == 1:
                        continue
                    x_axis, y_axis = x + x_axis, y + y_axis
                    if x_axis < 0 or x_axis >= self.room.shape[0] or y_axis < 0 or y_axis >= self.room.shape[1]:
                        continue
                    else:
                        if k >= 4:
                            index = k - 1
                        else:
                            index = k
                        feature[index] = self.room[x_axis][y_axis]
                feature.append(label)
                feature.insert(0, y)
                feature.insert(0, x)
                print(feature, self.room.shape)
                dataset.append(np.asarray(feature))
            self.data.append(np.stack(dataset, axis=0))
        return self.data

    def positionSample(self, pos):
        res_total = []
        for direct in Robot.directions:
            dx, dy = self.samp_step * direct[0], self.samp_step * direct[1]
            res_array = self.directionSample(pos, (dx, dy))
            res_total.extend(res_array)
        res_total = list(res_total)  # remove repetition samples
        return np.array(res_total, dtype=int)

    def directionSample(self, pos, dstep):
        res_arr = []
        ptr, temp = pos, pos
        while len(res_arr) < detection_max:
            if not (ptr[0] in range(self.room.shape[0]) and ptr[1] in range(self.room.shape[1])):
                break
            if self.isBlocked(ptr):
                break
            if ptr not in res_arr:
                res_arr.append((ptr[0], ptr[1], self.detecting(0)))  # Sample a point which isn't blocked
            temp = updatePos(temp, dstep)
            ptr = floor(temp[0]), floor(temp[1])
        if len(res_arr) < detection_max:
            res_arr.append((ptr[0], ptr[1], self.detecting(1)))  # Sample the blocked point
        return res_arr

    def detecting(self, no_bias: int):
        return random.gauss(mu=no_bias, sigma=sqrt(self.samp_var))

    def setPath(self, critical_list, var=0):
        """
        Get path from a list of critical points, using CriticalPath.getPath().

        CAUTION: Guarantee no obstacle between every adjacent coordinates in
        your critical path list.

        :param var: variance used in generating rand critical points.
        :param critical_list: list of critical points.
        :return: list of aggregate moving path.
        """
        for pos in critical_list:
            self.cp.randCriticalPoint(pos, var)
        self.path = self.cp.getPath()

    def isBlocked(self, pos):
        prob = 0.9              # sample correct probability
        x, y = pos
        assert x in range(self.room.shape[0]) and y in range(self.room.shape[1])  # Size overflow
        if self.room[x, y] == 1:
            return np.random.random() < prob
        return np.random.random() > prob
