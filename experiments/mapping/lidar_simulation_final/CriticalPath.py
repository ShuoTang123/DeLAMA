from math import sqrt, floor
import numpy as np


def distance(pos1, pos2=(0, 0)):
    x1, y1 = pos1
    x2, y2 = pos2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class CriticalPath(object):
    def __init__(self, matrix, step):
        self.room = matrix
        self.critical_points = []
        self.step = step

    def getPath(self):
        """
        Given the ordered critical points, return a path that passes through all
        critical points and connection points among them.

        :return: list[tuple(x, y)], which contains all the positions of the path.
        """
        if len(self.critical_points) == 0:
            raise Exception("You didn't add any critical points!")
        if len(self.critical_points) == 1:
            raise Exception("You should add one more critical point!")

        start = self.critical_points[0]
        res = [start]
        for end in self.critical_points[1:]:
            res.extend(self.connect2Point(start, end))
            res.append(end)
            start = end

        return res

    def randCriticalPoint(self, pos, var):
        """Give random gaussian noise to critical points."""
        x, y = pos
        while True:
            rand = np.random.randint((-var, -var), (var + 1, var + 1))
            current_pos = x + rand[0], y + rand[1]
            if not self.isBlocked(current_pos):
                self.critical_points.append(current_pos)
                break

    def connect2Point(self, pos1, pos2):
        """
        Connect 2 critical points(excluded), sampling the connection-line at step_length.

        :param pos1: tuple(x0, y0) The starting position.
        :param pos2: tuple(x1, y1) The ending position
        :return: list[tuple(x, y)] excluding pos1 and pos2
        """
        d = distance(pos1, pos2)
        samp_num = floor(d / self.step) - 1
        if samp_num <= 0:
            return []
        dx0, dy0 = pos2[0] - pos1[0], pos2[1] - pos1[1]
        dx, dy = dx0 / d * self.step, dy0 / d * self.step

        res, temp = [pos1, pos2], pos1
        for _ in range(samp_num):
            temp = temp[0] + dx, temp[1] + dy
            ptr = round(temp[0]), round(temp[1])  # update ptr position
            if ptr not in res:
                if not self.isBlocked(ptr):
                    res.append(ptr)
        return res[2:]  # ignore starting and ending position

    def isBlocked(self, pos):
        x, y = pos
        assert x in range(self.room.shape[0]) \
               and y in range(self.room.shape[1])  # Size overflow
        return self.room[pos] == 1
