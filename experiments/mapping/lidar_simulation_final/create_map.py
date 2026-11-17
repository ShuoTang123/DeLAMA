import os.path
import random
import matplotlib.pyplot as plt
import numpy as np
import torch


def create_one_map(triangle_num=1, is_show=False):
    if is_show:
        plt.figure()

    init_map = torch.randint(0, 1, size=(100, 100)).to(torch.float)

    for k in range(triangle_num):
        random_position = torch.randint(0, 50, size=(2, 1)).squeeze()

        length, width = random.randint(20, 50), random.randint(20, 50)
        for i in range(random_position[0], random_position[0] + length):
            for j in range(random_position[1], random_position[1] + width):
                init_map[i][j] = 1

        if is_show:
            plt.imshow(init_map.numpy())

    for i in range(5):      # edge processing
        for j in range(100):
            init_map[i][j], init_map[99-i][j] = 0, 0
            init_map[j][i], init_map[j][99-i] = 0, 0

    if is_show:
        plt.show()
    return init_map


def create_check_set(size=200):
    train_path = 'dataset/mapping_check_train.npy'
    test_path = 'dataset/mapping_check_test.npy'
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    train_set, test_set = [], []
    for i in range(size):
        num = random.randint(1, 3)
        train_map = create_one_map(triangle_num=num).numpy()
        test_map = create_one_map(triangle_num=num).numpy()

        train_set.append(train_map)
        test_set.append(test_map)

    train_set_data = np.stack(train_set, axis=0)
    test_set_data = np.stack(test_set, axis=0)

    print(train_set_data.shape, test_set_data.shape)

    np.save(train_path, train_set_data)
    np.save(test_path, test_set_data)


# create_check_set(size=200)


