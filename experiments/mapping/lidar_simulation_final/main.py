import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from classification_train import MappingUnroll

# Loading datasets
train_mapset_path = 'dataset/mapping_check_train.npy'
test_mapset_path = 'dataset/mapping_check_test.npy'
train_check_set = np.load('dataset/train_mapset.npy', allow_pickle=True)
test_check_set = np.load('dataset/test_mapset.npy', allow_pickle=True)

# Configure
train_save_path = 'dataset/sample_check_train.pkl'
test_save_path = 'dataset/sample_check_test.pkl'

agent_num = 4
t = 12
feature_dim = 10

model_path = os.path.join('model', 'nocollab_phi_0.001_4_10_directions90.pt')
hyper_param_path = os.path.join('param', 'nocollab_param_0.001_4_10_directions90.npy')
if not os.path.exists('imgs'):
    os.mkdir('imgs')

solver = MappingUnroll(hyper_param=torch.tensor([0, 0.1, 0.1]),
                       feature_dim=feature_dim, T=t, agent_num=agent_num)
# solver.train(lr=1e-3, batch_size=4, epochs=10, with_collaboration=False)
# solver.train(lr=[1e-3, 1e-3], batch_size=4, epochs=10, with_collaboration=True)

sample_index = 1
img_name = os.path.join('imgs', 'gray_room_%s.pdf' % sample_index)
fig, axes = plt.subplots(2, 5, figsize=(20, 10))

test_f1, axes[0, :] = solver.display(sample_index=sample_index, model_path=model_path,
                                     hyper_param_path=hyper_param_path,
                                     with_collaboration=False, mode='test', axes=axes[0, :])
# train_f1 = solver.display(sample_index=sample_index, model_path=model_path, hyper_param_path=hyper_param_path,
#                           with_collaboration=False, mode='train')

model_path = os.path.join('model', 'collaboration_phi_[0.001, 0.001]_4_10_direction90.pt')
hyper_param_path = os.path.join('param', 'collaboration_param_[0.001, 0.001]_4_10_directions90.npy')

test_f1_col, axes[1, :] = solver.displayWithTime(sample_index=sample_index, model_path=model_path,
                                                 hyper_param_path=hyper_param_path, axes=axes[1, :],
                                                 with_collaboration=True, mode='test')
# train_f1_col = solver.display(sample_index=sample_index, model_path=model_path, hyper_param_path=hyper_param_path,
#                               with_collaboration=True, mode='train')

# print(np.around(test_f1_col, 4))
# print(np.around(test_f1, 4))

# print(test_f1_col, test_f1)
# plt.show()
# exit(0)

plt.savefig(img_name, format='pdf', bbox_inches='tight')
plt.close()
ifplot = False

if ifplot:
    time = np.arange(1, t + 1)
    plt.plot(time, np.array(test_f1), time, np.array(test_f1_col))
    plt.xlabel('time')
    plt.ylabel('F1 Score')
    plt.legend(['no collaboration', 'collaboration'])
    plt.savefig(os.path.join('imgs', f'gray_score_room_{sample_index}.pdf'), format='pdf', bbox_inches='tight')
