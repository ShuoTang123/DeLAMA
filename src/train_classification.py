from classification_v2 import CollaborativeLearning
import torch
import os


if __name__ == '__main__':
    device = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{device}'
    solver = CollaborativeLearning(agent_num=6, n_classes=5, feature_dim=50, T=10, float_type=torch.float32)

    solver.train(lr=[1e-3, 1e-3], batch_size=2, dataset_type='cifar10', epochs=100, with_collaboration=True,
                param_path='param/collaboration_param_[0.0001, 0.001]_2_cifar10_100.npy',
                model_path='model/collaboration_phi_[0.0001, 0.001]_2_cifar10_100.pt')

    
