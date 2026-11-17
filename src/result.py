import numpy as np
import torch

from classification_v2 import ClassificationResultDisplay
# from classification import ClassificationResultDisplay
import os


def result(
        task,
        task_path, param_path, model_path,
        graph_method, graph_param, is_collab, is_mem
        ):
    acc1, loss1, adj1 = result_displayer.result(
        task, 
        param_path,
        model_path, 
        with_collaboration=is_collab, 
        with_memory=is_mem, 
        method=graph_method,
        graph_hyper=graph_param
    )

    print(np.mean(acc1.numpy(), axis=-1))
    collab_flag = 'collab' if is_collab else 'nocollab'
    mem_flag = 'mem' if is_mem else 'nomem'
    
    np.save(os.path.join(task_path, f'acc_{collab_flag}_{mem_flag}_{graph_method}_{graph_param}.npy'), acc1.numpy())
    np.save(os.path.join(task_path, f'adj_{collab_flag}_{mem_flag}_{graph_method}_{graph_param}.npy'), adj1.numpy())
    
    print(loss1)

if __name__ == '__main__':

    base_dirctory = 'result'

    if not os.path.exists(base_dirctory):
        os.mkdir(base_dirctory)
    classification = ['mnist', 'cifar10']
    regression = ['regression']
    task = 'cifar10'
    graph_methods = [
        'dual_ascend', 'glasso', 'oracle',
        'mtrl', 'pds', 'admm',
        'gl_sigrep', 'gl_logdet', 'neighbor_select'
    ]

    IS_COLLAB = True
    IS_MEM = True

    param_path = f'param/collaboration_param_[0.0001, 0.001]_2_{task}_100.npy'
    model_path = f'model/collaboration_phi_[0.0001, 0.001]_2_{task}_100.pt'

    float_type = torch.float32 if task == 'cifar10' else torch.float32

    task_path = os.path.join(base_dirctory, task)
    if not os.path.exists(task_path):
        os.mkdir(task_path)

    if task in classification:
        result_displayer = ClassificationResultDisplay(agent_num=6, n_classes=5, feature_dim=50, T=10,
                                                        float_type=float_type)
        
        for graph_method in graph_methods:
            print(f'####Doing the method: {graph_method}')
            if graph_method not in ['dual_ascend', 'oracle', 'mtrl', 'pds', 'admm']:
                for graph_param in [0.01, 0.1, 1, 10]:
                    print(f'Using hyper param: {graph_param}')
                    result(task=task,
                            task_path=task_path, param_path=param_path, model_path=model_path,
                            graph_method=graph_method, graph_param=graph_param,
                            is_collab=IS_COLLAB, is_mem=IS_MEM)
            else:
                result(task=task,
                        task_path=task_path, param_path=param_path, model_path=model_path,
                        graph_method=graph_method, graph_param=0.1,
                        is_collab=IS_COLLAB, is_mem=IS_MEM)
        
        


# oracle 
# [0.47246668 0.6148667  0.6731666  0.71016675 0.74033326 0.75823325 0.7642667  0.7715001  0.78006667 0.78356665]

# dual_ascend
# [0.42366663 0.55979997 0.65356666 0.7075334  0.74716663 0.76430005 0.77003336 0.7774001  0.7873334  0.7887666 ]

# gl_sigrep
# lambda=10
# [0.42626667 0.55573326 0.6329333  0.6743     0.7033333  0.7193666 0.7273667  0.7350001  0.7439666  0.74639994]

# lambda=1
# [0.4263333  0.5572333  0.6367     0.67996675 0.71029997 0.72650003 0.7349334  0.7414333  0.7520666  0.7534334 ]

# lambda=0.1
# [0.4226333  0.57346666 0.6708334  0.7230334  0.7559333  0.7689999 0.77290004 0.77893335 0.7880333  0.7893333 ]

# lambda=0.01
# [0.3743666  0.5234334  0.6083999  0.66483337 0.69696647 0.7202 0.73946667 0.7520333  0.76690006 0.7705999 ]

# gl_logdet
# lambda=10
# [0.33129993 0.43123332 0.51443326 0.56506675 0.609      0.6501 0.67953324 0.6993     0.7209334  0.7328    ]

# lambda=1
# [0.3469999  0.4198334  0.50573343 0.572      0.6157334  0.6546666 0.68236667 0.7066     0.7295     0.7433999 ]

# lambda=0.1
# [0.34316668 0.51846665 0.6427334  0.70546675 0.7492666  0.76476663 0.77113336 0.7788001  0.78689986 0.79076654]

# lambda=0.01
# [0.3541     0.5244667  0.6422667  0.6915     0.7376     0.7551999 0.7635667  0.76940006 0.7816667  0.78609985]
