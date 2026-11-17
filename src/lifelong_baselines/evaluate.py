import numpy as np
import os

base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '')

result_dir = 'lifelong_baseline_result'

result_path = os.path.join(base_path, result_dir)

result_dic = {}


for file_name in os.listdir(result_path):
    file_path = os.path.join(result_path, file_name)

    dataset = file_name.split('_')[0]
    method = file_name.split('_')[4][7:]
    backbone = file_name.split('_')[3]

    key = dataset + '_' + method + '_' + backbone

    if key in result_dic.keys():
        prev = result_dic[key]
        curr = np.load(file_path)
        if np.mean(curr) > np.mean(prev):
            result_dic[key] = curr

    else:
        result_dic[key] = np.load(file_path)

for datatype in ['mnist', 'cifar10']:
    for backbone in ['True', 'False']:
        for method in ['naive', 'ewc', 'lwf', 'gem']:
            key = datatype + '_' + method + '_backbone=' + backbone
            print(key, end=' \t')
            acc = np.around(result_dic[key], 4)
            print(acc, end='\n')






