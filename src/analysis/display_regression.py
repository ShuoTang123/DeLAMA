import numpy as np

import os
import re

dir = 'regression'

graph_methods = [
        'dual_ascend', 'glasso',
        'mtrl', 'pds', 'admm',
        'gl_sigrep', 'gl_logdet', 'neighbor_select'
    ]

file_list = os.listdir(dir)

for graph_method in graph_methods:

    mse_pattern = re.compile(f'mse_{graph_method}')
    gmse_pattern = re.compile(f'gmse_{graph_method}')

    matched_mse = [f for f in file_list if mse_pattern.match(f)][0]
    matched_gmse = [f for f in file_list if gmse_pattern.match(f)][0]

    mse = np.load(os.path.join(dir, matched_mse))
    gmse = np.load(os.path.join(dir, matched_gmse))

    print(f'Method: {graph_method}')
    print(np.around(mse, 4))
    print(np.around(gmse, 4))