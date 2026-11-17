from nonlinear_regression import RegressionResultDisplay
import torch
import numpy as np
import os

graph_methods = [
        'dual_ascend', 'glasso',
        'mtrl', 'pds', 'admm',
        'gl_sigrep', 'gl_logdet', 'neighbor_select'
    ]

param_path = 'param/Param_params_lr=0.0001_phi_lr=0.001_type=0_Mon_Nov_14_22_20_59_2022.npy'
model_path = 'model/Phi_params_lr=0.0001_phi_lr=0.001_type=0_Mon_Nov_14_22_20_59_2022.pt'
save_dir = 'result/regression'



def bench_mark(setting='graph', device=0):
    if setting  == 'graph':
        for graph_method in graph_methods:
            print(f'Processing method {graph_method}')
            diplayer = RegressionResultDisplay(method=graph_method, device=device)
            if graph_method == 'glasso':
                graph_params = [1e-4, 1e-3, 1e-2, 1e-1, 1]
            elif graph_method in ['gl_sigrep', 'gl_logdet', 'neighbor_select']:
                graph_params = [0.1, 1, 10, 1e2, 1e3]
            else:
                graph_params = [0.1]

            results, gmse_results = [], []
            min_index, min_value = -1, 1e23
            for i, graph_param in enumerate(graph_params):
                res, gmse = diplayer.regression_res_display(model_path, param_path, graph_param)
                results.append(res.numpy())
                gmse_results.append(gmse.numpy())
                
                if torch.mean(res).item() < min_value:
                    min_value = torch.mean(res).item()
                    min_index = i

            mse_path = os.path.join(save_dir, f'mse_{graph_method}_{graph_param}.npy')
            gmse_path = os.path.join(save_dir, f'gmse_{graph_method}_{graph_param}.npy')
            mse_data, gmse_data = results[min_index], gmse_results[min_index]
            np.save(mse_path, mse_data)
            np.save(gmse_path, gmse_data)
    else:
        graph_method = 'dual_ascend'
        delama = RegressionResultDisplay(setting='colla_mem', method=graph_method, device=device)
        delama_wm = RegressionResultDisplay(setting='colla_nomem', method=graph_method, device=device)
        delama_wc = RegressionResultDisplay(setting='nocolla_mem', device=device)

        delama_mse_path = os.path.join(save_dir, f'delama_mse_{graph_method}.npy')
        delama_gmse_path = os.path.join(save_dir, f'delama_gmse_{graph_method}.npy')
        delama_mse_data, delama_gmse_data = delama.regression_res_display(model_path, param_path)
        np.save(delama_mse_path, delama_mse_data.numpy())
        np.save(delama_gmse_path, delama_gmse_data.numpy())
        print('delama', np.around(delama_mse_data.numpy(), 4))

        delama_wm_mse_path = os.path.join(save_dir, f'delama_wm_mse_{graph_method}.npy')
        delama_wm_gmse_path = os.path.join(save_dir, f'delama_wm_gmse_{graph_method}.npy')
        delama_wm_mse_data, delama_wm_gmse_data = delama_wm.regression_res_display(model_path, param_path)
        np.save(delama_wm_mse_path, delama_wm_mse_data.numpy())
        np.save(delama_wm_gmse_path, delama_wm_gmse_data.numpy())
        print('delama_wm', np.around(delama_wm_mse_data.numpy(), 4))

        delama_wc_mse_path = os.path.join(save_dir, f'delama_wc_mse.npy')
        delama_wc_gmse_path = os.path.join(save_dir, f'delama_wc_gmse.npy')
        delama_wc_mse_data, delama_wc_gmse_data = delama_wc.regression_res_display(model_path, param_path)
        np.save(delama_wc_mse_path, delama_wc_mse_data.numpy())
        np.save(delama_wc_gmse_path, delama_wc_gmse_data.numpy())
        print('delama_wc', np.around(delama_wc_mse_data.numpy(), 4))




if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    setting = 'graph'
    device = 6
    bench_mark(setting=setting, device=device)