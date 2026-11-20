import torch
from sklearn.covariance import GraphicalLasso
import time
try:
    import cvxpy as cp
except ImportError:
    cp = None
import numpy as np

# device = torch.device('cuda:0')


def normalize(adj: torch.tensor):
    assert adj.shape[0] == adj.shape[1]

    diagonal = torch.diag(torch.diag(adj))
    adj = adj - diagonal
    edge_num = adj.shape[0] * (adj.shape[0] - 1) / 2
    adj = adj / (torch.sum(adj) + 1e-5) * edge_num * 2
    return adj


def distance_vec(beta_s):
    agent_num = len(beta_s)
    device = beta_s.device

    beta_tmp = torch.repeat_interleave(beta_s.unsqueeze(-1), agent_num, dim=2)
    permuted_beta = torch.permute(beta_tmp, (2, 1, 0))
    diff_map = torch.sum((permuted_beta - beta_tmp) ** 2, dim=1) + 1e10 * torch.eye(agent_num, device=device)

    return diff_map


def dual_ascent(beta_s, param, *args):
    """
    :param beta_s: model parameters
    :param param: hyper parameter
    t: dual ascent step size
    """
    t_start = time.time()
    agent_num = len(beta_s)
    device = beta_s.device

    lambda2, lambda3 = param[1], param[2]
    edge_num = int(agent_num * (agent_num - 1) / 2)
    b = 1e-4
    y = distance_vec(beta_s)

    if len(args) >= 1:
        communication_graph = args[0]
    else:
        communication_graph = torch.ones((agent_num, agent_num), device=device) - torch.eye(agent_num, device=device)

    edge_mask = (communication_graph > 0).to(dtype=y.dtype, device=device)
    edge_mask = edge_mask * (1.0 - torch.eye(agent_num, device=device, dtype=y.dtype))
    y_masked = y + 1e10 * (1.0 - edge_mask)
    z = -lambda2 * torch.min(y_masked)
    for k in range(5):
        u = (-1 / lambda3) * (lambda2 * y + z)
        sp = (u + (u ** 2 + b) ** 0.5) / 2
        sp_p = 0.5 * (1 + u / (u ** 2 + b) ** 0.5)
        sp_pp = b / (2 * (u ** 2 + b) ** 1.5)
        w = sp * edge_mask

        f = torch.sum((u * sp_p + sp * (1 - sp_p)) * edge_mask)
        f_hat = torch.sum((2 * sp_p - sp_p ** 2 + (u - sp) * sp_pp) * edge_mask)

        f = f - 2 * edge_num
        f_hat = (-1 / lambda3) * f_hat

        z = z - f / f_hat

    adj_matrix = w / (torch.sum(w) + 1e-5) * 2 * edge_num
    # print(torch.sum(adj_matrix))
    t_end = time.time()
    return adj_matrix, t_end-t_start


def distributed_dual_ascent(beta_s, param, collab_will, *args):
    t_start = time.time()
    agent_num = len(beta_s)
    assert len(collab_will) == agent_num
    device = beta_s.device

    if len(args) >= 1:
        communication_graph = args[0]

    else:
        communication_graph = torch.ones((agent_num, agent_num), device=device) - torch.eye(agent_num, device=device)

    communication_mask = communication_graph + torch.eye(agent_num, device=device)

    lambda2, lambda3 = param[1], param[2]
    b = 1e-4
    y = distance_vec(beta_s) * communication_mask
    # print(np.around(y.cpu().numpy(), 2))

    z = - lambda2 * torch.min(y + 1e10 * (1-communication_mask), dim=0).values
    for k in range(30):
        u = (-1 / lambda3) * (lambda2 * y  + z) * communication_mask
        sp = (u + (u ** 2 + b) ** 0.5) / 2
        sp_p = 0.5 * (1 + u / (u ** 2 + b) ** 0.5)
        sp_pp = b / (2 * (u ** 2 + b) ** 1.5)
        w = sp

        f = torch.sum(u * sp_p + sp * (1 - sp_p), dim=0)
        f_hat = torch.sum(2 * sp_p - sp_p ** 2 + (u - sp) * sp_pp, dim=0)

        f = f - collab_will
        f_hat = (-1 / lambda3) * f_hat
        z = z - f / f_hat

    adj_matrix = w.T
    t_end = time.time()
    return adj_matrix, t_end - t_start


def DMTL(beta_s):
    t_start = time.time()
    agent_num = len(beta_s)
    prod = beta_s @ beta_s.T
    u, lamda, v = torch.linalg.svd(prod)
    adj = u @ torch.sqrt(torch.diag(lamda)) @ v
    adj = normalize(adj)
    t_end = time.time()
    return adj, t_end-t_start


def glasso(betas, alpha):
    t_start = time.time()
    device, dtype = betas.device, betas.dtype
    betas = betas.cpu().numpy()
    agent_num = len(betas)
    # try:
    #     solver = GraphicalLasso(alpha=1e-5, mode='lars', tol=0.001, max_iter=1000, verbose=False, assume_centered=True)

    #     solver.fit(betas.T)
    #     adj_matrix = solver.get_precision()
    #     adj_matrix = torch.tensor(adj_matrix, device=device, dtype=dtype)
    # except:
    #     print('Float error')
    #     adj_matrix = torch.zeros((agent_num, agent_num), device=device)
    Sigma = betas @ betas.T / agent_num
    W = cp.Variable((agent_num, agent_num))

    prob = cp.Problem(cp.Minimize(-cp.log_det(W) + cp.trace(Sigma @ W) + alpha * cp.pnorm(W, 1)))

    prob.solve(max_iters=100)

    W = torch.tensor(W.value, device=device, dtype=dtype)

    adj = normalize(W)

    t_end = time.time()
    return adj, t_end-t_start


def findD(num, device, dtype):
    D = torch.zeros((num, num * (num - 1) // 2), device=device, dtype=dtype)
    start = 0
    for i in range(num):
        for j in range(start, start + num - 1 - i):
            D[i, j] = 1
        for j in range(i + 1, num):
            k = start + j - i - 1
            D[j, k] = 1
        start += num - 1 - i
    return D


def pds(theta_s, gamma=1e-2, alpha=1e2, beta=1e-2):
    t_start = time.time()
    y = getDistance(theta_s)
    max_iter = 500
    m = theta_s.shape[0]
    agent_num = len(theta_s)
    device, dtype = theta_s.device, theta_s.dtype

    D = findD(m, device, dtype)
    n = D.shape[1]
    w, v = torch.zeros((n, 1), device=device, dtype=dtype)+1e-3, torch.zeros((m, 1), device=device, dtype=dtype)+1e-3
    w_prev = w + 100
    cnt = 0
    while torch.linalg.norm(w - w_prev, 1) >= 1e-3 and cnt < max_iter:
        r1 = w - gamma * (2 * beta * w + 2 * y + D.T @ v)
        r2 = v + gamma * D @ w
        p1 = torch.relu(r1)
        p2 = 0.5 * (r2 - torch.sqrt(torch.square(r2) + 4 * alpha * gamma))
        q1 = p1 - gamma * (2 * beta * p1 + 2 * y + D.T @ p2)
        q2 = p2 + gamma * D @ p1
        w_prev = w
        w = w - r1 + q1
        v = v - r2 + q2
        cnt += 1
    # w -> W
    W = torch.zeros((m, m), device=device, dtype=dtype)
    cnt = 0
    for i in range(0, m):
        for j in range(i + 1, m):
            W[i, j] = w[cnt]
            cnt += 1
    W = W + W.T
    W = normalize(W)
    t_end = time.time()
    return W, t_end-t_start


def admm(theta_s, gamma=1e-2, alpha=1e2, beta=1e-2, lamda=1.8):
    t_start = time.time()
    y = getDistance(theta_s)
    max_iter = 500
    m = theta_s.shape[0]
    agent_num = len(theta_s)
    device, dtype = theta_s.device, theta_s.dtype

    D = findD(m, device, dtype)
    n = D.shape[1]
    w, v = torch.zeros((n, 1), device=device, dtype=dtype), torch.zeros((m, 1), device=device, dtype=dtype)

    for i in range(max_iter):
        r1 = w - gamma * (2 * beta * w + 2 * y + D.T @ v)
        p1 = torch.relu(r1)
        r2 = v + gamma * D @ (2 * r1 - w)
        p2 = 0.5 * (r2 - torch.sqrt(torch.square(r2) + 4 * alpha * gamma))
        w = w + lamda * (p1 - w)
        v = v + lamda * (p2 - v)

    # print(D @ w)

    W = torch.zeros((m, m), device=device, dtype=dtype)
    cnt = 0
    for i in range(0, m):
        for j in range(i + 1, m):
            W[i, j] = w[cnt]
            cnt += 1
    W = W + W.T
    W = normalize(W)
    t_end = time.time()

    return W, t_end-t_start


def getDistance(theta):
    m, n = theta.shape
    device, dtype = theta.device, theta.dtype
    y = torch.zeros((m * (m - 1) // 2, 1), device=device, dtype=dtype)
    cnt = 0
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            y[cnt] = torch.sum(torch.square(theta[i, :] - theta[j, :]))
            cnt += 1
    return y


def gl_sigrep(theta, lamda):
    t_start = time.time()
    agent_num = len(theta)
    m = agent_num * (agent_num - 1)
    device, dtype = theta.device, theta.dtype
    L = cp.Variable((agent_num, agent_num))
    theta = theta.to('cpu').numpy()
    prob = cp.Problem(cp.Minimize(cp.trace(theta.T @ L @ theta) + lamda * cp.sum_squares(L)), 
                      [cp.trace(L) == m,
                       L == L.T,
                       L - cp.diag(cp.diag(L)) <= 0,
                       L @ np.ones(agent_num) == np.zeros(agent_num)])
    
    prob.solve()
    laplacian = L.value
    adj = np.diag(np.diag(laplacian)) - laplacian
    adj = torch.tensor(adj, device=device, dtype=dtype)
    # adj = normalize(adj)
    t_end =time.time()
    return adj, t_end - t_start
    



def gl_logdet(theta, lamda):
    t_start = time.time()
    agent_num, p = theta.shape[0], theta.shape[1]
    device, dtype = theta.device, theta.dtype
    theta = theta.to('cpu').numpy()
    A  = theta @ theta.T

    W = cp.Variable((agent_num, agent_num))
    L_pre = cp.Variable((agent_num, agent_num))
    sigma_square = cp.Variable()
    
    prob = cp.Problem(cp.Minimize(cp.trace(L_pre @ A / p) - cp.log_det(L_pre) + lamda * cp.pnorm(W, 1) / p),
                    [
                        cp.diag(W) == np.zeros(agent_num),
                        W >= 0,
                        -sigma_square <= 0,
                        L_pre >> 0,
                        L_pre  == (cp.diag(W @ np.ones(agent_num)) - W)  + np.eye(agent_num) * sigma_square
                    ])
    
    prob.solve()
    W = W.value
    t_end = time.time()
    return torch.tensor(W, device=device, dtype=dtype), t_end - t_start



def neighbor_select(theta, lamda):
    t_start = time.time()
    w_set = []
    device, dtype = theta.device, theta.dtype
    theta = theta.to('cpu').numpy()
    agent_num = len(theta)
    for i in range(agent_num):
        target_param = theta[i]
        others_param = np.delete(theta, i, axis=0)
        w = cp.Variable(agent_num-1)
        
        prob = cp.Problem(cp.Minimize(cp.sum_squares(target_param - others_param.T @ w) + lamda * cp.pnorm(w, 1)))

        prob.solve(solver=cp.ECOS)

        w = w.value
        if i != agent_num - 1:
            w = np.insert(w, i, 0)
        else:
            w = np.append(w, 0)
        w_set.append(w)
    
    W = np.stack(w_set)

    adj = torch.tensor(W, dtype=dtype, device=device)
    t_end = time.time()
    return normalize(adj), t_end - t_start



# if __name__ == '__main__':
#     param_path = 'param/collaboration_param_[0.001, 0.0001]_2_mnist_100.npy'
#     communicate_graph_set_path = 'dataset/communication_graph_BA_low.npy'
#     device = torch.device('cpu')
#     import numpy as np
#     import networkx as nx
#     param = torch.tensor(np.load(param_path), device=device)
#     theta = torch.tensor(np.load('theta.npy'), device=device)

#     # adj = torch.tensor(np.load(communicate_graph_set_path)[0], device=device)
#     # print(adj)

#     # res1, _ = dual_ascent(theta,  param, adj)
#     # print(np.around(res1.numpy(), 1))
#     # res2, _ = dual_ascent(theta, param)
#     # print(np.around(res2.numpy(), 1))
#     # gl_sigrep(theta, 10)
#     print(gl_logdet(theta, 1))







