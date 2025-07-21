import torch
import torch.nn.functional as F

from utils import save_as_pkl, get_sparse_matrix_alpha
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds

from torch import Tensor
from torch_geometric.utils import spmm
from torch_geometric.data import HeteroData
import numpy as np
from gurobipy import * 

# root mean square error
def rmse_loss(pred: Tensor, target: Tensor): 
    assert pred.shape == target.shape
    return F.mse_loss(pred, target).sqrt()

def nrmse_loss(pred: Tensor, target: Tensor): 
    assert pred.shape == target.shape
    target_range = torch.max(target) - torch.min(target)
    return F.mse_loss(pred, target).sqrt() / target_range

# huber loss
def huber_loss(pred: Tensor, target: Tensor, delta: float = 0.25): 
    assert pred.shape == target.shape
    return F.huber_loss(pred, target, reduction='mean', delta=delta)

# pinball loss
def pinball_loss(pred: Tensor, target: Tensor, tau: float): 
    assert pred.shape == target.shape
    return torch.mean(torch.max((tau - 1) * (target - pred), tau * (target - pred)))

def node_v_recon_loss(data: HeteroData, alphas: Tensor):
    # u, v set
    x = data['demand'].x_origin.unsqueeze(1)
    y = data['measurement'].x_origin.unsqueeze(1)

    num_node_u = data['demand'].num_nodes
    num_node_v = data['measurement'].num_nodes
    # edge
    adj = data['measurement', 'rev_contributes_to', 'demand'].edge_index # [[to], [from]] => \alpha_ji * x_i
    assert adj.shape[-1] == alphas.shape[0], f'(2, E) = {adj.shape} | (E, ) = {alphas.shape}'

    sparse_alpha = torch.sparse_coo_tensor(indices=adj, values=alphas, size=(num_node_v, num_node_u), 
                                            dtype=x.dtype, device=x.device).to_sparse_csr()
    y_pred = spmm(src=sparse_alpha, other=x)
    return nrmse_loss(y_pred, y)
    # return huber_loss(y_pred, y, delta=30.0)

def node_u_recon_loss(data: HeteroData, alphas: Tensor): 
    # vector x [n \times 1], y [m \times 1]
    x = data['demand'].x_origin.unsqueeze(1) 
    y = data['measurement'].x_origin.unsqueeze(1)
    num_node_u = data['demand'].num_nodes
    num_node_v = data['measurement'].num_nodes
    # sparse matrix A [m \times n]
    adj = data['measurement', 'rev_contributes_to', 'demand'].edge_index
    adj_t = data['demand', 'contributes_to', 'measurement'].edge_index
    A = get_sparse_matrix_alpha(indices=adj, values=alphas, size=(num_node_v, num_node_u), data_type=x.dtype, device=x.device)
    AT = get_sparse_matrix_alpha(indices=adj_t, values=alphas, size=(num_node_u, num_node_v), data_type=x.dtype, device=x.device)
    AT_Y = spmm(src=AT, other=y)
    A_X = spmm(src=A, other=x)
    AT_A_X = spmm(src=AT, other=A_X)
    return nrmse_loss(AT_Y, AT_A_X)
    # return huber_loss(AT_Y, AT_A_X, delta=30.0)

    


