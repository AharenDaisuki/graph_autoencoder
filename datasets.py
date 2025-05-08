import os
import glob
import torch
import numpy as np
import torch_geometric.transforms as transforms

from tqdm import tqdm
from utils import load_from_pkl
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

def get_sumo_dataloader(data_dir: str, batch_size: int = 1, shuffle: bool = True) -> DataLoader: 
    def _check_data_integrity(folder):
        all_files = [os.path.join(folder, raw_file) for raw_file in raw_files] + [os.path.join(folder, add_file) for add_file in add_files]
        return all([os.path.exists(file) for file in all_files]) 
    assert os.path.exists(data_dir)
    data_list = []
    raw_files = ['fcd.xml', 'q.pkl', 'k.pkl', 'v.pkl', 'x.pkl']
    add_files = ['a.pkl']
    num_edges = 0
    num_node_u = 0
    num_node_v = 0
    num_graphs = 0
    bar = tqdm(glob.glob(os.path.join(data_dir, '*', '*')), desc=f'load {data_dir}...'.ljust(20))
    for folder in bar: 
        _check_data_integrity(folder = folder)
        q_mat = load_from_pkl(pkl_path=os.path.join(folder, raw_files[1]))
        k_mat = load_from_pkl(pkl_path=os.path.join(folder, raw_files[2]))
        v_mat = load_from_pkl(pkl_path=os.path.join(folder, raw_files[3]))
        x_mat = load_from_pkl(pkl_path=os.path.join(folder, raw_files[4]))
        a_mat = load_from_pkl(pkl_path=os.path.join(folder, add_files[0]))
        assert isinstance(q_mat, np.ndarray)    
        assert isinstance(k_mat, np.ndarray)
        assert isinstance(v_mat, np.ndarray)
        assert isinstance(x_mat, np.ndarray)
        assert isinstance(a_mat, dict)
        I, J, K = x_mat.shape
        L, T = q_mat.shape
        E = len(a_mat)
        
        data = HeteroData()
        # traffic demand feature (I * J * K = 10 * 10 * 20 = 2000)
        u_features = x_mat.reshape(-1, 1)
        data['demand'].x = torch.tensor(u_features, dtype=torch.float) # [num_origin_destination, num_features] => [I * J * K, 1(?)]
        assert u_features.shape == (I * J * K, 1)
        # measurement feature (L * T = 300 * 20 = 6000)
        v_features = np.array([[q_mat[l, t], k_mat[l, t], v_mat[l, t]] for l in range(L) for t in range(T)], dtype=np.float64)
        data['measurement'].x = torch.tensor(v_features, dtype=torch.float) # [num_detectors, num_features] => [L * T, 3]
        assert v_features.shape == (L * T, 3)
        # adjacent matrix
        adj_matrix = np.array([[i*J+j*K+k for (i, j, k, l, t) in a_mat.keys()],[l*T+t for (i, j, k, l, t) in a_mat.keys()]], dtype=np.int32)
        data['demand', 'contributes_to', 'measurement'].edge_index =  torch.tensor(adj_matrix, dtype=torch.long)# [2, num_edges] => ([ [from], [to] ])
        assert adj_matrix.shape == (2, E)
        # edge weight
        edge_features = np.array([[v] for v in a_mat.values()], dtype=np.float64)
        data['demand', 'contributes_to', 'measurement'].edge_attr = torch.tensor(edge_features, dtype=torch.float) # [num_edges, num_features] 
        assert edge_features.shape == (E, 1)
        # edge label
        data['demand', 'contributes_to', 'measurement'].edge_label = torch.tensor(list(a_mat.values()))

        # data transformation
        transform_modules = transforms.Compose([
            transforms.RemoveIsolatedNodes(), 
            transforms.RemoveDuplicatedEdges(),
            transforms.ToDevice(device=0)
        ])
        data = transform_modules(data)
        data_list.append(data)
        num_graphs += 1
        num_node_u += data['demand'].num_nodes
        num_node_v += data['measurement'].num_nodes
        num_edges  += data['demand', 'contributes_to', 'measurement'].num_edges
    print(f'[summary] #node_u: {num_node_u} | #node_v: {num_node_v} | #node_total: {num_node_u + num_node_v} | #edges: {num_edges} | #graphs: {num_graphs}\n')
    return DataLoader(data_list, batch_size = batch_size, shuffle = shuffle)
    
