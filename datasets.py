import os
import glob
import torch
import numpy as np
import torch_geometric.transforms as transforms

from tqdm import tqdm
from utils import load_from_pkl
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

# TODO: [debug] get sumo dataloader
def get_sumo_trainloader(data_dir: str, batch_size: int = 1, shuffle: bool = True, swap_prob: float = 0.5) -> DataLoader: 
    def _check_data_integrity(folder):
        # all_files = [os.path.join(folder, raw_file) for raw_file in raw_files] + [os.path.join(folder, add_file) for add_file in add_files]
        all_files = [os.path.join(folder, raw_file) for raw_file in raw_files]
        return all([os.path.exists(file) for file in all_files]) 
    assert os.path.exists(data_dir)
    data_list = []
    raw_files = ['q.pkl', 'k.pkl', 'v.pkl', 'x.pkl', 'a.pkl', 'dist.pkl']
    add_files = ['metadata.pkl', 'fcd.xml']
    num_edges = 0
    num_node_u = 0
    num_node_v = 0
    num_graphs = 0
    avg_density = 0
    # sim_state => sim_scene => sim_index
    bar = tqdm(glob.glob(os.path.join(data_dir, '*', 'scenario_*')), desc=f'load {data_dir}...'.ljust(20))
    for sim_scene in bar:
        dist_data = load_from_pkl(pkl_path=os.path.join(sim_scene, raw_files[-1]))
        mu, sigma = dist_data['mu'], dist_data['sigma']
        for sim_index in glob.glob(os.path.join(sim_scene, '[1-1000]')): 
            _check_data_integrity(folder = sim_index)
            # raw data
            q_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[0]))
            k_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[1]))
            v_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[2]))
            x_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[3]))
            a_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[4]))
            # additional data
            if os.path.exists(os.path.join(sim_index, add_files[0])): 
                meta_data = load_from_pkl(pkl_path=os.path.join(sim_index, add_files[0]))
            else: 
                meta_data = None
            
            assert isinstance(q_mat, np.ndarray)    
            assert isinstance(k_mat, np.ndarray)
            assert isinstance(v_mat, np.ndarray)
            assert isinstance(x_mat, np.ndarray)
            assert isinstance(mu, np.ndarray)
            assert isinstance(sigma, np.ndarray)
            assert isinstance(a_dict, dict)

            x_mask = np.random.normal(loc=mu, scale=sigma, size=x_mat.shape)
            if meta_data: 
                q_mag, k_mag, v_mag, x_mag = meta_data['q_mag'], meta_data['k_mag'], meta_data['v_mag'], meta_data['x_mag']
                x_mask /= x_mag

            I, J, K = x_mat.shape
            L, T = q_mat.shape
            E = len(a_dict)
            
            data = HeteroData()
            # TODO: setup1
            # traffic demand feature (I * J * K = 10 * 10 * 20 = 2000)
            # u_features = x_mat.reshape(-1, 1)
            # data['demand'].x = torch.tensor(u_features, dtype=torch.float) # [num_origin_destination, num_features] => [I * J * K, 1(?)]
            # assert u_features.shape == (I * J * K, 1)

            # TODO: setup2
            # if swap_prob:
            #     u_features = np.array([
            #         [x_mat[i, j, k], mu[i, j, k], sigma[i, j, k]] if np.random.rand() > swap_prob else [mu[i, j, k], x_mat[i, j, k], sigma[i, j, k]] 
            #         for i in range(I) for j in range(J) for k in range(K)
            #     ], dtype=np.float64) 
            # else: 
            #     u_features = np.array([
            #         [x_mat[i, j, k], mu[i, j, k], sigma[i, j, k]] 
            #         for i in range(I) for j in range(J) for k in range(K)
            #     ], dtype=np.float64)
            # assert u_features.shape == (I * J * K, 2)
            # data['demand'].x = torch.tensor(u_features, dtype=torch.float) # [num_origin_destination, num_features] => [I * J * K, 2]
            
            # traffic demand feature (I * J * K = 10 * 10 * 20 = 2000)
            if swap_prob:
                u_features = np.array([
                    [x_mat[i, j, k], x_mask[i, j, k]] if np.random.rand() > swap_prob else [x_mask[i, j, k], x_mat[i, j, k]] 
                    for i in range(I) for j in range(J) for k in range(K)
                ], dtype=np.float64) 
            else: 
                u_features = np.array([
                    [x_mat[i, j, k], x_mask[i, j, k]] 
                    for i in range(I) for j in range(J) for k in range(K)
                ], dtype=np.float64)
            assert u_features.shape == (I * J * K, 2)
            data['demand'].x = torch.tensor(u_features, dtype=torch.float) # [num_origin_destination, num_features] => [I * J * K, 2]

            # measurement feature (L * T = 260 * 20 = 5200)
            v_features = np.array([[q_mat[l, t], k_mat[l, t], v_mat[l, t]] for l in range(L) for t in range(T)], dtype=np.float64)
            assert v_features.shape == (L * T, 3)
            data['measurement'].x = torch.tensor(v_features, dtype=torch.float) # [num_detectors, num_features] => [L * T, 3]

            if meta_data: 
                data['demand'].x_origin = torch.tensor(u_features * x_mag, dtype=torch.float)
                tmp = np.array([[q_mat[l, t] * q_mag, k_mat[l, t] * k_mag, v_mat[l, t] * v_mag] for l in range(L) for t in range(T)], dtype=np.float64)
                data['measurement'].x_origin = torch.tensor(tmp, dtype=torch.float) # [num_detectors, num_features] => [L * T, 3]

            # adjacent matrix
            alpha_keys, alpha_values = a_dict.keys(), a_dict.values()
            adj_matrix = np.array([[i*J*K+j*K+k for (i, j, k, l, t) in alpha_keys],[l*T+t for (i, j, k, l, t) in alpha_keys]], dtype=np.int32) # NOTE: index mapping
            data['demand', 'contributes_to', 'measurement'].edge_index =  torch.tensor(adj_matrix, dtype=torch.long)# [2, num_edges] => ([ [from], [to] ])
            assert adj_matrix.shape == (2, E)
            # edge weight
            edge_features = np.array([[v] for v in alpha_values], dtype=np.float64)
            data['demand', 'contributes_to', 'measurement'].edge_attr = torch.tensor(edge_features, dtype=torch.float) # [num_edges, num_features] 
            assert edge_features.shape == (E, 1)
            # edge label
            data['demand', 'contributes_to', 'measurement'].edge_label = torch.tensor(list(alpha_values))
            
            # data transformation
            transform_modules = transforms.Compose([
                transforms.RemoveIsolatedNodes(), 
                transforms.RemoveDuplicatedEdges(),
                transforms.ToDevice(device=0)
            ])
            data = transform_modules(data)
            data_list.append(data) # add meta data
            num_graphs += 1
            num_node_u += data['demand'].num_nodes
            num_node_v += data['measurement'].num_nodes
            num_edges  += data['demand', 'contributes_to', 'measurement'].num_edges
            avg_density += num_edges / (num_node_u * num_node_v)
    avg_density /= num_graphs
    print(f'[summary] #node_u: {num_node_u} | #node_v: {num_node_v} | #node_total: {num_node_u + num_node_v} | #edges: {num_edges} | #graphs: {num_graphs} | avg_density: {avg_density}\n')
    return DataLoader(data_list, batch_size = batch_size, shuffle = shuffle)
    
