import os
import glob
import torch
import random
import numpy as np
import torch_geometric.transforms as transforms

from tqdm import tqdm
from utils import load_from_pkl
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

def get_sumo_trainloader(data_dir: str, batch_size: int = 1, shuffle: bool = True): 
    def _check_data_integrity(folder):
        # all_files = [os.path.join(folder, raw_file) for raw_file in raw_files] + [os.path.join(folder, add_file) for add_file in add_files]
        all_files = [os.path.join(folder, raw_file) for raw_file in raw_files]
        return all([os.path.exists(file) for file in all_files]) 
    assert os.path.exists(data_dir)
    data_list = []
    raw_files = ['q.pkl', 'k.pkl', 'v.pkl', 'x.pkl', 'a.pkl', 'b.pkl', 'u.pkl', 'y.pkl']
    add_files = ['metadata.pkl', 'dist.pkl']
    num_edges = 0
    num_node_u = 0
    num_node_v = 0
    num_graphs = 0
    avg_density = 0
    # sim_state => sim_scene => sim_index
    bar = tqdm(glob.glob(os.path.join(data_dir, '*', 'scenario_*')), desc=f'load {data_dir}...'.ljust(20))
    for sim_scene in bar:
        # dist_data = load_from_pkl(pkl_path=os.path.join(sim_scene, add_files[1]))
        # mu, sigma = dist_data['mu'], dist_data['sigma']
        for sim_index in glob.glob(os.path.join(sim_scene, '*')):
            if os.path.basename(sim_index) in add_files: 
                continue 
            assert _check_data_integrity(folder = sim_index)
            # raw data
            q_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[0]))
            k_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[1]))
            v_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[2]))
            x_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[3]))
            a_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[4]))
            b_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[5]))
            u_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[6]))
            v_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[7])) 
            assert isinstance(q_mat, np.ndarray)    
            assert isinstance(k_mat, np.ndarray)
            assert isinstance(v_mat, np.ndarray)
            assert isinstance(x_mat, np.ndarray)
            assert isinstance(a_dict, dict)
            assert isinstance(b_dict, dict)
            assert isinstance(u_dict, dict)
            assert isinstance(v_dict, dict)
            # additional data
            if os.path.exists(os.path.join(sim_index, add_files[0])): 
                meta_data = load_from_pkl(pkl_path=os.path.join(sim_index, add_files[0]))
            else: 
                meta_data = None
            # x_mask = np.random.normal(loc=mu, scale=sigma, size=x_mat.shape)
            # if meta_data: 
            #     q_mag, k_mag, v_mag, x_mag = meta_data['q_mag'], meta_data['k_mag'], meta_data['v_mag'], meta_data['x_mag']
                # x_mask /= x_mag

            I, J, K = x_mat.shape
            L, T = q_mat.shape
            E = len(a_dict)
            
            data = HeteroData()
            # TODO: setup1
            # traffic demand feature (I * J * K = 10 * 10 * 20 = 2000)
            # u_features = x_mat.reshape(-1, 1)
            # data['demand'].x = torch.tensor(u_features * x_mag, dtype=torch.float) # [num_origin_destination, num_features] => [I * J * K, 1]
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
            # assert u_features.shape == (I * J * K, 3)
            
            # traffic demand feature (I * J * K = 10 * 10 * 20 = 2000)
            # if swap_prob:
            #     u_features = np.array([
            #         [x_mat[i, j, k], x_mask[i, j, k]] if np.random.rand() > swap_prob else [x_mask[i, j, k], x_mat[i, j, k]] 
            #         for i in range(I) for j in range(J) for k in range(K)
            #     ], dtype=np.float64) 
            # else: 
            #     u_features = np.array([
            #         [x_mat[i, j, k], x_mask[i, j, k]] 
            #         for i in range(I) for j in range(J) for k in range(K)
            #     ], dtype=np.float64)
            # assert u_features.shape == (I * J * K, 2)
            # data['demand'].x = torch.tensor(u_features, dtype=torch.float) # [num_origin_destination, num_features] => [I * J * K, 2]
            data['demand'].x = torch.arange(I * J * K)

            # measurement feature (L * T = 260 * 20 = 5200)
            v_features = np.array([[q_mat[l, t], k_mat[l, t], v_mat[l, t]] for l in range(L) for t in range(T)], dtype=np.float64)
            assert v_features.shape == (L * T, 3)
            data['measurement'].x = torch.tensor(v_features, dtype=torch.float) # [num_detectors, num_features] => [L * T, 3]
            
            # meta data
            # TODO: modify original feature
            if meta_data: 
                x_mag = meta_data['x_mag']
                # u set origin 
                # u_orig_list = []
                # for i in range(I): 
                #     for j in range(J): 
                #         for k in range(K):
                #             if (i, j, k) in u_dict:  
                #                 u_orig_list.append(u_dict[i, j, k]) 
                # data['demand'].x_origin = torch.tensor(u_orig_list, dtype=torch.float)

                u_orig_list = x_mat.reshape(-1)
                # assert u_orig_list.shape[0] == I * J * K
                data['demand'].x_origin = torch.tensor(u_orig_list * x_mag, dtype=torch.float) # [num_origin_destination, num_features] => [I * J * K, 1]

                # v set origin
                v_orig_list = []
                for l in range(L): 
                    for t in range(T): 
                            if (l, t) in v_dict:  
                                v_orig_list.append(v_dict[l, t]) 
                data['measurement'].x_origin = torch.tensor(v_orig_list, dtype=torch.float)

            # adjacent matrix
            # alpha_keys, alpha_values = a_dict.keys(), a_dict.values()
            alpha_keys, alpha_values = b_dict.keys(), b_dict.values()
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
                # transforms.RemoveDuplicatedEdges(),
                transforms.ToDevice(device=0)
            ])
            data = transform_modules(data)
            node_u_n = data['demand'].num_nodes
            node_v_n = data['measurement'].num_nodes
            edge_n = data['demand', 'contributes_to', 'measurement'].num_edges
            data_list.append(data) # add meta data
            num_graphs += 1
            num_node_u += node_u_n
            num_node_v += node_v_n
            num_edges  += edge_n
            avg_density += edge_n / (node_u_n * node_v_n)
    avg_density /= num_graphs
    print(f'[summary] #node_u: {num_node_u} | #node_v: {num_node_v} | #node_total: {num_node_u + num_node_v} | #edges: {num_edges} | #graphs: {num_graphs} | avg_density: {round(100 * avg_density, 4)}%\n')
    return DataLoader(data_list, batch_size = batch_size, shuffle = shuffle), (I, J, K), (L, T)

def get_sumo_dataloaders(data_dir: str, batch_size: int = 1, shuffle: bool = True, train_test_split: float = 0.9):
    """
    Get train data loader and test data loader.

    Args:
        data_dir (str): directory of data.
        batch_size (int, optional): how many samples per batch to load (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
        train_test_split (float, optional): ratio of training samples in dataset (default: ``0.9``).

    Returns:
        tuple[DataLoader, DataLoader, tuple[Unbound | int, Unbound | int, Unbound | int], tuple[Unbound | int, Unbound | int]]: train data loader, test data loader, shape of demand matrix, shape of measurement matrix.
    """
    def _check_data_integrity(folder):
        all_files = [os.path.join(folder, raw_file) for raw_file in raw_files]
        return all([os.path.exists(file) for file in all_files]) 
    assert os.path.exists(data_dir)
    raw_files = ['q.pkl', 'k.pkl', 'v.pkl', 'x.pkl', 'a.pkl', 'b.pkl', 'u.pkl', 'y.pkl']
    add_files = ['metadata.pkl', 'dist.pkl']
    train_data = []
    test_data = []
    train_edge_n = 0
    train_node_u = 0
    train_node_v = 0
    train_graph_n = 0
    train_density = 0.0
    test_edge_n = 0
    test_node_u = 0
    test_node_v = 0
    test_graph_n = 0
    test_density = 0.0
    # sim state => sim scene
    bar = tqdm(glob.glob(os.path.join(data_dir, '*', 'scenario_*')), desc=f'load {data_dir}...'.ljust(20))
    for sim_scene in bar:
        # dist_data = load_from_pkl(pkl_path=os.path.join(sim_scene, add_files[1]))
        # mu, sigma = dist_data['mu'], dist_data['sigma']
        simulation_list = glob.glob(os.path.join(sim_scene, '*'))
        simulation_n = len(simulation_list) - 1
        train_split = random.sample(range(simulation_n), int(train_test_split * simulation_n))
        split = [0 if i in train_split else 1 for i in range(simulation_n)]
        graph_index = 0
        for sim_index in simulation_list:
            # check data integrity
            if os.path.basename(sim_index) in add_files: 
                continue 
            assert _check_data_integrity(folder = sim_index)
            # load data
            q_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[0]))
            k_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[1]))
            v_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[2]))
            x_mat = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[3]))
            a_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[4]))
            b_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[5]))
            u_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[6]))
            v_dict = load_from_pkl(pkl_path=os.path.join(sim_index, raw_files[7])) 
            assert isinstance(q_mat, np.ndarray)    
            assert isinstance(k_mat, np.ndarray)
            assert isinstance(v_mat, np.ndarray)
            assert isinstance(x_mat, np.ndarray)
            assert isinstance(a_dict, dict)
            assert isinstance(b_dict, dict)
            assert isinstance(u_dict, dict)
            assert isinstance(v_dict, dict)
            # build hetero graph
            I, J, K = x_mat.shape
            L, T = q_mat.shape
            E = len(a_dict)

            data = HeteroData()
            # node feature u
            data['demand'].x = torch.arange(I * J * K)
            # node feature v
            v_features = np.array([[q_mat[l, t], k_mat[l, t], v_mat[l, t]] for l in range(L) for t in range(T)], dtype=np.float64)
            assert v_features.shape == (L * T, 3)
            data['measurement'].x = torch.tensor(v_features, dtype=torch.float) # [num_detectors, num_features] => [L * T, 3] 
            # node feature u origin 
            u_orig_list = []
            for i in range(I): 
                for j in range(J): 
                    for k in range(K):
                        if (i, j, k) in u_dict:  
                            u_orig_list.append(u_dict[i, j, k]) 
            data['demand'].x_origin = torch.tensor(u_orig_list, dtype=torch.float)
            # node feature v origin
            v_orig_list = []
            for l in range(L): 
                for t in range(T): 
                        if (l, t) in v_dict:  
                            v_orig_list.append(v_dict[l, t]) 
            data['measurement'].x_origin = torch.tensor(v_orig_list, dtype=torch.float)
            # adjacent matrix
            # NOTE: predict \alpha or \beta
            # alpha_keys, alpha_values = a_dict.keys(), a_dict.values()
            alpha_keys, alpha_values = b_dict.keys(), b_dict.values()
            adj_matrix = np.array([[i*J*K+j*K+k for (i, j, k, l, t) in alpha_keys],[l*T+t for (i, j, k, l, t) in alpha_keys]], dtype=np.int32) 
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
                # transforms.RemoveDuplicatedEdges(),
                transforms.ToDevice(device=0)
            ])
            data = transform_modules(data)
            node_u_n = data['demand'].num_nodes
            node_v_n = data['measurement'].num_nodes
            edge_n = data['demand', 'contributes_to', 'measurement'].num_edges
            if split[graph_index] == 0: 
                # train
                train_data.append(data)     
                train_graph_n += 1   
                train_node_u += node_u_n
                train_node_v += node_v_n
                train_edge_n += edge_n
                train_density += edge_n / (node_u_n * node_v_n)
            else: 
                # test
                test_data.append(data)     
                test_graph_n += 1   
                test_node_u += node_u_n
                test_node_v += node_v_n
                test_edge_n += edge_n
                test_density += edge_n / (node_u_n * node_v_n)
            graph_index += 1
    train_density /= train_graph_n
    test_density /= test_graph_n
    print(f'[train] #node_u: {train_node_u} | #node_v: {train_node_v} | #node_total: {train_node_u + train_node_v} | #edges: {train_edge_n} | #graphs: {train_graph_n} | avg_density: {round(100 * train_density, 4)}%\n')
    print(f'[test] #node_u: {test_node_u} | #node_v: {test_node_v} | #node_total: {test_node_u + test_node_v} | #edges: {test_edge_n} | #graphs: {test_graph_n} | avg_density: {round(100 * test_density, 4)}%\n')
    return DataLoader(train_data, batch_size = batch_size, shuffle = shuffle), DataLoader(test_data, batch_size = batch_size, shuffle = shuffle), (I, J, K), (L, T)



    
