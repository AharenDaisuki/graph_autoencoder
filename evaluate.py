import os
import shutil
import argparse
import subprocess

import torch 
import numpy as np
import xml.etree.ElementTree as ET

from argparse import Namespace
from utils import save_as_pkl, get_sparse_matrix_alpha, tensor_to_ndarray, clear_tmp_dir
from visual import plot_pearson_correlation_scatter_arr, plot_pearson_correlation_scatter_lis, plot_line_chart
from models import Bipartite_link_pred
from datasets import get_sumo_trainloader
from loss import nrmse_loss, rmse_loss
from simulations import SumoSimulation
# from loss import rmse_loss, node_u_recon_loss, node_v_recon_loss

from torch import Tensor
# from torch.sparse import mm, spsolve
# from torch.linalg import solve
import torch_geometric.transforms as T
from torch_geometric.utils import spmm
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def reconstruct_graph(index: int, data: HeteroData, alpha: Tensor, u_dim: tuple, v_dim: tuple, save_dir = 'tmp'): 
    def reconstruct_node(alpha: np.ndarray, adj: np.ndarray, x_vec: np.ndarray, y_vec: np.ndarray):
        # def map_ijk(index):
        #     # i = (index // (J_ * K_))
        #     # j = (index - i * J_ * K_) // K_
        #     ij = (index // K_)
        #     k = (index % K_)
        #     return ij, k
        def map_lt(index): 
            l = (index // T_)
            t = (index % T_)
            return l, t
        def dict2list(d: dict): 
            x_tmp = sorted(d.items(), key=lambda x: x[0])
            return [i[1] for i in x_tmp]
        
        x_ijkt = {}
        # s_tmp = set()
        for i in range(len(alpha)): 
            ijk, lt = adj[0, i], adj[1, i]
            # s_tmp.add(ijk)
            l, t = map_lt(lt)
            x_ijkt.setdefault((ijk, t), 0) 
            tmp = alpha[i] * y_vec[lt]
            if x_ijkt.setdefault((ijk, t), 0) < tmp: 
                x_ijkt[(ijk, t)] = tmp 
        # assert len(s_tmp) == len(x_vec), f'{len(s_tmp)} | {len(x_vec)}'
        # for idx, val in np.ndenumerate(alpha): 
        #     ijk, lt = idx
        #     l, t = map_lt(lt)
        #     x_dic.setdefault((ijk, t), 0)
        #     x_dic[(ijk, t)] += val * y[lt]
        x_ijk = {}
        x_low = {}
        for key, val in x_ijkt.items(): 
            ijk, t = key
            if x_ijk.setdefault(ijk, 0) < val: 
                tmp = 0 if (ijk, t+1) not in x_ijkt else x_ijkt[(ijk, t+1)] # +1 is better
                x_ijk[ijk] = val + tmp
                x_low[ijk] = val
        # TODO: two consecutive intervals
        # for key, val in x_ijkt.items(): 
        #     ijk, t = key
        #     t1 = val
        #     t2 = 0 if (ijk, t+1) not in x_ijkt else x_ijkt[(ijk, t+1)]
        #     if x_ijk.setdefault(ijk, 0) < t1 + t2: 
        #         # tmp = 0 if (ijk, t+1) not in x_ijkt else x_ijkt[(ijk, t+1)]
        #         x_ijk[ijk] = t1 + t2
        #         x_low[ijk] = max(t1, t2)

        # assert len(x_ijk) == len(x_vec), f'{len(x_ijk)} | {len(x_vec)}'
        # x_pred = np.zeros((I_ * (J_ - 1) * K_), dtype=np.float64)
        # for key, val in x_ijk.items(): 
        #     x_pred[key] = val

        # x_tmp = sorted(x_ijk.items(), key=lambda x: x[0])
        # x_tmp_lower = sorted(x_low.items(), key=lambda x: x[0])
        # u_list = [i[1] for i in x_tmp]
        # u_list_lower = [i[1] for i in x_tmp_lower]
        x_upper = dict2list(x_ijk)
        x_lower = dict2list(x_low)
        x_pred = np.array(x_upper, dtype=np.float64) * 10.0
        x_pred_lower = np.array(x_lower, dtype=np.float64) * 10.0
        # u_list = []
        # for i in range(I_): 
        #     for j in range(J_): 
        #         for k in range(K_): 
        #             x = 0.0
        #             flag = False
        #             for t in range(k, K_): 
        #                 key = (i*J_*K_+j*K_+k, t)
        #                 if key in x_dic: 
        #                     if x < x_dic[key]: 
        #                         x = x_dic[key]
        #                         flag = True
        #             if flag: 
        #                 u_list.append(x)
        # x_pred = x_pred.reshape(-1)
        x_true = x_vec.astype(dtype=np.float64)
        # x_pred_ = [sum(lis) * 10 for lis in x_pred]
        # x_true_ = [sum(lis) for lis in x_true] # unit: veh/h
        loss = nrmse_loss(pred=torch.tensor(x_pred), target=torch.tensor(x_true))
        # plot scatter
        plot_pearson_correlation_scatter_lis(x_true, x_pred, savefig=os.path.join(save_dir, f'{index}-node(nrmse = {loss}).png'), xlabel='Ground Truth (unit: veh/h)', ylabel='Prediction (unit: veh/h)')
        # plot line chart
        assert len(x_pred) == len(x_true)
        length = len(x_true)
        od_pair_n = (length // K_)
        for i in range(od_pair_n): 
            plot_line_chart(array_x=x_true[i*K_:(i+1)*K_], array_y_upper=x_pred[i*K_:(i+1)*K_], array_y_lower=x_pred_lower[i*K_:(i+1)*K_], savefig=os.path.join(save_dir, f'od_pair_{i}.png'))
        # reconstruct traffic 
        return loss, x_true, x_pred

    I_, J_, K_ = u_dim
    L_, T_ = v_dim
    x = data['demand'].x_origin 
    y = data['measurement'].x_origin
    a = data['demand', 'measurement'].edge_label
    adj = data['demand', 'measurement'].edge_index
    q = data['measurement'].x[:,0]
    k = data['measurement'].x[:,1]
    v = data['measurement'].x[:,2]
    num_node_u = data['demand'].num_nodes
    num_node_v = data['measurement'].num_nodes
    
    # reconstruct edge
    beta_pred = tensor_to_ndarray(alpha)
    beta_true = tensor_to_ndarray(a)
    y_vector = tensor_to_ndarray(y)
    x_vector = tensor_to_ndarray(x)
    adj_mat = tensor_to_ndarray(adj)
    plot_pearson_correlation_scatter_arr(array_x=beta_true, array_y=beta_pred, upper=1.0, savefig=os.path.join(save_dir, f'{index}-edge.png'))

    # reconstruct node
    loss, x_true, x_pred = reconstruct_node(alpha=beta_pred, adj=adj_mat, x_vec=x_vector, y_vec=y_vector)
    # plot_pearson_correlation_scatter(array_x=x_true, array_y=x_pred, savefig=os.path.join(save_dir, f'{index}-node.png'))
    # # sparse matrix A [m \times n]
    # adj = data['measurement', 'rev_contributes_to', 'demand'].edge_index
    # adj_t = data['demand', 'contributes_to', 'measurement'].edge_index
    # A = get_sparse_matrix_alpha(indices=adj, values=alpha, size=(num_node_u, num_node_v), data_type=x.dtype, device=x.device)
    # AT = get_sparse_matrix_alpha(indices=adj_t, values=alphas, size=(num_node_u, num_node_v), data_type=x.dtype, device=x.device)
    # AT_Y = spmm(src=AT, other=y)
    # X = spmm(src=A, other=y.unsqueeze(-1)).squeeze(-1)
    # x_pred = tensor_to_ndarray(X) * 2
    # x_true = tensor_to_ndarray(x)
    # plot_pearson_correlation_scatter_arr(array_x=x_true, array_y=x_pred, savefig=os.path.join(save_dir, f'{index}-node.png'))

    # AT_A = mm(AT, A).to_dense() # 
    # # AT_A_X = spmm(src=AT, other=A_X)
    # # reconstruct edge
    # a_gap = a - alphas 
    # a_gap = a_gap.detach().cpu().numpy()
    # plot_histogram(array_x=a_gap, bin_n=20, range=(-1.0, 1.0), savefig=os.path.join(save_dir, f'{index}-edge.png'))
    # # reconstruct node v
    # y_pred = A_X.squeeze(-1).detach().cpu().numpy()
    # y_true = y.squeeze(-1).detach().cpu().numpy()
    # plot_pearson_correlation_scatter(array_x=y_true, array_y=y_pred, upper=np.max(y_true), savefig=os.path.join(save_dir, f'{index}-node-v.png'))
    # # reconstruct node u
    # # TODO: [debug] Runtime Error: Calling linear solver with sparse tensors requires compiling PyTorch with CUDA cuDSS and is not supported in ROCm build
    # # x_pred = spsolve(AT_A, AT_Y)
    # x_pred = solve(AT_A, AT_Y)
    # x_gap = x - x_pred][]
    # x_gap = x_gap.squeeze(-1).detach().cpu().numpy()
    # # x_pred = x_pred.squeeze(-1).detach().cpu().numpy() 
    # # x_true = x.squeeze(-1).detach().cpu().numpy() 
    # plot_histogram(array_x=x_gap, bin_n=20, range=(-10.0, 10.0), savefig=os.path.join(save_dir, f'{index}-node-u.png'))
    # # plot_pearson_correlation_scatter(array_x=x_true, array_y=x_pred, upper=np.max(x_true), savefig=os.path.join(save_dir, f'{index}-node-u.png'))
    return loss, x_true, x_pred

def reconstruct_traffic(index: int, x_pred: np.array, x_true: np.array, matrix_dim: tuple, sim_tmp_dir: str, simulation_args):
    def generate_rou_file(filepath, matrix_dim, od_matrix, period):
        ''' generate sumo route file ''' 
        with open(filepath, 'w') as f:
            f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
            I, J, K = matrix_dim
            # assert matrix_dim == od_matrix.shape
            for k in range(K): 
                for i in range(I): 
                    for j in range(J):
                        if i != j:
                            if od_matrix[i, j, k] > 0.0:   
                                f.write(f'\t<flow id="{i}-{j}-{k}" begin="{float(period * k)}" end="{float(period * (k+1))}" vehsPerHour="{od_matrix[i, j, k]}" fromTaz="taz_{i}" toTaz="taz_{j}"/>\n')
            f.write('</routes>\n')
        f.close()
    def generate_cfg_file(config_path, route_file, duration):
        ''' generate sumo configuration '''
        with open(config_path, 'w') as f: 
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
            # input
            f.write('\t<input>\n')
            # net files
            f.write(f'\t\t<net-file value="{ROADNETWORK_FILE}"/>\n')
            # route files
            f.write(f'\t\t<route-files value="{route_file}"/>\n')
            # additional files
            additional_files = [DETECTOR_FILE, TAZ_FILE]
            ADD_FILES = ','.join(additional_files)
            f.write(f'\t\t<additional-files value="{ADD_FILES}"/>\n')
            f.write('\t</input>\n')
            # time
            f.write('\t<time>\n')
            f.write('\t\t<begin value="0"/>\n')
            f.write(f'\t\t<end value="{duration}"/>\n')
            f.write(f'\t\t<step-length value="{SUMO_STEP_LENGTH}"/>\n')
            f.write('\t</time>\n')
            f.write('</configuration>\n')
        f.close()
    def od2routes(road_network, add_taz, od_demand, output_routes): 
        assert isinstance(road_network, str)
        assert isinstance(add_taz, str)
        assert isinstance(od_demand, str)
        assert isinstance(output_routes, str)
        # trips to routes
        cmd = ['duarouter']
        # osm
        cmd.extend(['-n', road_network])
        # taz
        cmd.extend(['--additional-files', add_taz])
        # rou
        cmd.extend(['--route-files', od_demand])
        # output
        cmd.extend(['-o', output_routes])
        # misc
        cmd.extend(['--ignore-errors'])
        subprocess.run(cmd, check=True)
    def sumo_objective_function_nrmse(sim_measurements: list, obs_measurements: list):
        obj = 0
        loss_list = []
        weight_list = []
        # calculate weight, loss
        for sim_measurement, obs_measurement in zip(sim_measurements, obs_measurements): 
            assert sim_measurement.shape == obs_measurement.shape
            L, K = sim_measurement.shape
            sigma_sim, sigma_obs = np.std(sim_measurement), np.std(obs_measurement)
            square_error = np.sum((sim_measurement - obs_measurement) * (sim_measurement - obs_measurement))
            weight = square_error / L * K * (sigma_obs - sigma_sim) ** 2 
            loss = np.sqrt(L * K * square_error) / np.sum(obs_measurement)
            loss_list.append(loss)
            weight_list.append(weight)
        # calculate objective
        for loss, weight in zip(loss_list, weight_list):
            alpha = weight / sum(weight_list)
            obj += alpha * loss
        return obj
    I, J, K = matrix_dim
    assert os.path.exists(sim_tmp_dir)

    ROADNETWORK_FILE = 'osm.net.xml'
    # SUMOCONFIG_FILE = 'sim.sumocfg'
    DETECTOR_FILE = 'sparse.add.xml'
    # DEMAND_FILE = 'flow.rou.xml'
    # ROUTE_FILE = 'trips.rou.xml'
    TAZ_FILE = 'taz.add.xml'
    FCD_FILE = 'fcd.xml'
    # MEASUREMENT_FILE = 'detectors.xml'

    SUMO_STEP_LENGTH = 2.0
    # copy osm.net.xml
    orig_net_path = ROADNETWORK_FILE
    dest_net_path = os.path.join(sim_tmp_dir, ROADNETWORK_FILE)
    shutil.copy(orig_net_path, dest_net_path)

    # copy det.add.xml
    orig_det_path = DETECTOR_FILE
    dest_det_path = os.path.join(sim_tmp_dir, DETECTOR_FILE)
    shutil.copy(orig_det_path, dest_det_path)

    # copy taz.add.xml
    orig_taz_path = TAZ_FILE
    dest_taz_path = os.path.join(sim_tmp_dir, TAZ_FILE)
    shutil.copy(orig_taz_path, dest_taz_path)

    # generate trips.rou.xml
    # x, mu, sigma, od_demand_file = generate_od_matrix(matrix_dim=matrix_dim, 
    #                 simulation_state=simulation_state, 
    #                 flow_save_dir=sim_tmp_dir,
    #                 period=(simulation_args.duration // K))
    od_mat_pred = np.zeros(matrix_dim)
    od_mat_true = np.zeros(matrix_dim)
    n = 0
    for i in range(I): 
        for j in range(J): 
            for k in range(K): 
                if i != j and n < len(x_pred): 
                    od_mat_pred[i, j, k] = x_pred[n]
                    od_mat_true[i, j, k] = x_true[n]
                    n += 1
    # prediction                
    demand_file_pred = os.path.join(sim_tmp_dir, 'pred.rou.xml')
    route_file_pred = os.path.join(sim_tmp_dir, 'pred-trip.rou.xml')
    generate_rou_file(filepath=demand_file_pred, matrix_dim=matrix_dim, od_matrix=od_mat_pred, period=(simulation_args.duration // K))
    od2routes(road_network=dest_net_path, add_taz=dest_taz_path, od_demand=demand_file_pred, output_routes=route_file_pred)
    # true
    demand_file_true = os.path.join(sim_tmp_dir, 'true.rou.xml')
    route_file_true = os.path.join(sim_tmp_dir, 'true-trip.rou.xml')
    generate_rou_file(filepath=demand_file_true, matrix_dim=matrix_dim, od_matrix=od_mat_true, period=(simulation_args.duration // K))
    od2routes(road_network=dest_net_path, add_taz=dest_taz_path, od_demand=demand_file_true, output_routes=route_file_true)

    # create temp sim.sumocfg
    sumocfg_pred = os.path.join(sim_tmp_dir, 'pred.sumocfg')
    sumocfg_true = os.path.join(sim_tmp_dir, 'true.sumocfg')
    generate_cfg_file(config_path=os.path.join(sim_tmp_dir, 'pred.sumocfg'), route_file='pred-trip.rou.xml', duration=simulation_args.duration)
    generate_cfg_file(config_path=os.path.join(sim_tmp_dir, 'true.sumocfg'), route_file='true-trip.rou.xml', duration=simulation_args.duration)
    
    # simulation
    hash_index = 0 
    link_hash  = {}
    tree = ET.parse(DETECTOR_FILE) # NOTE: hard-code detector additional file
    root = tree.getroot()
    for detector in root.iter('inductionLoop'): 
        link_id = detector.get('id')[:-2]
        if link_hash.setdefault(link_id, -1) < 0: 
            link_hash[link_id] = hash_index
            hash_index += 1
    # true
    simulation_args.config = sumocfg_true
    simulation_args.data = os.path.join(sim_tmp_dir, 'simulation')
    simulation_args.fcd_output = os.path.join(sim_tmp_dir, FCD_FILE)
    simulation_args.mute_warnings = True
    simulation_args.mute_step_logs = True
    if not os.path.exists(simulation_args.data): 
        os.makedirs(simulation_args.data, exist_ok=True)
    simulator = SumoSimulation(simulation_args, link_hash=link_hash)
    q1, k1, v1, a, b, u_orig, v_orig = simulator.run_sumo(save_fcd = False) # NOTE: do not save fcd
    clear_tmp_dir(simulation_args.data)
    # pred
    simulation_args.config = sumocfg_pred
    if not os.path.exists(simulation_args.data): 
        os.makedirs(simulation_args.data, exist_ok=True)
    simulator = SumoSimulation(simulation_args, link_hash=link_hash)
    q2, k2, v2, a, b, u_orig, v_orig = simulator.run_sumo(save_fcd = False) # NOTE: do not save fcd
    clear_tmp_dir(simulation_args.data)
    # score = sumo_objective_function_nrmse(sim_measurements=[q1, k1, v1], obs_measurements=[q2, k2, v2])
    plot_pearson_correlation_scatter_arr(array_x=q2.reshape(-1), array_y=q1.reshape(-1), savefig=os.path.join(sim_tmp_dir, f'{index}-q.png'))
    plot_pearson_correlation_scatter_arr(array_x=k2.reshape(-1), array_y=k1.reshape(-1), savefig=os.path.join(sim_tmp_dir, f'{index}-k.png'), upper=200.0)
    plot_pearson_correlation_scatter_arr(array_x=v2.reshape(-1), array_y=v1.reshape(-1), savefig=os.path.join(sim_tmp_dir, f'{index}-v.png'))
    # print(score)
    # return score

def evaluate(args):
    checkpoint = args.checkpoint
    hidden_channels = args.hidden_channels
    out_channels = args.out_channels
    test_set = args.data
    eval_dir = args.dir
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    assert os.path.exists(checkpoint) and os.path.exists(test_set)

    checkpoint_dict = torch.load(checkpoint)
    model = Bipartite_link_pred(hidden_channels=hidden_channels, out_channels=out_channels).to(device) 
    model.load_state_dict(checkpoint_dict['model'])
    test_loader, u_dim, v_dim = get_sumo_trainloader(data_dir=test_set)
    graph_index = 0
    loss_list = []
    for data in test_loader: 
        # unidirected graph
        data = T.ToUndirected()(data)
        del data['measurement', 'rev_contributes_to', 'demand'].edge_label
        metapath = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]
        data = T.AddMetaPaths(metapaths=[metapath])(data)

        # # Apply normalization to filter the metapath:
        _, edge_weight = gcn_norm(
            data['measurement', 'measurement'].edge_index,
            num_nodes=data['measurement'].num_nodes,
            add_self_loops=False,
        )
        edge_index = data['measurement', 'measurement'].edge_index[:, edge_weight > 0.005]
        data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index

        model.eval()
        pred = model(data.x_dict, 
                    data.edge_index_dict, 
                    data['demand', 'measurement'].edge_index)
        # target = data['demand', 'measurement'].edge_label
        save_dir = os.path.join('benchmark', eval_dir, str(graph_index))
        os.makedirs(save_dir, exist_ok=True)
        assert os.path.exists(save_dir)
        x_loss, x_true, x_pred = reconstruct_graph(graph_index, data, pred, u_dim, v_dim, save_dir)
        simulation_options = {'seed': 2025, 'duration': 7200, 'period': 360} # NOTE: hard code some simulation parameters
        reconstruct_traffic(graph_index, x_pred, x_true, u_dim, sim_tmp_dir=save_dir, simulation_args=Namespace(**simulation_options))
        graph_index += 1
        loss_list.append(x_loss)
    eval_metric = np.mean(loss_list)
    print(f'Normalized root mean squared error: {eval_metric}\n')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type = int, default = 64, help = 'number of hidden channels (encoder)')
    parser.add_argument('--out_channels', type = int, default = 128, help = 'number of output channels (encoder)')
    parser.add_argument('--checkpoint', type = str, help = 'load model x')
    parser.add_argument('--data', type = str, default = 'sim_dataset_low', help = 'testing data folder')
    parser.add_argument('--dir', type=str, default='eval-l', help='benchmark directory')
    args = parser.parse_args()
    evaluate(args)
