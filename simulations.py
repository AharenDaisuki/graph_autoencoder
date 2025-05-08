import os
import sys
import shutil
import argparse
import subprocess
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from glob import glob
from utils import save_as_pkl
from multiprocessing.pool import ThreadPool

TASK_SUCCESS = 0
TASK_ERROR = -1

# setup: 
## 1. initialize traffic assignment zone
## 2. initialize detector layout
## 3. prepare road network

def generate_od_matrix(matrix_dim, simulation_state, flow_save_dir, period=180):
    ''' generate od matrix numpy nd array for scenario A, state B and simulation C '''

    def generate_rou_file(filepath, matrix_dim, od_matrix, period=180):
        ''' generate sumo route file''' 
        with open(filepath, 'w') as f:
            f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
            I, J, K = matrix_dim
            # assert matrix_dim == od_matrix.shape
            for k in range(K): 
                for i in range(I): 
                    for j in range(J):
                        if i != j:  
                            f.write(f'\t<flow id="{i}-{j}-{k}" begin="{float(period * k)}" end="{float(period * (k+1))}" vehsPerHour="{od_matrix[i, j, k]}" fromTaz="taz_{i}" toTaz="taz_{j}"/>\n')
            f.write('</routes>\n')
        f.close()

    # x \in [0, 40]
    assert simulation_state in [0, 1, 2] # 0 => low, 1 => medium, 2 => high
    I, J, K = matrix_dim 
    # TODO: hard-code distribution parameters
    means = [5, 10, 20]
    stds = [1, 3, 6]
    node_state_prob = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]] 
    x_min, x_max = 1.0, 40.0

    # generate od matrix
    node_state  = np.random.choice(a = [0, 1, 2], size = I * J * K, p = node_state_prob[simulation_state])
    mu = list(map(lambda x: means[x], node_state))
    sigma = list(map(lambda x: stds[x], node_state))
    od_matrix = np.random.normal(loc = mu, scale = sigma, size = I * J * K)
    od_matrix = np.clip(od_matrix, a_min = x_min, a_max = x_max)
    od_matrix = od_matrix.reshape(matrix_dim)
    # generate sumo route
    flow_file_path = os.path.join(flow_save_dir, 'flow.rou.xml')
    generate_rou_file(filepath=flow_file_path, matrix_dim=matrix_dim, od_matrix=od_matrix, period=period)
    return od_matrix, flow_file_path

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

def sumo_simulation_task(*args):
    def _make_tmp_config(config_path, duration):
        with open(config_path, 'w') as f: 
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">\n')
            # input
            f.write('\t<input>\n')
            # net files
            f.write(f'\t\t<net-file value="{ROADNETWORK_FILE}"/>\n')
            # route files
            f.write(f'\t\t<route-files value="{DEMAND_FILE}"/>\n')
            # additional files
            additional_files = [DETECTOR_FILE, TAZ_FILE]
            ADD_FILES = ','.join(additional_files)
            f.write(f'\t\t<additional-files value="{ADD_FILES}"/>\n')
            f.write('\t</input>\n')
            # time
            f.write('\t<time>\n')
            f.write('\t\t<begin value="0"/>\n')
            f.write(f'\t\t<end value="{duration}"/>\n')
            f.write(f'\t\t<step-length value="1"/>\n')
            f.write('\t</time>\n')
            f.write('</configuration>\n')
        f.close()
    
    def _clear_tmp_config(tmp_dir): 
        try: 
            shutil.rmtree(tmp_dir)
        except FileNotFoundError:
            assert False, f'Directory [{tmp_dir}] is not found!'

    # receive arguments
    matrix_dim, simulation_scenario, simulation_state, simulation_index, simulation_args, link_hash = args
    assert simulation_state in [0, 1, 2]
    assert isinstance(simulation_index, int)
    assert isinstance(simulation_scenario, str)
    assert isinstance(matrix_dim, tuple)
    assert isinstance(link_hash, dict)
    
    categories = ['low', 'medium', 'high']
    ROADNETWORK_FILE = 'osm.net.xml'
    SUMOCONFIG_FILE = 'sim.sumocfg'
    DETECTOR_FILE = 'det.add.xml'
    DEMAND_FILE = 'trips.rou.xml'
    TAZ_FILE = 'taz.add.xml'
    FCD_FILE = 'fcd.xml'

    try: 
        # create temporary directory
        sim_tmp_dir = os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index), 'tmp')
        sim_dat_dir = os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index), 'tmp', 'simulation')
        output_dir  = os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index))
        # clear cache
        if os.path.exists(sim_tmp_dir): 
            _clear_tmp_config(sim_tmp_dir) 
        # make directory
        os.makedirs(sim_dat_dir)
        assert os.path.exists(output_dir) and os.path.exists(sim_tmp_dir)

        # - output_dir
        # --tmp
            # -- osm.net.xml
            # -- det.add.xml
            # -- taz.add.xml
            # -- flow.rou.xml
            # -- trips.rou.xml
            # -- sim.sumocfg
            # -- simulation
                # -- detectors.xml 

        # -- q.pkl, k.pkl, v.pkl *
        # -- fcd.xml * 
        # -- alpha.pkl *
        # -- x.pkl * 

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
        x, od_demand_file = generate_od_matrix(matrix_dim=matrix_dim, 
                        simulation_state=simulation_state, 
                        flow_save_dir=sim_tmp_dir,
                        period=simulation_args.period)
        od2routes(road_network=dest_net_path, add_taz=dest_taz_path, od_demand=od_demand_file, output_routes=os.path.join(sim_tmp_dir, DEMAND_FILE))

        # create temp sim.sumocfg
        orig_cfg_path = os.path.join(sim_tmp_dir, SUMOCONFIG_FILE)
        _make_tmp_config(config_path=orig_cfg_path, duration=simulation_args.duration)

        # simulation
        simulation_args.config = orig_cfg_path
        simulation_args.data = sim_dat_dir
        simulation_args.flow = os.path.join(sim_tmp_dir, DEMAND_FILE)
        simulation_args.fcd_output = os.path.join(output_dir, FCD_FILE)
        simulation_args.mute_warnings = True
        simulation_args.mute_step_logs = True
        simulator = SumoSimulation(simulation_args, link_hash=link_hash)

        q, k, v, a = simulator.run_sumo()
        # TODO: feature normalization?
        q_magnitude = np.linalg.norm(q)
        k_magnitude = np.linalg.norm(k)
        v_magnitude = np.linalg.norm(v)
        x_magnitude = np.linalg.norm(x)
        assert q_magnitude > 0 and k_magnitude > 0 and v_magnitude > 0 and x_magnitude > 0
        normalized_q = q / q_magnitude
        normalized_k = k / k_magnitude
        normalized_v = v / v_magnitude
        normalized_x = x / x_magnitude
        # normalized_a = np.zeros((I * J * K, L * T))
        # for (i, j, k, l, t), alpha in a.items():
        #     normalized_a[i * J + j * K + k, l * T + t] = alpha

        # numpy nd array
        save_as_pkl(normalized_q, pkl_path=os.path.join(output_dir, 'q.pkl'))
        save_as_pkl(normalized_k, pkl_path=os.path.join(output_dir, 'k.pkl'))
        save_as_pkl(normalized_v, pkl_path=os.path.join(output_dir, 'v.pkl'))
        save_as_pkl(normalized_x, pkl_path=os.path.join(output_dir, 'x.pkl'))
        save_as_pkl(a, pkl_path=os.path.join(output_dir, 'a.pkl'))

        _clear_tmp_config(tmp_dir = sim_tmp_dir)
        return TASK_SUCCESS
    except:
        assert False, 'Error!' 
        return TASK_ERROR

class SumoSimulation(object): 
    def __init__(self, args, link_hash=None):
        self.duration   = args.duration
        self.period     = args.period
        self.data_dir   = args.data
        self.seed       = args.seed
        self.config     = args.config
        self.fcd_output = args.fcd_output
        self.mute_warnings  = args.mute_warnings
        self.mute_step_logs = args.mute_step_logs
        self.link_hash  = link_hash
        self.interval_n = (args.duration // args.period)
        self.link_n  = len(link_hash)
        self.counter = 0
        # check environment variable
        if 'SUMO_HOME' not in os.environ:
            assert False, f'Check SUMO_HOME environment variable'
        else:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)

    def _extract_measurements(self, single_file_mode=False):
        def extract_single_file(filename): 
            data = {}
            tree = ET.parse(filename)
            root = tree.getroot()
            # load
            for interval in root.iter('interval'): 
                v_t, v_s = float(interval.get('speed')), float(interval.get('harmonicMeanSpeed'))
                time_mean_speed  = 0.0 if v_t == -1.00 else v_t 
                space_mean_speed = 0.0 if v_s == -1.00 else v_s 
                begin = float(interval.get('begin'))
                flow  = float(interval.get('flow'))
                k = int(begin / self.period)
                data.setdefault(k, [])
                data[k].append((time_mean_speed, space_mean_speed, flow))
            # process
            q, k, v = [], [], []
            for _, lanes in data.items(): 
                q_, k_, v_ = 0, 0, 0
                tmp_vt, tmp_vs = 0, 0
                # for m lanes
                for lane in lanes:
                    v_t, v_s, f = lane
                    q_     += f 
                    tmp_vt += f * v_t
                    tmp_vs += f * v_s
                k_ = (q_ * q_) / tmp_vs if tmp_vs > 0 else 0
                v_ = tmp_vt / q_ if q_ > 0 else 0
                q.append(q_)
                k.append(k_)
                v.append(v_)
            # TODO: add fcd
            return q, k, v
        
        def extract_all_files(filename): 
            assert self.link_hash is not None, 'No link hash!'
            data = {}
            tree = ET.parse(filename)
            root = tree.getroot()
            # load
            for interval in root.iter('interval'): 
                v_t, v_s = float(interval.get('speed')), float(interval.get('harmonicMeanSpeed'))
                time_mean_speed  = 0.0 if v_t == -1.00 else v_t 
                space_mean_speed = 0.0 if v_s == -1.00 else v_s 
                begin = float(interval.get('begin'))
                flow  = float(interval.get('flow'))
                l = self.link_hash[interval.get('id')[:-2]]
                k = int(begin / self.period)
                data.setdefault((l, k), [])
                if time_mean_speed > 0.0 and space_mean_speed > 0.0 and flow > 0.0: 
                    data[l, k].append((time_mean_speed, space_mean_speed, flow))
            # process
            q_arr = np.zeros((self.link_n, self.interval_n))
            k_arr = np.zeros((self.link_n, self.interval_n))
            v_arr = np.zeros((self.link_n, self.interval_n))
            for (l, t), lanes in data.items(): 
                q_, k_, v_ = 0, 0, 0
                tmp_vt, tmp_vs = 0, 0
                # for m lanes
                for lane in lanes:
                    v_t, v_s, f = lane
                    q_ += f 
                    tmp_vt += f * v_t
                    tmp_vs += f * v_s
                k_ = (q_ * q_) / tmp_vs if tmp_vs > 0 else 0
                v_ = tmp_vt / q_ if q_ > 0 else 0
                q_arr[l, t], k_arr[l, t], v_arr[l, t] = q_, k_, v_
            # floating car data
            if self.fcd_output: 
                tree = ET.parse(self.fcd_output)
                root = tree.getroot()
                tmp = {}
                for time_step in root.iter('timestep'): 
                    time = float(time_step.get('time'))
                    t = int(time / self.period)
                    for vehicle in time_step.iter('vehicle'):       
                        id = vehicle.get('id')
                        edge = vehicle.get('lane')[:-2]
                        detector_id = f'det-{edge}'
                        if detector_id in self.link_hash: 
                            l = self.link_hash[detector_id]
                            tmp.setdefault((l, t), set())
                            tmp[(l, t)].add(id)
                alpha = {}
                for (l, t), vehicles in tmp.items():
                    vehicle_list = list(vehicles)
                    for vehicle_id in vehicle_list: 
                        ijk = vehicle_id.split('.')[0]
                        o, d, k = int(ijk.split('-')[0]), int(ijk.split('-')[1]), int(ijk.split('-')[2])
                        alpha.setdefault((o, d, k, l, t), 0) 
                        alpha[(o, d, k, l, t)] += 1
                for (o, d, k, l, t), val in alpha.items(): 
                    alpha[(o, d, k, l, t)] = val / len(tmp[(l, t)])
            return q_arr, k_arr, v_arr, alpha

        ''' simulation measurements: {q[l, k], k[l, k], v[l, k]} '''
        if single_file_mode: 
            file_list = glob(os.path.join(self.data_dir, '*.xml'))
            assert len(file_list) == 1
            return extract_all_files(file_list[0])
        else: 
            q, k, v = [], [], []
            for file in glob(os.path.join(self.data_dir, '*.xml')):
                q_l, k_l, v_l = extract_single_file(file)
                assert len(q_l) == self.interval_n
                q.append(q_l)
                k.append(k_l)
                v.append(v_l)
            return np.array(q), np.array(k), np.array(v)

    def run_sumo(self, tripinfo_output=None):
        self.counter += 1 
        # run sumo
        cmd = ['sumo', '-c', self.config]
        # seed for reproductivity
        cmd.extend(['--seed', str(self.seed)])
        # mute warnings
        cmd.extend(['-W', str(self.mute_warnings)])
        # no step log
        cmd.extend(['--no-step-log', str(self.mute_step_logs)])
        # no teleport
        cmd.extend(['--time-to-teleport', str(-1)])
        # fcd output
        cmd.extend([ '--fcd-output', self.fcd_output])
        # trip info
        if tripinfo_output is not None:
            cmd.extend(['--tripinfo-output', tripinfo_output])

        subprocess.run(cmd, check=True)
        return self._extract_measurements(single_file_mode=True)
    
class SumoParallelSimulationHandler(object): 
    def __init__(self, sumo_sim_args, simulation_scenario: str, matrix_dim: tuple):
        self.sumo_sim_args = sumo_sim_args
        self.simulation_scenario = simulation_scenario
        self.matrix_dim = matrix_dim
        self.link_hash = {}

        # link hash
        hash_index = 0 
        tree = ET.parse('det.add.xml') # NOTE: hard-code detector additional file
        root = tree.getroot()
        for detector in root.iter('inductionLoop'): 
            link_id = detector.get('id')[:-2]
            if self.link_hash.setdefault(link_id, -1) < 0: 
                self.link_hash[link_id] = hash_index
                hash_index += 1

    def parallel_simulations(self, thread_n: int, simulation_n: int):
        tasks = [(self.matrix_dim, self.simulation_scenario, state, index, self.sumo_sim_args, self.link_hash) for state in [0, 1, 2] for index in range(simulation_n)]
        print(f'# of tasks: {len(tasks)}')
        with ThreadPool(min(thread_n, len(tasks))) as pool: 
            pool.starmap(func=sumo_simulation_task, iterable=tasks, chunksize=1)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type = str, default = 'demo.sumocfg', help = 'sumo configuration path')
    parser.add_argument('--data',       type = str, default = 'simulation', help = 'sensor xml data path')
    parser.add_argument('--flow',       type = str, default = 'demo.rou.xml', help = 'demand xml data path')
    parser.add_argument('--fcd_output', type = str, default = 'fcd.xml', help = 'floating car data path')
    parser.add_argument('--duration',   type = int, default = 7200, help = 'simulation time')
    parser.add_argument('--period',     type = int, default = 360, help = 'interval time')
    parser.add_argument('--seed',       type = int, default = 2025, help = 'for simulation reproductive')    
    parser.add_argument('--mute_warnings',  action='store_true', default=False, help='mute sumo warnings')
    parser.add_argument('--mute_step_logs', action='store_true', default=False, help='mute step logs')
    args = parser.parse_args()
    # consider 10 \times 9 = 90 tazs
    taz_n = 10 # TODO: check number of taz
    matrix_dim = (taz_n, taz_n, (args.duration // args.period))
    # configs
    simulation_handler = SumoParallelSimulationHandler(sumo_sim_args=args, simulation_scenario='scenario_0', matrix_dim=matrix_dim)
    simulation_handler.parallel_simulations(thread_n=10, simulation_n=1000)