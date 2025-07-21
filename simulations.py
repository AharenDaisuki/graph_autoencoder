import os
import sys
import shutil
import random
import argparse
import subprocess
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from glob import glob
from utils import save_as_pkl, clear_tmp_dir
from multiprocessing.pool import ThreadPool

TASK_SUCCESS = 0
TASK_ERROR = -1

ROADNETWORK_FILE = 'osm.net.xml'
SUMOCONFIG_FILE = 'sim.sumocfg'
# DETECTOR_FILE = 'det.add.xml'
DETECTOR_FILE = 'sparse.add.xml'
DEMAND_FILE = 'flow.rou.xml'
ROUTE_FILE = 'trips.rou.xml'
TAZ_FILE = 'taz.add.xml'
FCD_FILE = 'fcd.xml'
MEASUREMENT_FILE = 'detectors.xml'

MATX_FILE = 'x.pkl'
FLOW_FILE = 'q.pkl'
DENSITY_FILE = 'k.pkl'
VELOCITY_FILE = 'v.pkl'
DAR_A_FILE = 'a.pkl'
DAR_B_FILE = 'b.pkl'
UORIG_FILE = 'u.pkl'
VORIG_FILE = 'y.pkl'
DISTRIBUTION_FILE = 'dist.pkl'
META_FILE = 'metadata.pkl'

# setup: 
## 1. initialize traffic assignment zone
## 2. initialize detector layout
## 3. prepare road network

# simulation params: 
NODE_STATE_PROB = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
MEANS = [25.0, 50.0, 70.0]
STDS  = [5.0, 7.5, 10.0]
X_MIN, X_MAX = 10.0, 100.0

# MEANS = [10, 20, 40]
# STDS = [3, 5, 10]
# X_MIN, X_MAX = 1.0, 70.0

DETECTOR_WORKING_CYCLE = 30.0 # NOTE: detector working cycle is 30s by default
SUMO_STEP_LENGTH = 2.0
LINK_MIN_LENGTH = 2.0
LINK_MAX_NUM = 200

def generate_od_matrix(matrix_dim, simulation_state, flow_save_dir = None, period = 180):
    ''' generate od matrix numpy nd array for scenario A, state B and simulation C '''

    def generate_rou_file(filepath, matrix_dim, od_matrix, period):
        ''' generate sumo route file''' 
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

    # x \in [0, 40]
    assert simulation_state in [0, 1, 2] # 0 => low, 1 => medium, 2 => high
    I, J, K = matrix_dim 
    # means = [10, 20, 40]
    # stds = [3, 5, 10]
    # node_state_prob = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]] 
    # x_min, x_max = 1.0, 70.0

    # generate od matrix
    node_state  = np.random.choice(a = [0, 1, 2], size = I * J * K, p = NODE_STATE_PROB[simulation_state])
    mu = list(map(lambda x: MEANS[x], node_state))
    sigma = list(map(lambda x: STDS[x], node_state))
    od_matrix = np.random.normal(loc = mu, scale = sigma, size = I * J * K)
    od_matrix = np.clip(od_matrix, a_min = X_MIN, a_max = X_MAX)
    od_matrix = od_matrix.reshape(matrix_dim)
    mu = np.array(mu, dtype=np.float64).reshape(matrix_dim)
    sigma = np.array(sigma, dtype=np.float64).reshape(matrix_dim)
    # generate sumo route
    flow_file_path = os.path.join(flow_save_dir, DEMAND_FILE) if flow_save_dir else DEMAND_FILE
    if flow_save_dir: 
        generate_rou_file(filepath=flow_file_path, matrix_dim=matrix_dim, od_matrix=od_matrix, period=period)
    return od_matrix, mu, sigma, flow_file_path

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

# def sumo_objective_function_nrmse(sim_measurements: list, obs_measurements: list):
#     obj = 0
#     loss_list = []
#     weight_list = []
#     # calculate weight, loss
#     for sim_measurement, obs_measurement in zip(sim_measurements, obs_measurements): 
#         assert sim_measurement.shape == obs_measurement.shape
#         L, K = sim_measurement.shape
#         sigma_sim, sigma_obs = np.std(sim_measurement), np.std(obs_measurement)
#         square_error = np.sum((sim_measurement - obs_measurement) * (sim_measurement - obs_measurement))
#         weight = square_error / L * K * (sigma_obs - sigma_sim) ** 2 
#         loss = np.sqrt(L * K * square_error) / np.sum(obs_measurement)
#         loss_list.append(loss)
#         weight_list.append(weight)
#     # calculate objective
#     for loss, weight in zip(loss_list, weight_list):
#         alpha = weight / sum(weight_list)
#         obj += alpha * loss
#     return obj

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
            f.write(f'\t\t<route-files value="{ROUTE_FILE}"/>\n')
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

    # receive arguments
    matrix_dim, simulation_dataset, simulation_state, simulation_scenario, simulation_index, simulation_args, link_hash, normalized, save_fcd = args
    assert simulation_state in [0, 1, 2]
    assert isinstance(simulation_index, int)
    assert isinstance(simulation_scenario, int)
    assert isinstance(simulation_dataset, str)
    assert isinstance(matrix_dim, tuple)
    assert isinstance(link_hash, dict)
    assert isinstance(normalized, bool)
    assert isinstance(save_fcd, bool)
    
    categories = ['low', 'medium', 'high']

    try: 
        # create temporary directory
        sim_tmp_dir = os.path.join(simulation_dataset, categories[simulation_state], f'scenario_{simulation_scenario}', str(simulation_index), 'tmp')
        sim_dat_dir = os.path.join(simulation_dataset, categories[simulation_state], f'scenario_{simulation_scenario}', str(simulation_index), 'tmp', 'simulation')
        output_dir  = os.path.join(simulation_dataset, categories[simulation_state], f'scenario_{simulation_scenario}', str(simulation_index))
        sim_scene_dir = os.path.join(simulation_dataset, categories[simulation_state], f'scenario_{simulation_scenario}')
        # clear cache
        if os.path.exists(sim_tmp_dir): 
            clear_tmp_dir(sim_tmp_dir) 
        # make directory
        os.makedirs(sim_dat_dir)
        assert os.path.exists(output_dir) and os.path.exists(sim_tmp_dir) and os.path.exists(sim_scene_dir)

        # - output_dir
            # --tmp
                # -- osm.net.xml
                # -- det.add.xml
                # -- taz.add.xml
                # -- flow.rou.xml
                # -- trips.rou.xml
                # -- sim.sumocfg
                # -- detectors.xml 

            # -- q.pkl, k.pkl, v.pkl *
            # -- fcd.xml * 
            # -- alpha.pkl *
            # -- x.pkl * 

        I, J, K = matrix_dim

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
        x, mu, sigma, od_demand_file = generate_od_matrix(matrix_dim=matrix_dim, 
                        simulation_state=simulation_state, 
                        flow_save_dir=sim_tmp_dir,
                        period=(simulation_args.duration // K))
        od2routes(road_network=dest_net_path, add_taz=dest_taz_path, od_demand=od_demand_file, output_routes=os.path.join(sim_tmp_dir, ROUTE_FILE))

        # create temp sim.sumocfg
        orig_cfg_path = os.path.join(sim_tmp_dir, SUMOCONFIG_FILE)
        _make_tmp_config(config_path=orig_cfg_path, duration=simulation_args.duration)

        # simulation
        simulation_args.config = orig_cfg_path
        simulation_args.data = sim_dat_dir
        # simulation_args.flow = os.path.join(sim_tmp_dir, ROUTE_FILE)
        simulation_args.fcd_output = os.path.join(output_dir, FCD_FILE)
        simulation_args.mute_warnings = True
        simulation_args.mute_step_logs = True
        simulator = SumoSimulation(simulation_args, link_hash=link_hash)

        q, k, v, a, b, u_orig, v_orig = simulator.run_sumo(save_fcd = save_fcd) # NOTE: do not save fcd
        if normalized: 
            # q_magnitude = np.linalg.norm(q)
            # k_magnitude = np.linalg.norm(k)
            # v_magnitude = np.linalg.norm(v)
            # x_magnitude = np.linalg.norm(x)
            q_magnitude = np.max(q)
            k_magnitude = np.max(k)
            v_magnitude = np.max(v)
            x_magnitude = np.max(x)
            assert q_magnitude > 0 and k_magnitude > 0 and v_magnitude > 0 and x_magnitude > 0
            q = q / q_magnitude
            k = k / k_magnitude
            v = v / v_magnitude
            x = x / x_magnitude
            meta_data = {'q_mag': q_magnitude, 'k_mag': k_magnitude, 'v_mag': v_magnitude, 'x_mag': x_magnitude}
            save_as_pkl(meta_data, pkl_path=os.path.join(output_dir, META_FILE)) 
        
        dist_data = {'mu': mu, 'sigma': sigma}

        # numpy nd array
        save_as_pkl(q, pkl_path=os.path.join(output_dir, FLOW_FILE))
        save_as_pkl(k, pkl_path=os.path.join(output_dir, DENSITY_FILE))
        save_as_pkl(v, pkl_path=os.path.join(output_dir, VELOCITY_FILE))
        save_as_pkl(x, pkl_path=os.path.join(output_dir, MATX_FILE))
        save_as_pkl(a, pkl_path=os.path.join(output_dir, DAR_A_FILE))
        save_as_pkl(b, pkl_path=os.path.join(output_dir, DAR_B_FILE))
        save_as_pkl(u_orig, pkl_path=os.path.join(output_dir, UORIG_FILE))
        save_as_pkl(v_orig, pkl_path=os.path.join(output_dir, VORIG_FILE))
        save_as_pkl(dist_data, pkl_path=os.path.join(sim_scene_dir, DISTRIBUTION_FILE)) # distribution data

        clear_tmp_dir(tmp_dir = sim_tmp_dir)
        return TASK_SUCCESS
    except:
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
        self.link_n  = len(self.link_hash)
        self.counter = 0
        # check environment variable
        if 'SUMO_HOME' not in os.environ:
            assert False, f'Check SUMO_HOME environment variable'
        else:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)

    def set_simulation_args(self, args): 
        self.duration   = args.duration
        self.period     = args.period
        self.data_dir   = args.data
        self.seed       = args.seed
        self.config     = args.config
        self.fcd_output = args.fcd_output
        self.mute_warnings  = args.mute_warnings
        self.mute_step_logs = args.mute_step_logs
        self.interval_n = (args.duration // args.period)

    def set_link_hash(self, link_hash): 
        self.link_hash = link_hash
        self.link_n = len(self.link_hash)

    def _extract_measurements(self, single_file_mode=False, save_fcd=True):
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
                q.append(q_ * DETECTOR_WORKING_CYCLE / self.period)
                k.append(k_ * DETECTOR_WORKING_CYCLE / self.period)
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
                q_arr[l, t] = (q_ * DETECTOR_WORKING_CYCLE / self.period)
                k_arr[l, t] = (k_ * DETECTOR_WORKING_CYCLE / self.period)
                v_arr[l, t] =  v_
            # floating car data
            # \alpha_{ijk}^{lt} = x_{ijk}^{lt} / x_{ijk}
            # \beta_{ijk}^{lt} = x_{ijk}^{lt} / y_{lt}
            if self.fcd_output: 
                tree = ET.parse(self.fcd_output)
                root = tree.getroot()
                tmp = {}
                u_origin, v_origin = {}, {}
                for time_step in root.iter('timestep'): 
                    time = float(time_step.get('time'))
                    t = int(time / self.period)
                    for vehicle in time_step.iter('vehicle'):       
                        id = vehicle.get('id')
                        # ijk
                        ijk = id.split('.')[0]
                        o, d, k = int(ijk.split('-')[0]), int(ijk.split('-')[1]), int(ijk.split('-')[2])
                        # n
                        n = int(id.split('.')[1])
                        # x_ijk = max{n}
                        u_origin.setdefault((o, d, k), 0)
                        u_origin[(o, d, k)] = max(u_origin[o, d, k], n+1)

                        edge = vehicle.get('lane')[:-2]
                        detector_id = f'det-{edge}'
                        if detector_id in self.link_hash: 
                            l = self.link_hash[detector_id]
                            tmp.setdefault((l, t), set())
                            tmp[(l, t)].add((o, d, k, n))
                if not save_fcd:
                    os.remove(self.fcd_output) 
                alpha, beta = {}, {} 
                for (l, t), vehicles_data in tmp.items():
                    vehicle_list = list(vehicles_data)
                    v_origin[(l, t)] = len(vehicle_list)
                    for data in vehicle_list: 
                        o, d, k, _ = data
                        alpha.setdefault((o, d, k, l, t), 0) 
                        alpha[(o, d, k, l, t)] += 1
                for (o, d, k, l, t), val in alpha.items(): 
                    assert val <= u_origin[(o, d, k)]
                    alpha[(o, d, k, l, t)] = val / u_origin[(o, d, k)]  
                    beta[(o, d, k, l, t)] = val / v_origin[(l, t)]
            return q_arr, k_arr, v_arr, alpha, beta, u_origin, v_origin

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

    def run_sumo(self, tripinfo_output=None, save_fcd=True):
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
        return self._extract_measurements(single_file_mode=True, save_fcd=save_fcd)
    
class SumoParallelSimulationHandler(object): 
    def __init__(self, sumo_sim_args, simulation_dataset: str, matrix_dim: tuple):
        self.sumo_sim_args = sumo_sim_args
        self.simulation_dataset = simulation_dataset
        self.matrix_dim = matrix_dim
        self.link_hash = {}
        self.link_n = 0
        # warm up
        # self._sumo_simulation_warmup()
        self._link_hash()

    def get_link_n(self) -> int: 
        return self.link_n
    
    def _link_hash(self): 
        # link hash
        hash_index = 0 
        self.link_hash = {}
        tree = ET.parse(DETECTOR_FILE) 
        root = tree.getroot()
        for detector in root.iter('inductionLoop'): 
            link_id = detector.get('id')[:-2]
            if self.link_hash.setdefault(link_id, -1) < 0: 
                self.link_hash[link_id] = hash_index
                hash_index += 1
        self.link_n = hash_index
        assert hash_index == len(self.link_hash)

    def _sumo_simulation_warmup(self):
        ''' run a warm-up simulation to deploy detectors on road network and hash links with detectors '''
        def deploy_detectors(osm_file, period, edge_set = None, filtered_length = 0.0):
            ''' deploy detectors (period = x) for all edges longer than filtered_length ''' 
            cnt = 0
            with open(DETECTOR_FILE, 'w') as f: 
                # write header
                f.write('<?xml version="1.0" encoding="utf-8"?>\n') 
                f.write('<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">\n')
                # read roadnetwork
                assert os.path.exists(osm_file), 'road network does not exist.'
                tree = ET.parse(osm_file)
                root = tree.getroot()
                for edge in root.iter('edge'): 
                    # skip internal
                    if edge.get('function') == 'internal': 
                        continue
                    edge_id = edge.get('id')
                    # skip remote edges
                    if edge_set: 
                        if edge_id not in edge_set: 
                            continue
                    # deploy detectors
                    add_detector = False
                    for lane in edge.iter('lane'): 
                        lane_id_ = lane.get('id')
                        len_ = float(lane.get('length'))
                        detector_id_ = f'det-{lane_id_}'
                        file_ = os.path.join('simulation', MEASUREMENT_FILE)
                        if len_ > filtered_length: 
                            f.write(f'\t<inductionLoop id="{detector_id_}" lane="{lane_id_}" pos="{float(len_/2)}" period="{period}" file="{file_}"/>\n')
                            add_detector = True
                    if add_detector: 
                        cnt += 1
                f.write('</additional>\n')
            f.close()
            return cnt
        
        print('sumo simulation warm up...')
        # deploy detectors
        deploy_detectors(osm_file = ROADNETWORK_FILE, 
                         period = DETECTOR_WORKING_CYCLE)
        self._link_hash()
        # warm up
        tmp_dir = 'sim_warmup'
        edge_set = set()
        is_normalized = False
        save_fcd = True
        for simulation_state in [0, 1, 2]: 
            sumo_simulation_task(self.matrix_dim, tmp_dir, simulation_state, 0, 0, self.sumo_sim_args, self.link_hash, is_normalized, save_fcd)
        for fcd_path in glob(os.path.join(tmp_dir, '*', '*', '*', FCD_FILE)): 
            tree = ET.parse(fcd_path)
            root = tree.getroot()
            for time_step in root.iter('timestep'): 
                for vehicle in time_step.iter('vehicle'):       
                    edge_id = vehicle.get('lane')[:-2]
                    edge_set.add(edge_id)
        # try: 
        #     edge_set = random.sample(population=edge_set, k=LINK_MAX_NUM)
        #     assert len(edge_set) == LINK_MAX_NUM
        # except ValueError:
        #     edge_set = edge_set
        #     assert False, 'maximum number of links is too large!'

        deploy_detectors(osm_file = ROADNETWORK_FILE, 
                         period = DETECTOR_WORKING_CYCLE, 
                         edge_set = edge_set, filtered_length=LINK_MIN_LENGTH)
        self._link_hash()
        clear_tmp_dir(tmp_dir = tmp_dir)

    def parallel_simulations(self, thread_n: int, scenario_n: int, simulation_n: int, normalized: bool, save_fcd: bool):
        simulation_states = [0, 1, 2]
        tasks = [(self.matrix_dim, self.simulation_dataset, state, scenario, index, self.sumo_sim_args, self.link_hash, normalized, save_fcd) 
                 for state in simulation_states 
                 for scenario in range(scenario_n) 
                 for index in range(simulation_n)]
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
    matrix_dim = (10, 10, 20)
    # configs
    simulation_handler = SumoParallelSimulationHandler(sumo_sim_args=args, simulation_dataset='sim_dataset_v1', matrix_dim=matrix_dim)
    simulation_handler.parallel_simulations(thread_n=10, scenario_n=25, simulation_n=40, normalized=True, save_fcd=False)