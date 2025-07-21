import os
import unittest
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from unittest import TestCase
from simulations import sumo_simulation_task, SumoParallelSimulationHandler
from utils import load_from_pkl

class TestSumoSimulation(TestCase): 
    def test_1_sumo_parallel_simulation_handler(self): 
        # args 1
        matrix_dim = (10, 10, 20)
        simulation_dataset = 'sim_dataset_test'

        # args 2
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
        simulation_args = parser.parse_args()
        simulation_handler = SumoParallelSimulationHandler(sumo_sim_args=simulation_args, simulation_dataset=simulation_dataset, matrix_dim=matrix_dim)
        link_n = simulation_handler.get_link_n()
        print(f'{link_n} links with detectors')
        print('#' * 50 + '\n')
    
    def test_2_sumo_simulation_task(self):
        # args 1
        simulation_dataset = 'sim_dataset_test'
        simulation_scenario = 0
        simulation_state = 0
        simulation_index = 0
        categories = ['low', 'medium', 'high']
        I, J, K = 10, 10, 20
        matrix_dim = (I, J, K)
        # args 2
        parser = argparse.ArgumentParser()
        parser.add_argument('--config',     type = str, default = 'demo.sumocfg', help = 'sumo configuration path')
        parser.add_argument('--data',       type = str, default = 'simulation', help = 'sensor xml data path')
        parser.add_argument('--flow',       type = str, default = 'demo.rou.xml', help = 'demand xml data path')
        parser.add_argument('--fcd_output', type = str, default = 'fcd.xml', help = 'floating car data path')
        parser.add_argument('--duration',   type = int, default = 7200, help = 'simulation time')
        parser.add_argument('--period',     type = int, default = 180, help = 'interval time')
        parser.add_argument('--seed',       type = int, default = 2025, help = 'for simulation reproductive')    
        parser.add_argument('--mute_warnings',  action='store_true', default=False, help='mute sumo warnings')
        parser.add_argument('--mute_step_logs', action='store_true', default=False, help='mute step logs')
        simulation_args = parser.parse_args()
        T = (simulation_args.duration / simulation_args.period)
        # args 3
        hash_index = 0 
        link_hash  = {}
        tree = ET.parse('sparse.add.xml') # NOTE: hard-code detector additional file
        root = tree.getroot()
        for detector in root.iter('inductionLoop'): 
            link_id = detector.get('id')[:-2]
            if link_hash.setdefault(link_id, -1) < 0: 
                link_hash[link_id] = hash_index
                hash_index += 1
        L = len(link_hash)
        # test single simulation: normalized + save fcd
        finish_code = sumo_simulation_task(matrix_dim, simulation_dataset, simulation_state, simulation_scenario, simulation_index, simulation_args, link_hash, True, True) 
        self.assertEqual(finish_code, 0)
        output_dir = os.path.join(simulation_dataset, categories[simulation_state], f'scenario_{simulation_scenario}', str(simulation_index))
        scene_dir = os.path.join(simulation_dataset, categories[simulation_state], f'scenario_{simulation_scenario}')
        q = load_from_pkl(pkl_path=os.path.join(output_dir, 'q.pkl'))
        k = load_from_pkl(pkl_path=os.path.join(output_dir, 'k.pkl'))
        v = load_from_pkl(pkl_path=os.path.join(output_dir, 'v.pkl'))
        x = load_from_pkl(pkl_path=os.path.join(output_dir, 'x.pkl'))
        a = load_from_pkl(pkl_path=os.path.join(output_dir, 'a.pkl'))
        b = load_from_pkl(pkl_path=os.path.join(output_dir, 'b.pkl'))
        u = load_from_pkl(pkl_path=os.path.join(output_dir, 'u.pkl'))
        y = load_from_pkl(pkl_path=os.path.join(output_dir, 'y.pkl'))
        dist = load_from_pkl(pkl_path=os.path.join(scene_dir, 'dist.pkl'))
        self.assertEqual(isinstance(q, np.ndarray), True)
        self.assertEqual(isinstance(k, np.ndarray), True)
        self.assertEqual(isinstance(v, np.ndarray), True)
        self.assertEqual(isinstance(x, np.ndarray), True)
        self.assertEqual(isinstance(a, dict), True)
        self.assertEqual(isinstance(b, dict), True)
        self.assertEqual(isinstance(u, dict), True)
        self.assertEqual(isinstance(y, dict), True)
        self.assertEqual(isinstance(dist, dict), True)
        self.assertEqual(x.shape, (I, J, K))
        self.assertEqual(q.shape, (L, T))
        self.assertEqual(k.shape, (L, T))
        self.assertEqual(v.shape, (L, T))
        self.assertEqual(dist['mu'].shape, (I, J, K))
        self.assertEqual(dist['sigma'].shape, (I, J, K))
        print(f'(I, J, K) = {x.shape}; (L, T) = {q.shape}')
        if os.path.exists(os.path.join(output_dir, 'metadata.pkl')): 
            metadata = load_from_pkl(pkl_path=os.path.join(output_dir, 'metadata.pkl'))
            self.assertEqual(isinstance(metadata, dict), True)
            print('|q| = {}, |k| = {}, |v| = {}, |x| = {}'.format(metadata['q_mag'], metadata['k_mag'], metadata['v_mag'], metadata['x_mag']))
        print('#' * 50 + '\n')
        
unittest.main()
