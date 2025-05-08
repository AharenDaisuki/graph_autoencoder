import os
import unittest
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from unittest import TestCase
from simulations import sumo_simulation_task
from utils import load_from_pkl

class TestSumoSimulation(TestCase): 
    def test_sumo_simulation_task(self):
        # args 1
        matrix_dim = (10, 10, 20)
        simulation_scenario = 'scenario_test'
        simulation_state = 0
        simulation_index = 0
        categories = ['low', 'medium', 'high']
        link_n = 300
        I, J, K, L, T = 10, 10, 20, 300, 20
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
        # args 3
        hash_index = 0 
        link_hash  = {}
        tree = ET.parse('det.add.xml') # NOTE: hard-code detector additional file
        root = tree.getroot()
        for detector in root.iter('inductionLoop'): 
            link_id = detector.get('id')[:-2]
            if link_hash.setdefault(link_id, -1) < 0: 
                link_hash[link_id] = hash_index
                hash_index += 1
        # test single simulation
        finish_code = sumo_simulation_task(matrix_dim, simulation_scenario, simulation_state, simulation_index, simulation_args, link_hash) 
        self.assertEqual(finish_code, 0)
        q = load_from_pkl(pkl_path=os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index), 'q.pkl'))
        k = load_from_pkl(pkl_path=os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index), 'k.pkl'))
        v = load_from_pkl(pkl_path=os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index), 'v.pkl'))
        x = load_from_pkl(pkl_path=os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index), 'x.pkl'))
        a = load_from_pkl(pkl_path=os.path.join(simulation_scenario, categories[simulation_state], str(simulation_index), 'a.pkl'))
        self.assertEqual(isinstance(q, np.ndarray), True)
        self.assertEqual(isinstance(k, np.ndarray), True)
        self.assertEqual(isinstance(v, np.ndarray), True)
        self.assertEqual(isinstance(x, np.ndarray), True)
        self.assertEqual(isinstance(a, dict), True)
        self.assertEqual(x.shape, (I, J, K))
        self.assertEqual(q.shape, (L, T))
        self.assertEqual(k.shape, (L, T))
        self.assertEqual(v.shape, (L, T))

unittest.main()
