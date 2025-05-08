import unittest
import argparse

from unittest import TestCase
from train import train_bipartite_link_pred

class TestTrain(TestCase): 
    def test_train_bipartite_link_pred(self): 
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_channels', type = int, default = 64, help = 'number of hidden channels (encoder)')
        parser.add_argument('--out_channels', type = int, default = 64, help = 'number of output channels (encoder)')
        parser.add_argument('--lr', type = float, default = 5e-4, help = 'learning rate')
        parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'adam weight decay')
        parser.add_argument('--epoch', type = int, default = 200, help = 'number of training epochs')
        parser.add_argument('--period', type = int, default = 20, help = 'save model every n epochs')
        parser.add_argument('--data',  type = str, default = 'scenario_test', help = 'training data folder')
        parser.add_argument('--seed',  type = int, default = 2025, help = 'training reproductivity')    
        parser.add_argument('--save',  type = str, default = 'checkpoints', help = 'checkpoint save path')
        parser.add_argument('--model', type = str, default = 'BipartiteLinkPred', help = 'model name')      
        parser.add_argument('--resume', type = str, help = 'load model from checkpoints')  
        args = parser.parse_args()
        train_bipartite_link_pred(args)

unittest.main()