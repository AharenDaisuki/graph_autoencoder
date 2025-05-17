import torch
import unittest
import torch_geometric.transforms as T

from unittest import TestCase
from datasets import get_sumo_trainloader
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class TestSumoDataLoader(TestCase): 
    def test_cuda_available(self):
        self.assertEqual(torch.cuda.is_available(), True)

    def test_get_sumo_trainloader(self): 
        dataloader = get_sumo_trainloader(data_dir='sim_dataset_test', batch_size=1, shuffle=True, swap_prob=0.3)
        for data in dataloader:
            # unidirected graph
            data = T.ToUndirected()(data)
            del data['measurement', 'rev_contributes_to', 'demand'].edge_label
            # link-level split
            train_data, val_data, test_data = T.RandomLinkSplit(
                num_val  = 0.1, # 10% validation
                num_test = 0.1, # 10% test
                neg_sampling_ratio=0.0,
                edge_types=[('demand', 'contributes_to', 'measurement')],
                rev_edge_types=[('measurement', 'rev_contributes_to', 'demand')],
            )(data)
            # Generate the co-occurence matrix of movies<>movies:
            metapath = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]
            train_data = T.AddMetaPaths(metapaths=[metapath])(train_data)

            # Apply normalization to filter the metapath:
            _, edge_weight = gcn_norm(
                train_data['measurement', 'measurement'].edge_index,
                num_nodes=train_data['measurement'].num_nodes,
                add_self_loops=False,
            )
            edge_index = train_data['measurement', 'measurement'].edge_index[:, edge_weight > 0.01]

            train_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
            val_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
            test_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
            print('train data: \n')
            print(train_data)
            print('validation data: \n')
            print(val_data)
            print('test data: \n')
            print(test_data)

unittest.main()