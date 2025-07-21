import torch
import unittest
import torch_geometric.transforms as T

from unittest import TestCase
from statistics import quantiles
from datasets import get_sumo_trainloader, get_sumo_dataloaders
from loss import node_u_recon_loss, node_v_recon_loss
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class TestSumoDataLoader(TestCase): 
    def test_1_cuda_available(self):
        self.assertEqual(torch.cuda.is_available(), True)

    def test_2_get_sumo_trainloader(self): 
        # NOTE: name of dataset
        dataloader = get_sumo_trainloader(data_dir='sim_dataset_test', batch_size=1, shuffle=True, swap_prob=0.0)
        quantile   = [0] * 10
        sample_n   = 0
        for data in dataloader:
            # unidirected graph
            data = T.ToUndirected()(data)
            del data['measurement', 'rev_contributes_to', 'demand'].edge_label
            # loss_u = node_u_recon_loss(data=data, alphas=data['demand', 'measurement'].edge_label)
            # loss_v = node_v_recon_loss(data=data, alphas=data['demand', 'measurement'].edge_label)
            # self.assertAlmostEqual(float(loss_u), 0.0, places=4)
            # self.assertAlmostEqual(float(loss_v), 0.0, places=4)
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
            edge_weight_quantile = quantiles(data=edge_weight.tolist(), n = 10+1)
            for idx, val in enumerate(edge_weight_quantile): 
                quantile[idx] += val
            sample_n += 1

            edge_index = train_data['measurement', 'measurement'].edge_index[:, edge_weight > 0.006]

            train_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
            val_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
            test_data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index

        quantile_ = [x / sample_n for x in quantile]
        print(quantile_)

    def test_3_get_sumo_dataloader(self): 
        train_loader, test_loader = get_sumo_dataloaders(data_dir='sim_dataset_v2')
        train_quantile = [0] * 10
        test_quantile = [0] * 10
        train_sample = 0
        test_sample = 0
        # train
        for data in train_loader: 
            # unidirected graph
            data = T.ToUndirected()(data)
            del data['measurement', 'rev_contributes_to', 'demand'].edge_label
            # check data 
            # loss_u = node_u_recon_loss(data=data, alphas=data['demand', 'measurement'].edge_label)
            # loss_v = node_v_recon_loss(data=data, alphas=data['demand', 'measurement'].edge_label)
            # self.assertAlmostEqual(float(loss_u), 0.0, places=4)
            # self.assertAlmostEqual(float(loss_v), 0.0, places=4)
            # Generate the co-occurence matrix of movies<>movies:
            metapath = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]
            data = T.AddMetaPaths(metapaths=[metapath])(data)
            # Apply normalization to filter the metapath:
            _, edge_weight = gcn_norm(
                data['measurement', 'measurement'].edge_index,
                num_nodes=data['measurement'].num_nodes,
                add_self_loops=False,
            )
            # statistics
            edge_weight_quantile = quantiles(data=edge_weight.tolist(), n = 10+1)
            for idx, val in enumerate(edge_weight_quantile): 
                train_quantile[idx] += val
            train_sample += 1
            edge_index = data['measurement', 'measurement'].edge_index[:, edge_weight > 0.006]
            data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        train_quantile_ = [x / train_sample for x in train_quantile]
        print(train_quantile_)
        # test
        for data in test_loader: 
            # unidirected graph
            data = T.ToUndirected()(data)
            del data['measurement', 'rev_contributes_to', 'demand'].edge_label
            # check data 
            # loss_u = node_u_recon_loss(data=data, alphas=data['demand', 'measurement'].edge_label)
            # loss_v = node_v_recon_loss(data=data, alphas=data['demand', 'measurement'].edge_label)
            # self.assertAlmostEqual(float(loss_u), 0.0, places=4)
            # self.assertAlmostEqual(float(loss_v), 0.0, places=4)
            # Generate the co-occurence matrix of movies<>movies:
            metapath = [('measurement', 'rev_contributes_to', 'demand'), ('demand', 'contributes_to', 'measurement')]
            data = T.AddMetaPaths(metapaths=[metapath])(data)
            # Apply normalization to filter the metapath:
            _, edge_weight = gcn_norm(
                data['measurement', 'measurement'].edge_index,
                num_nodes=data['measurement'].num_nodes,
                add_self_loops=False,
            )
            # statistics
            edge_weight_quantile = quantiles(data=edge_weight.tolist(), n = 10+1)
            for idx, val in enumerate(edge_weight_quantile): 
                test_quantile[idx] += val
            test_sample += 1
            edge_index = data['measurement', 'measurement'].edge_index[:, edge_weight > 0.006]
            data['measurement', 'metapath_0', 'measurement'].edge_index = edge_index
        test_quantile_ = [x / test_sample for x in test_quantile]
        print(test_quantile_)

unittest.main()