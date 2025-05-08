''' Implementations for encoders '''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing, GraphConv, SAGEConv, GATConv, GCNConv

''' 
config.size_u
config.size_z
config.edge_n
config.drop_prob
config.aggr
'''
# # GCMC
# class RGCLayer(MessagePassing):
#     ''' regularized graph convolution layer: 
#         x_i^k = \gamma_k (x_i^(k-1), \acc_{j \in N(i)} \phi_k(x_i^(k-1), x_j^(k-1), e_{ij})), 
#         where \gamma denotes update function, \acc denotes aggtqdmregation function (sum, mean, max), 
#         \phi denotes message passing function. Note that \gamma and \phi are differentiatable. ''' 
#     def __init__(self, aggr, config, weight_init):
#         # super(RGCLayer, self).__init__()
#         super().__init__(aggr)
        
#         self.in_channels = None
#         self.out_channels = None

#         self.od_n = config.size_u
#         self.detector_n = config.size_v

#         self.node_n = config.size_u + config.size_v
#         self.basis_n = config.basis_n

#         self.drop_prob = config.drop_prob
#         self.weight_init = weight_init
#         # self.aggr = config.aggr
#         self.relu = nn.ReLU()

#         # weight initialization (NOTE: split stack not implemented)  
#         assert not self.aggr == 'split_stack', 'split stack not implemented!'

#         if self.aggr == 'split_stack':
#             pass
#         else: 
#             # |R| tensors of size 1 * in_c * out_c
#             ord_basis = [nn.Parameter(torch.Tensor(1, self.in_channels * self.out_channels)) for _ in range(self.basis_n)]
#             self.ord_basis = nn.ParameterList(ord_basis)

#         if self.aggr == 'stack': 
#             self.bn = nn.BatchNorm1d(self.in_channels * self.basis_n)
#         else: 
#             self.bn = nn.BatchNorm1d(self.in_channels)

#         self._init_weight(weight_init)

#     # def __init__(self, aggr = 'sum', *, aggr_kwargs = None, flow = "source_to_target", node_dim = -2, decomposed_layers = 1):
#     #     super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)

#     def _init_weight(self, weight_init): 
#         if self.aggr == 'split_stack': 
#             pass
#         else: 
#             for basis in self.ord_basis:
#                 weight_init(basis, self.in_channels, self.out_channels)

#     def reset_parameters(self):
#         return super().reset_parameters()
            
#     # TODO: implement random substitution
#     def _node_dropout(self, weight: torch.Tensor): 
#         mask = torch.rand(self.in_channels) + (1 - self.drop_prob)
#         mask = torch.floor(mask).type(torch.float)
#         if self.accum == 'split_stack':
#             pass
#         else:
#             mask = torch.cat([mask for _ in range(self.basis_n)], dim=0).unsqueeze(1)
#         mask = mask.expand(mask.size(0), self.out_channels)
#         assert weight.shape == mask.shape
#         weight = weight * mask
#         return weight
    
#     def forward(self, x, edge_index, edge_type, edge_norm=None):
#         return self.propagate()
    
#     # def propagate(self, edge_index, size = None, **kwargs):
#     #     # message
#     #     # aggregation
#     #     # update
#     #     return super().propagate(edge_index, size, **kwargs)

#     def propagate(self, edge_index, size = None, **kwargs):
#         mutable_size = self._check_input(edge_index, size)
#         coll_dict = self._collect(self._user_args, edge_index,
#                                     mutable_size, kwargs)
#         # message
#         msg_kwargs = self.inspector.collect_param_data(
#             'message', coll_dict)
#         for hook in self._message_forward_pre_hooks.values():
#             res = hook(self, (msg_kwargs, ))
#             if res is not None:
#                 msg_kwargs = res[0] if isinstance(res, tuple) else res
#         out = self.message(**msg_kwargs)
#         for hook in self._message_forward_hooks.values():
#             res = hook(self, (msg_kwargs, ), out)
#             if res is not None:
#                 out = res
#         # TODO: aggregation
#         if self.aggr == 'split_stack':
#             pass
#             # out = split_stack(out, edge_index[0], kwargs['edge_type'], dim_size=size)
#         elif self.aggr == 'stack':
#             pass
#             # out = stack(out, edge_index[0], kwargs['edge_type'], dim_size=size)
#         else:
#             pass
#             # out = scatter_(aggr, out, edge_index[0], dim_size=size)
#         # update
#         update_kwargs = self.inspector.collect_param_data('update', coll_dict)
#         out = self.update(out, **update_kwargs)
#         return super().propagate(edge_index, size, **kwargs)

#     def message(self, x_j, edge_type, edge_norm):
#         ''' implement message passing function \phi '''
#         if self.aggr == 'split_stack': 
#             pass
#         else: 
#             for basis in range(self.basis_n): 
#                 if basis == 0: 
#                     weight = self.ord_basis[basis]
#                 else: 
#                     # weight = [basis_0, basis_0 + basis_1, ..., \sum_i basis_i]
#                     weight = torch.cat((weight, weight[-1] + self.ord_basis[basis]), 0)
#             # R \times (in_c * out_c) => (R * in_c) \times out_c
#             weight = weight.reshape(-1, self.out_channels)
#             # (r, j) => i
#             index  = edge_type * self.in_channels + x_j
#         weight = self._node_dropout(weight)
#         out = weight[index] if edge_norm is None else weight[index] * edge_norm.reshape(-1, 1)
#         return out

#     def update(self, inputs):
#         ''' implement update function \gamma '''
#         # inputs of size [n, out_c]
#         outputs = inputs
#         if self.bn: 
#             outputs = self.bn(outputs.unsqueeze(0)).squeeze(0)
#         if self.relu: 
#             outputs = self.relu(outputs)
#         return outputs
        
# # TODO: modify
# class DenseLayer(nn.Module): 
#     def __init__(self, config, weight_init, bias=False):
#         super(DenseLayer, self).__init__()
#         # TODO: configurations
#         in_channels, out_channels = None, None

#         self.bn = None
#         self.activate = None
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p = config.drop_prob)
#         self.fc = nn.Linear(in_channels, out_channels, bias = bias)

#         # TODO: assert config.accum
#         if config.accum == 'stack':
#             self.bn_u = nn.BatchNorm1d()
#             self.bn_v = nn.BatchNorm1d()
#         else: 
#             self.bn_u = nn.BatchNorm1d(config.size_u)
#             self.bn_v = nn.BatchNorm1d(config.size_v) 

#     def forward(self, u_features, v_features):
#         # for u features (drop => linear => normalize => activate)
#         u_features = self.dropout(u_features) 
#         u_features = self.fc(u_features)
#         if self.bn: 
#             u_features = torch.unsqueeze(u_features, 0)
#             u_features = torch.squeeze(self.bn(u_features))
#         if self.activate: 
#             u_features = self.relu(u_features)
#         # for v features
#         v_features = self.dropout(v_features) 
#         v_features = self.fc(v_features)
#         if self.bn: 
#             v_features = torch.unsqueeze(v_features, 0)
#             v_features = torch.squeeze(self.bn(v_features))
#         if self.activate: 
#             v_features = self.relu(v_features)
#         return u_features, v_features
    
# class Encoder_GCMC(nn.Module): 
#     def __init__(self, config):
#         super(Encoder_GCMC, self).__init__()
#         self.rgc_layer = RGCLayer(config)
#         self.dense_layer = DenseLayer(config)
#         self.edge_n = config.edge_n
#         self.od_n = None

#     def forward(self, x):
#         features = self.rgc_layer()
#         u_features, v_features = self._seperate_features(features)
#         u_features, v_features = self.dense_layer(u_features, v_features)
#         return u_features, v_features

#     def _seperate_features(self, features): 
#         ''' seperate features if stacked '''
#         # TODO: check accum
#         if self.accum == 'stack':
#             node_n = int(features.shape[0] / self.edge_n)
#             for r in range(self.edge_n):
#                 if r == 0:
#                     u_features = features[:self.od_n]
#                     v_features = features[self.od_n: (r+1) * node_n]
#                 else:
#                     u_features = torch.cat((u_features,
#                         features[r * node_n: r * node_n + self.od_n]), dim=0)
#                     v_features = torch.cat((v_features,
#                         features[r * node_n + self.od_n: (r+1) * node_n]), dim=0)
#         else:
#             u_features = features[:self.od_n]
#             v_features = features[self.od_n:]

#         return u_features, v_features

class Encoder_GNN(nn.Module): 
    def __init__(self, hidden_channels: int, out_channels: int, gconv=GraphConv, edge_weighted=True):
        super().__init__()
        assert gconv in [GraphConv, SAGEConv, GATConv, GCNConv]
        self.gconv = gconv
        self.conv1 = gconv((-1, -1), hidden_channels)
        self.conv2 = gconv((-1, -1), out_channels)
        # self.lin = Linear(hidden_channels, out_channels)
        self.edge_weighted = edge_weighted

    def forward(self, x, edge_index, edge_weight=None):
        if self.gconv not in [GraphConv, GCNConv]: 
            edge_weight = None
        if edge_weight is not None: 
            edge_weight = F.sigmoid(edge_weight) if self.edge_weighted else None
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x

class Encoder_GNN_u_weighted(nn.Module):
    def __init__(self, hidden_channels, out_channels, gconv=GraphConv, edge_weighted=True):
        super().__init__()
        assert gconv in [GraphConv, SAGEConv, GATConv, GCNConv]
        self.gconv = gconv
        self.conv1 = gconv((-1, -1), hidden_channels)
        self.conv2 = gconv((-1, -1), hidden_channels)
        self.conv3 = gconv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)
        self.edge_weighted = edge_weighted

    def forward(self, x_dict, edge_index_dict, edge_weight=None):
        if self.gconv not in [GraphConv, GCNConv]: 
            edge_weight = None
        if edge_weight is not None: 
            edge_weight = F.sigmoid(edge_weight) if self.edge_weighted else None
        movie_x = self.conv1(
            x_dict['measurement'],
            edge_index_dict[('measurement', 'metapath_0', 'measurement')]
        ).relu()

        user_x = self.conv2(
            (x_dict['measurement'], x_dict['demand']),
            edge_index_dict[('measurement', 'rev_contributes_to', 'demand')],
            edge_weight = edge_weight
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x),
            edge_index_dict[('measurement', 'rev_contributes_to', 'demand')],
            edge_weight = edge_weight
        ).relu()

        return self.lin(user_x)
    
class Encoder_GNN_v_weighted(nn.Module): 
    def __init__(self, hidden_channels, out_channels, gconv=GraphConv):
        super().__init__()
        assert gconv in [GraphConv, SAGEConv, GATConv, GCNConv]
        self.gconv = gconv
        self.conv1 = gconv(-1, hidden_channels)
        self.conv2 = gconv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


    







        
