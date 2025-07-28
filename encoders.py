''' Implementations for encoders '''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn.conv import GraphConv, SAGEConv, GATConv, GCNConv

''' 
config.size_u
config.size_z
config.edge_n
config.drop_prob
config.aggr
'''

# baseline
# class Encoder_GNN(nn.Module): 
#     def __init__(self, hidden_channels: int, out_channels: int, gconv=GraphConv, edge_weighted=True):
#         super().__init__()
#         assert gconv in [GraphConv, SAGEConv, GATConv, GCNConv]
#         self.gconv = gconv
#         self.conv1 = gconv((-1, -1), hidden_channels)
#         self.conv2 = gconv((-1, -1), out_channels)
#         # self.lin = Linear(hidden_channels, out_channels)
#         self.edge_weighted = edge_weighted

#     def forward(self, x, edge_index, edge_weight=None):
#         if self.gconv not in [GraphConv, GCNConv]: 
#             edge_weight = None
#         if edge_weight is not None: 
#             edge_weight = F.sigmoid(edge_weight) if self.edge_weighted else None
#         x = self.conv1(x, edge_index, edge_weight).relu()
#         x = self.conv2(x, edge_index, edge_weight)
#         return x
    
# weighted 
class Encoder_GNN_u_weighted(nn.Module):
    """
    GNN Encoder for u set features (embeddings of OD matrix indices) in a bipartite graph. 

    Args: 
        hidden_channels (int): number of hidden channels
        out_channels (int): number of output channels
        gconv (torch.nn.Module): graph convolutional layer (default: ``GraphConv``)
    """
    def __init__(self, hidden_channels, out_channels, gconv=GraphConv):
        super().__init__()
        assert gconv in [GraphConv, SAGEConv, GATConv, GCNConv]
        self.gconv = gconv
        self.conv1 = gconv((-1, -1), hidden_channels)
        self.conv2 = gconv((-1, -1), hidden_channels)
        self.conv3 = gconv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight=None):
        if edge_weight is None: 
            movie_x = self.conv1(
                x_dict['measurement'],
                edge_index_dict[('measurement', 'metapath_0', 'measurement')]
            ).relu()
        else: 
            movie_x = self.conv1(
                x_dict['measurement'],
                edge_index_dict[('measurement', 'metapath_0', 'measurement')], 
                edge_weight
            ).relu()

        user_x = self.conv2(
            (x_dict['measurement'], x_dict['demand']),
            edge_index_dict[('measurement', 'rev_contributes_to', 'demand')],
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x),
            edge_index_dict[('measurement', 'rev_contributes_to', 'demand')],
        ).relu()
        return self.lin(user_x)
    
class Encoder_GNN_v_weighted(nn.Module): 
    """
    GNN Encoder for v set features (measurement matrix) in a bipartite graph. 

    A measurement matrix is a concatenation of flow, density, velocity data with shape (3, L, T).
    L is the number of detectors; T is the number of time intervals.

    Args: 
        hidden_channels (int): number of hidden channels
        out_channels (int): number of output channels
        gconv (torch.nn.Module): graph convolutional layer (default: ``SAGEConv``)
    """
    def __init__(self, hidden_channels, out_channels, gconv=SAGEConv):
        super().__init__()
        assert gconv in [GraphConv, SAGEConv, GATConv, GCNConv]
        self.gconv = gconv
        self.conv1 = gconv(-1, hidden_channels)
        self.conv2 = gconv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)
    
##################################################################################################
    
# two hop encoding

# v1
# class GCNLayer_v1(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, alpha: float = None):
#         super(GCNLayer_v1, self).__init__()
#         self.gconv = GraphConv(in_channels=-1, out_channels=out_channels, aggr="mean")
#         self.activate = nn.LeakyReLU(alpha) if alpha else nn.ReLU()

#     def forward(self, x, edge_index, edge_weight=None): 
#         x = self.gconv(x, edge_index, edge_weight) 
#         return self.activate(x) 
    
# class GCNLayers_two_hop_v1(nn.Module): 
#     def __init__(self, feature_u_dim: int, feature_v_dim: int, hidden_channels: int, out_channels: int, alpha: float = None):
#         super(GCNLayers_two_hop_v1, self).__init__()
#         self.gconv_u1 = GCNLayer_v1(feature_u_dim, hidden_channels, alpha)
#         self.gconv_u2 = GCNLayer_v1(hidden_channels, feature_u_dim, alpha)
#         self.gconv_v1 = GCNLayer_v1(feature_v_dim, hidden_channels, alpha)
#         self.gconv_v2 = GCNLayer_v1(hidden_channels, feature_v_dim, alpha)
#         self.lin_u = Linear(2 * feature_u_dim, out_channels)
#         self.lin_v = Linear(2 * feature_v_dim, out_channels)

#     def forward(self, u_features, v_features, uv_adj, vu_adj, edge_weight=None): 
#         u_ = self.gconv_u1((u_features, v_features), uv_adj, edge_weight) # u => v 
#         u_ = self.gconv_u2((u_, u_features), vu_adj, edge_weight) # v => u
#         v_ = self.gconv_v1((v_features, u_features), vu_adj, edge_weight) # v => u
#         v_ = self.gconv_v2((v_, v_features), uv_adj, edge_weight) # u => v
#         u = torch.cat((u_, u_features), dim=1)
#         v = torch.cat((v_, v_features), dim=1)
#         u = self.lin_u(u)
#         v = self.lin_v(v)
#         return F.relu(u), F.relu(v)

# class Encoder_GNN_two_hop(nn.Module): 
#     def __init__(self, hidden_channels: int, out_channels: int, layer_n: int, 
#                  feature_u_dim: int = 2, 
#                  feature_v_dim: int = 3, 
#                  alpha: float = None, 
#                  dropout: float = None):
#         super(Encoder_GNN_two_hop, self).__init__()
#         self.layer_n = layer_n
#         self.dropout = dropout
#         gcn_layers = []
#         for _ in range(self.layer_n):
#             gcn_layers.append(GCNLayers_two_hop_v1(feature_u_dim, feature_v_dim, hidden_channels, out_channels, alpha)) # NOTE: two symmetric modules 
#         self.layers = nn.ModuleList(gcn_layers)

#     def forward(self, x_dict, edge_index_dict, edge_weight=None):  
#         feature_u = x_dict['demand']
#         feature_v = x_dict['measurement']
#         for layer in self.layers: 
#             feature_u = F.dropout(feature_u, self.dropout, training=self.training)
#             feature_v = F.dropout(feature_v, self.dropout, training=self.training)
#             feature_u, feature_v = layer(u_features = feature_u, 
#                                          v_features = feature_v, 
#                                          uv_adj = edge_index_dict[('demand', 'contributes_to', 'measurement')], 
#                                          vu_adj = edge_index_dict[('measurement', 'rev_contributes_to', 'demand')], 
#                                          edge_weight = edge_weight)
#         return feature_u, feature_v
    
# # v2

# # [deprecated]
# class GraphConv_bipartite(GraphConv): 
#     def __init__(self, in_channels, out_channels, aggr = 'add', bias = True, **kwargs):
#         super().__init__(in_channels, out_channels, aggr, bias, **kwargs)

#     def forward(self, x, edge_index, edge_weight = None, size = None) -> torch.Tensor:

#         if isinstance(x, torch.Tensor):
#             x = (x, x)

#         # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
#                              size=size)
#         out = self.lin_rel(out)

#         # NOTE: do not add transformed root node features to the output
#         # x_r = x[1]
#         # if x_r is not None:
#         #     out = out + self.lin_root(x_r)

#         return out

# class GCNLayer_v2(nn.Module): 
#     def __init__(self, hidden_channels: int = 64, out_channels: int = 64):
#         super(GCNLayer_v2, self).__init__()

#         # self.conv1 = GraphConv_bipartite((-1, -1), hidden_channels, aggr='add')
#         self.conv1 = SAGEConv((-1, -1), hidden_channels, aggr='mean', root_weight=False)
#         self.conv2 = SAGEConv((-1, -1), hidden_channels, aggr='mean', root_weight=False)
#         self.conv3 = SAGEConv((-1, -1), hidden_channels, aggr='mean', root_weight=False)
#         self.conv4 = SAGEConv((-1, -1), hidden_channels, aggr='mean', root_weight=False)
#         self.lin_u1 = Linear(hidden_channels, out_channels)
#         self.lin_v1 = Linear(hidden_channels, out_channels)
#         self.lin_u2 = Linear(2 * hidden_channels, out_channels)
#         self.lin_v2 = Linear(2 * hidden_channels, out_channels)

#     def forward(self, u: torch.Tensor, v: torch.Tensor, edge_index_dict, edge_weight=None):
#         if edge_weight is None: 
#             # weighted + add
#             v_hat = self.conv1((u, v), edge_index_dict['demand', 'contributes_to', 'measurement']).relu()
#             u_hat = self.conv3((v, u), edge_index_dict['measurement', 'rev_contributes_to', 'demand']).relu()
#             u_bar = self.conv2((v_hat, u_hat), edge_index_dict['measurement', 'rev_contributes_to', 'demand']).relu()
#             v_bar = self.conv4((u_hat, v_hat), edge_index_dict['demand', 'contributes_to', 'measurement']).relu()
#         else: 
#             raise NotImplementedError
        
#         if u.shape[1] == u_bar.shape[1] and v.shape[1] == v_bar.shape[1]:
#             u_cat = torch.cat((u, u_bar), dim=1)
#             v_cat = torch.cat((v, v_bar), dim=1)
#             u_out = self.lin_u2(u_cat)
#             v_out = self.lin_v2(v_cat)
#         else: 
#             u_out = self.lin_u1(u_bar)
#             v_out = self.lin_v1(v_bar)
#         return F.relu(u_out), F.relu(v_out)
    
# class Encoder_GNN_2hop_v2(nn.Module): 
#     def __init__(self, hidden_channels: int, out_channels: int, layer_n: int, alpha: float = None, dropout: float = None):
#         super(Encoder_GNN_2hop_v2, self).__init__()
#         self.layer_n = layer_n
#         self.dropout = dropout
#         self.out_channels = out_channels
#         gcn_layers = []
#         for _ in range(self.layer_n):
#             gcn_layers.append(GCNLayer_v2(hidden_channels, out_channels)) 
#         self.layers = nn.ModuleList(gcn_layers)

#     def forward(self, x_dict, edge_index_dict, edge_weight=None):  
#         feature_u = x_dict['demand']
#         feature_v = x_dict['measurement']
#         for layer in self.layers: 
#             # feature_u = F.dropout(feature_u, self.dropout, training=self.training)
#             # feature_v = F.dropout(feature_v, self.dropout, training=self.training)
#             feature_u, feature_v = layer(feature_u, feature_v, edge_index_dict)
#             assert feature_u.shape[1] == self.out_channels
#             assert feature_v.shape[1] == self.out_channels
#         return feature_u, feature_v
        
