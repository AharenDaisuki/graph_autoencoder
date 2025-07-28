import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import Encoder_GNN_u_weighted, Encoder_GNN_v_weighted
from decoders import Decoder_bipartite
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models.autoencoder import GAE
from torch_geometric.nn.conv import GraphConv, SAGEConv, GATConv, GCNConv
    
class Bipartite_link_pred(nn.Module): 
    """
    Graph autoencoder for bipartite graph link regression.

    Args: 
        hidden_channels (int): number of hidden channels
        out_channels (int): number of output channels
    """
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.user_emb = nn.Embedding(2000, hidden_channels) # TODO: hard code
        self.user_encoder = Encoder_GNN_u_weighted(hidden_channels, out_channels, gconv=GraphConv)
        self.movie_encoder = Encoder_GNN_v_weighted(hidden_channels, out_channels, gconv=SAGEConv)
        self.decoder = Decoder_bipartite(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_weight = None):
        """
        
        """
        z_dict = {}
        x_dict['demand'] = self.user_emb(x_dict['demand'])
        # if edge_weight is not None: 
        #     edge_weight = F.sigmoid(edge_weight)

        z_dict['demand'] = self.user_encoder(x_dict, edge_index_dict, edge_weight = edge_weight)
        z_dict['measurement'] = self.movie_encoder(
            x_dict['measurement'],
            edge_index_dict[('measurement', 'metapath_0', 'measurement')]
        )
        return self.decoder(z_dict['demand'], z_dict['measurement'], edge_label_index)
    
class Bipartite_LinkQuantileRegression_GAE(nn.Module): 
    def __init__(self, num_node_u: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.embedding_u = nn.Embedding(num_node_u, hidden_channels)
        self.encoder_u = Encoder_GNN_u_weighted(hidden_channels, 3 * out_channels, gconv=GraphConv)
        self.encoder_v = Encoder_GNN_v_weighted(hidden_channels, 3 * out_channels, gconv=SAGEConv)
        self.decoder = Decoder_bipartite(out_channels)
        # self.decoder_2 = Decoder_bipartite(out_channels)
        # self.decoder_3 = Decoder_bipartite(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_weight=None): 
        z_1, z_2, z_3 = {}, {}, {}
        out_dim = self.out_channels
        # id embedding [0-1999]
        x_dict['demand'] = self.embedding_u(x_dict['demand'])
        # encoder
        z_u = self.encoder_u(x_dict, edge_index_dict)
        z_v = self.encoder_v(x_dict['measurement'], edge_index_dict[('measurement', 'metapath_0', 'measurement')])
        z_1['demand'], z_2['demand'], z_3['demand'] = z_u[:,:out_dim], z_u[:,out_dim:2*out_dim], z_u[:,2*out_dim:]
        z_1['measurement'], z_2['measurement'], z_3['measurement'] = z_v[:,:out_dim], z_v[:,out_dim:2*out_dim], z_v[:,2*out_dim:]
        # decoder
        a_1 = self.decoder(z_1['demand'], z_1['measurement'], edge_label_index)
        a_2 = self.decoder(z_2['demand'], z_2['measurement'], edge_label_index)
        a_3 = self.decoder(z_3['demand'], z_3['measurement'], edge_label_index)
        return a_1, a_2, a_3
    
# class Bipartite_link_pred_2hop(nn.Module): 
#     def __init__(self, hidden_channels: int, out_channels: int, layer_n: int, alpha: float = None, dropout: float = None):
#         super(Bipartite_link_pred_2hop, self).__init__()
#         self.encoder = Encoder_GNN_2hop_v2(hidden_channels, out_channels, layer_n, alpha=alpha, dropout=dropout)
#         self.decoder = Decoder_bipartite(out_channels)

#     def forward(self, x_dict, edge_index_dict, edge_label_index, edge_weight = None): 
#         z_dict = {}
#         z_dict['demand'], z_dict['measurement'] = self.encoder(x_dict, edge_index_dict, edge_weight)
#         return self.decoder(z_dict['demand'], z_dict['measurement'], edge_label_index)