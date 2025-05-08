import torch
import torch.nn as nn

from encoders import Encoder_GNN, Encoder_GNN_u_weighted, Encoder_GNN_v_weighted
from decoders import EdgeDecoder, Decoder_bipartite
from torch_geometric.nn import to_hetero
from torch_geometric.nn.models.autoencoder import GAE

# GCMC
# class GAE_GCMC(nn.Module): 
#     def __init__(self, config):
#         super(GAE_GCMC, self).__init__()
#         self.encoder = Encoder_GCMC()
#         self.decoder = Decoder_GCMC()

#     def forward(self, x): 
#         u_features, v_features = self.encoder(x)
#         adjacent_matrix = self.decoder(u_features, v_features)
#         return adjacent_matrix
    
# others
class GAE_Bipartite(GAE): 
    def __init__(self, encoder: nn.Module, decoder = None):
        super().__init__(
            encoder = encoder,
            decoder = EdgeDecoder() if decoder is None else decoder
        )
    
    def recon_loss(self, z, pos_edge_index, pos_edge_weight, neg_edge_index = None):
        ''' def recon_loss(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Optional[Tensor] = None) -> Tensor: '''
        edge_weight_pred = self.decoder(z, pos_edge_index, sigmoid=False)
        return torch.nn.functional.mse_loss(edge_weight_pred, pos_edge_weight) # TODO: modify loss function
    
class GAE_hetero_link_pred(nn.Module): 
    def __init__(self, hidden_channels, out_channels, aggr='sum', metadata=(['demand', 'measurement'], [('demand', 'contributes_to', 'measurement')])):
        super().__init__()
        self.encoder = Encoder_GNN(hidden_channels, out_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr=aggr)
        self.decoder = EdgeDecoder(hidden_channels=out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_weight):
        z_dict = self.encoder(x_dict, edge_index_dict, edge_weight)
        return self.decoder(z_dict, edge_label_index, sigmoid=True)
    
class Bipartite_link_pred(nn.Module): 
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # self.user_emb = Embedding(num_users, hidden_channels)
        self.user_encoder = Encoder_GNN_u_weighted(hidden_channels, out_channels)
        self.movie_encoder = Encoder_GNN_v_weighted(hidden_channels, out_channels)
        self.decoder = Decoder_bipartite(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_weight=None):
        z_dict = {}
        # x_dict['user'] = self.user_emb(x_dict['user'])
        z_dict['demand'] = self.user_encoder(x_dict, edge_index_dict, edge_weight=edge_weight) # TODO: modify edge weight in u side
        z_dict['measurement'] = self.movie_encoder(
            x_dict['measurement'],
            edge_index_dict[('measurement', 'metapath_0', 'measurement')]
        )
        return self.decoder(z_dict['demand'], z_dict['measurement'], edge_label_index)