import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder_GCMC(nn.Module): 
    def __init__(self, config):
        ''' bilinear decoder '''
        super(Decoder_GCMC, self).__init__()
        pass

    def forward(self): 
        pass

class EdgeDecoder(nn.Module):
     """
     Edge Decoder module to infer the predictions. 

     Args:
     hidden_channels (int): The number of hidden channels.
     out_channels (int): The number of out_channels.
     """

     def __init__(self, hidden_channels=8, out_channels=1):

         super().__init__()

         self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
         self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

     def forward(self, z, edge_label_index, sigmoid=False):
         """
         Forward pass of the EdgeDecoder module.

         Args:
         z_dict (dict): node type as keys and temporal node embeddings 
         for each node as values. 
         edge_label_index (torch.Tensor): see previous section.

         Returns:
         torch.Tensor: Predicted edge labels.
         """
         row, col = edge_label_index

         z = torch.cat([z['demand'][row], z['measurement'][col]], dim=-1) # TODO: debug
         z = self.lin1(z).relu()
         # z = F.leaky_relu(z, negative_slope=0.1)
         z = self.lin2(z)
         z = torch.sigmoid(z) if sigmoid else z
         return z.view(-1)
     
class Decoder_bipartite(nn.Module): 
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index, sigmoid=True):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        z = torch.sigmoid(z) if sigmoid else z # TODO: sigmoid
        return z.view(-1)