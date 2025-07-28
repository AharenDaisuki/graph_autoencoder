import torch
import torch.nn as nn
     
class Decoder_bipartite(nn.Module): 
    """
    Base decoder for graph autoencoders.

    Args: 
        hidden_channels (int): number of hidden channels
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        z = torch.sigmoid(z)
        return z.view(-1)