import torch
import torch.nn.functional as F


from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class MolecularGCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(MolecularGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv = GCNConv(input_channels, hidden_channels)
        
    def forward(self, node_feat, edge_index, batch):
        x = self.conv(node_feat, edge_index)
        x = x.relu()
        gmp_x = global_mean_pool(x, batch)
        
        return gmp_x



class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.p_conv = MolecularGCN(1280, 64)
        self.r_conv = MolecularGCN(120, 64)
        self.lin = Linear(128, 2)

    def forward(self, data):
        p_conv = self.p_conv(data[0].node_feat.cuda(), data[0].edge_index.cuda(), data[0].batch.cuda())
        r_conv = self.r_conv(data[1].node_feat.cuda(), data[1].edge_index.cuda(), data[1].batch.cuda())

        p_r = torch.concatenate((p_conv, r_conv), dim=1)
        p_r = F.dropout(p_r, p=0.5, training=self.training)
        p_r = self.lin(p_r)

        return p_r
