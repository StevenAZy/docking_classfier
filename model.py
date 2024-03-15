import torch
import torch.nn.functional as F


from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class PGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(PGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv = GCNConv(1280, hidden_channels)

    def forward(self, node_feat, edge_index, batch):
        # 1. Obtain node embeddings 
        p = self.conv(node_feat, edge_index)
        p = p.relu()
        gmp_p = global_mean_pool(p, batch)
        
        return gmp_p
    

class RGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(RGCN, self).__init__()
        torch.manual_seed(12345)
        self.p_conv = GCNConv(1280, hidden_channels)
        self.r_conv = GCNConv(120, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, p_node_feat, p_edge_index, r_node_feat, r_edge_index, batch):
        # 1. Obtain node embeddings 
        # p = self.p_conv(p_node_feat, p_edge_index)
        # p = p.relu()
        # gmp_p = global_mean_pool(p, batch)
        p = 0

        r = self.r_conv(r_node_feat, r_edge_index)
        r = r.relu()
        gmp_r = global_mean_pool(r, batch)

        p_r = torch.concatenate((p, r), dim=0)
        # 2. Readout layer 
        x = global_mean_pool(p_r, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(p_r, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# model = GCN(hidden_channels=64)