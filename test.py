import torch

from torch_geometric.loader import DataLoader

from data import P_RPairDataset
from model import GCN


dataset = P_RPairDataset(data_path='./data_process/train_data.csv')
dataloader  = DataLoader(dataset=dataset, batch_size=3)

model = GCN(hidden_channels=64).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for data in dataloader:
    out = model(data.p_node_feat.cuda(), data.p_edge_index.cuda(), data.r_node_feat.cuda(), data.r_edge_index.cuda(), data.batch)