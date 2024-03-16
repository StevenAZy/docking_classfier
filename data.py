"""
pairdata processing
"""
import csv

from torch_geometric.data import Batch, Dataset

from config import *
from data_process.construct_graph import rna_graph, protein_graph


class P_RPairDataset(Dataset):
    def __init__(self, data_path, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(transform, pre_transform, pre_filter)
        self.data_path = data_path

        self.all_data = []
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            for _, row in enumerate(reader):
                self.all_data.append(f'{row[0]}-{row[1]}-{row[2]}')

    def len(self):
        return len(self.all_data)
    
    def get(self, idx):
        idx_data = self.all_data[idx].split('-')

        p_graph = protein_graph(PDB_PROTEIN_PATH, GRAPH_PROTEIN_PATH, idx_data[0])
        r_graph = rna_graph(PDB_RNA_PATH, GRAPH_RNA_PATH, idx_data[1])
        label = idx_data[2]

        # p_r_pairdata = Data(p_node_feat = p_graph.node_feat, p_edge_index = p_graph.edge_index,
        #                     r_node_feat = r_graph.node_feat, r_edge_index = r_graph.edge_index,
        #                     label = label)

        return p_graph, r_graph, label
    

def collate(data_list):
    batchP = Batch.from_data_list([data[0] for data in data_list])
    batchR = Batch.from_data_list([data[1] for data in data_list])
    batchL = Batch.from_data_list([data[2] for data in data_list])

    return batchP, batchR, batchL
