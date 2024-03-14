# from graphein.protein.config import ProteinGraphConfig
# from graphein.protein.graphs import construct_graph
# from graphein.protein.visualisation import plotly_protein_structure_graph
# from graphein.rna.config import RNAGraphConfig
# from graphein.rna.graphs import construct_rna_graph_3d
# from graphein.rna.visualisation import plotly_rna_structure_graph
# from graphein.rna.edges import (
#     add_all_dotbracket_edges,
#     add_pseudoknots,
#     add_phosphodiester_bonds,
#     add_base_pairing_interactions
# )

# p_config = ProteinGraphConfig()
# g_p = construct_graph(config=p_config, path='data/1ASY_r_u.pdb')

# p_1 = plotly_protein_structure_graph(
#     g_1,
#     colour_edges_by="kind",
#     colour_nodes_by="degree",
#     label_node_ids=False,
#     plot_title="Peptide backbone graph. Nodes coloured by degree.",
#     node_size_multiplier=1
#     )
# p_1.show()

# r_config = RNAGraphConfig()
# g_r = construct_rna_graph_3d(path='data/pdb_rna/1AQO.pdb')

# p_1 = plotly_rna_structure_graph(g_r)
# p_1.show()


# if __name__ == '__main__':
#     from RNABERT.utils.bert import Load_RNABert_Model
#     model = Load_RNABert_Model('/home/steven/code/docking_classfier/RNABERT/RNABERT.pth')
#     emb = model.predict_embedding('AUGC')
#     print(emb.size())


import torch
import pickle
import torch.nn.functional as F

from base_model import GCN
from torch.nn import Linear
from torch_geometric.data import Data, Batch

class forwardmodel(torch.nn.Module):
    def __init__(
        self,
        protein_dim1,
        protein_dim2,
        protein_dim3,
        rna_dim1,
        rna_dim2,
        hidden_dim,
        hidden_dim2,
    ):
        super(forwardmodel, self).__init__()
        self.protein_GCN = GCN(protein_dim1, protein_dim2, protein_dim3)
        self.rna_GCN = GCN(rna_dim1, rna_dim2, hidden_dim)

        self.cat_MLP = MLP(hidden_dim, hidden_dim2, 1)
        self.fc1 = Linear(protein_dim3, protein_dim3)
        self.fc2 = Linear(protein_dim3, hidden_dim)

    def forward(self, batch_data):
        protein_node_feat = batch_data.p_node_feat
        protein_edge_index = batch_data.p_node_feat
        rna_node_feat = batch_data.p_node_feat
        rna_edge_index = batch_data.p_node_feat
        protein_emb = self.protein_GCN(protein_node_feat, protein_edge_index)
        rna_emb = self.rna_GCN(rna_node_feat, rna_edge_index)

        input = torch.cat((protein_emb, rna_emb))

        pred = self.cat_MLP(input)

        return pred

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim)
        self.lin3 = Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = F.relu(self.lin1(input))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)

        return x
    

if __name__ == '__main__':
    protein1_graph_path = 'data_process/graph_protein/3amt.pkl'
    rna1_graph_path = 'data_process/graph_rna/3amt.pkl'

    protein2_graph_path = 'data_process/graph_protein/3j7a.pkl'
    rna2_graph_path = 'data_process/graph_rna/3j7a.pkl'

    with open(protein1_graph_path, 'rb') as f:
        protein1_graph = pickle.load(f)

    with open(rna1_graph_path, 'rb') as f:
        rna1_graph = pickle.load(f)

    with open(protein2_graph_path, 'rb') as f:
        protein2_graph = pickle.load(f)

    with open(rna2_graph_path, 'rb') as f:
        rna2_graph = pickle.load(f)


    data_1 = Data(p_node_feat = protein1_graph.node_feat,
                  p_edge_index = protein1_graph.edge_index,
                  r_node_feat = rna1_graph.node_feat,
                  r_edge_index = rna1_graph.edge_index,
                  label = 1)
    data_2 = Data(p_node_feat = protein2_graph.node_feat,
                  p_edge_index = protein2_graph.edge_index,
                  r_node_feat = rna2_graph.node_feat,
                  r_edge_index = rna2_graph.edge_index,
                  label = 1)
    
    batch_data = Batch.from_data_list([data_1, data_2])

    # p1_node_feat, p1_edge_index = protein1_graph.node_feat, protein1_graph.edge_index
    # p2_node_feat, p2_edge_index = protein2_graph.node_feat, protein2_graph.edge_index
    # r1_node_feat, r1_edge_index = rna1_graph.node_feat, rna1_graph.edge_index
    # r2_node_feat, r2_edge_index = rna2_graph.node_feat, rna2_graph.edge_index

    # p_node_feat = torch.stack((p1_node_feat, p2_node_feat))
    # r_node_feat = torch.cat((r1_node_feat, r2_node_feat))
    # p_edge_index = torch.cat((p1_edge_index, p2_edge_index))
    # r_edge_index = torch.cat((r1_edge_index, r2_edge_index))

    model = forwardmodel(protein_dim1=1280, protein_dim2=512, protein_dim3=256, rna_dim1=120, rna_dim2=512, hidden_dim=256, hidden_dim2=64)
    model = model.to('cuda')

    out = model(batch_data)

    print(out)

