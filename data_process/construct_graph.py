"""
construct graph for protein and rna
"""
import os
import esm
import torch
import pickle
import numpy as np
import networkx as nx

from rdkit import Chem
from functools import partial
from torch_geometric.data import Data
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_distance_threshold

from features import atom_to_feature_vector,bond_to_feature_vector

protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
protein_model.eval()

new_edge_funcs = {
    "edge_construction_functions": [
        partial(add_distance_threshold, long_interaction_threshold=0, threshold=8)
    ]
}
config = ProteinGraphConfig(**new_edge_funcs)


def pretrain_protein(data):
    _, _, batch_tokens = batch_converter(data)
    results = protein_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    feat = token_representations.squeeze(0)[1 : len(data[0][1]) + 1]
    return feat


def adj2table(adj):
    edge_index = [[], []]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if int(adj[i][j]) != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(edge_index, dtype=torch.long)


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
                     torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def protein_graph_node(id, seq):
    if len(seq) > 1022:
        seq_feat = []
        for i in range(len(seq) // 1022):
            data = [(id, seq[i * 1022 : (i + 1) * 1022])]
            seq_feat.append(pretrain_protein(data))
        data = [(id, seq[(i + 1) * 1022 :])]
        seq_feat.append(pretrain_protein(data))
        seq_feat = torch.cat(seq_feat, dim=0)
    else:
        data = [(id, seq)]
        seq_feat = pretrain_protein(data)
    return seq_feat


def protein_graph(pdb_protein_path, graph_protein_path, id):
    file = f"{graph_protein_path}/{id}.pkl"

    if os.path.exists(file):
        with open(file, "rb") as f:
            graph = pickle.load(f)
        return graph
        # return graph.edge_index, graph.node_feat

    g = construct_graph(config=config, path=f"{pdb_protein_path}/{id}.pdb")
    A = nx.to_numpy_array(g, nonedge=0, weight="distance")
    edge_index = adj2table(A)

    seq = ""
    for key in g.graph.keys():
        if key[:9] == "sequence_":
            seq += g.graph[key]
    if len(seq) != g.number_of_nodes():
        raise RuntimeError("number of nodes mismatch")
    node_feat = protein_graph_node(id, seq)
    graph = Data(node_feat=node_feat.detach(), edge_index=edge_index.detach())

    with open(f"{graph_protein_path}/{id}.pkl", "wb") as f:
        pickle.dump(graph, f)
    return edge_index, node_feat.detach()


def rna_graph(pdb_rna_path, graph_rna_path, id):

    file = f"{graph_rna_path}/{id}.pkl"
    if os.path.exists(file):
        with open(file, "rb") as f:
            try:
                graph = pickle.load(f)
            except:
                print(file)
                exit()
        return Data(node_feat=graph['node_feat'], edge_index=graph['edge_index'])
        # return graph
    
    mol = Chem.MolFromPDBFile(f'{pdb_rna_path}/{id}.pdb')

    if mol is None:
        return

    # atoms
    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)


    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    node_feat = convert_to_single_emb(torch.tensor(x))
    graph = dict()
    graph['edge_index'] = torch.tensor(edge_index)
    graph['edge_feat'] = torch.tensor(edge_attr)
    graph['node_feat'] = node_feat
    graph['num_nodes'] = len(node_feat)
    with open(f"{graph_rna_path}/{id}.pkl", "wb") as f:
        pickle.dump(graph, f)
    return graph

# if __name__ == "__main__":
#     pdb_protein_path = "pdb_protein"
#     graph_protein_path = "graph_protein"

#     pdb_rna_path = "pdb_rna"
#     graph_rna_path = "graph_rna"

    # ids = [name.split('.')[0] for name in os.listdir(pdb_protein_path)]

    # for id in ids:
    #     try:
    #         protein_graph(pdb_protein_path, graph_protein_path, id)
    #     except:
    #         print(id)
    #         continue

    # ids = [name.split(".")[0] for name in os.listdir(pdb_rna_path)]

    # for id in ids:
    #     rna_graph(pdb_rna_path, graph_rna_path, id)
