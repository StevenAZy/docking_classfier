"""
construct graph for protein and rna
"""
import os
import esm
import torch
import pickle
import networkx as nx

from functools import partial
from torch_geometric.data import Data
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.rna.graphs import construct_rna_graph_3d
from graphein.protein.edges.distance import add_distance_threshold

from data_process.RNABERT.utils.bert import Load_RNABert_Model

protein_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
protein_model.eval()

rna_model = Load_RNABert_Model("data_process/RNABERT/RNABERT.pth")
rna_model.eval()

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

def rna_graph_node(seq):
    if len(seq) > 440:
        seq_feat = []
        for i in range(len(seq) // 440):
            data = seq[i * 440 : (i + 1) * 440]
            seq_feat.append(rna_model.predict_embedding(data))
        data = seq[(i + 1) * 440 :]
        seq_feat.append(rna_model.predict_embedding(data))
        seq_feat = torch.cat(seq_feat, dim=0)
    else:
        seq_feat = rna_model.predict_embedding(seq)
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
    return graph
    # return edge_index, node_feat.detach()


def rna_graph(pdb_rna_path, graph_rna_path, id):
    file = f"{graph_rna_path}/{id}.pkl"

    if os.path.exists(file):
        with open(file, "rb") as f:
            graph = pickle.load(f)
        return graph
        # return graph.edge_index, graph.node_feat

    g = construct_rna_graph_3d(path=f"{pdb_rna_path}/{id}.pdb")
    A = nx.to_numpy_array(g, nonedge=0, weight="distance")
    edge_index = adj2table(A)

    seq = ""
    for key in g.graph.keys():
        if key[:9] == "sequence_":
            seq += g.graph[key]
    if len(seq) != g.number_of_nodes():
        raise RuntimeError("number of nodes mismatch")
    node_feat = rna_graph_node(seq)
    graph = Data(node_feat=node_feat.detach(), edge_index=edge_index.detach())

    with open(f"{graph_rna_path}/{id}.pkl", "wb") as f:
        pickle.dump(graph, f)
    return graph
    # return edge_index, node_feat.detach()

if __name__ == "__main__":
    # convert protein pdb into graph
    pdb_protein_path = "pdb_protein"
    graph_protein_path = "graph_protein"

    ids = [name.split('.')[0] for name in os.listdir(pdb_protein_path)]

    for id in ids:
        try:
            protein_graph(pdb_protein_path, graph_protein_path, id)
        except:
            print(id)
            continue
    
    # convert rna pdb into graph
    pdb_rna_path = "pdb_rna"
    graph_rna_path = "graph_rna"
    ids = [name.split(".")[0] for name in os.listdir(pdb_rna_path)]

    for id in ids:
        rna_graph(pdb_rna_path, graph_rna_path, id)
