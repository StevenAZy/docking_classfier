"""
data processing
"""
import csv
import copy
import random
import numpy as np

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GNN_DataLoader
from pytorch_lightning import LightningDataModule

from config import PDB_RNA_PATH, GRAPH_RNA_PATH
from data_process.construct_graph import rna_graph


class RNADataset:
    def __init__(self, rna_graphs, rna_labels, rna_ID, names):
        super().__init__()
        self.rna_graphs = rna_graphs
        self.rna_labels = rna_labels
        self.rna_ID = rna_ID
        self.names = names
        self.data = []

        for i in range(len(rna_graphs)):
            data = Data(
                x=rna_graphs[i]["node_feat"],
                edge_index=rna_graphs[i]["edge_index"],
                y=rna_labels[i],
            )
            data.protein_pdbID = names[i]
            data.edge_num = rna_graphs[i]["edge_index"].shape[1]
            self.data.append(data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.rna_graphs)


def train_rnas(pdb_rna_path, graph_rna_path):
    np.random.seed(42)
    rna_graphs = {}
    rna_labels = {}

    with open("data_process/train_data.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for _, row in tqdm(enumerate(reader)):
            if row[1] not in rna_graphs.keys():
                rna_graphs[row[1]] = []
                rna_labels[row[1]] = []
            if float(row[2]) == 1:
                rna_labels[row[1]].append(1)
            elif float(row[2]) == 0:
                rna_labels[row[1]].append(0)
            graph = rna_graph(pdb_rna_path, graph_rna_path, row[1])
            rna_graphs[row[1]].append(graph)

        train_rna_graphs = []
        train_rna_labels = []
        train_names = []
        train_rna_ID = {}
        start_index = 0
        for train_name in rna_graphs.keys():
            train_rna_ID[train_name] = [
                i for i in range(start_index, start_index + len(rna_graphs[train_name]))
            ]
            start_index = start_index + len(rna_graphs[train_name])
            train_names.extend([train_name for _ in range(len(rna_graphs[train_name]))])
            train_rna_graphs.extend(rna_graphs[train_name])
            train_rna_labels.extend(rna_labels[train_name])

        train_set = RNADataset(
            train_rna_graphs, train_rna_labels, train_rna_ID, train_names
        )

    rna_graphs = {}
    rna_labels = {}

    with open("data_process/val_data.csv", "r") as f1:
        reader = csv.reader(f1)
        next(reader)
        for row in tqdm(reader):
            if row[1] not in rna_graphs.keys():
                rna_graphs[row[1]] = []
                rna_labels[row[1]] = []
            if float(row[2]) == 1:
                rna_labels[row[1]].append(1)
            elif float(row[2]) == 0:
                rna_labels[row[1]].append(0)
            graph = rna_graph(pdb_rna_path, graph_rna_path, row[1])
            rna_graphs[row[1]].append(graph)

        valid_rna_ID = {}
        valid_rna_graphs = []
        valid_rna_labels = []
        valid_names = []
        start_index = 0

        for valid_name in rna_graphs.keys():
            valid_rna_ID[valid_name] = [
                i for i in range(start_index, start_index + len(rna_graphs[valid_name]))
            ]
            start_index = start_index + len(rna_graphs[valid_name])
            valid_names.extend([valid_name for _ in range(len(rna_graphs[valid_name]))])
            valid_rna_graphs.extend(rna_graphs[valid_name])
            valid_rna_labels.extend(rna_labels[valid_name])

        val_set = RNADataset(
            valid_rna_graphs, valid_rna_labels, valid_rna_ID, valid_names
        )

    return train_set, val_set


def test_rnas(pdb_rna_path, graph_rna_path):
    rna_graphs = {}
    rna_labels = {}

    with open("data_process/test_data.csv", "r") as f1:
        reader = csv.reader(f1)
        next(reader)
        for row in tqdm(reader):
            if row[0] not in rna_graphs.keys():
                rna_graphs[row[0]] = []
                rna_labels[row[0]] = []
            if float(row[4]) < 1000:
                rna_labels[row[0]].append(1)
            elif float(row[4]) >= 100000:
                rna_labels[row[0]].append(0)
            else:
                continue
            graph = rna_graph(pdb_rna_path, graph_rna_path, row[1])
            rna_graphs[row[0]].append(graph)

    test_rna_ID = {}
    test_rna_graphs = []
    test_rna_labels = []
    test_names = []
    start_index = 0
    for test_name in rna_graphs.keys():
        test_rna_ID[test_name] = [
            i for i in range(start_index, start_index + len(rna_graphs[test_name]))
        ]
        start_index = start_index + len(rna_graphs[test_name])
        test_names.extend([test_name for _ in range(len(rna_graphs[test_name]))])
        test_rna_graphs.extend(rna_graphs[test_name])
        test_rna_labels.extend(rna_labels[test_name])

    test_set = RNADataset(test_rna_graphs, test_rna_labels, test_rna_ID, test_names)
    return test_set


class FewShotBatchSampler:
    def __init__(self, train_rna_datasets, K_shot, K_query, batch_size):
        self.rna_labels = train_rna_datasets.rna_labels
        self.rna_ID = train_rna_datasets.rna_ID
        self.N_way = 2
        self.K_shot = K_shot
        self.batch_size = batch_size
        self.K_query = K_query

        self.classes = [0, 1]
        self.protein_name = list(self.rna_ID.keys())
        self.iterations = 0
        rna_ID = copy.deepcopy(self.rna_ID)
        proteins = list(rna_ID.keys())
        self.ratio_dict = {}
        for selected_protein in proteins:
            positive_num = 0
            negative_num = 0
            for index in rna_ID[selected_protein]:
                if self.rna_labels[index] == 1:
                    positive_num += 1
                else:
                    negative_num += 1
            self.ratio_dict[selected_protein] = positive_num / (
                positive_num + negative_num
            )

        while len(proteins) >= self.batch_size:
            random.seed(42)
            group_weights = [len(rna_ID[protein]) for protein in proteins]
            total_count = sum(group_weights)
            group_weights = [count / total_count for count in group_weights]
            selected_proteins = random.choices(
                proteins, weights=group_weights, k=self.batch_size
            )
            while len(set(selected_proteins)) != self.batch_size:
                selected_proteins = random.choices(
                    proteins, weights=group_weights, k=self.batch_size
                )
            for selected_protein in selected_proteins:
                positive_list = []
                negative_list = []
                for index in rna_ID[selected_protein]:
                    if self.rna_labels[index] == 1:
                        positive_list.append(index)
                    else:
                        negative_list.append(index)
                selected_positive = random.sample(positive_list, self.K_shot)
                selected_negative = random.sample(negative_list, self.K_shot)
                for positive_index in selected_positive:
                    rna_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_negative:
                    rna_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)

                if (
                    len(positive_list) >= self.K_query // 2
                    and len(negative_list) >= self.K_query // 2
                ):
                    selected_query_positive = random.sample(
                        positive_list, self.K_query // 2
                    )
                    selected_query_negative = random.sample(
                        negative_list, self.K_query // 2
                    )
                else:
                    length = min(len(positive_list), len(negative_list))
                    selected_query_positive = random.sample(positive_list, length)
                    selected_query_negative = random.sample(negative_list, length)
                for positive_index in selected_query_positive:
                    rna_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_query_negative:
                    rna_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)
                if (
                    len(positive_list) <= self.K_shot
                    or len(negative_list) <= self.K_shot
                ):
                    rna_ID.pop(selected_protein)
                    proteins.remove(selected_protein)
            self.iterations += 1

    def weighted_sample(self, rna_ID):
        select_proteins = []
        for _ in range(self.batch_size):
            weight = [len(value) for value in rna_ID.values()]
            select_protein = random.choices(list(rna_ID.keys()), weights=weight)
            rna_ID.pop(select_protein[0])
            select_proteins.append(select_protein[0])
        return select_proteins

    def __iter__(self):
        rna_ID = copy.deepcopy(self.rna_ID)
        proteins = list(rna_ID.keys())
        for _ in range(self.iterations):
            index_batch = []
            random.seed(42)
            group_weights = [len(rna_ID[protein]) for protein in proteins]
            total_count = sum(group_weights)
            group_weights = [count / total_count for count in group_weights]
            selected_proteins = random.choices(
                proteins, weights=group_weights, k=self.batch_size
            )
            while len(set(selected_proteins)) != self.batch_size:
                selected_proteins = random.choices(
                    proteins, weights=group_weights, k=self.batch_size
                )
            for selected_protein in selected_proteins:
                positive_list = []
                negative_list = []
                for index in rna_ID[selected_protein]:
                    if self.rna_labels[index] == 1:
                        positive_list.append(index)
                    else:
                        negative_list.append(index)
                selected_positive = random.sample(positive_list, self.K_shot)
                selected_negative = random.sample(negative_list, self.K_shot)
                for positive_index in selected_positive:
                    rna_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_negative:
                    rna_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)

                if (
                    len(positive_list) >= self.K_query // 2
                    and len(negative_list) >= self.K_query // 2
                ):
                    selected_query_positive = random.sample(
                        positive_list, self.K_query // 2
                    )
                    selected_query_negative = random.sample(
                        negative_list, self.K_query // 2
                    )
                else:
                    length = min(len(positive_list), len(negative_list))
                    selected_query_positive = random.sample(positive_list, length)
                    selected_query_negative = random.sample(negative_list, length)
                for positive_index in selected_query_positive:
                    rna_ID[selected_protein].remove(positive_index)
                    positive_list.remove(positive_index)
                for negative_index in selected_query_negative:
                    rna_ID[selected_protein].remove(negative_index)
                    negative_list.remove(negative_index)
                if (
                    len(positive_list) <= self.K_shot
                    or len(negative_list) <= self.K_shot
                ):
                    rna_ID.pop(selected_protein)
                    proteins.remove(selected_protein)
                index_batch.extend(selected_positive)
                index_batch.extend(selected_negative)
                selected_query = selected_query_positive + selected_query_negative
                random.shuffle(selected_query)
                index_batch.extend(selected_query)
            yield index_batch

    def __len__(self):
        return self.iterations


class TrainBatchSampler:
    def __init__(self, train_rna_datasets, train_shot):
        self.rna_labels = train_rna_datasets.rna_labels
        self.rna_ID = train_rna_datasets.rna_ID

        self.train_shot = train_shot
        self.iterations = 0
        for protein_name in list(self.rna_ID.keys()):
            if len(self.rna_ID[protein_name]) % self.train_shot == 0:
                self.iterations += len(self.rna_ID[protein_name]) // self.train_shot
            else:
                self.iterations += len(self.rna_ID[protein_name]) // self.train_shot + 1

    def __iter__(self):
        for protein_name in list(self.rna_ID.keys()):
            if len(self.rna_ID[protein_name]) % self.train_shot == 0:
                n = len(self.rna_ID[protein_name]) // self.train_shot
            else:
                n = len(self.rna_ID[protein_name]) // self.train_shot + 1
            for i in range(n):
                yield self.rna_ID[protein_name][
                    i * self.train_shot : (i + 1) * self.train_shot
                ]

    def __len__(self):
        return self.iterations


class Test_valBatchSampler:
    def __init__(self, Test_val_rna_datasets, val_shot):
        self.rna_labels = Test_val_rna_datasets.rna_labels
        self.rna_ID = Test_val_rna_datasets.rna_ID

        self.val_shot = val_shot
        self.iterations = 0
        for protein_name in list(self.rna_ID.keys()):
            if len(self.rna_ID[protein_name]) % self.val_shot == 0:
                self.iterations += len(self.rna_ID[protein_name]) // self.val_shot
            else:
                self.iterations += len(self.rna_ID[protein_name]) // self.val_shot + 1

    def __iter__(self):
        for protein_name in list(self.rna_ID.keys()):
            if len(self.rna_ID[protein_name]) % self.val_shot == 0:
                n = len(self.rna_ID[protein_name]) // self.val_shot
            else:
                n = len(self.rna_ID[protein_name]) // self.val_shot + 1
            for i in range(n):
                yield self.rna_ID[protein_name][
                    i * self.val_shot : (i + 1) * self.val_shot
                ]

    def __len__(self):
        return self.iterations


def collate(items, K_shot, K_query):
    node_feats, edge_indexs, edge_attrs, ys, protein_pdbIDs, edge_nums = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    batch_size = len(items) // (2 * K_shot + K_query)
    for item in items:
        node_feats.append(item.x)
        edge_indexs.append(item.edge_index)
        edge_attrs.append(item.edge_attr)
        ys.append(item.y)
        protein_pdbIDs.append(item.protein_pdbID)
        edge_nums.append(item.edge_num)
    support_datas = []
    proteins = []
    for m in range(batch_size):
        support_data = []
        for n_p in range(2 * K_shot):
            ID = n_p + m * (2 * K_shot + K_query)
            data = Data(
                x=node_feats[ID],
                edge_index=edge_indexs[ID],
                edge_attr=edge_attrs[ID],
                y=ys[ID],
            )
            support_data.append(data)
        support_datas.append(support_data)
        proteins.append(protein_pdbIDs[m * (2 * K_shot + K_query)])
    query_datas = []
    for l in range(batch_size):
        query_data = []
        for ll in range(K_query):
            ID = ll + l * (2 * K_shot + K_query) + 2 * K_shot
            data = Data(
                x=node_feats[ID],
                edge_index=edge_indexs[ID],
                edge_attr=edge_attrs[ID],
                y=ys[ID],
            )
            query_data.append(data)
        query_datas.append(query_data)
    return support_datas, query_datas, proteins


class GCNRNADataModule(LightningDataModule):
    def __init__(
        self, num_workers, batch_size, k_shot, k_query, val_shot, test, explanation
    ):
        super(GCNRNADataModule, self).__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_shot = val_shot
        self.k_shot = k_shot
        self.k_query = k_query
        self.test = test
        self.explanation = explanation

        if self.explanation:
            self.explan_batch = Test_valBatchSampler(self.explan_rna, self.val_shot)
        elif self.test:
            self.test_rna = test_rnas()
            self.test_batch = Test_valBatchSampler(self.test_rna, self.val_shot)
        else:
            self.train_rna, self.val_rna = train_rnas(PDB_RNA_PATH, GRAPH_RNA_PATH)
            self.val_batch = Test_valBatchSampler(self.val_rna, self.val_shot)
            self.train_batch = TrainBatchSampler(self.train_rna, self.k_shot)
            # self.train_batch = FewShotBatchSampler(
            #     self.train_rna,
            #     batch_size=self.batch_size,
            #     K_shot=self.k_shot,
            #     K_query=self.k_query,
            # )
            self.iterations = self.train_batch.iterations

    def train_dataloader(self):
        loader = GNN_DataLoader(
            dataset=self.train_rna,
            batch_sampler=self.train_batch,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self):
        loader = GNN_DataLoader(
            dataset=self.val_rna,
            batch_sampler=self.val_batch,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader

    def test_dataloader(self):
        loader = GNN_DataLoader(
            dataset=self.test_rna,
            batch_sampler=self.test_batch,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return loader
