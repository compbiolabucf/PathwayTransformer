from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from ogb.graphproppred import PygGraphPropPredDataset

from utils.graph_utils import build_edge_input, floyd_warshall


def normalize_meta_info(meta_info):
    if isinstance(meta_info, pd.Series):
        meta_info = meta_info.copy()
        for key in ("additional node files", "additional edge files"):
            if key in meta_info and pd.isna(meta_info[key]):
                meta_info[key] = "None"
        if "data type" in meta_info and pd.isna(meta_info["data type"]):
            meta_info["data type"] = ""
        return meta_info

    normalized = dict(meta_info)
    for key in ("additional node files", "additional edge files"):
        if key in normalized and pd.isna(normalized[key]):
            normalized[key] = "None"
    if "data type" in normalized and pd.isna(normalized["data type"]):
        normalized["data type"] = ""
    return normalized


def convert_to_single_embedding(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    feature_count = x.size(1) if x.dim() > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_count * offset, offset, dtype=torch.long, device=x.device)
    return x + feature_offset


def preprocess_graph_item(item):
    edge_attr, edge_index, node_features = item.edge_attr, item.edge_index, item.x
    num_nodes = node_features.size(0)
    node_features = convert_to_single_embedding(node_features)

    adjacency = torch.zeros([num_nodes, num_nodes], dtype=torch.bool)
    adjacency[edge_index[0, :], edge_index[1, :]] = True

    if edge_attr.dim() == 1:
        edge_attr = edge_attr[:, None]
    attention_edge_type = torch.zeros([num_nodes, num_nodes, edge_attr.size(-1)], dtype=torch.long)
    attention_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_embedding(edge_attr) + 1

    shortest_path_result, path_matrix = floyd_warshall(adjacency.to(dtype=torch.int64).cpu().numpy())
    max_distance = int(np.amax(shortest_path_result))
    edge_input = build_edge_input(max_distance, path_matrix, attention_edge_type.cpu().numpy())
    spatial_positions = torch.from_numpy(shortest_path_result).long()
    attention_bias = torch.zeros([num_nodes + 1, num_nodes + 1], dtype=torch.float)

    item.x = node_features
    item.adj = adjacency
    item.attn_bias = attention_bias
    item.attn_edge_type = attention_edge_type
    item.spatial_pos = spatial_positions
    item.in_degree = adjacency.long().sum(dim=1).view(-1)
    item.out_degree = adjacency.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None, meta_dict=None):
        self.name = name

        if meta_dict is None:
            self.dir_name = "_".join(name.split("-"))
            if osp.exists(osp.join(root, self.dir_name + "_pyg")):
                self.dir_name = self.dir_name + "_pyg"

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), "meta_info.csv"), index_col=0)
            if self.name not in master:
                error_mssg = f"Invalid dataset name {self.name}.\n"
                error_mssg += "Available datasets are as follows:\n"
                error_mssg += "\n".join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = normalize_meta_info(master[self.name])
        else:
            self.dir_name = meta_dict["dir_path"]
            self.original_root = ""
            self.root = meta_dict["dir_path"]
            self.meta_info = normalize_meta_info(meta_dict)

        self.download_name = self.meta_info["download_name"]
        self.num_tasks = int(self.meta_info["num tasks"])
        self.eval_metric = self.meta_info["eval metric"]
        self.task_type = self.meta_info["task type"]
        self.__num_classes__ = int(self.meta_info["num classes"])
        self.binary = self.meta_info["binary"] == "True"

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_graph_item(item)
        return self.index_select(idx)
