import io
import os
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache, partial

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from ogb.graphproppred import Evaluator
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils.graph_utils import collate_graph_batch
from wrapper import MyGraphPropPredDataset


class GraphClassificationEvaluator(Evaluator):
    def __init__(self, name: str):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__), "meta_info.csv"), index_col=0)
        if self.name not in meta_info:
            error_mssg = f"Invalid dataset name {self.name}.\n"
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(meta_info.keys())
            raise ValueError(error_mssg)

        self.num_tasks = int(meta_info[self.name]["num tasks"])
        self.eval_metric = meta_info[self.name]["eval metric"]


@lru_cache(maxsize=None)
def get_dataset(dataset_name: str = "ogbg_mol_breast_cancer", dataset_root: str = "processed_data"):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        pyg_dataset = MyGraphPropPredDataset("ogbg_mol_breast_cancer", root=dataset_root)
    return {
        "num_class": 1,
        "loss_fn": F.binary_cross_entropy_with_logits,
        "metric": "auroc",
        "evaluator_metric": "rocauc",
        "metric_mode": "max",
        "evaluator": GraphClassificationEvaluator("ogbg_mol_breast_cancer"),
        "dataset": pyg_dataset,
        "max_node": 512,
    }


class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
        self,
        dataset_name: str = "ogbg_mol_breast_cancer",
        dataset_root: str = "processed_data",
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.dataset = get_dataset(self.dataset_name, self.dataset_root)
        self.num_workers = int(num_workers)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.multi_hop_max_dist = int(multi_hop_max_dist)
        self.spatial_pos_max = int(spatial_pos_max)
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphData")
        parser.add_argument("--dataset_name", type=str, default="ogbg_mol_breast_cancer")
        parser.add_argument("--dataset_root", type=str, default="processed_data")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--multi_hop_max_dist", type=int, default=5)
        parser.add_argument("--spatial_pos_max", type=int, default=1024)
        return parent_parser

    def setup(self, stage: str | None = None) -> None:
        if self.dataset_train is not None and self.dataset_val is not None and self.dataset_test is not None:
            return

        split_idx = self.dataset["dataset"].get_idx_split()
        print(f"splits: train={len(split_idx['train'])} valid={len(split_idx['valid'])} test={len(split_idx['test'])}")
        self.dataset_train = self.dataset["dataset"][split_idx["train"]]
        self.dataset_val = self.dataset["dataset"][split_idx["valid"]]
        self.dataset_test = self.dataset["dataset"][split_idx["test"]]

    def _build_dataloader(self, split_dataset, shuffle: bool):
        return DataLoader(
            split_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            collate_fn=partial(
                collate_graph_batch,
                max_node=get_dataset(self.dataset_name, self.dataset_root)["max_node"],
                multi_hop_max_dist=self.multi_hop_max_dist,
                spatial_pos_max=self.spatial_pos_max,
            ),
        )

    def train_dataloader(self):
        return self._build_dataloader(self.dataset_train, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.dataset_val, shuffle=False)

    def test_dataloader(self):
        return self._build_dataloader(self.dataset_test, shuffle=False)
