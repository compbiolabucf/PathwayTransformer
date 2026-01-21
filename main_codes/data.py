# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Copied and modified from Graphormer (https://github.com/microsoft/Graphormer)


from collator import collator
from wrapper import MyGraphPropPredDataset
import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import ogb
import ogb.lsc
from ogb.graphproppred import Evaluator
from functools import partial


dataset = None


### Evaluator for graph classification
class eval(Evaluator):
    def __init__(self, name):
        self.name = name

        meta_info = pd.read_csv(os.path.join(os.path.dirname(__file__),'meta_info.csv'), index_col = 0)
        if not self.name in meta_info:
            print(self.name)
            error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
            error_mssg += 'Available datasets are as follows:\n'
            error_mssg += '\n'.join(meta_info.keys())
            raise ValueError(error_mssg)

        self.num_tasks = int(meta_info[self.name]['num tasks'])
        self.eval_metric = meta_info[self.name]['eval metric']



def get_dataset(dataset_name='abaaba'):
    global dataset
    if dataset is not None:
        return dataset

    dataset = {
        'num_class': 1,
        'loss_fn': F.binary_cross_entropy_with_logits,
        'metric': 'rocauc',
        'metric_mode': 'max',
        'evaluator': eval('ogbg_mol_breast_cancer'),
        'dataset': MyGraphPropPredDataset('ogbg_mol_breast_cancer', root='dataset'),
        'max_node': 512,
    }


    print(f' > {dataset_name} loaded!')
    print(dataset)
    print(f' > dataset info ends')
    return dataset


class GraphDataModule(LightningDataModule):
    name = "OGB-GRAPH"

    def __init__(
        self,
        dataset_name: str = 'ogbg_mol_breast_cancer',
        num_workers: int = 0,
        batch_size: int = 256,
        seed: int = 42,
        multi_hop_max_dist: int = 5,
        spatial_pos_max: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_name = dataset_name
        self.dataset = get_dataset(self.dataset_name)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def setup(self, stage: str = None):
        if self.dataset_name == 'ZINC':
            self.dataset_train = self.dataset['train_dataset']
            self.dataset_val = self.dataset['valid_dataset']
            self.dataset_test = self.dataset['test_dataset']
        else:
            split_idx = self.dataset['dataset'].get_idx_split()
            print('lengths: ',len(split_idx["train"]), len(split_idx["valid"]), len(split_idx["test"]))
            self.dataset_train = self.dataset['dataset'][split_idx["train"]]
            self.dataset_val = self.dataset['dataset'][split_idx["valid"]]
            self.dataset_test = self.dataset['dataset'][split_idx["test"]]

    def train_dataloader(self):
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(val_dataloader)', len(loader))
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=partial(collator, max_node=get_dataset(self.dataset_name)[
                               'max_node'], multi_hop_max_dist=self.multi_hop_max_dist, spatial_pos_max=self.spatial_pos_max),
        )
        print('len(test_dataloader)', len(loader))
        return loader
