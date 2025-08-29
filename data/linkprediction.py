import logging
import torch
import numpy as np
import random

from collections import defaultdict
from torch.utils.data import Dataset
from typing import List

logger = logging.getLogger(__name__)

class LinkPrediction(Dataset):
    def __init__(
        self,
        root: str,
        closure: float,
        split: str,
        group_size: int,
        sampling: str = "weighted",
    ):
        self.root = root
        self.group_size = group_size
        self.num_negatives = group_size - 1
        self.split = split
        self.sampling = sampling

        self.idx_to_node = {}
        with open(f"./{root}/noun_closure.tsv.vocab", "r") as f:
            for line in f:
                idx, node = line.strip().split("\t")
                self.idx_to_node[int(idx)] = node
        self.node_to_idx = {n : i for i, n in self.idx_to_node.items()}
        self.all_indices = set(self.idx_to_node.keys())
        logger.info(f"{len(self.node_to_idx)} nodes")

        pos_edge_path = {
            "train" : f"./{root}/noun_closure.tsv.train_{int(100 * closure)}percent",
            "val" : f"./{root}/noun_closure.tsv.valid",
            "test" : f"./{root}/noun_closure.tsv.test",
        }

        neg_edge_path = {
            "train" : f"./{root}/noun_closure.tsv.full_neg",
            "val" : f"./{root}/noun_closure.tsv.valid_neg",
            "test" : f"./{root}/noun_closure.tsv.test_neg",
        }

        self.positive_edges = []
        self.counts = defaultdict(int)
        with open(pos_edge_path[split], "r") as f:
            for line in f:
                target, source = line.strip().split("\t")
                self.positive_edges.append([int(source), int(target)])
                self.counts[int(target)] += 1
        logger.info(f"Total {len(self.positive_edges)} positive edges")

        freq_array = np.array([self.counts[i] for i in range(len(self.idx_to_node))], dtype=np.float64)
        freq_array_pwr = np.power(freq_array, 0.75)

        self.node_probabilities = freq_array_pwr / freq_array_pwr.sum()
        self.node_probabilities_cumsum = np.cumsum(self.node_probabilities)

        if split in ("val", "test"):
            self.negative_edges = []
            with open(neg_edge_path[split], "r") as f:
                for line in f:
                    target, source = line.strip().split("\t")
                    self.negative_edges.append([int(source), int(target)])

            logger.info(f"Total {len(self.negative_edges)} negative edges")

    def __len__(self):
        return len(self.positive_edges)
    
    def negative_sampling(self, idx: int, num_negatives: int) -> List[int]:
        if self.sampling == "weighted":
            uniform_0_1_numbers = np.random.rand(num_negatives)
            negs = list(np.searchsorted(self.node_probabilities_cumsum, uniform_0_1_numbers))
        else:
            negs = random.sample(self.all_indices - set({idx}), num_negatives)
        return negs

    def __getitem__(self, idx):
        src, tgt = self.positive_edges[idx]
        if self.split == "train":
            neg_src = self.negative_sampling(src, self.num_negatives // 2)
            neg_tgt = self.negative_sampling(tgt, self.num_negatives // 2)
            neg_edges = [[src, n] for n in neg_src] + [[n, tgt] for n in neg_tgt]
        else:
            neg_edges = self.negative_edges[idx * 10 : (idx + 1) * 10]
        return [[src, tgt]] + neg_edges
    
    def collate_fn(self, batch):
        edges = torch.tensor(batch, dtype=torch.long)
        return edges