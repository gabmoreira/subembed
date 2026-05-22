import logging
import torch
import random
import networkx as nx

from networkx import DiGraph
from torch.utils.data import Dataset
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class Reconstruction(Dataset):
    def __init__(
        self,
        graph: DiGraph,
        group_size: int,
        split: str,
        seed: Optional[int] = 42,
        node_to_idx: Optional[Dict[str, int]] = None,
    ):
        random.seed(seed)
        self.group_size = group_size
        self.graph = graph
        self.split = split
        self.closure = nx.transitive_closure(graph)
        self.nodes = list(self.closure.nodes())
        self.positive_edges = list(self.closure.edges())
        
        logger.info(f"Graph nodes: {len(self.graph.nodes())}, edges: {len(self.graph.edges())}")
        logger.info(f"Full closure nodes: {len(self.closure.nodes())}, edges: {len(self.closure.edges())}")

        if node_to_idx is not None:
            self.node_to_idx = node_to_idx
        else:
            self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        self.idx_to_node = {v: k for k, v in self.node_to_idx.items()}

        logger.info("Precomputing positives & negatives...")
        self.total_indices = set(self.node_to_idx.values())
        self.all_neighbors = {
            self.node_to_idx[n]: set(
                self.node_to_idx[nei] for nei in nx.all_neighbors(self.closure, n)
            ).union({self.node_to_idx[n]})
            for n in self.nodes
            if n in self.node_to_idx
        }

        self.all_out_neighbors = {
            self.node_to_idx[n]: set(
                self.node_to_idx[nei] for nei in nx.neighbors(self.closure, n)
            ).union({self.node_to_idx[n]})
            for n in self.nodes
            if n in self.node_to_idx
        }

    def __len__(self):
        return len(self.positive_edges)

    def _sample_negatives(self, pos_set, count):
        negs = []
        seen = set()
        max_idx = len(self.nodes) - 1
        while len(negs) < count:
            candidate = random.randint(0, max_idx)
            if candidate not in pos_set and candidate not in seen:
                seen.add(candidate)
                negs.append(candidate)
        return negs

    def __getitem__(self, idx):
        src, tgt = self.positive_edges[idx]
        anchor = self.node_to_idx[src]
        positive = [self.node_to_idx[tgt]]

        pos_set = self.all_neighbors[anchor]
        neg_pool = None

        if self.group_size is not None:
            negatives = self._sample_negatives(pos_set, self.group_size - 1)
        else:
            neg_pool = list(self.total_indices - pos_set)
            negatives = neg_pool
        posnegs = positive + negatives
        return anchor, posnegs
    
    def collate_fn(self, batch):
        anchors, posnegs = zip(*batch)
        anchors = torch.tensor(anchors, dtype=torch.long)
        posnegs = torch.tensor(posnegs, dtype=torch.long)
        return anchors, posnegs