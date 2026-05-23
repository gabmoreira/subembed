import sys
import logging
import torch
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score
from collections import defaultdict

from data import build_hypernym_graph
from subspaces import ridge_projector
from train_wordnet_reconstruction import ReconstructionData

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate WordNet Subspace Embeddings.")
    p.add_argument("--device",     type=str, default="cuda:0")
    p.add_argument("--embed-path", type=str, required=True)
    return p.parse_args()


def build_adjacency(graph, node_to_idx):
    """Build adjacency dict[int, set[int]] from the transitive closure of the graph."""
    try:
        closure = nx.transitive_closure_dag(graph)
    except nx.NetworkXError:
        # Fallback if graph isn't recognized as a DAG
        closure = nx.transitive_closure(graph)
    adj = defaultdict(set)
    for src, tgt in closure.edges():
        src_str = str(src)
        tgt_str = str(tgt)
        if src_str in node_to_idx and tgt_str in node_to_idx:
            u = node_to_idx[src_str]
            v = node_to_idx[tgt_str]
            adj[u].add(v)
    return dict(adj)


def main():
    args = parse_args()

    try:
        emb_data = ReconstructionData.load(args.embed_path)
    except Exception as e:
        logger.error(f"Error loading data from {args.embed_path}: {e}")
        return

    graph = build_hypernym_graph(type=emb_data.synset, root=None)

    n_nodes = len(emb_data.node_to_idx)
    embeddings = nn.Embedding(
        num_embeddings=n_nodes,
        embedding_dim=emb_data.N * emb_data.D,
        sparse=True,
        device="cpu"
    )

    logger.info("Precomputing projections")
    proj = ridge_projector(
        emb_data.state_dict["weight"].view(len(graph), emb_data.N, emb_data.D), lbd=emb_data.lbd
    )
    embeddings.weight.data.copy_(proj.flatten(1,2))
    embeddings.weight.requires_grad = False
    embeddings = embeddings.to(args.device)

    # All embeddings as a single tensor for ranking
    all_emb = embeddings.weight.data  # (nodes, N*D)

    adj = build_adjacency(graph, emb_data.node_to_idx)
    source_nodes = sorted(adj.keys())
    total_edges = sum(len(v) for v in adj.values())

    logger.info(
        f"Transitive closure: {total_edges} edges over {len(source_nodes)} source nodes"
    )
    logger.info(
        f"Evaluating against {n_nodes} total nodes, D={emb_data.D}"
    )

    ranksum = 0.0
    nranks = 0.0
    ap_sum = 0.0
    n_sources = 0

    labels = np.empty(n_nodes, dtype=np.float32)

    pbar = tqdm(source_nodes, desc="Evaluating", leave=True, dynamic_ncols=True)

    with torch.no_grad():
        for src in pbar:
            neighbors = np.array(sorted(adj[src]), dtype=np.int32)
            if len(neighbors) == 0:
                continue

            # Distance from src to every node
            dists = - all_emb @ all_emb[src]
            dists[src] = 1e12

            # Sort and find where neighbors land
            _, sorted_idx = dists.sort()
            sorted_idx_np = sorted_idx.cpu().numpy()
            ranks, = np.where(np.in1d(sorted_idx_np, neighbors))
            ranks += 1  # 1-indexed

            # Adjusted rank: subtract contribution of other positives
            # that appear before each positive (see Facebook's code)
            N = len(neighbors)
            ranksum += ranks.sum() - (N * (N - 1)) / 2
            nranks += N

            # MAP via sklearn (all neighbors as positives)
            labels.fill(0)
            labels[neighbors] = 1
            dists_np = dists.cpu().numpy()
            ap_sum += average_precision_score(labels, -dists_np)
            n_sources += 1

            pbar.set_postfix(
                MR=f"{ranksum / nranks:.2f}",
                MAP=f"{ap_sum / n_sources:.4f}",
            )

    mean_rank = ranksum / nranks
    mean_ap = ap_sum / n_sources

    logger.info(f"Mean Rank:             {mean_rank:.2f}")
    logger.info(f"Mean Average Precision: {mean_ap:.4f}")


if __name__ == "__main__":
    main()
