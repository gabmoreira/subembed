import torch
import argparse
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from data import Reconstruction
from data import build_hypernym_graph
from subspaces import ridge_projector
from train_wordnet_reconstruction import ReconstructionData

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Reconstruction embeddings.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computations (e.g., 'cuda:0', 'cpu')."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the ReconstructionData file."
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        required=False,
        default=1,
        help="Number of chunks for splitting negatives."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    try:
        emb_data = ReconstructionData.load(args.path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.path}")
        return
    except Exception as e:
        print(f"Error loading data from {args.path}: {e}")
        return

    graph = build_hypernym_graph(type=emb_data.synset, root=None)

    embeddings = nn.Embedding(
        num_embeddings=len(graph),
        embedding_dim=emb_data.N * emb_data.D,
        sparse=True,
        device="cpu"
    )

    print("Precomputing projections")
    proj = ridge_projector(
        emb_data.state_dict["weight"].view(len(graph), emb_data.N, emb_data.D), lbd=emb_data.lbd
    )
    embeddings.weight.data.copy_(proj.flatten(1,2))
        
    embeddings.weight.requires_grad = False
    embeddings = embeddings.to(args.device)

    dataset = Reconstruction(
        graph,
        group_size=None,
        node_to_idx=emb_data.node_to_idx,
        split="test",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=2,
    )

    rank_sum = 0.0
    ap_sum = 0.0
    progress_bar = tqdm(range(len(loader)), desc=f"Eval", leave=True, dynamic_ncols=True) 
    for it, (node_i, node_j) in enumerate(loader):
        node_i, node_j = node_i.to(args.device), node_j.to(args.device)

        emb_i = embeddings(node_i) # (B, N * D)
        emb_j = embeddings(node_j.flatten()) # (B * G, N * D)
        emb_j = emb_j.view(node_j.shape[0], node_j.shape[1], -1) # (B, G, N * D)
            
        chunks = torch.chunk(emb_j, chunks=args.num_chunks, dim=1)
        scores = []
        for chunk in chunks:
            score = torch.einsum("bd,bgd->bg", emb_i, chunk)
            scores.append(score) 
        scores = torch.cat(scores, dim=-1)
            
        sorted_indices = torch.argsort(-scores, dim=-1)
        r = (sorted_indices == 0).nonzero(as_tuple=True)[1].float() + 1
            
        rank_sum += r.item()
        ap_sum += 1.0 / r.item()
            
        progress_bar.set_postfix(
            mean_rank=f"{rank_sum / (it + 1):.4f}",
            map=f"{ap_sum / (it + 1):.4f}",
            group_size=f"{emb_j.shape[1]}",
        )
        progress_bar.update()     
    progress_bar.close()


if __name__ == "__main__":
    main()