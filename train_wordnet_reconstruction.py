import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass, asdict, fields
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import Mean
from torch import Tensor
from typing import Dict, Any, Union, Optional

from data import build_hypernym_graph, Reconstruction
from subspaces import ridge_projector

logging.basicConfig(
    level=logging.INFO,  # INFO for training, DEBUG if debugging internals
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

@dataclass
class ReconstructionData:
    node_to_idx: Optional[Dict[str, int]] = None
    batch_size: Optional[int] = None # Batch size
    N: Optional[int] = None # Num vectors per node
    D: Optional[int] = None # Embedding dimension
    lbd: Optional[float] = None # Regularization
    group_size: Optional[int] = None # Num positives (1) + num negatives
    synset: Optional[str] = None # "n" - nouns, "v" - verbs
    epoch: Optional[int] = None
    epochs: Optional[int] = None
    state_dict: Optional[Dict[str, Any]] = None
    std_init: Optional[float] = None
    lr: Optional[float] = None

    def save(self, path: str) -> None:
        """Save the dataclass as a torch .pt dictionary."""
        torch.save(asdict(self), path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "ReconstructionData":
        data = torch.load(path, map_location=map_location)
        field_names = {f.name for f in fields(ReconstructionData)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return ReconstructionData(**filtered_data)

def compute_similarity(
    emb_i: Tensor,
    emb_j: Tensor,
    D: int,
    lbd: float,
) -> Tensor:
    proj_i = ridge_projector(emb_i, lbd, chol=False)
    proj_j = ridge_projector(emb_j.flatten(0,1), lbd, chol=False)
    proj_j = proj_j.view(emb_j.shape[0], emb_j.shape[1], D, D)
    scores = torch.einsum("bd,bgd->bg", proj_i.flatten(1,2), proj_j.flatten(2,3))
    return scores

def parse_args():
    parser = argparse.ArgumentParser(description="Train graph embeddings for reconstruction.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--synset", type=str, required=True, help="Synset: n (nouns) or v (verbs).")
    parser.add_argument("--group_size", type=int, default=20, help="Number of positives + negatives per sample.")
    parser.add_argument("--N", type=int, default=128, help="Number of vectors per node.")
    parser.add_argument("--D", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--lbd", type=float, default=0.2, help="Regularization parameter λ.")
    parser.add_argument("--std_init", type=float, default=1e-4, help="Std-dev for weight initialization.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    return parser.parse_args()

def train_one_epoch(
    embeddings: nn.Embedding,
    dataloader: DataLoader,
    optimizer,
    metrics,
    emb_data: ReconstructionData,
    epoch: int,
    pbar: tqdm,
    device: Union[torch.device, str],
) -> int:
    step_count = 0
    metrics["loss"].reset()
    metrics["mean_rank"].reset()
    metrics["map"].reset()

    for it, (node_i, node_j) in enumerate(dataloader):
        step_count += 1
        node_i = node_i.to(device)
        node_j = node_j.to(device)

        emb_i = embeddings(node_i)
        emb_i = emb_i.view(-1, emb_data.N, emb_data.D) # (B, N, D)
        emb_j = embeddings(node_j.flatten())
        emb_j = emb_j.view(-1, emb_data.group_size, emb_data.N, emb_data.D) # (B, G, N, D)

        scores = compute_similarity(emb_i, emb_j, emb_data.D, emb_data.lbd)
        target = torch.zeros(len(scores), device=scores.device, dtype=torch.long)
        loss = F.cross_entropy(scores, target, reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        sorted_indices = torch.argsort(-scores, dim=-1)
        r = (sorted_indices == 0).nonzero(as_tuple=True)[1].float() + 1.0
            
        metrics["loss"].update(loss.detach().cpu())
        metrics["mean_rank"].update(r.mean().cpu())
        metrics["map"].update((1.0 / r).mean().cpu())

        cur_loss = metrics["loss"].compute()
        cur_mean_rank = metrics["mean_rank"].compute()
        cur_map = metrics["map"].compute()

        pbar.set_postfix(
            loss=f"{cur_loss:.6f}",
            mean_rank=f"{cur_mean_rank:.4f}",
            map=f"{cur_map:.4f}",
            epoch=f"{epoch + it / (len(dataloader) + 1) :.4f}",
        )
        pbar.update() 
    return step_count

def main():
    args = parse_args()
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 16
    emb_data = ReconstructionData(**vars(args))
    
    root_dir = f"./wordnet_embeddings/reconstruction_" \
               f"{emb_data.synset}_{emb_data.N}x{emb_data.D}_" \
               f"{emb_data.lbd}_{emb_data.group_size}"
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{root_dir}/config.pt"

    G = build_hypernym_graph(type=emb_data.synset, root=None)

    dataset = Reconstruction(
        graph=G,
        group_size=emb_data.group_size,
        split="train",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=emb_data.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    init = torch.normal(
        mean=0.0,
        std=emb_data.std_init,
        size=(len(dataset.nodes), emb_data.N, emb_data.D),
        device=device,
        dtype=torch.bfloat16,
    )

    embeddings = nn.Embedding(
        num_embeddings=len(dataset.nodes),
        embedding_dim=emb_data.N * emb_data.D,
        sparse=True, 
        device=device,
    )
    with torch.no_grad():
        embeddings.weight.copy_(init.view(len(dataset.nodes), -1))

    optimizer = torch.optim.SparseAdam(embeddings.parameters(), lr=emb_data.lr)
    writer = SummaryWriter(log_dir=root_dir)

    emb_data.node_to_idx = dataset.node_to_idx
    emb_data.state_dict = embeddings.state_dict

    train_metrics = {"loss" : Mean(), "mean_rank" : Mean(), "map" : Mean()}

    step_count = 0
    total_steps = emb_data.epochs * len(dataloader)
    desc = f"Training {emb_data.N}x{emb_data.D} | Group-size {emb_data.group_size} | lbd {emb_data.lbd}"
    pbar = tqdm(total=total_steps, desc=desc, leave=True, dynamic_ncols=True) 
    for epoch in range(emb_data.epochs):            
        steps = train_one_epoch(
            embeddings, dataloader, optimizer, train_metrics, emb_data, epoch, pbar, device
        )
        step_count += steps
        for k, metric in train_metrics.items():
            writer.add_scalar(f"train/{k}", metric.compute().item(), step_count)

        if epoch % 5 == 0:
            emb_data.state_dict = embeddings.state_dict()
            emb_data.epoch = epoch
            emb_data.save(filename)
            logger.info(f"Saved to {filename}")
    pbar.close()
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()