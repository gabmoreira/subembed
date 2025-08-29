import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, Union
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import SparseAdam
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import BinaryAUPRC, Mean
from tqdm.auto import tqdm

from subspaces import ridge_projector
from data import LinkPrediction

logging.basicConfig(
    level=logging.INFO,  # INFO for training, DEBUG if debugging internals
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

EPS = 1e-6

@dataclass
class LinkPredictionData:
    """
    Container for experiment configuration and model state.
    Saved/loaded as a Torch .pt dictionary for reproducibility.
    """
    root: Optional[str] = None
    closure: Optional[float] = None # Fraction of transitive closure edges
    node_to_idx: Optional[Dict[str, int]] = None
    batch_size: Optional[int] = None # Batch size
    N: Optional[int] = None # Num vectors per node
    D: Optional[int] = None # Embedding dimension
    lbd: Optional[float] = None # Regularization
    group_size: Optional[int] = None # Num positives + num negative
    epochs: Optional[int] = None
    epoch: Optional[int] = None
    gamma_pos: Optional[float] = None
    gamma_neg: Optional[float] = None
    state_dict: Optional[Dict[str, Any]] = None
    std_init: Optional[float] = None
    lr: Optional[float] = None

    def save(self, path: str) -> None:
        """Save the dataclass as a torch .pt dictionary."""
        torch.save(asdict(self), path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "LinkPredictionData":
        """Load a saved configuration from disk."""
        data = torch.load(path, map_location=map_location)
        field_names = {f.name for f in fields(LinkPredictionData)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return LinkPredictionData(**filtered_data)

def parse_args():
    parser = argparse.ArgumentParser(description="Train graph embeddings for link prediction.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the Link Prediction dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--group_size", type=int, default=11, help="Number of positives + negatives per sample.")
    parser.add_argument("--closure", type=float, required=True, help="Fraction <= 1 of non-basic edges.")
    parser.add_argument("--N", type=int, default=128, help="Number of vectors per node.")
    parser.add_argument("--D", type=int, default=128, help="Ambient space dimension.")
    parser.add_argument("--lbd", type=float, default=0.2, help="Regularization parameter λ.")
    parser.add_argument("--gamma_pos", type=float, default=0.8, help="Positive margin γ+.")
    parser.add_argument("--gamma_neg", type=float, default=0.1, help="Negative margin γ−.")
    parser.add_argument("--std_init", type=float, default=1e-4, help="Std-dev for weight initialization.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    return parser.parse_args()

def compute_scores(
    embeddings: nn.Embedding,
    nodes: Tensor,
    N: int,
    D: int,
    lbd: float,
    eps: float = EPS,
) -> Tensor:
    """
    Compute normalized inclusion scores for link prediction.
    nodes: LongTensor of shape (..., 2) with (i, j) node indices.
    Returns: Tensor of shape (B, G) with scores in (0, 1).
    """
    nodes_i = nodes[...,0]
    nodes_j = nodes[...,1]
    emb_i = embeddings(nodes_i).view(*nodes_i.shape, N, D)
    emb_j = embeddings(nodes_j).view(*nodes_j.shape, N, D)
    proj_i = ridge_projector(emb_i, lbd=lbd)
    proj_j = ridge_projector(emb_j, lbd=lbd)
    prior_i = torch.einsum("bgii->bg", proj_i)
    intersections = torch.einsum("bgii->bg", proj_i @ proj_j)
    scores = torch.clamp(intersections / (eps + prior_i), eps, 1.0 - eps) # (B, G)
    return scores

def train_one_epoch(
        embeddings: nn.Embedding,
        dataloader: DataLoader,
        optimizer,
        metrics,
        emb_data: LinkPredictionData,
        epoch: int,
        pbar: tqdm,
        device: Union[torch.device, str],
    ) -> int:
    """
    Train for one epoch. Returns the number of steps (batches) processed.
    """
    embeddings.train()
    metrics["loss"].reset()
    metrics["auprc"].reset()

    step_count = 0
    for it, nodes in enumerate(dataloader):
        step_count += 1
        nodes = nodes.to(device)
        scores = compute_scores(embeddings, nodes, emb_data.N, emb_data.D, emb_data.lbd)
        targets = torch.zeros_like(scores, device=device, dtype=torch.long)
        targets[..., 0] = 1

        scores_pos, scores_neg = scores[:, 0], scores[:, 1:]
        loss = (
            F.relu(emb_data.gamma_pos - scores_pos).sum() +
            F.relu(scores_neg - emb_data.gamma_neg).mean(-1).sum()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics["loss"].update(loss.detach())
        metrics["auprc"].update(scores.flatten(), targets.flatten())

        cur_loss = metrics["loss"].compute().item()
        cur_auprc = metrics["auprc"].compute().item()

        pbar.set_postfix(
            train_loss=f"{cur_loss:.6f}",
            train_auprc=f"{cur_auprc:.4f}",
            epoch=f"{epoch + it / (len(dataloader) + 1) :.4f}",
        )
        pbar.update() 

    return step_count

def main():
    args = parse_args()
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 16

    field_names = {f.name for f in fields(LinkPredictionData)}
    args_dict = {k: v for k, v in vars(args).items() if k in field_names}
    emb_data = LinkPredictionData(**args_dict)
    
    root_dir = f"./wordnet_embeddings/linkprediction_" \
               f"{int(100 * emb_data.closure)}_wordnet_" \
               f"subspace_{emb_data.N}x{emb_data.D}_{emb_data.lbd}_{emb_data.group_size}"
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{root_dir}/config.pt"

    train_dataset = LinkPrediction(
        root=args.dataset_path,
        closure=emb_data.closure,
        split="train",
        group_size=emb_data.group_size,
    )
    val_dataset = LinkPrediction(
        root=args.dataset_path,
        closure=emb_data.closure,
        split="val", 
        group_size=emb_data.group_size,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=emb_data.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=val_dataset.collate_fn,
        pin_memory=False,
        num_workers=4,
    )
    embeddings = nn.Embedding(
        num_embeddings=len(train_dataset.node_to_idx),
        embedding_dim=emb_data.N * emb_data.D,
        sparse=True, 
        device=device,
    )
    init = torch.normal(
        mean=0.0,
        std=emb_data.std_init,
        size=(len(train_dataset.node_to_idx), emb_data.N, emb_data.D),
        device=device,
    )

    with torch.no_grad():
        embeddings.weight.copy_(init.view(len(train_dataset.node_to_idx), -1))

    optimizer = SparseAdam(embeddings.parameters(), lr=emb_data.lr)
    writer = SummaryWriter(log_dir=root_dir)

    emb_data.node_to_idx = train_dataset.node_to_idx
    emb_data.state_dict = embeddings.state_dict()
    
    train_metrics = {
        "auprc" : BinaryAUPRC(device=device),
        "loss" :  Mean(device=device),
    }
    val_auprc = BinaryAUPRC(device=device)

    desc = f"Training {emb_data.N}x{emb_data.D} | Group-size {emb_data.group_size} | lbd {emb_data.lbd}"
    total_steps = emb_data.epochs * len(train_dataloader)
    pbar = tqdm(total=total_steps, desc=desc, leave=True, dynamic_ncols=True) 

    step_count = 0
    best_val_auprc = 0.0

    for epoch in range(emb_data.epochs):
        steps = train_one_epoch(
            embeddings, train_dataloader, optimizer, train_metrics, emb_data, epoch, pbar, device
        )
        step_count += steps
        writer.add_scalar('train/loss', train_metrics["loss"].compute().item(), step_count)
        writer.add_scalar('train/auprc', train_metrics["auprc"].compute().item(), step_count)

        # VAL
        embeddings.eval()
        val_auprc.reset()
        with torch.no_grad():
            for nodes in val_loader:
                nodes = nodes.to(device)
                scores = compute_scores(embeddings, nodes, emb_data.N, emb_data.D, emb_data.lbd)
                targets = torch.zeros_like(scores, device=scores.device, dtype=torch.long) 
                targets[...,0] = 1
                val_auprc.update(scores.flatten(), targets.flatten())
        
        val_score = val_auprc.compute().item()
        writer.add_scalar('val/auprc', val_score, step_count)

        if val_score >= best_val_auprc:
            best_val_auprc = val_score
            emb_data.state_dict = embeddings.state_dict()
            emb_data.epoch = epoch
            emb_data.save(filename)
            logger.info(f"Best val AUPRC: {best_val_auprc:.4f}. Saved to {filename}")

    pbar.close()
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()