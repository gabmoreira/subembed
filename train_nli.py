import sys
import logging
import argparse
import torch

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict, fields
from tqdm.auto import tqdm
from typing import Dict, Any

from subspaces import *
from data import SNLI
from model import TransformerSubspaceEmbedder

logging.basicConfig(
    level=logging.INFO,  # INFO for training, DEBUG if debugging internals
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

CACHE_DIR = "./.cache"

@dataclass
class NLITrainingData:
    base_model_name: Optional[str] = None
    pretrained: Optional[str] = None
    max_length: Optional[int] = None
    two_way: Optional[bool] = None
    N: Optional[int] = None # Num vectors per node
    D: Optional[int] = None # Embedding dimension
    lbd: Optional[float] = None # Regularization
    batch_size: Optional[int] = None # Batch size
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    epochs: Optional[int] = None
    state_dict: Optional[Dict[str, Any]] = None
    epoch: Optional[int] = None

    def save(self, path: str) -> None:
        """Save the dataclass as a torch .pt dictionary."""
        torch.save(asdict(self), path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "NLITrainingData":
        """Load a saved configuration from disk."""
        data = torch.load(path, map_location=map_location)
        field_names = {f.name for f in fields(NLITrainingData)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return NLITrainingData(**filtered_data)

def parse_args():
    parser = argparse.ArgumentParser(description="Train subspace transformer model.")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base transformer model.") # sentence-transformers/all-MiniLM-L6-v2 , sentence-transformers/all-mpnet-base-v2
    parser.add_argument("--pretrained", type=str, default="", help="Pretrained model name.")
    parser.add_argument("--max_length", type=int, default=35, help="Maximum sequence length.")
    parser.add_argument("--two_way", action="store_true", help="Two-way (entail vs non-entail).")
    parser.add_argument("--N", type=int, default=128, help="Number of vectors per node.")
    parser.add_argument("--D", type=int, default=128, help="Ambient space dimension.")
    parser.add_argument("--lbd", type=float, default=0.05, help="Regularization parameter λ.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Optimizer weight-decay.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 8
    enable_autocast = True

    field_names = {f.name for f in fields(NLITrainingData)}
    args_dict = {k: v for k, v in vars(args).items() if k in field_names}
    train_data = NLITrainingData(**args_dict)

    save_to = f"./runs/{train_data.base_model_name.split('/')[1]}_" \
              f"{train_data.N}x{train_data.D}_lbd{train_data.lbd}_" \
              f"context{train_data.max_length}_" \
              f"{'2way' if train_data.two_way else ''}"
    Path(save_to).mkdir(parents=True, exist_ok=True)

    model = TransformerSubspaceEmbedder(
        base_model_name=train_data.base_model_name,
        N=train_data.N,
        D=train_data.D,
        lbd=train_data.lbd,
        two_way=train_data.two_way,
        cache_dir=CACHE_DIR,
    )
    model.to(device)
    
    if len(train_data.pretrained) > 0:
        loaded_data = NLITrainingData.load(train_data.pretrained, map_location=torch.device("cpu"))
        model.load_state_dict(loaded_data.state_dict, strict=False)
        logger.info(f"Loaded {train_data.pretrained}")

    train_dataset = SNLI(max_length=train_data.max_length, split="train", two_way=train_data.two_way)
    val_dataset = SNLI(max_length=train_data.max_length, split="validation", two_way=train_data.two_way)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_data.batch_size,
        shuffle=True,
        collate_fn=lambda b: train_dataset.collate_fn(b, model.tokenizer),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_data.batch_size,
        shuffle=False,
        collate_fn=lambda b: train_dataset.collate_fn(b, model.tokenizer),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_data.lr, weight_decay=train_data.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=enable_autocast)
    writer = SummaryWriter(log_dir=save_to)
    best_acc = 0.0

    num_train_steps = 0
    for epoch in range(train_data.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for enc_pre, enc_hyp, targets in pbar:
            for k in enc_pre:
                enc_pre[k] = enc_pre[k].to(device)
                enc_hyp[k] = enc_hyp[k].to(device)
            targets = targets.to(device)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=enable_autocast):                
                x_pre = model.forward_backbone(**enc_pre)
                x_hyp = model.forward_backbone(**enc_hyp)

            P_pre = model.to_projection(x_pre.to(torch.float32))  # (B, D, D)
            P_hyp = model.to_projection(x_hyp.to(torch.float32))  # (B, D, D)

            logits, loss = model.classify(P_pre, P_hyp, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == targets).float().mean()

            pbar.set_postfix(train_loss=loss.item(), train_acc=acc.item(), lr=f"{scheduler.get_lr()[0]:.6f}")
            
            if num_train_steps % 20 == 0:
                writer.add_scalar('train/loss', loss.item(), num_train_steps)
                writer.add_scalar('train/accuracy', acc.item(), num_train_steps)

            num_train_steps += 1
        scheduler.step()

        # ------ EVAL ------
        model.eval()
        logits, labels = [], []
        for enc_pre, enc_hyp, targets in val_loader:
            targets = targets.to(device)
            for k in enc_pre:
                enc_pre[k] = enc_pre[k].to(device)
                enc_hyp[k] = enc_hyp[k].to(device)
            
            with torch.no_grad():
                P_pre = model(**enc_pre)
                P_hyp = model(**enc_hyp)
                logits_, _ = model.classify(P_pre, P_hyp, targets)
                logits.append(logits_)
                labels.append(targets)

        labels = torch.cat(labels)
        logits = torch.cat(logits)
        acc = (torch.argmax(logits.cpu(), dim=-1) == labels.cpu()).float().mean().item()
        writer.add_scalar('val/accuracy', acc, num_train_steps)

        if acc >= best_acc:
            state_dict = model.state_dict()
            train_data.epoch = epoch
            train_data.state_dict = state_dict
            train_data.save(f"{save_to}.pt")
            best_acc = acc
            logger.info(f"Best validation accuracy: {best_acc:.4f}. Saved to {save_to}.pt")
