import sys
import logging
import argparse
import torch
import time

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from dataclasses import dataclass, asdict, fields
from tqdm.auto import tqdm
from typing import Dict, Any

from subspaces import *
from data import SNLI
from model import BoxTransformerClassifier

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
    box_dim: Optional[int] = None
    label_smoothing: Optional[float] = None # Label Smoothing
    batch_size: Optional[int] = None # Batch size
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    epochs: Optional[int] = None
    state_dict: Optional[Dict[str, Any]] = None
    epoch: Optional[int] = None
    seed: Optional[int] = None

    def save(self, path: str) -> None:
        """Save the dataclass as a torch .pt dictionary."""
        torch.save(asdict(self), path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "NLITrainingData":
        """Load a saved configuration."""
        data = torch.load(path, map_location=map_location, weights_only=False)
        field_names = {f.name for f in fields(NLITrainingData)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return NLITrainingData(**filtered_data)

def parse_args():
    parser = argparse.ArgumentParser(description="Train subspace transformer model.")
    parser.add_argument("--base-model-name", type=str, required=True, help="Base transformer model e.g., sentence-transformers/all-MiniLM-L6-v2 , sentence-transformers/all-mpnet-base-v2") 
    parser.add_argument("--pretrained", type=str, default="", help="Pretrained model name.")
    parser.add_argument("--max-length", type=int, default=35, help="Maximum sequence length.")
    parser.add_argument("--box-dim", default=128, type=int, help="Box dim.")
    parser.add_argument("--two-way", action="store_true", help="Two-way (entail vs non-entail).")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Optimizer weight-decay.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--benchmark", action="store_true", help="Set to benchmark time and memory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    field_names = {f.name for f in fields(NLITrainingData)}
    args_dict = {k: v for k, v in vars(args).items() if k in field_names}
    config = NLITrainingData(**args_dict)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    device = args.device
    num_workers = 8
    difference = True
    enable_autocast = True

    prefix = f"box{config.box_dim}d"
    save_to = f"./nli_models/{prefix}_{config.base_model_name.split('/')[1]}_" \
              f"context{config.max_length}_seed{config.seed}" \
              f"{'_2way' if config.two_way else ''}" \
              f"{'_benchmark' if args.benchmark else ''}"
    Path(save_to).mkdir(parents=True, exist_ok=True)

    model = BoxTransformerClassifier(
        base_model_name=config.base_model_name,
        box_dim=config.box_dim,
        cache_dir=CACHE_DIR,
    )
    model.to(device)
    
    if len(config.pretrained) > 0:
        loaded_data = NLITrainingData.load(config.pretrained, map_location=torch.device("cpu"))
        model.load_state_dict(loaded_data.state_dict, strict=False)
        logger.info(f"Loaded {config.pretrained}")

    train_dataset = SNLI(max_length=config.max_length, split="train", two_way=config.two_way)
    val_dataset = SNLI(max_length=config.max_length, split="validation", two_way=config.two_way)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: train_dataset.collate_fn(b, model.tokenizer),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda b: train_dataset.collate_fn(b, model.tokenizer),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=enable_autocast)
    writer = SummaryWriter(log_dir=save_to)
    best_acc = 0.0

    num_train_steps = 0
    for epoch in range(config.epochs):
        if args.benchmark:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            epoch_start = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for enc_pre, enc_hyp, targets in pbar:
            for k in enc_pre:
                enc_pre[k] = enc_pre[k].to(device)
                enc_hyp[k] = enc_hyp[k].to(device)
            targets = targets.to(device)

            with torch.autocast(device_type=device, dtype=torch.float16, enabled=enable_autocast):                
                x_pre = model(**enc_pre)
                x_hyp = model(**enc_hyp)

            logits, loss = model.classify(x_pre, x_hyp, targets, config.label_smoothing)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == targets).float().mean()

            pbar.set_postfix(loss=loss.item(), acc=acc.item(), lr=f"{scheduler.get_lr()[0]:.6f}")
            
            if num_train_steps % 20 == 0:
                writer.add_scalar('train/loss', loss.item(), num_train_steps)
                writer.add_scalar('train/accuracy', acc.item(), num_train_steps)

            num_train_steps += 1
        scheduler.step()

        if args.benchmark:
            torch.cuda.synchronize()
            epoch_end = time.time()
            print(f"Epoch {epoch+1} time: {epoch_end - epoch_start:.2f} seconds")
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Peak GPU memory usage: {peak_memory:.2f} MB")

        # ------ EVAL ------
        model.eval()
        logits, labels = [], []
        for enc_pre, enc_hyp, targets in val_loader:
            for k in enc_pre:
                enc_pre[k] = enc_pre[k].to(device)
                enc_hyp[k] = enc_hyp[k].to(device)
            targets = targets.to(device)

            with torch.no_grad():
                x_pre = model(**enc_pre)
                x_hyp = model(**enc_hyp)
                logits_ = model.classify(x_pre, x_hyp)
                logits.append(logits_)
                labels.append(targets)

        labels = torch.cat(labels)
        logits = torch.cat(logits)
        acc = (torch.argmax(logits.cpu(), dim=-1) == labels.cpu()).float().mean().item()
        writer.add_scalar('val/accuracy', acc, num_train_steps)

        if acc >= best_acc:
            state_dict = model.state_dict()
            config.epoch = epoch
            config.state_dict = state_dict
            config.save(f"{save_to}/config.pt")
            best_acc = acc
            logger.info(f"Best validation accuracy: {best_acc:.4f}. Saved to {save_to}.pt")
