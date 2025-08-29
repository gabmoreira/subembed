import sys
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torcheval.metrics import BinaryF1Score, BinaryAUPRC
from tqdm.auto import tqdm

from data import LinkPrediction
from train_wordnet_lp import LinkPredictionData, compute_scores
from sklearn.metrics import f1_score

logging.basicConfig(
    level=logging.INFO,  # INFO for training, DEBUG if debugging internals
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

EPS = 1e-6

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate link prediction model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computations (e.g., 'cuda:0', 'cpu')."
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        required=True,
        help="Path to the LinkPredictionData file."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the Link Prediction dataset."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch-size."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers."
    )
    args = parser.parse_args()
    return args

def main():
    """
    Link prediction evaluation.

    Loads data and embeddings, calibrates a classification threshold 
    on the validation set, and evaluates F1-score and AUPRC on the test set.
    """
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    try:
        emb_data = LinkPredictionData.load(args.embed_path)
    except FileNotFoundError:
        print(f"Error: Embed file not found at {args.embed_path}")
        return
    except Exception as e:
        print(f"Error loading data from {args.path}: {e}")
        return

    val_dataset = LinkPrediction(
        root=args.dataset_path, closure=emb_data.closure, split="val", group_size=11
    )
    test_dataset = LinkPrediction(
        root=args.dataset_path, closure=emb_data.closure, split="test", group_size=11
    )
    
    dataloader_common_args = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": args.num_workers,
    }

    val_loader = DataLoader(
        val_dataset,
        collate_fn=val_dataset.collate_fn,
        **dataloader_common_args,
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate_fn,
        **dataloader_common_args,
    )

    embeddings = nn.Embedding(
        num_embeddings=len(test_dataset.node_to_idx),
        embedding_dim=emb_data.N * emb_data.D,
        sparse=True,
        device="cpu"
    )
    
    embeddings.load_state_dict(emb_data.state_dict)
    embeddings = embeddings.to(device)
    embeddings.weight.requires_grad = False
    embeddings.eval()

    val_scores, val_targets = [], []
    for nodes in tqdm(val_loader, desc="Calibration on validation set"):
        nodes = nodes.to(device) # (B, G, 2)
        scores = compute_scores(embeddings, nodes, emb_data.N, emb_data.D, emb_data.lbd, EPS) # (B, G)
        targets = torch.zeros_like(scores, device=scores.device, dtype=torch.long) 
        targets[...,0] = 1
        val_scores.append(scores.flatten().cpu())
        val_targets.append(targets.flatten().cpu())

    val_scores = torch.cat(val_scores)
    val_targets = torch.cat(val_targets)
    threshold = max(
        np.linspace(0, 1, 100),
        key=lambda t: f1_score(val_targets, val_scores >= t)
    )
    logger.info(f"Classification threshold: {threshold:.4f}")

    test_f1score = BinaryF1Score(threshold=threshold)
    test_auprc = BinaryAUPRC()
    test_auprc.reset()
    test_f1score.reset()
    for nodes in tqdm(test_loader, "Test"):
        nodes = nodes.to(device) # (B, G, 2)
        scores = compute_scores(embeddings, nodes, emb_data.N, emb_data.D, emb_data.lbd, EPS) # (B, G)
        targets = torch.zeros_like(scores, device=scores.device, dtype=torch.long) 
        targets[...,0] = 1
        test_f1score.update(scores.flatten(), targets.flatten())
        test_auprc.update(scores.flatten(), targets.flatten())

    logger.info(f"Test results for {args.embed_path}")
    logger.info(f"Test F1-Score: {test_f1score.compute():.4f}")
    logger.info(f"Test AUPRC: {test_auprc.compute():.4f}")

if __name__ == "__main__":
    main()