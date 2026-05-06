import sys
import os
import re
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import SNLI
from model import TransformerClassifier, TransformerSubspaceEmbedder
from train_nli_baseline import NLITrainingData as NLITrainingDataBaseline
from train_nli import NLITrainingData

logging.basicConfig(
    level=logging.INFO,  # INFO for training, DEBUG if debugging internals
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Eval SNLI-trained models.")
    parser.add_argument(
        "--root",
        type=str,
        default="./nli_models",
        help="Root folder of model runs."
    ) 
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Num workers."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch-size."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Singular value threshold for compression. Negative = No Compression"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Model name."
    )
    return parser.parse_args()

def main():
    """
    Example:
        Eval specific model with singular value threshold:
            python eval_snli.py --root ./nli_models/ --model-name all-mpnet-base-v2_128x128_lbd0.05_context35_seed24_2way --threshold 0.8
        Eval all models: 
            python eval_snli.py --root ./nli_models/ 
    """
    args = parse_args()
    root = args.root
    device = args.device
    num_workers = args.num_workers
    batch_size = args.batch_size

    if len(args.model_name) == 0:
        model_list = os.listdir(root)
    else:
        model_list = [args.model_name]

    evals = {re.sub(r"_seed\d+", "", model_name) : [] for model_name in model_list}
    logger.info(f"Evaluating {len(evals)} models")

    for model_name in model_list:
        config_path = os.path.join(root, model_name, "config.pt")
        if "baseline" in model_name:
            config = NLITrainingDataBaseline.load(config_path)
            difference = "p-h" in model_name
            model = TransformerClassifier(
                base_model_name=config.base_model_name,
                difference=difference,
                two_way=config.two_way,
                cache_dir="./.cache",
            )
        else:
            config = NLITrainingData.load(config_path)
            model = TransformerSubspaceEmbedder(
                base_model_name=config.base_model_name,
                N=config.N,
                D=config.D,
                lbd=config.lbd,
                two_way=config.two_way,
                cache_dir="./.cache",
            )

        model.load_state_dict(config.state_dict, strict=False)
        model.eval()
        model.to(device)

        test_dataset = SNLI(
            max_length=config.max_length,
            split="test",
            two_way=config.two_way
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: test_dataset.collate_fn(b, model.tokenizer),
            pin_memory=device != "cpu",
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        logits, labels = [], []
        ranks = []
        for enc_pre, enc_hyp, targets in test_loader:
            targets = targets.to(device)
            for k in enc_pre:
                enc_pre[k] = enc_pre[k].to(device)
                enc_hyp[k] = enc_hyp[k].to(device)
            
            with torch.no_grad():
                x_pre = model(**enc_pre)
                x_hyp = model(**enc_hyp)

                if args.threshold > 0.0:
                    U, S, Vh = torch.linalg.svd(x_pre, full_matrices=False)
                    mask = (S >= args.threshold)
                    x_pre = (mask.unsqueeze(1) * (U * S.unsqueeze(1))) @ (Vh * mask.unsqueeze(2))
                    compressed_pre_rank = mask.sum(-1)
                    U, S, Vh = torch.linalg.svd(x_hyp, full_matrices=False)
                    mask = (S >= args.threshold)
                    x_hyp = (mask.unsqueeze(1) * (U * S.unsqueeze(1))) @ (Vh * mask.unsqueeze(2))
                    compressed_hyp_rank = mask.sum(-1)
                    ranks.append(compressed_pre_rank)
                    ranks.append(compressed_hyp_rank)

                logits_ = model.classify(x_pre, x_hyp)

            logits.append(logits_)
            labels.append(targets)

        labels = torch.cat(labels)
        logits = torch.cat(logits)

        preds = torch.argmax(logits.cpu(), dim=-1)
        acc = (preds == labels.cpu()).float().mean().item()
        info_str = f"{model_name} | Test accuracy: {acc * 100:.2f}%"

        if len(ranks) > 0:
            ranks = torch.cat(ranks)
            average_rank = ranks.float().mean().item()
            info_str += f" | Average rank: {average_rank:.2f}"

        logger.info(info_str)

        evals[re.sub(r"_seed\d+", "", model_name)].append(acc)
        
    for k, v in evals.items():
        logger.info(f"{k} (averaged over {len(v)} runs): {np.mean(v):.4f} ± {np.std(v):.4f}")

if __name__ == "__main__":
    main()