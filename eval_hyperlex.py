import sys
import logging
import argparse
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from tqdm.auto import tqdm
from data.graph import *
from subspaces import *
from train_wordnet_reconstruction import ReconstructionData

logging.basicConfig(
    level=logging.INFO,  # INFO for training, DEBUG if debugging internals
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WordNet embeddings on HyperLex.")
    parser.add_argument("--embed_path", type=str, default="./", help="Path to embeddings file")
    parser.add_argument("--hyperlex_path", type=str, default="./", help="Path to hyperlex hyperlex-nouns.txt file")
    parser.add_argument("--lbd", type=float, default=0.2, help="Regularization parameter λ")
    return parser.parse_args()

def main():
    args = parse_args()
    device = device = "cuda" if torch.cuda.is_available() else "cpu"

    emb_data = ReconstructionData.load(args.embed_path)

    embeddings = nn.Embedding(
        num_embeddings=len(emb_data.state_dict["weight"]),
        embedding_dim=emb_data.N * emb_data.D,
        sparse=True,
        device="cpu",
    )
    embeddings.load_state_dict(emb_data.state_dict)
    embeddings = embeddings.to(device)
    embeddings.weight.requires_grad = False

    logger.info("Precomputing projectors.")
    proj = ridge_projector(
        embeddings.weight.view(-1, emb_data.N, emb_data.D), lbd=args.lbd, chol=False
    )
    logger.info(f"Projectors: {proj.shape[0]}x{proj.shape[1]}x{proj.shape[2]}")

    hyperlex = []
    with open(args.hyperlex_path, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            parts = line.strip().split()
            word1 = parts[0]
            word2 = parts[1]
            avg_score = float(parts[5])
            hyperlex.append((word1, word2, avg_score))

    gt_scores = []
    es_scores = []
    for tuple in tqdm(hyperlex):
        word1, word2, gt_score = tuple

        synsets1 = wn.synsets(word1, pos="n")
        synsets2 = wn.synsets(word2, pos="n")

        best_score = 0.0
        synset1 = synsets1[0]
        synset2 = synsets2[0]

        for s1 in synsets1:
            for s2 in synsets2:
                key1, key2 = s1.name(), s2.name()
                i = emb_data.node_to_idx[key1]
                j = emb_data.node_to_idx[key2]
                score = torch.trace(proj[i] @ proj[j])
                if score > best_score:
                    best_score = score
                    synset1, synset2 = s1, s2

        i = emb_data.node_to_idx[synset1.name()]
        j = emb_data.node_to_idx[synset2.name()]
        
        estimated_scores = torch.trace(proj[i] @ proj[j]) / (torch.trace(proj[i]))
        es_scores.append(estimated_scores.item())
        gt_scores.append(gt_score)

    corr, p_value = spearmanr(gt_scores, es_scores)
    logger.info(f"Results for {args.embed_path}")
    logger.info(f"Spearman rank-correlation  (lbd={args.lbd}): {corr:.4f}")

if __name__ == "__main__":
    main()