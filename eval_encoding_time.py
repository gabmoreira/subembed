import sys
import argparse
import time
import logging
import torch
import numpy as np
from datasets import load_dataset

from model import TransformerSubspaceEmbedder
from train_nli import NLITrainingData

logging.basicConfig(
    level=logging.INFO,  # INFO for training, DEBUG if debugging internals
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

CACHE_DIR = "./.cache"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate encoding time.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computations (e.g., 'cuda:0', 'cpu')."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Embedding batch size."
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=100,
        help="Number of batches to encode."
    )
    parser.add_argument(
        "--sph",
        action="store_true",
        help="Use SPH module."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = args.device
    batch_size = args.batch_size
    num_trials = args.num_trials

    train_data = NLITrainingData.load(args.model_path)

    model = TransformerSubspaceEmbedder(
        train_data.base_model_name,
        train_data.N,
        train_data.D,
        train_data.lbd,
        two_way=train_data.two_way,
        cache_dir=CACHE_DIR,
    )
    model.load_state_dict(train_data.state_dict, strict=False)
    model.eval()
    model.to(device)

    dataset = load_dataset("nlphuji/flickr30k", cache_dir=CACHE_DIR, split="test")

    captions = []
    for i in range(len(dataset)):
        cs = dataset[i]["caption"]
        captions.extend(cs)

    db = []
    for i in range(num_trials):
        tokens = model.tokenizer(
            captions[i*batch_size:(i+1)*batch_size],
            return_tensors='pt',
            padding="max_length",
            max_length=train_data.max_length,
            truncation=True
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        db.append(tokens)

   # Warm-up
    for i in range(5):
        with torch.inference_mode():
            _ = model(**db[i])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timings = {"base_model": [], "attention": [], "mlp": [], "projection": []}
    for i in range(num_trials):
        torch.cuda.synchronize()
        
        with torch.inference_mode():
            t0 = time.perf_counter()
            hidden_state = model.forward_base_model(**db[i])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings["base_model"].append(t1 - t0)

            x = model.forward_attention(hidden_state)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            timings["attention"].append(t2 - t1)

            x = model.mlp(x)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            timings["mlp"].append(t3 - t2)

            proj = model.to_projection(x)
            torch.cuda.synchronize()
            t4 = time.perf_counter()
            timings["projection"].append(t4 - t3)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logger.info(f"Peak GPU memory: {peak_mem:.1f} MB")

    for k, v in timings.items():
        batch_time = np.mean(v)
        input_time = batch_time / batch_size
        logger.info(
            f"{k:>12s} | Batch: {batch_time*1000:.2f} ms ± {np.std(v)*1000:.2f} | Per input: {input_time*1000:.3f} ms"
        )
            
if __name__ == "__main__":
    main()