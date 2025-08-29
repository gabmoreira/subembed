import os
import logging
import torch

#os.environ["HF_DATASETS_OFFLINE"] = "1"
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_HUB_OFFLINE"] = "1"

from datasets import load_dataset
from torch.utils.data import Dataset
from torch import Tensor
from typing import Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CACHE_DIR = "./.cache"

class SNLI(Dataset):
    def __init__(self, max_length: int, split: str, two_way: bool) -> None:
        self.max_length = max_length
        self.split = split
        self.two_way = two_way
        
        self.dataset = load_dataset("stanfordnlp/snli", split=split, cache_dir=CACHE_DIR)
        self.dataset = self.dataset.filter(lambda x: x['label'] != -1)  
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx):
       return self.dataset[idx]
    
    def collate_fn(self, batch, tokenizer) -> Tuple[Tensor, Tensor, Tensor]:
        premises = [ex['premise'] for ex in batch]
        hypotheses = [ex['hypothesis'] for ex in batch]
        labels = [ex['label'] for ex in batch]
    
        encoded_pre = tokenizer(
            premises,
            return_tensors='pt',
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        encoded_hyp = tokenizer(
            hypotheses,
            return_tensors='pt', 
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        if self.two_way:
            # Entailment: 0, Non-Entailment: 1
            labels = (torch.tensor(labels) >= 1).long()
        else:
            # Entailment: 0, Neutral: 1, Contradiction: 2
            labels = torch.tensor(labels)

        return encoded_pre, encoded_hyp, labels
