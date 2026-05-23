# Native Hierarchical and Compositional Representations with Subspace Embeddings

Code for the KDD 2026 paper *Native Hierarchical and Compositional Representations with Subspace Embeddings*.

## WordNet reconstruction

The WordNet noun hierarchy is fetched from `nltk.corpus`, so no manual download is required.

Train (128×128 projection matrices, ridge λ = 0.2, noun synset):

```bash
python train_wordnet_reconstruction.py --N 128 --D 128 --lbd 0.2 --synset n
```

The resulting `ReconstructionData` (optimized embeddings + optimization config) is saved to:

```
./wn_r_embeddings/{synset}_{N}x{D}_{lbd}_{group_size}/
```

Evaluate:

```bash
python eval_wordnet_reconstruction.py \
    --embed-path <path to the ReconstructionData saved above> \
    --device cuda
```

## HyperLex

Download HyperLex from https://github.com/cambridgeltl/hyperlex, then:

```bash
python eval_hyperlex.py \
    --embed-path <path to the ReconstructionData> \
    --hyperlex-path <hyperlex>/nouns-verbs/hyperlex-nouns.txt
```

## WordNet link prediction

Download the WordNet splits from https://github.com/lapras-inc/disk-embedding/tree/master/data/maxn. The directory should contain:

- `noun_closure.tsv.vocab`
- `noun_closure.tsv.train_{percent}percent`
- `noun_closure.tsv.valid`
- `noun_closure.tsv.test`
- `noun_closure.tsv.full_neg`
- `noun_closure.tsv.valid_neg`
- `noun_closure.tsv.test_neg`

Train (10% closure coverage, 128×128 projection matrices, ridge 0.2, γ⁺ = 0.8, γ⁻ = 0.1):

```bash
python train_wordnet_lp.py \
    --dataset-path <root folder of the files above> \
    --closure 0.1 \
    --gamma-pos 0.8 --gamma-neg 0.1 \
    --N 128 --D 128 --lbd 0.2
```

The resulting `LinkPredictionData` is saved to:

```
./wn_lp_embeddings/{seed}_{int(100*closure)}_wordnet_subspace_{N}x{D}_{lbd}_{group_size}/
```

Evaluate:

```bash
python eval_wordnet_lp.py \
    --embed-path <path to the LinkPredictionData saved above> \
    --dataset-path <same root folder used for training>
```

## SNLI

Train (`sentence-transformers/all-mpnet-base-v2`, 128×128 projection matrices, two-way):

```bash
python train_nli.py \
    --base-model-name sentence-transformers/all-mpnet-base-v2 \
    --N 128 --D 128 --two-way
```

The `NLITrainingData` (state dict + training config) is saved to:

```
./nli_models/{base_model}_{N}x{D}_lbd{lbd}_context{max_length}_seed{seed}[_2way][_benchmark]/
```

Evaluate:

```bash
python eval_snli.py --root ./nli_models --model-name <name generated above>
```

## Citation

```bibtex
@inproceedings{moreira2026native,
  author    = {Moreira, Gabriel and Marinho, Zita and Marques, Manuel and Costeira, Jo{\~a}o Paulo and Xiong, Chenyan},
  title     = {Native Hierarchical and Compositional Representations with Subspace Embeddings},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026},
}
```
