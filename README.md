# Geometry-Aware Directional Alignment for Coherent Model Merging

Official code for **MDA**, accepted at **NeurIPS 2026**.

## Installation

```bash
conda create --name concrete python=3.11
conda activate concrete
pip install -r requirements.txt
```

## Usage

### CLIP (TA and AM)

```bash
python scripts/MDA.py model=ViT-B-32
python scripts/MDA.py model=ViT-B-16
python scripts/MDA.py model=ViT-L-14
```

### CLIP generalization

```bash
python scripts/MDA_generalization.py --config-name clip_generalization_exp1 model=ViT-B-32
python scripts/MDA_generalization.py --config-name clip_generalization_exp2 model=ViT-B-32
```

### Flan-T5 (TA)

```bash
python scripts/MDA_flant5.py models=flan-t5-base
python scripts/MDA_flant5.py models=flan-t5-base peft=lora-16
```

Feature-space adaptation uses prediction argmax pseudo-labels for alignment and prototypes. Ground-truth labels are used only for evaluation.

## Hyperparameters

Shared settings: `config/mda/default.yaml`. Override them from the command line:

```bash
python scripts/MDA.py model=ViT-B-32 mda.dimension=2400 mda.alignment_alpha=0.8 mda.rotation_weight=0.2
```

- `mda.k`: SVD retention fraction. Default `null` uses **1 / number of merged tasks**; generalization counts seen tasks only. Set a fraction in `(0, 1]` for ablations.
- `mda.dimension`: parameter-space ETF reference class dimension; default `2400`.
- `mda.parameter_alpha`: reconstruction/ETF blending weight; default `1.0`.
- `mda.lambda_init`: initial merging coefficient; default `1.0`.
- `mda.alignment_alpha`: feature alignment weight; default `1.0`.
- `mda.rotation_weight`: Procrustes penalty multiplier; default `0.0`. The effective penalty weight is `rotation_weight * current_alignment_alpha`.
- `mda.alignment_warmup_epochs`: alignment warmup iterations; default `0`.
- `mda.lr`, `mda.betas`, `mda.weight_decay`: Adam settings; defaults `0.001`, `[0.9, 0.999]`, and `0.0`.
- `mda.epochs`: total adaptation iterations; default `1000`.
- `mda.batch_size`, `mda.prototype_batch_size`: adaptation/prototype batch sizes; defaults `16` and `4`.
- `mda.eval_interval`: evaluation interval; default `500` iterations.
- `mda.resume`: path to an `alignment.pt` checkpoint; default `null`.

Experiment settings:

- `config/mda/generalization.yaml`: warmup `300`, rotation multiplier `0.2`, evaluation interval `200`.
- `config/mda/flan_t5.yaml`: learning rate `0.0001`.

## Code Base

- [AdaMerging](https://github.com/EnnengYang/AdaMerging)
- [Task Singular Vectors](https://github.com/AntoAndGar/task_singular_vectors)
