## 1.  setup environment 

```bash
conda create --name concrete python=3.11
conda activate concrete
pip install -r requirements.txt
```

## 2. Experiment Scripts

### CLIP experiments

Evaluate our method (MDA) —— both TA and AM

```bash
# CLIP-ViT-B/32
python scripts/MDA.py model=ViT-B-32
# CLIP-ViT-B/16
python scripts/MDA.py model=ViT-B-16
# CLIP-ViT-L/14
python scripts/MDA.py model=ViT-L-14
```

### CLIP Generalization experiments

Evaluate our method (MDA) —— both TA and AM

```bash
# CLIP-ViT-B/32 generation_exp1
python scripts/MDA_generalization.py \
    --config-name clip_generalization_exp1 \
    exp_name=generalization_exp1 \
    model=ViT-B-32
# CLIP-ViT-B/32 generation_exp2
python scripts/MDA_generalization.py \
    --config-name clip_generalization_exp2 \
    exp_name=generalization_exp2 \
    model=ViT-B-32
```

### Flan-T5 Experiments

Evaluate our method (MDA)

```bash
# flan-t5-base
python scripts/MDA_flant5.py models=flan-t5-base
python scripts/MDA_flant5.py models=flan-t5-base peft=lora-16
```

## MDA hyperparameters

All MDA experiment entry points load the shared `mda` Hydra configuration from
`config/mda/default.yaml`. Override values on the command line, for example:

```bash
python scripts/MDA.py model=ViT-B-32 mda.k=64 mda.dimension=2400
python scripts/MDA.py mda.sv_reduction=0.0625 mda.parameter_alpha=0.8
python scripts/MDA.py mda.alignment_alpha=0.8 mda.rotation_weight=0.2 mda.lr=0.0005 mda.epochs=500
python scripts/MDA_generalization.py --config-name clip_generalization_exp2 mda.k=32 mda.dimension=1200
python scripts/MDA_flant5.py models=flan-t5-base peft=lora-16 mda.k=16 mda.dimension=2400
```

- `mda.k`: optional absolute SVD rank retained per task and layer. It must be a
  positive integer no larger than the smallest dimension of each eligible layer.
- `mda.sv_reduction`: fraction of each layer's available SVD rank, default `0.125`.
  Used when `k` is `null`: `k = max(1, floor(min(layer.shape) * sv_reduction))`.
  An explicit `k` takes precedence. This preserves the original fixed `1/8`
  default; use a different fraction when running another task count.
- `mda.dimension`: ETF reference class dimension, default `2400`. This was the
  hard-coded `num_classes` in parameter-space alignment; it is not the model's
  hidden width or the number of feature-space dataset labels. Encoder feature
  width is inferred from the model, and feature-space class count from datasets.
- `mda.parameter_alpha`: reconstruction/ETF blending weight, default `1.0`.
- `mda.lambda_init`: initial merging coefficient, default `1.0`.
- `mda.alignment_alpha`, `mda.rotation_weight`, `mda.alignment_warmup_epochs`:
  feature alignment strength, rotation penalty multiplier, and warmup duration.
  The effective rotation penalty is `rotation_weight * current_alignment_alpha`.
- `mda.lr`, `mda.betas`, `mda.weight_decay`: Adam settings.
- `mda.epochs`, `mda.batch_size`, `mda.prototype_batch_size`, `mda.eval_interval`:
  adaptation iterations, adaptation/prototype batch sizes, and evaluation interval.
  Use positive integers for these counts and a nonnegative warmup duration.

`config/mda/generalization.yaml` preserves the generalization script's original
300-iteration warmup, `0.2` rotation multiplier, and 200-iteration evaluation
interval. `config/mda/flan_t5.yaml` preserves its `0.0001` learning rate. The
Flan-T5 script currently performs TA evaluation only; its commented-out AM loop
is not enabled by these options. CLIP defaults remain 1000 iterations, learning
rate `0.001`, alignment weight `1.0`, and rotation weight `0.0`.

Configuration checks (no models, datasets, or GPU required):

```bash
python -m unittest discover -s tests -v
```

## Acknowledgements

Some of the code in this repository is based on https://github.com/EnnengYang/AdaMerging.
