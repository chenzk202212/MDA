# Geometry-Aware Directional Alignment for Coherent Model Merging

Our paper has been accepted at **NeurIPS 2026**!

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
python scripts/MDA.py model=ViT-B-32 mda.k=0.125 mda.dimension=2400
python scripts/MDA.py mda.k=0.0625 mda.parameter_alpha=0.8
python scripts/MDA.py mda.alignment_alpha=0.8 mda.rotation_weight=0.2 mda.lr=0.0005 mda.epochs=500
python scripts/MDA_generalization.py --config-name clip_generalization_exp2 mda.k=0.0625 mda.dimension=1200
python scripts/MDA_flant5.py models=flan-t5-base peft=lora-16 mda.k=0.125 mda.dimension=2400
```

- `mda.k`: SVD retention fraction, **not an absolute rank**. The default `null`
  automatically uses `k = 1 / T`, where `T` is the actual number of task vectors
  being merged. Each layer retains `max(1, floor(min(layer.shape) / T))` singular
  components. Thus 8, 14, and 20 merged tasks use ratios `1/8`, `1/14`, and `1/20`;
  generalization uses the count of seen tasks being merged, excluding unseen tasks.
  For ablations, explicitly set a fraction in `(0, 1]`, e.g. `mda.k=0.0625`.
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

## Feature-space rotation training

Both CLIP entry points optimize the merging coefficients and task-specific rotation
matrices. Rotated features feed the frozen classification heads during training
and evaluation. After every Adam step, each matrix is projected onto `SO(d)`
(orthogonal with determinant `+1`). This projection adds a per-task SVD cost.
Unseen tasks in generalization experiments use the merged encoder without a
task-specific rotation; no unseen-task labels are used for adaptation.

The Procrustes target follows the same convention as `F.linear(features, R)`,
i.e. `features @ R.T`. It is fit separately per task, using unrotated class means
for classes present in that batch. `mda.rotation_weight=0.2` enables its penalty;
the default CLIP value `0.0` disables that penalty but still trains rotations
through entropy/alignment losses and enforces `SO(d)` after each update.
The alignment target is fixed throughout training. When feature dimension is
below `C-1`, the normalized truncated simplex target is an approximation, not an
exact simplex ETF.

Each evaluation interval and the final training iteration save `alignment.pt`
alongside the legacy coefficient-only `.pt` file in the result directory. The
alignment checkpoint includes learned coefficients, rotations and task order,
Adam state, completed iteration count, and ETF targets. Resume with the same
model and task configuration, for example:

```bash
python scripts/MDA.py mda.resume=results/ViT-B-32/layer_wise_adamerging/alignment.pt mda.epochs=1500
```

`mda.epochs` is the total desired iteration count, not the number of extra
iterations after resuming. Legacy coefficient-only files cannot restore a
rotation training session. Data-loader/RNG state is not checkpointed, so a
resumed run does not promise the same minibatch sequence as an uninterrupted run.
The tests above exercise real CPU tensor gradients, both entry-point loops with
tiny models/data, proper Procrustes rotations, prediction consistency, and
checkpoint restoration; they do not reproduce full CLIP benchmark accuracies.

## Acknowledgements

Our code builds on the following codebases:

- [AdaMerging](https://github.com/EnnengYang/AdaMerging)
- [Task Singular Vectors](https://github.com/AntoAndGar/task_singular_vectors)
