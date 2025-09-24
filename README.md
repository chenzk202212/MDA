# From Coefficients to Directions: Rethinking Model Merging with Directional Alignment

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

Generalization experiments

Evaluate our method (MDA) —— both TA and AM

```bash
# CLIP-ViT-B/32 generation_exp1
python scripts/MDA.py \
    --config-name clip_generalization_exp1 \
    exp_name=generalization_exp1 \
    model=ViT-B-32
# CLIP-ViT-B/32 generation_exp2
python scripts/MDA.py \
    --config-name clip_generalization_exp2 \
    exp_name=generalization_exp2 \
    model=ViT-B-32
```

## Flan-T5 Experiments

Evaluate our method (MDA)

```bash
# flan-t5-base
python scripts/MDA_flant5.py.py models=flan-t5-base
python scripts/MDA_flant5.py.py models=flan-t5-base peft=lora-16
```

## Acknowledgements

Some of the code in this repository is based on https://github.com/EnnengYang/AdaMerging.
