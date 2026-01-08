# Reproducible Codebase — Self-Supervised Plant Disease Detection under Domain Shifts

This repository provides **fully runnable training/evaluation code**, a **final Jupyter notebook**, and **scripts** to reproduce the thesis experiments:

> *Self-Supervised Plant Disease Detection under Domain Shifts: Comparative Evaluation of CNN, Transformer, and Hybrid Architectures*

## 0) Hardware & OS
- Tested for **RunPod GPU** (e.g., RTX 3090) and Ubuntu
- Python **3.10** recommended

## 1) Environment (Pinned)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **CUDA note (RunPod):** install the CUDA-enabled PyTorch build that matches your base image if needed.

## 2) Datasets
This repo expects datasets under `data/` (dataset folders are **not** included).

### 2.1 PlantVillage (38 classes)
Download the PlantVillage “New Plant Diseases Dataset (Augmented)” from Kaggle:
- Kaggle: `vipoooool/new-plant-diseases-dataset`

You may have either:
- **Option A (recommended):** a single `PlantVillage/raw/` folder with class subfolders (no split). Then run our deterministic split script.
- **Option B:** Kaggle-provided split folders (`train/valid/test`). In that case, you can skip split creation.

Expected layouts:

**Option A**
```
data/PlantVillage/raw/<class_name>/*.jpg
```

**Option B**
```
data/PlantVillage/train/<class_name>/*.jpg
.../valid/<class_name>/*.jpg
.../test/<class_name>/*.jpg
```

### 2.2 PlantDoc (cross-domain, 27 classes)
Download from Kaggle:
- Kaggle: `abdulhasibuddin/plant-doc-dataset`

Expected:
```
data/PlantDoc/plantdoc/<class_name>/*.jpg
```

### 2.3 Class alignment (PlantDoc → PlantVillage)
The evaluation script filters PlantDoc samples to the **intersection** of PlantVillage classes.

## 3) Deterministic splits & labeled subsets (seed=42)
If you use **Option A** for PlantVillage:
```bash
python scripts/create_splits.py --data_dir data --seed 42
```
This writes `data/splits/` including:
- 80/10/10 train/val/test split indices
- stratified labeled subsets: **25%** and **10%** (seed=42)

## 4) Reproducing thesis results (Tables/Figures)
All commands below assume `data/` is prepared.

### Table 6.1 (PlantVillage test — full data)
Train each model (full data):
```bash
python scripts/train.py --model resnet --data_dir data --epochs 50 --fraction 1.0 --seed 42
python scripts/train.py --model vit    --data_dir data --epochs 50 --fraction 1.0 --seed 42
python scripts/train.py --model hybrid --data_dir data --epochs 50 --fraction 1.0 --seed 42
```
Evaluate:
```bash
python scripts/evaluate.py --model resnet --weights checkpoints/resnet_full_seed42.pth --data_dir data --dataset plantvillage
python scripts/evaluate.py --model vit    --weights checkpoints/vit_full_seed42.pth    --data_dir data --dataset plantvillage
python scripts/evaluate.py --model hybrid --weights checkpoints/hybrid_full_seed42.pth --data_dir data --dataset plantvillage
```

### Table 6.2 (Low-label: 25% and 10%)
```bash
python scripts/train.py --model resnet --data_dir data --epochs 100 --fraction 0.25 --seed 42
python scripts/train.py --model resnet --data_dir data --epochs 100 --fraction 0.10 --seed 42
# repeat for vit + hybrid
```

### Table 6.3 (Robustness: blur/noise)
```bash
python scripts/evaluate.py --model hybrid --weights checkpoints/hybrid_full_seed42.pth --data_dir data --dataset plantvillage --corruption blur
python scripts/evaluate.py --model hybrid --weights checkpoints/hybrid_full_seed42.pth --data_dir data --dataset plantvillage --corruption noise
```

### Table 6.4 (PlantDoc cross-domain — average of 4 seeds)
```bash
python scripts/evaluate.py --model hybrid --weights checkpoints/hybrid_full_seed42.pth --data_dir data --dataset plantdoc --plantdoc_seeds 0 1 2 3 --average
```

### Figure 6.3 (Confusion matrix)
Run the notebook section **“Confusion Matrix (PlantVillage Test)”**.

### Figure 6.4 (Grad-CAM)
```bash
python scripts/visualize_cam.py --model hybrid --weights checkpoints/hybrid_full_seed42.pth --image path/to/sample.jpg --output gradcam.png
```

## 5) Main Notebook (all inline)
Open and run:
- `notebooks/final_thesis_reproduction.ipynb`

The notebook includes:
- training (full + low-label)
- evaluation tables
- corruption robustness
- PlantDoc averaging (4 seeds)
- confusion matrix
- Grad-CAM visualizations

## Notes
- Seeds are enforced via `utils/repro.py`.
- Minor numerical differences can still occur across GPUs/drivers, but protocol and splits are fixed.
