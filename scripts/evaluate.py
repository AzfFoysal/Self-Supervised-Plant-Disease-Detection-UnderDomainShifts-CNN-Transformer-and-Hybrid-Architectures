"""Evaluation entrypoint.

Supports:
  - PlantVillage test evaluation (index-based split JSON)
  - PlantVillage robustness evaluation with corruptions:
      * Gaussian Blur: kernel=5, sigma=1.0 (thesis)
      * Gaussian Noise: N(0, 0.05) (thesis)
  - PlantDoc cross-domain evaluation averaged over 4 seeds (thesis)

Examples:
  python scripts/evaluate.py --model hybrid --weights checkpoints/hybrid_full_seed42.pth --data_dir data --seed 42
  python scripts/evaluate.py --model hybrid --weights checkpoints/hybrid_full_seed42.pth --data_dir data --seed 42 --corruption blur
  python scripts/evaluate.py --model vit --weights checkpoints/vit_full_seed42.pth --data_dir data --dataset plantdoc --plantdoc_seeds 0 1 2 3 --average
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms.functional import gaussian_blur

from models.resnet50_model import ResNet50Classifier
from models.vit_dino_model import ViTClassifier
from models.hybrid_cnn_vit_model import HybridCNNViTModel
from utils.augmentation import val_transform_cnn, val_transform_vit
from utils.dataset_prep import HybridDataset
from utils.metrics import compute_metrics
from utils.repro import set_global_seed


class PathLabelDataset(Dataset):
    """A minimal dataset that stores (path,label) pairs and applies PIL->tensor transform."""

    def __init__(self, samples: List[Tuple[str, int]], loader, transform):
        self.samples = samples
        self.loader = loader
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


class CorruptDataset(Dataset):
    """Applies deterministic corruptions then transform."""

    def __init__(self, samples: List[Tuple[str, int]], loader, transform, blur: bool, noise: bool):
        self.samples = samples
        self.loader = loader
        self.transform = transform
        self.blur = blur
        self.noise = noise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        # Corruptions (thesis): blur kernel=5 sigma=1.0; noise std=0.05
        if self.blur:
            img = gaussian_blur(img, kernel_size=[5, 5], sigma=[1.0, 1.0])
        if self.noise:
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.clip(arr + np.random.normal(0, 0.05, arr.shape), 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


def _predict(model, batch, model_type: str, device):
    if model_type == "hybrid":
        (x_cnn, x_vit), y = batch
        x_cnn = x_cnn.to(device, non_blocking=True)
        x_vit = x_vit.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x_cnn, x_vit)
    else:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
    return logits, y


@torch.no_grad()
def evaluate_model(model, loader, model_type: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    y_true, y_pred = [], []
    for batch in loader:
        logits, y = _predict(model, batch, model_type, device)
        preds = logits.argmax(dim=1)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
    return compute_metrics(y_true, y_pred)


def load_split_indices(split_json: str):
    with open(split_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["train_idx"], obj["val_idx"], obj["test_idx"], obj.get("class_to_idx")


def build_plantvillage_loader(
    raw_dir: str,
    split_json: str,
    split_name: str,
    model_kind: str,
    batch_size: int,
    num_workers: int,
    corruption: str = "none",
):
    base = datasets.ImageFolder(raw_dir)
    train_idx, val_idx, test_idx, _ = load_split_indices(split_json)
    idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    indices = idx_map[split_name]
    samples = [base.samples[i] for i in indices]

    if model_kind == "hybrid":
        # HybridDataset expects base.dataset.samples; easiest: subset base then wrap
        # We keep base without transform; HybridDataset applies two transforms.
        subset_base = Subset(base, indices)
        # Subset returns (img,label) because base has loader; but HybridDataset expects .samples.
        # So we re-create a thin ImageFolder-like object by copying samples.
        class ThinBase:
            def __init__(self, base, samples):
                self.samples = samples
                self.loader = base.loader

        thin = ThinBase(base, samples)
        ds = HybridDataset(thin, transform_cnn=val_transform_cnn, transform_vit=val_transform_vit)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return loader

    tfm = val_transform_vit if model_kind == "vit" else val_transform_cnn
    if corruption == "none":
        ds = PathLabelDataset(samples, base.loader, tfm)
    else:
        ds = CorruptDataset(
            samples,
            base.loader,
            tfm,
            blur=(corruption == "blur"),
            noise=(corruption == "noise"),
        )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader


def build_plantdoc_loader(
    plantdoc_dir: str,
    model_kind: str,
    batch_size: int,
    num_workers: int,
):
    tfm = val_transform_vit if model_kind == "vit" else val_transform_cnn
    ds = datasets.ImageFolder(plantdoc_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader


def load_model(kind: str, num_classes: int, weights_path: str):
    if kind == "resnet":
        model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
        model_type = "resnet"
    elif kind == "vit":
        model = ViTClassifier(num_classes=num_classes, pretrained=False)
        model_type = "vit"
    else:
        model = HybridCNNViTModel(num_classes=num_classes)
        model_type = "hybrid"
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model, model_type


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on PlantVillage/PlantDoc")
    parser.add_argument("--model", choices=["resnet", "vit", "hybrid"], required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split_json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["plantvillage", "plantdoc"], default="plantvillage")
    parser.add_argument("--corruption", choices=["none", "blur", "noise"], default="none")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--plantdoc_seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--average", action="store_true", help="Average PlantDoc results over --plantdoc_seeds")
    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.split_json is None:
        args.split_json = os.path.join(args.data_dir, "splits", "plantvillage_split_seed42.json")

    raw_dir = os.path.join(args.data_dir, "PlantVillage", "raw")
    plantdoc_dir = os.path.join(args.data_dir, "PlantDoc", "plantdoc")

    # Determine num_classes from PlantVillage raw (thesis uses 38)
    pv_base = datasets.ImageFolder(raw_dir)
    num_classes = len(pv_base.classes)

    model, model_type = load_model(args.model, num_classes=num_classes, weights_path=args.weights)

    if args.dataset == "plantvillage":
        bs = 32 if args.model == "vit" else 64
        loader = build_plantvillage_loader(
            raw_dir=raw_dir,
            split_json=args.split_json,
            split_name="test",
            model_kind=args.model,
            batch_size=bs,
            num_workers=args.num_workers,
            corruption=args.corruption,
        )
        acc, prec, rec, f1 = evaluate_model(model, loader, model_type)
        if args.corruption == "none":
            print(f"PlantVillage TEST | Acc: {acc*100:.2f}% | Prec(macro): {prec*100:.2f}% | Rec(macro): {rec*100:.2f}% | F1(macro): {f1*100:.2f}%")
        else:
            print(f"PlantVillage TEST ({args.corruption}) | Acc: {acc*100:.2f}% | Prec(macro): {prec*100:.2f}% | Rec(macro): {rec*100:.2f}% | F1(macro): {f1*100:.2f}%")
        return

    # PlantDoc
    bs = 32 if args.model == "vit" else 64
    if not args.average:
        loader = build_plantdoc_loader(plantdoc_dir, args.model, bs, args.num_workers)
        acc, prec, rec, f1 = evaluate_model(model, loader, model_type)
        print(f"PlantDoc | Acc: {acc*100:.2f}% | Prec(macro): {prec*100:.2f}% | Rec(macro): {rec*100:.2f}% | F1(macro): {f1*100:.2f}%")
        return

    # Average over multiple seeds (thesis requirement).
    results = []
    for s in args.plantdoc_seeds:
        set_global_seed(s)
        loader = build_plantdoc_loader(plantdoc_dir, args.model, bs, args.num_workers)
        results.append(evaluate_model(model, loader, model_type))

    arr = np.array(results, dtype=np.float32)  # shape (k,4)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    print(
        "PlantDoc (avg over seeds: "
        + ",".join(map(str, args.plantdoc_seeds))
        + ") | "
        + f"Acc: {mean[0]*100:.2f}±{std[0]*100:.2f}% | "
        + f"Prec: {mean[1]*100:.2f}±{std[1]*100:.2f}% | "
        + f"Rec: {mean[2]*100:.2f}±{std[2]*100:.2f}% | "
        + f"F1: {mean[3]*100:.2f}±{std[3]*100:.2f}%"
    )


if __name__ == "__main__":
    main()
