import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import argparse

from utils.repro import set_global_seed

from models.resnet50_model import ResNet50Classifier
from models.vit_dino_model import ViTClassifier
from models.hybrid_cnn_vit_model import HybridCNNViTModel
from utils.augmentation import train_transform_cnn, train_transform_vit, val_transform_cnn, val_transform_vit
from utils.stratified_split import stratified_subsample
from utils.dataset_prep import HybridDataset

def get_optimizer(model, model_type="resnet"):
    # Set different learning rates for different parts of the model
    if model_type == "resnet":
        params = [
            {"params": model.model.fc.parameters(), "lr": 5e-4},      # classifier head
            {"params": model.model.parameters(), "lr": 5e-4}
        ]
    elif model_type == "vit":
        params = [
            {"params": model.classifier.parameters(), "lr": 5e-4},    # classifier head
            {"params": model.backbone.parameters(), "lr": 5e-5}
        ]
    elif model_type == "hybrid":
        params = [
            {"params": model.fc1.parameters(), "lr": 1e-4},          # fusion MLP
            {"params": model.fc2.parameters(), "lr": 1e-4},
            {"params": model.cnn_branch.parameters(), "lr": 5e-5},   # CNN backbone
            {"params": model.vit_branch.parameters(), "lr": 5e-6}    # ViT backbone
        ]
    else:
        params = [{"params": model.parameters(), "lr": 1e-4}]
    optimizer = optim.AdamW(params, weight_decay=1e-4)
    return optimizer

def train_model(model, train_loader, val_loader, model_type="resnet", epochs=50, amp: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = amp and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model = model.to(device)
    optimizer = get_optimizer(model, model_type=model_type)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss = float('inf')
    best_state = None
    patience = 5
    wait = 0
    # Phase 1: warm-up (freeze backbone, train only classifier)
    warmup_epochs = 5
    if model_type == "resnet":
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.model.fc.parameters():
            param.requires_grad = True
    elif model_type == "vit":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_type == "hybrid":
        for param in model.cnn_branch.parameters():
            param.requires_grad = False
        for param in model.vit_branch.parameters():
            param.requires_grad = False
        for param in list(model.fc1.parameters()) + list(model.fc2.parameters()):
            param.requires_grad = True
    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            if model_type == "hybrid":
                img_cnn = images[0].to(device)
                img_vit = images[1].to(device)
                labels = labels.to(device)
                outputs = model(img_cnn, img_vit)
            else:
                images = images.to(device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * labels.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        train_loss = train_loss_sum / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0
        # Validation
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if model_type == "hybrid":
                    img_cnn = images[0].to(device)
                    img_vit = images[1].to(device)
                    labels = labels.to(device)
                    outputs = model(img_cnn, img_vit)
                else:
                    images = images.to(device)
                    labels = labels.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        val_loss = val_loss_sum / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        scheduler.step()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
        # Unfreeze backbone after warm-up
        if epoch == warmup_epochs:
            if model_type == "resnet":
                for param in model.model.parameters():
                    param.requires_grad = True
            elif model_type == "vit":
                for param in model.backbone.parameters():
                    param.requires_grad = True
            elif model_type == "hybrid":
                for param in model.cnn_branch.parameters():
                    param.requires_grad = True
                for param in model.vit_branch.parameters():
                    param.requires_grad = True
    # Load best model weights
    if best_state:
        model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on PlantVillage dataset (reproducible splits).")
    parser.add_argument("--model", choices=["resnet", "vit", "hybrid"], required=True, help="Model type to train")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory of datasets")
    parser.add_argument("--split_json", type=str, default=None,
                        help="Path to split JSON from scripts/create_splits.py. If not set, defaults to data/splits/plantvillage_split_seed42.json")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of TRAINING data to use (1.0 = full, 0.25, 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training (recommended on GPU)")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing for ViT backbones (if supported)")
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=True)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.split_json is None:
        args.split_json = os.path.join(args.data_dir, "splits", f"plantvillage_split_seed{args.seed}.json")

    raw_dir = os.path.join(args.data_dir, "PlantVillage", "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Expected PlantVillage raw folder at: {raw_dir}\n"
            "Place images as data/PlantVillage/raw/<class_name>/*.jpg and run scripts/create_splits.py first."
        )
    if not os.path.isfile(args.split_json):
        raise FileNotFoundError(
            f"Split JSON not found: {args.split_json}\n"
            "Run: python scripts/create_splits.py --data_dir data --seed 42"
        )

    with open(args.split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    # Base dataset (no transform) only for indexing and target extraction
    base_ds = datasets.ImageFolder(raw_dir)

    # Sanity check class order
    if "classes" in split and split["classes"] != base_ds.classes:
        raise ValueError(
            "Class order mismatch between raw dataset and split JSON. "
            "Re-create splits on the same raw folder."
        )

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    # Subsample only training indices for low-label regimes
    if args.fraction < 1.0:
        # stratified_subsample expects dataset-like with .targets; we can wrap subset for targets
        subset_labels = [base_ds.targets[i] for i in train_idx]
        # build a tiny proxy dataset object
        class Proxy: pass
        proxy = Proxy()
        proxy.targets = subset_labels
        # selected indices relative to train_idx
        rel = stratified_subsample(proxy, args.fraction, seed=args.seed)
        train_idx = [train_idx[i] for i in rel]

    # Build datasets per model
    if args.model == "resnet":
        train_dataset = Subset(datasets.ImageFolder(raw_dir, transform=train_transform_cnn), train_idx)
        val_dataset = Subset(datasets.ImageFolder(raw_dir, transform=val_transform_cnn), val_idx)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        model = ResNet50Classifier(num_classes=len(base_ds.classes), pretrained=True)
        model_type = "resnet"
    elif args.model == "vit":
        train_dataset = Subset(datasets.ImageFolder(raw_dir, transform=train_transform_vit), train_idx)
        val_dataset = Subset(datasets.ImageFolder(raw_dir, transform=val_transform_vit), val_idx)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        model = ViTClassifier(num_classes=len(base_ds.classes), pretrained=True, grad_ckpt=args.grad_ckpt)
        model_type = "vit"
    else:
        base_train = Subset(base_ds, train_idx)
        base_val = Subset(base_ds, val_idx)
        train_dataset = HybridDataset(base_train, transform_cnn=train_transform_cnn, transform_vit=train_transform_vit)
        val_dataset = HybridDataset(base_val, transform_cnn=val_transform_cnn, transform_vit=val_transform_vit)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        model = HybridCNNViTModel(num_classes=len(base_ds.classes), grad_ckpt=args.grad_ckpt)
        model_type = "hybrid"

    model_trained, history = train_model(
        model, train_loader, val_loader, model_type=model_type, epochs=args.epochs, amp=args.amp
    )

    frac_tag = "full" if args.fraction >= 1.0 else f"{int(args.fraction*100)}pct"
    save_path = os.path.join(args.out_dir, f"{args.model}_{frac_tag}_seed{args.seed}.pth")
    torch.save(model_trained.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
