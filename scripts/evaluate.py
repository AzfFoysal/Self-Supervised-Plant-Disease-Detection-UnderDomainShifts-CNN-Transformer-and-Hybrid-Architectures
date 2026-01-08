import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from PIL import Image
from torchvision.transforms.functional import gaussian_blur

from models.resnet50_model import ResNet50Classifier
from models.vit_dino_model import ViTClassifier
from models.hybrid_cnn_vit_model import HybridCNNViTModel
from utils.augmentation import val_transform_cnn, val_transform_vit
from utils.dataset_prep import HybridDataset
from utils.metrics import compute_metrics
from utils.repro import set_global_seed


def corrupt_image(img: Image.Image, blur: bool = False, noise: bool = False) -> Image.Image:
    if blur:
        img = gaussian_blur(img, kernel_size=[5,5], sigma=[1.0,1.0])
    if noise:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0, 0.05, arr.shape), 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))
    return img


class CorruptDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, blur=False, noise=False, transform=None):
        self.base_dataset = base_dataset
        self.blur = blur
        self.noise = noise
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        img = self.base_dataset.loader(path)
        img = corrupt_image(img, blur=self.blur, noise=self.noise)
        if self.transform:
            img = self.transform(img)
        return img, label


def _predict(model, batch, model_type: str, device):
    if model_type == "hybrid":
        (x_cnn, x_vit), y = batch
        x_cnn = x_cnn.to(device)
        x_vit = x_vit.to(device)
        y = y.to(device)
        logits = model(x_cnn, x_vit)
    else:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
    return logits, y


def evaluate_model(model, loader, model_type: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits, y = _predict(model, batch, model_type, device)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return compute_metrics(y_true, y_pred)


def evaluate_hybrid_corruption(model, data_dir: str, blur: bool = False, noise: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    test_dir = os.path.join(data_dir, "PlantVillage", "test")
    base = datasets.ImageFolder(test_dir)

    correct = 0
    total = 0
    with torch.no_grad():
        for path, label in base.samples:
            img = Image.open(path).convert("RGB")
            img = corrupt_image(img, blur=blur, noise=noise)
            x_cnn = val_transform_cnn(img).unsqueeze(0).to(device)
            x_vit = val_transform_vit(img).unsqueeze(0).to(device)
            logits = model(x_cnn, x_vit)
            pred = logits.argmax(dim=1).item()
            correct += int(pred == label)
            total += 1
    return correct / max(1, total)


def load_model(model_name: str, weights_path: str):
    num_classes = 38
    if model_name == "resnet":
        model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
        model_type = "resnet"
    elif model_name == "vit":
        model = ViTClassifier(num_classes=num_classes, pretrained=False)
        model_type = "vit"
    else:
        model = HybridCNNViTModel(num_classes=num_classes, pretrained=False)
        model_type = "hybrid"

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    return model, model_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "vit", "hybrid"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", choices=["plantvillage", "plantdoc"], default="plantvillage")
    parser.add_argument("--corruption", choices=["none", "blur", "noise"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plantdoc_seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--average", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=True)
    model, model_type = load_model(args.model, args.weights)

    
    if args.dataset == "plantvillage":
        if args.split_json is None:
            args.split_json = os.path.join(args.data_dir, "splits", f"plantvillage_split_seed{args.seed}.json")
        raw_dir = os.path.join(args.data_dir, "PlantVillage", "raw")
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"Expected raw PlantVillage at {raw_dir}.")
        if not os.path.isfile(args.split_json):
            raise FileNotFoundError(f"Split JSON not found: {args.split_json}")

        import json
        with open(args.split_json, "r", encoding="utf-8") as f:
            split = json.load(f)
        base_ds = datasets.ImageFolder(raw_dir)

        test_idx = split["test_idx"]

        if args.model == "hybrid":
            base_test = Subset(base_ds, test_idx)
            test_dataset = HybridDataset(base_test, transform_cnn=val_transform_cnn, transform_vit=val_transform_vit)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        else:
            transform = val_transform_vit if args.model == "vit" else val_transform_cnn
            test_dataset = Subset(datasets.ImageFolder(raw_dir, transform=transform), test_idx)
            test_loader = DataLoader(test_dataset, batch_size=(32 if args.model=="vit" else 64), shuffle=False, num_workers=4, pin_memory=True)

        # Apply corruption if specified
        if args.corruption != "none":
            if args.model == "hybrid":
                acc = evaluate_hybrid_corruption(model, blur=(args.corruption=="blur"), noise=(args.corruption=="noise"),
                                                data_dir=args.data_dir, split_json=args.split_json, seed=args.seed)
                print(f"Hybrid model accuracy on {args.corruption} PlantVillage test: {acc*100:.2f}%")
                exit(0)
            else:
                # CorruptDataset expects an ImageFolder-like .samples; build from raw and then subset via indices list
                base_test_raw = datasets.ImageFolder(raw_dir)
                # create a thin wrapper with only samples in test_idx
                class Wrapper(torch.utils.data.Dataset):
                    def __init__(self, base, indices):
                        self.base=base
                        self.indices=indices
                        self.samples=[base.samples[i] for i in indices]
                        self.loader=base.loader
                    def __len__(self): return len(self.indices)
                    def __getitem__(self, idx):
                        p,l = self.samples[idx]
                        return p,l
                w = Wrapper(base_test_raw, test_idx)
                test_dataset = CorruptDataset(w,
                                              blur=(args.corruption=="blur"), noise=(args.corruption=="noise"),
                                              transform=(val_transform_vit if args.model=="vit" else val_transform_cnn))
                test_loader = DataLoader(test_dataset, batch_size=(32 if args.model=="vit" else 64), shuffle=False, num_workers=4, pin_memory=True)

        acc, prec, rec, f1 = evaluate_model(model, test_loader, model_type=model_type)
        print(f"Accuracy: {acc*100:.2f}%, Precision (macro): {prec*100:.2f}%, Recall (macro): {rec*100:.2f}%, F1 (macro): {f1*100:.2f}%")
else:
            tfm = val_transform_vit if args.model == "vit" else val_transform_cnn
            base = datasets.ImageFolder(test_dir, transform=tfm)
            if args.corruption != "none":
                ds = CorruptDataset(datasets.ImageFolder(test_dir),
                                    blur=args.corruption == "blur",
                                    noise=args.corruption == "noise",
                                    transform=tfm)
            else:
                ds = base

        if args.model == "hybrid" and args.corruption != "none":
            acc = evaluate_hybrid_corruption(model, args.data_dir,
                                             blur=args.corruption == "blur",
                                             noise=args.corruption == "noise")
            print(f"Hybrid accuracy ({args.corruption}) = {acc*100:.2f}%")
            return

        loader = DataLoader(ds, batch_size=(32 if args.model == "vit" else 64), shuffle=False)
        acc, prec, rec, f1 = evaluate_model(model, loader, model_type)
        print(f"Accuracy: {acc*100:.2f}% | Precision(macro): {prec*100:.2f}% | Recall(macro): {rec*100:.2f}% | F1(macro): {f1*100:.2f}%")
        return

    # PlantDoc (cross-domain)
    pd_dir = os.path.join(args.data_dir, "PlantDoc", "plantdoc")
    tfm = val_transform_vit if args.model == "vit" else val_transform_cnn
    base_pd = datasets.ImageFolder(pd_dir, transform=tfm)

    pv_classes = set(datasets.ImageFolder(os.path.join(args.data_dir, "PlantVillage", "train")).classes)
    keep_idx = [i for i, (_, y) in enumerate(base_pd.samples) if base_pd.classes[y] in pv_classes]
    pd_ds = Subset(base_pd, keep_idx)

    loader = DataLoader(pd_ds, batch_size=(32 if args.model == "vit" else 64), shuffle=False)

    if not args.average:
        acc, prec, rec, f1 = evaluate_model(model, loader, model_type)
        print(f"PlantDoc — Accuracy: {acc*100:.2f}% | Precision(macro): {prec*100:.2f}% | Recall(macro): {rec*100:.2f}% | F1(macro): {f1*100:.2f}%")
        return

    # Average over multiple seeds (deterministic data order, but noise/augmentation could depend on seed)
    runs = []
    for s in args.plantdoc_seeds:
        set_global_seed(s, deterministic=True)
        acc, prec, rec, f1 = evaluate_model(model, loader, model_type)
        runs.append([acc, prec, rec, f1])
        print(f"Seed {s}: acc={acc*100:.2f} f1={f1*100:.2f}")

    arr = np.array(runs)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    print("\nPlantDoc (mean±std over seeds)")
    print(f"Accuracy: {mean[0]*100:.2f}±{std[0]*100:.2f}%")
    print(f"Precision(macro): {mean[1]*100:.2f}±{std[1]*100:.2f}%")
    print(f"Recall(macro): {mean[2]*100:.2f}±{std[2]*100:.2f}%")
    print(f"F1(macro): {mean[3]*100:.2f}±{std[3]*100:.2f}%")


if __name__ == "__main__":
    main()
