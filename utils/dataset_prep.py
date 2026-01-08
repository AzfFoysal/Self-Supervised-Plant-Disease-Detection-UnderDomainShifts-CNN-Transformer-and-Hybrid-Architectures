import os
import zipfile
from torchvision import datasets
from torchvision.datasets import ImageFolder

# Attempt to use Kaggle API if available for downloading datasets
try:
    from kaggle import api
except ImportError:
    api = None

def download_datasets(data_dir="data"):
    """Download PlantVillage and PlantDoc datasets (requires Kaggle API credentials)."""
    os.makedirs(data_dir, exist_ok=True)
    if api is not None:
        print("Downloading PlantVillage dataset...")
        api.dataset_download_files('vipoooool/new-plant-diseases-dataset', path=data_dir, quiet=False)
        pv_zip = os.path.join(data_dir, 'new-plant-diseases-dataset.zip')
        if os.path.exists(pv_zip):
            with zipfile.ZipFile(pv_zip, 'r') as zf:
                zf.extractall(os.path.join(data_dir, 'PlantVillage'))
            os.remove(pv_zip)
        print("Downloading PlantDoc dataset...")
        api.dataset_download_files('abdulhasibuddin/plant-doc-dataset', path=data_dir, quiet=False)
        pd_zip = os.path.join(data_dir, 'plant-doc-dataset.zip')
        if os.path.exists(pd_zip):
            with zipfile.ZipFile(pd_zip, 'r') as zf:
                zf.extractall(os.path.join(data_dir, 'PlantDoc'))
            os.remove(pd_zip)
    else:
        print("Kaggle API not available. Please download datasets manually.")

def load_plantvillage_datasets(root_dir="data/PlantVillage"):
    """Load PlantVillage train/valid/test sets as ImageFolder datasets."""
    train_set = ImageFolder(os.path.join(root_dir, "train"))
    val_set = ImageFolder(os.path.join(root_dir, "valid"))
    test_set = ImageFolder(os.path.join(root_dir, "test"))
    return train_set, val_set, test_set

def load_plantdoc_dataset(root_dir="data/PlantDoc"):
    """Load PlantDoc dataset and align classes to PlantVillage classes."""
    pd_path = os.path.join(root_dir, "plantdoc")
    pd_dataset = ImageFolder(pd_path)
    # Optionally map class names if needed (assuming they mostly match PlantVillage)
    mapping = {}  # e.g., {"Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust"} if needed
    if mapping:
        pd_dataset.class_to_idx = {mapping.get(c, c): idx for c, idx in pd_dataset.class_to_idx.items()}
        pd_dataset.classes = [mapping.get(c, c) for c in pd_dataset.classes]
    # Filter out classes not in PlantVillage
    pv_classes = set(datasets.ImageFolder(os.path.join(root_dir.replace("PlantDoc", "PlantVillage"), "train")).classes)
    indices = [i for i, (_, label) in enumerate(pd_dataset.samples) if pd_dataset.classes[label] in pv_classes]
    if len(indices) < len(pd_dataset):
        from torch.utils.data import Subset
        pd_dataset = Subset(pd_dataset, indices)
    return pd_dataset

class HybridDataset:
    """Dataset wrapper to return paired (cnn_image, vit_image) for hybrid model."""
    def __init__(self, base_dataset, transform_cnn, transform_vit):
        self.base_dataset = base_dataset
        self.transform_cnn = transform_cnn
        self.transform_vit = transform_vit
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        image = self.base_dataset.loader(path)
        img_cnn = self.transform_cnn(image)
        img_vit = self.transform_vit(image)
        return (img_cnn, img_vit), label


from pathlib import Path
import json
from torch.utils.data import Subset

def load_plantvillage_from_splits(data_dir: str, split_json: str, transform_train=None, transform_val=None, transform_test=None):
    """Load PlantVillage from a single raw ImageFolder plus a split JSON produced by scripts/create_splits.py.

    Expected structure:
      data/PlantVillage/raw/<class_name>/*.jpg

    split_json must contain keys: classes, train_idx, val_idx, test_idx.
    """
    data_dir = Path(data_dir)
    raw_dir = data_dir / "PlantVillage" / "raw"
    ds = ImageFolder(str(raw_dir))
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    # sanity: ensure class order matches
    if "classes" in split and split["classes"] != ds.classes:
        raise ValueError(
            "Class order mismatch between raw dataset and split JSON. "
            "Re-create splits on the same raw folder or ensure identical class folder names."
        )

    train_ds = Subset(ImageFolder(str(raw_dir), transform=transform_train), split["train_idx"])
    val_ds   = Subset(ImageFolder(str(raw_dir), transform=transform_val),   split["val_idx"])
    test_ds  = Subset(ImageFolder(str(raw_dir), transform=transform_test),  split["test_idx"])
    return train_ds, val_ds, test_ds, ds.classes
