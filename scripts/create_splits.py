import os
import argparse
import json
from pathlib import Path

import numpy as np
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit

from utils.repro import set_global_seed
from utils.stratified_split import stratified_subsample


def make_train_val_test_indices(labels, seed: int = 42, train_frac: float = 0.8, val_frac: float = 0.1):
    """Return (train_idx, val_idx, test_idx) with stratified 80/10/10."""
    assert abs(train_frac + 2 * val_frac - 1.0) < 1e-6
    labels = np.asarray(labels)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - train_frac), random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(labels)), labels))

    temp_labels = labels[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(np.zeros(len(temp_labels)), temp_labels))

    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=True)

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "PlantVillage" / "raw"
    out_dir = data_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Expected PlantVillage raw directory at {raw_dir}. "
            "If you already have train/valid/test folders, you can skip this script."
        )

    ds = datasets.ImageFolder(str(raw_dir))
    labels = np.array(ds.targets)

    train_idx, val_idx, test_idx = make_train_val_test_indices(labels, seed=args.seed)

    split_payload = {
        "seed": args.seed,
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "classes": ds.classes,
        "class_to_idx": ds.class_to_idx,
    }

    (out_dir / f"plantvillage_split_seed{args.seed}.json").write_text(json.dumps(split_payload, indent=2))

    # Labeled subsets from TRAIN ONLY
    class Dummy:
        pass
    dummy = Dummy()
    dummy.targets = labels[train_idx].tolist()
    dummy.samples = [ds.samples[i] for i in train_idx]

    idx_25_rel = stratified_subsample(dummy, 0.25, seed=args.seed)
    idx_10_rel = stratified_subsample(dummy, 0.10, seed=args.seed)

    # Convert relative indices back to original indices
    train_idx_arr = np.array(train_idx)
    idx_25 = train_idx_arr[np.array(idx_25_rel)].tolist()
    idx_10 = train_idx_arr[np.array(idx_10_rel)].tolist()

    subset_payload = {
        "seed": args.seed,
        "train_subset_25": idx_25,
        "train_subset_10": idx_10,
    }
    (out_dir / f"plantvillage_subsets_seed{args.seed}.json").write_text(json.dumps(subset_payload, indent=2))

    print(f"Wrote splits: {out_dir / f'plantvillage_split_seed{args.seed}.json'}")
    print(f"Wrote subsets: {out_dir / f'plantvillage_subsets_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
