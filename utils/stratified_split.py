import numpy as np


def stratified_subsample(dataset, fraction: float, seed: int = 42):
    """Return indices for a stratified random subset.

    Ensures each class is sampled proportionally (at least 1 per class),
    using the dataset.targets attribute (ImageFolder provides it).
    """
    assert 0.0 < fraction <= 1.0
    rng = np.random.default_rng(seed)
    labels = np.asarray(getattr(dataset, "targets", None))
    if labels is None:
        # Fallback: infer from samples list
        labels = np.array([y for _, y in dataset.samples], dtype=int)

    indices = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        k = int(round(len(cls_idx) * fraction))
        k = max(1, k)
        selected = rng.choice(cls_idx, size=k, replace=False)
        indices.extend(selected.tolist())

    rng.shuffle(indices)
    return indices
