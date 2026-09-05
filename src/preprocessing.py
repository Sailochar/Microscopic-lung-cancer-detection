"""
preprocessing.py  -  Step 1: Non-IID data audit + DataLoader factory
======================================================================
Run standalone to audit the Non-IID hospital split:
    python src/preprocessing.py

Import get_hospital_loaders() and get_test_loader() in other scripts.

Directory layout (paths are relative to project root):
    data/
        Hospital_1/{lung_aca, lung_n, lung_scc}/
        Hospital_2/{lung_aca, lung_n, lung_scc}/
        Hospital_3/{lung_aca, lung_n, lung_scc}/
        test/    {lung_aca, lung_n, lung_scc}/
"""

import collections
import os
import sys

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

# ── Root of the project (two levels up from src/) ─────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

HOSPITAL_PATHS = {
    1: os.path.join(DATA_ROOT, "Hospital_1"),
    2: os.path.join(DATA_ROOT, "Hospital_2"),
    3: os.path.join(DATA_ROOT, "Hospital_3"),
}
TEST_PATH = os.path.join(DATA_ROOT, "test")

IMAGE_SIZE = (224, 224)

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25,
                           saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = VAL_TRANSFORM


# ── Helpers ───────────────────────────────────────────────────────────────────
def _compute_class_weights(dataset):
    """Inverse-frequency class weights  shape=(num_classes,)."""
    counts = collections.Counter(dataset.targets)
    nc     = len(dataset.classes)
    total  = len(dataset)
    w = torch.zeros(nc)
    for c in range(nc):
        w[c] = total / (nc * max(counts.get(c, 1), 1))
    return w


def _compute_sample_weights(dataset):
    """Per-sample weights for WeightedRandomSampler."""
    cw = _compute_class_weights(dataset)
    return [cw[label].item() for label in dataset.targets]


# ── Public API ────────────────────────────────────────────────────────────────
def get_hospital_loaders(hospital_id: int,
                          batch_size: int = 32,
                          val_split: float = 0.15,
                          num_workers: int = 0,
                          use_sampler: bool = True):
    """
    Returns: train_loader, val_loader, class_weights (Tensor), class_names (list)

    WeightedRandomSampler on the training split corrects within-hospital
    class imbalance when use_sampler=True. class_weights are for nn.CrossEntropyLoss(weight=...).
    """
    path         = HOSPITAL_PATHS[hospital_id]
    full_dataset = datasets.ImageFolder(path, transform=TRAIN_TRANSFORM)
    class_names  = full_dataset.classes
    class_weights = _compute_class_weights(full_dataset)

    n_total = len(full_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val
    gen     = torch.Generator().manual_seed(42)
    train_ds, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=gen
    )
    train_ds = torch.utils.data.Subset(full_dataset, train_ds.indices)
    val_dataset = datasets.ImageFolder(path, transform=VAL_TRANSFORM)
    val_ds = torch.utils.data.Subset(val_dataset, val_indices.indices)

    if use_sampler:
        sw_full  = _compute_sample_weights(full_dataset)
        sw_train = [sw_full[i] for i in train_ds.indices]

        sampler = WeightedRandomSampler(
            weights=sw_train, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, class_weights, class_names


def get_test_loader(batch_size: int = 32, num_workers: int = 0):
    """DataLoader for the held-out test set."""
    ds = datasets.ImageFolder(TEST_PATH, transform=TEST_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, ds.classes


# ── Standalone audit ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  NON-IID DATA AUDIT")
    print("=" * 62)

    for h_id in [1, 2, 3]:
        path = HOSPITAL_PATHS[h_id]
        ds   = datasets.ImageFolder(path, transform=TEST_TRANSFORM)
        cnt  = collections.Counter(ds.targets)
        print(f"\n  Hospital {h_id}  (total={len(ds):,}):")
        for ci, cname in enumerate(ds.classes):
            n   = cnt.get(ci, 0)
            pct = 100 * n / len(ds)
            bar = "=" * (n // 80)
            print(f"    {cname:<12}: {n:5d}  ({pct:5.1f}%)  {bar}")

    ds_test = datasets.ImageFolder(TEST_PATH, transform=TEST_TRANSFORM)
    cnt = collections.Counter(ds_test.targets)
    print(f"\n  Test Set  (total={len(ds_test):,}):")
    for ci, cname in enumerate(ds_test.classes):
        n = cnt.get(ci, 0)
        print(f"    {cname:<12}: {n:5d}  ({100*n/len(ds_test):5.1f}%)")

    print("\n" + "=" * 62)
    print("  CLASS WEIGHTS  (higher = loss penalises misclassification more)")
    print("=" * 62)
    for h_id in [1, 2, 3]:
        path = HOSPITAL_PATHS[h_id]
        ds   = datasets.ImageFolder(path, transform=TEST_TRANSFORM)
        w    = _compute_class_weights(ds)
        print(f"\n  Hospital {h_id}:")
        for ci, cname in enumerate(ds.classes):
            print(f"    {cname:<12}: {w[ci]:.4f}")

    print("\n  Audit complete.")
