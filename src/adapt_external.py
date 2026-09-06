"""Adapt a global checkpoint to a labelled external dataset.

This is a local adaptation step for a new hospital/domain. Raw images stay
local. The resulting checkpoint can be evaluated before replacing the global
checkpoint used by the dashboard.

Run from the project root:
    python src/adapt_external.py --data_root path/to/external_data \
        --checkpoint checkpoints/global_fedprox_model.pth \
        --save_name global_fedprox_external_adapted.pth
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import (
    EXTERNAL_CLASS_NAMES,
    TRAIN_TRANSFORM,
    VAL_TRANSFORM,
    load_external_dataset,
)

EXPECTED_CLASSES = EXTERNAL_CLASS_NAMES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_name", default="global_fedprox_external_adapted.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--normal_weight", type=float, default=2.0,
                        help="Extra loss weight for the lung_n class")
    args = parser.parse_args()

    full_train = load_external_dataset(args.data_root, TRAIN_TRANSFORM)
    if len(full_train) < 20:
        raise ValueError("At least 20 labelled external images are required.")

    validation_source = load_external_dataset(args.data_root, VAL_TRANSFORM)
    validation_count = max(1, int(len(full_train) * 0.2))
    train_count = len(full_train) - validation_count
    generator = torch.Generator().manual_seed(42)
    train_subset, validation_subset = random_split(
        range(len(full_train)), [train_count, validation_count], generator=generator
    )
    train_dataset = torch.utils.data.Subset(full_train, train_subset.indices)
    validation_dataset = torch.utils.data.Subset(
        validation_source, validation_subset.indices
    )

    counts = torch.bincount(torch.tensor([full_train.targets[i]
                                           for i in train_subset.indices]),
                            minlength=len(EXPECTED_CLASSES)).float()
    class_weights = 1.0 / counts.clamp_min(1.0)
    sample_weights = torch.tensor([
        class_weights[full_train.targets[i]].item()
        for i in train_subset.indices
    ])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    model = get_model(num_classes=len(EXPECTED_CLASSES),
                      freeze_backbone=False, pretrained=False).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, args.normal_weight, 1.0], device=DEVICE)
    )
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(DEVICE)), targets.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        actual, predicted = [], []
        with torch.no_grad():
            for images, targets in validation_loader:
                images = images.to(DEVICE)
                views = [images, torch.flip(images, dims=[3]),
                         torch.flip(images, dims=[2])]
                logits = torch.stack([model(view) for view in views]).mean(dim=0)
                predicted.extend(logits.argmax(1).cpu().tolist())
                actual.extend(targets.tolist())
        macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
        print(f"Epoch {epoch}/{args.epochs}: external val macro-F1={macro_f1:.4f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    output_path = os.path.join(ROOT_DIR, "checkpoints", args.save_name)
    torch.save(best_state, output_path)
    print(f"Saved adapted checkpoint: {output_path}")


if __name__ == "__main__":
    main()