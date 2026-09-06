"""Calibrate external-domain class decision boundaries.

Creates a small logit-bias file from a deterministic external validation split.
The calibration changes only the decision boundary; it does not alter weights.
"""

import argparse
import itertools
import json
import os
import sys

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import EXTERNAL_CLASS_NAMES, VAL_TRANSFORM, load_external_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    model = get_model(num_classes=len(EXTERNAL_CLASS_NAMES),
                      freeze_backbone=False, pretrained=False)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/external_logit_bias.json")
    args = parser.parse_args()

    dataset = load_external_dataset(args.data_root, VAL_TRANSFORM)
    validation_count = max(1, int(len(dataset) * 0.2))
    train_count = len(dataset) - validation_count
    _, validation_indices = random_split(
        range(len(dataset)), [train_count, validation_count],
        generator=torch.Generator().manual_seed(42),
    )
    validation_dataset = torch.utils.data.Subset(dataset, validation_indices.indices)
    loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    model = load_checkpoint(args.checkpoint)
    logits, labels = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            views = [images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2])]
            averaged = torch.stack([model(view) for view in views]).mean(dim=0)
            logits.append(averaged.cpu())
            labels.extend(targets.tolist())

    logits = torch.cat(logits)
    labels = torch.tensor(labels).numpy()
    best_score = -1.0
    best_bias = (0.0, 0.0, 0.0)
    for bias in itertools.product(
        [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], repeat=3
    ):
        predictions = (logits + torch.tensor(bias)).argmax(1).numpy()
        score = f1_score(labels, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_bias = bias

    output_path = os.path.join(ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump({"classes": EXTERNAL_CLASS_NAMES,
                   "bias": list(best_bias),
                   "validation_macro_f1": best_score}, output_file, indent=2)
    print(f"Best validation macro-F1: {best_score:.4f}")
    print(f"Logit bias: {list(best_bias)}")
    print(f"Saved calibration: {output_path}")


if __name__ == "__main__":
    main()
