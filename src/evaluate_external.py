"""Evaluate a checkpoint on a separately collected ImageFolder dataset.

The external dataset may use the project folders or these known aliases:
    lung_aca/aca_bd, lung_n/nor, lung_scc/scc_bd

Run from the project root:
    python src/evaluate_external.py --data_root path/to/external_data \
        --checkpoint checkpoints/global_fedprox_model.pth
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import EXTERNAL_CLASS_NAMES, VAL_TRANSFORM, load_external_dataset

EXPECTED_CLASSES = EXTERNAL_CLASS_NAMES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    model = get_model(num_classes=len(EXPECTED_CLASSES),
                      freeze_backbone=False, pretrained=False)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="External ImageFolder root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="results/external_confusion_matrix.png")
    parser.add_argument("--calibration", default="")
    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(
            f"External dataset not found: {args.data_root}. "
            "Create folders lung_aca, lung_n, and lung_scc first."
        )

    dataset = load_external_dataset(args.data_root, VAL_TRANSFORM)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = load_checkpoint(args.checkpoint)
    logit_bias = torch.zeros(len(EXPECTED_CLASSES), device=DEVICE)
    if args.calibration:
        with open(args.calibration, "r") as calibration_file:
            calibration = json.load(calibration_file)
        if calibration.get("classes") != EXPECTED_CLASSES:
            raise ValueError("Calibration classes do not match external classes.")
        logit_bias = torch.tensor(calibration["bias"], device=DEVICE)
    labels, predictions = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            views = [images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2])]
            logits = torch.stack([model(view) for view in views]).mean(dim=0)
            logits = logits + logit_bias
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(targets.tolist())

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    print(f"Device: {DEVICE}")
    print(f"Classes: {dataset.classes}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(labels, predictions,
                                target_names=EXPECTED_CLASSES,
                                zero_division=0))

    matrix = confusion_matrix(labels, predictions,
                              labels=list(range(len(EXPECTED_CLASSES))))
    output_path = os.path.join(ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=EXPECTED_CLASSES, yticklabels=EXPECTED_CLASSES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("External Dataset Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix: {output_path}")


if __name__ == "__main__":
    main()