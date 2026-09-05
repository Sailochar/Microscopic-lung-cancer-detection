"""
evaluate.py  -  Step 4: Sanity-check a single trained checkpoint
=================================================================
Run from project root BEFORE running FedProx.
Confirm lung_n recall > 0.5 for all hospitals.

Usage:
    python src/evaluate.py --checkpoint checkpoints/hospital1_model.pth --name "Hospital 1"
    python src/evaluate.py --checkpoint checkpoints/hospital2_model.pth --name "Hospital 2"
    python src/evaluate.py --checkpoint checkpoints/hospital3_model.pth --name "Hospital 3"

Saves confusion-matrix PNG to results/.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import get_test_loader

NUM_CLASSES  = 3
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
CKPT_DIR     = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    model = get_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def run_inference(model, loader, device):
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def evaluate_checkpoint(checkpoint_path: str, name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*62}")
    print(f"  Evaluating : {name}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Device     : {device}")
    print("=" * 62)

    loader, class_names = get_test_loader(batch_size=32)
    model = load_checkpoint(checkpoint_path, device)
    y_true, y_pred, _ = run_inference(model, loader, device)

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                 target_names=class_names, zero_division=0))

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} — Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    safe_name = name.replace(" ", "_").lower()
    out_path  = os.path.join(RESULTS_DIR, f"{safe_name}_cm.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {out_path}")

    # ── lung_n recall gating check ────────────────────────────────────────
    if "lung_n" in class_names:
        idx  = class_names.index("lung_n")
        mask = y_true == idx
        recall_n = (y_pred[mask] == idx).mean() if mask.any() else 0.0
        if recall_n < 0.50:
            print(f"\n  [WARNING] lung_n recall = {recall_n:.3f}  (< 0.50)")
            print("    Do NOT proceed to FedProx yet.")
            print("    Fix: more epochs or larger WeightedRandomSampler ratio.")
        else:
            print(f"\n  [OK] lung_n recall = {recall_n:.3f}.  Safe to proceed.")

    return acc, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name",       type=str, default="Model")
    args = parser.parse_args()

    # Support both absolute paths and paths relative to project root
    ckpt = args.checkpoint
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(ROOT_DIR, ckpt)

    evaluate_checkpoint(ckpt, args.name)
