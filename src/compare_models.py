"""
compare_models.py  -  Step 6: Compare all models side-by-side
==============================================================
Generates for EACH model:
  - Confusion matrix
  - ROC curve  (one-vs-rest, per class)
  - Precision-Recall curve

Comparison charts (all models together):
  - Grouped bar chart: Accuracy / Precision / Recall / Macro-F1
  - Per-class F1 line plot

Run from project root:
    & "C:\\Users\\sailo\\AppData\\Local\\Programs\\Python\\Python38\\python.exe" src/compare_models.py

Models evaluated (skipped if checkpoint not found):
    checkpoints/hospital1_model.pth         -> Hospital 1
    checkpoints/hospital2_model.pth         -> Hospital 2
    checkpoints/hospital3_model.pth         -> Hospital 3
    checkpoints/global_fedavg_model.pth     -> FedAvg
    checkpoints/global_fedprox_model.pth    -> FedProx
    checkpoints/global_fedprox_model_best.pth -> FedProx (Best)

Results saved to results/.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve,
    precision_recall_fscore_support, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import get_test_loader

CKPT_DIR    = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Model registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Hospital 1":    os.path.join(CKPT_DIR, "hospital1_model.pth"),
    "Hospital 2":    os.path.join(CKPT_DIR, "hospital2_model.pth"),
    "Hospital 3":    os.path.join(CKPT_DIR, "hospital3_model.pth"),
    "FedAvg":        os.path.join(CKPT_DIR, "global_fedavg_model.pth"),
    "FedProx":       os.path.join(CKPT_DIR, "global_fedprox_model.pth"),
    "FedProx(Best)": os.path.join(CKPT_DIR, "global_fedprox_model_best.pth"),
}

NUM_CLASSES  = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_model(path: str) -> nn.Module:
    model = get_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def infer(model, loader):
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def save_confusion_matrix(cm, class_names, title, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_roc(y_true, y_prob, class_names, title, path):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(7, 5))
    for i, (cname, col) in enumerate(zip(class_names, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=col, lw=2,
                  label=f"{cname}  (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_pr_curve(y_true, y_prob, class_names, title, path):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(7, 5))
    for i, (cname, col) in enumerate(zip(class_names, CLASS_COLORS)):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap      = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(r, p, color=col, lw=2,
                  label=f"{cname}  (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    loader, class_names = get_test_loader(batch_size=32)
    print(f"\n  Test set classes : {class_names}")
    print(f"  Device           : {DEVICE}\n")

    summary = {}   # model_name -> metrics dict

    for model_name, ckpt_path in MODEL_REGISTRY.items():
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {model_name:<14} — checkpoint not found")
            continue

        print(f"  Evaluating: {model_name} ...")
        model = load_model(ckpt_path)
        y_true, y_pred, y_prob = infer(model, loader)

        acc      = accuracy_score(y_true, y_pred)
        prec, rec, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        _, _, per_class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        summary[model_name] = {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1_mac,
            "per_class_f1": per_class_f1.tolist(),
        }

        print(f"    acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  macro-F1={f1_mac:.4f}")
        print(classification_report(y_true, y_pred,
                                     target_names=class_names, zero_division=0))

        safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        cm   = confusion_matrix(y_true, y_pred)

        save_confusion_matrix(
            cm, class_names,
            f"{model_name} — Confusion Matrix",
            os.path.join(RESULTS_DIR, f"{safe}_cm.png"))

        save_roc(
            y_true, y_prob, class_names,
            f"{model_name} — ROC Curve",
            os.path.join(RESULTS_DIR, f"{safe}_roc.png"))

        save_pr_curve(
            y_true, y_prob, class_names,
            f"{model_name} — Precision-Recall Curve",
            os.path.join(RESULTS_DIR, f"{safe}_pr.png"))

    if not summary:
        print("\n  No checkpoints found. Run the federated simulation first.")
        return

    # ── Grouped bar chart ─────────────────────────────────────────────
    df = pd.DataFrame(
        {m: {"Accuracy": v["acc"], "Precision": v["prec"],
              "Recall": v["rec"], "Macro-F1": v["f1"]}
         for m, v in summary.items()}
    ).T

    # Highlight FedProx bars in a different color
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = df.plot(kind="bar", figsize=(13, 6), ylim=(0, 1.08), width=0.72,
                 color=colors[:4])
    plt.title("Model Comparison — Accuracy / Precision / Recall / Macro-F1",
               fontsize=13, fontweight="bold")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    plt.legend(loc="lower right")
    plt.tight_layout()
    bar_path = os.path.join(RESULTS_DIR, "comparison_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"  Saved: {bar_path}")

    # ── Per-class F1 line plot ────────────────────────────────────────
    markers = ["o", "s", "^", "D", "P", "X"]
    plt.figure(figsize=(9, 5))
    for (mname, vals), marker in zip(summary.items(), markers):
        lw   = 2.5 if "FedProx" in mname else 1.5
        ls   = "-" if "FedProx" in mname else "--"
        plt.plot(class_names, vals["per_class_f1"],
                  marker=marker, label=mname, lw=lw, ls=ls)
    plt.title("Per-class F1 Score by Model", fontsize=13, fontweight="bold")
    plt.ylabel("F1 Score")
    plt.xlabel("Class")
    plt.ylim(0, 1.08)
    plt.legend()
    plt.tight_layout()
    f1_path = os.path.join(RESULTS_DIR, "per_class_f1.png")
    plt.savefig(f1_path, dpi=150)
    plt.close()
    print(f"  Saved: {f1_path}")

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10}"
          f" {'Recall':>10} {'Macro-F1':>10}")
    print("=" * 72)
    best_acc = max(v["acc"] for v in summary.values())
    for mname, vals in summary.items():
        star = " ★" if vals["acc"] == best_acc else ""
        print(f"  {mname:<20} {vals['acc']:>10.4f} {vals['prec']:>10.4f}"
              f" {vals['rec']:>10.4f} {vals['f1']:>10.4f}{star}")
    print("=" * 72)

    # ── Winner banner ─────────────────────────────────────────────────
    winner = max(summary, key=lambda m: summary[m]["acc"])
    fedprox_acc = summary.get("FedProx", {}).get("acc", 0.0)
    fedprox_best_acc = summary.get("FedProx(Best)", {}).get("acc", 0.0)
    fedavg_acc  = summary.get("FedAvg",  {}).get("acc", 0.0)
    best_fedprox = max(fedprox_acc, fedprox_best_acc)
    print(f"\n  🏆  WINNER: {winner}  (acc={summary[winner]['acc']:.4f})")
    if best_fedprox > 0 and fedavg_acc > 0:
        delta = best_fedprox - fedavg_acc
        if delta > 0:
            print(f"  ✅  FedProx outperforms FedAvg by {delta*100:.2f}%")
        else:
            print(f"  ⚠️   FedAvg leads by {-delta*100:.2f}% — try more rounds or higher mu")
    print(f"\n  All results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
