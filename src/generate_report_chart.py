"""Regenerate the comparison chart from the supplied held-out report values."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS = ["Hospital 1", "Hospital 2", "Hospital 3", "FedAvg", "FedProx"]
VALUES = {
    "Accuracy": [0.9987, 0.9887, 0.9973, 0.9947, 0.9993],
    "Precision": [0.9987, 0.9887, 0.9973, 0.9947, 0.9993],
    "Recall": [0.9987, 0.9887, 0.9973, 0.9947, 0.9993],
    "Macro-F1": [0.9987, 0.9887, 0.9973, 0.9947, 0.9993],
}
COLORS = ["#1686c5", "#ff8214", "#2ca02c", "#df272e"]

fig, axis = plt.subplots(figsize=(13, 6), facecolor="white")
x = np.arange(len(MODELS))
width = 0.18
for index, (label, values) in enumerate(VALUES.items()):
    axis.bar(x + (index - 1.5) * width, values, width, label=label, color=COLORS[index])
axis.set_title("Model Comparison - Accuracy / Precision / Recall / Macro-F1", fontweight="bold")
axis.set_ylabel("Score")
axis.set_ylim(0.97, 1.01)
axis.set_xticks(x, MODELS, rotation=18, ha="right")
axis.grid(axis="y", alpha=0.2)
axis.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "comparison_bar.png"), dpi=180)
plt.close(fig)
print("Regenerated results/comparison_bar.png from the supplied report values.")