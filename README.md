# PrivCanFed — Privacy-Preserving Federated Learning for Lung Cancer Detection

> **Review-3 project** | EfficientNet-B0 | FedProx | Flower 1.x | PyTorch 2.4 + CUDA 11.8

---

## Project Structure

```
Major_project_1/
├── data/                        # Image data (do NOT add scripts here)
│   ├── Hospital_1/              # Non-IID: near-balanced (~36/28/36 %)
│   ├── Hospital_2/              # Non-IID: lung_n dominant (~15/70/15 %)
│   ├── Hospital_3/              # Non-IID: lung_n starved  (~44/12/44 %)
│   └── test/                    # Held-out test set (balanced ~33% each)
│
├── src/                         # All Python source files
│   ├── preprocessing.py         # Step 1 – Non-IID audit + DataLoader factory
│   ├── model.py                 # Step 2 – EfficientNet-B0 model definition
│   ├── train_local.py           # Step 3 – Per-hospital local training
│   ├── evaluate.py              # Step 4 – Checkpoint sanity-check
│   ├── fedprox_simulation.py    # Step 5 – FedProx / FedAvg simulation
│   └── compare_models.py        # Step 6 – Full comparison + charts
│
├── checkpoints/                 # Saved .pth model files (auto-created)
├── results/                     # Confusion matrices, ROC, PR, bar charts
└── README.md
```

---

## Environment

| Package | Version |
|---------|---------|
| Python | 3.8 |
| PyTorch | 2.4.1+cu118 |
| torchvision | 0.19.1+cu118 |
| Flower (flwr) | 1.11.1 |
| scikit-learn | 1.3.0 |
| GPU | NVIDIA RTX 3050 (4 GB) |
| CUDA | 11.8 |

Install dependencies:
```bash
pip install "flwr[simulation]" scikit-learn matplotlib seaborn pandas
```

---

## Step-by-Step Run Guide

All commands are run from the **project root** (`Major_project_1/`).

### Step 1 — Audit the Non-IID split
```bash
python src/preprocessing.py
```
Prints per-hospital class counts and class weights.

### Step 2 — Model self-test (optional)
```bash
python src/model.py
```
Prints output shape and parameter count.

### Step 3 — Train each hospital locally
```bash
python src/train_local.py --hospital 1 --epochs 20 --batch_size 32
python src/train_local.py --hospital 2 --epochs 20 --batch_size 32
python src/train_local.py --hospital 3 --epochs 20 --batch_size 32
```
Saves: `checkpoints/hospital{N}_model.pth` and `checkpoints/hospital{N}_history.json`

### Step 4 — Sanity-check each local model
```bash
python src/evaluate.py --checkpoint checkpoints/hospital1_model.pth --name "Hospital 1"
python src/evaluate.py --checkpoint checkpoints/hospital2_model.pth --name "Hospital 2"
python src/evaluate.py --checkpoint checkpoints/hospital3_model.pth --name "Hospital 3"
```
**lung_n recall must be > 0.5 for all hospitals before proceeding.**

### Step 5 — FedProx simulation
```bash
# Run A: FedAvg baseline (mu=0 is mathematically plain FedAvg)
python src/fedprox_simulation.py --rounds 15 --proximal_mu 0.0 --save_name global_fedavg_model.pth

# Run B: FedProx (Review-3 deliverable)
python src/fedprox_simulation.py --rounds 15 --proximal_mu 0.01 --save_name global_fedprox_model.pth
```
If CUDA OOM → lower `--batch_size 8`.

### Step 6 — Compare all models
```bash
python src/compare_models.py
```
Generates all confusion matrices, ROC, PR curves, and comparison bar chart in `results/`.

---

## Non-IID Distribution Summary

| Hospital | lung_aca | lung_n | lung_scc | Key issue |
|----------|----------|--------|----------|-----------|
| Hospital 1 | 36.2% | 27.6% | 36.2% | Near-balanced |
| Hospital 2 | 15.2% | **69.6%** | 15.2% | lung_n dominant |
| Hospital 3 | 43.7% | **12.5%** | 43.8% | lung_n starved |
| Test Set   | 33.4% | 33.4% | 33.3% | Balanced |

Hospital 3 is the critical client. FedProx`s proximal term prevents Hospital 3
from drifting away from lung_n representation learned by Hospitals 1 and 2.

---

## Expected Results

| Model | Expected Macro-F1 |
|-------|-------------------|
| Hospital 1 (local) | ~0.85–0.92 |
| Hospital 2 (local) | ~0.75–0.88 |
| Hospital 3 (local) | ~0.70–0.85 |
| FedAvg (mu=0) | > avg of 3 locals |
| **FedProx (mu=0.01)** | **Best overall — beats FedAvg on Hospital 3** |
