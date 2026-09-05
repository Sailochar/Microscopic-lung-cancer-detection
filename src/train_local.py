"""
train_local.py  -  Step 3: Train one hospital model locally
============================================================
Fixes vs Review-2:
  1. WeightedRandomSampler + class-weighted CrossEntropyLoss -> fixes lung_n collapse
  2. Best checkpoint selected by validation MACRO-F1, not accuracy
  3. Mixed precision (torch.cuda.amp) + cudnn.benchmark
  4. Cosine LR schedule + early stopping on macro-F1 plateau

Run from project root:
    python src/train_local.py --hospital 1 --epochs 20 --batch_size 32
    python src/train_local.py --hospital 2 --epochs 20 --batch_size 32
    python src/train_local.py --hospital 3 --epochs 20 --batch_size 32

Saves checkpoints to:  checkpoints/hospital{N}_model.pth
Saves history to:      checkpoints/hospital{N}_history.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler, autocast

# Allow import of sibling modules in src/
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import get_hospital_loaders

CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def train_one_hospital(hospital_id: int,
                        epochs: int = 20,
                        batch_size: int = 32,
                        lr: float = 1e-4,
                        weight_decay: float = 1e-4,
                        patience: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"\n{'='*62}")
    print(f"  Hospital {hospital_id}  |  device={device}")
    print(f"{'='*62}")

    train_loader, val_loader, class_weights, class_names = get_hospital_loaders(
        hospital_id, batch_size=batch_size
    )
    print(f"  Classes      : {class_names}")
    print(f"  Weights      : {[f'{w:.3f}' for w in class_weights.tolist()]}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model     = get_model(num_classes=len(class_names), freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_val_f1    = -1.0
    epochs_no_imp  = 0
    history        = []
    ckpt_path      = os.path.join(CKPT_DIR, f"hospital{hospital_id}_model.pth")
    history_path   = os.path.join(CKPT_DIR, f"hospital{hospital_id}_history.json")

    for epoch in range(1, epochs + 1):
        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                out  = model(imgs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * imgs.size(0)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_loss = run_loss / total
        train_acc  = correct / total

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=(device.type == "cuda")):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                preds        = out.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss     /= val_total
        val_acc       = val_correct / val_total
        val_macro_f1  = f1_score(all_labels, all_preds,
                                  average="macro", zero_division=0)
        scheduler.step()
        dt = time.time() - t0

        print(f"  Ep {epoch:02d}/{epochs}  "
              f"train {train_loss:.4f}/{train_acc:.3f}  "
              f"val {val_loss:.4f}/{val_acc:.3f}  "
              f"macro-F1={val_macro_f1:.3f}  {dt:.1f}s")

        history.append({"epoch": epoch,
                         "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss,   "val_acc": val_acc,
                         "val_macro_f1": val_macro_f1})

        # ── CHECKPOINT / EARLY STOP ────────────────────────────────────────
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_no_imp = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> new best macro-F1={best_val_f1:.3f}, saved {ckpt_path}")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Hospital {hospital_id} done. Best val macro-F1 = {best_val_f1:.3f}")
    print(f"  Checkpoint : {ckpt_path}")
    return best_val_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital",   type=int,   required=True, choices=[1, 2, 3])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()

    train_one_hospital(
        args.hospital,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
