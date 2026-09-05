"""
fedprox_simulation.py  -  FedProx / FedAvg Federated Simulation
================================================================
Simulates 3 hospital clients on ONE GPU using Flower 1.x start_simulation.
No separate server/client terminals needed.

FedProx vs FedAvg difference:
    total_loss = CE_loss + (mu/2) * ||w_local - w_global||^2

The proximal term prevents client models from drifting too far from the
global model when local data is highly heterogeneous (Non-IID).

FedAvg and FedProx use the same optimizer settings so the comparison isolates
the proximal term instead of comparing different training budgets.

Run from project root (use Python 3.8 which has all packages):
    # FedAvg baseline (mu=0 → plain FedAvg, 5 local epochs to induce drift)
    & "C:\\Users\\sailo\\AppData\\Local\\Programs\\Python\\Python38\\python.exe" src/fedprox_simulation.py --rounds 20 --local_epochs 5 --proximal_mu 0.0 --save_name global_fedavg_model.pth

    # FedProx (same training budget, with the proximal term)
    & "C:\\Users\\sailo\\AppData\\Local\\Programs\\Python\\Python38\\python.exe" src/fedprox_simulation.py --rounds 20 --local_epochs 5 --proximal_mu 0.01 --save_name global_fedprox_model.pth

OOM note: Default batch_size=16. Lower to 8 if you get CUDA OOM on 4 GB GPU.
"""

import argparse
import os
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast

import flwr as fl
from flwr.common import (
    Code, Context, EvaluateIns, EvaluateRes, FitIns, FitRes,
    GetParametersIns, GetParametersRes, NDArrays, Parameters,
    Scalar, Status, ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from model import get_model
from preprocessing import get_hospital_loaders, get_test_loader

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
CKPT_DIR    = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(CKPT_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Enable cuDNN auto-tuner for faster convolutions
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ── Flower Client ─────────────────────────────────────────────────────────────
class HospitalClient(fl.client.Client):
    """One federated client = one hospital."""

    def __init__(self, hospital_id: int, proximal_mu: float,
                  local_epochs: int, batch_size: int, use_sampler: bool = True):
        self.hospital_id  = hospital_id
        self.proximal_mu  = proximal_mu
        self.local_epochs = local_epochs
        self.batch_size   = batch_size
        self.use_sampler  = use_sampler

        (self.train_loader, self.val_loader,
         self.class_weights, self.class_names) = get_hospital_loaders(
            hospital_id, batch_size=batch_size, use_sampler=use_sampler
        )
        self.model  = get_model(num_classes=NUM_CLASSES,
                                 freeze_backbone=False).to(DEVICE)
        self.scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # ── Flower protocol ───────────────────────────────────────────────
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        params = [v.cpu().numpy() for v in self.model.state_dict().values()]
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(params),
        )

    def _load_params(self, parameters: NDArrays):
        sd = OrderedDict(
            {k: torch.tensor(v)
             for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(sd, strict=True)

    def fit(self, ins: FitIns) -> FitRes:
        global_params = parameters_to_ndarrays(ins.parameters)
        self._load_params(global_params)

        # ── Frozen copy of global weights for the proximal term ──────
        # Only track actual parameters (not buffers like BatchNorm stats)
        global_tensors = []
        state_dict_keys = list(self.model.state_dict().keys())
        param_names = {n for n, _ in self.model.named_parameters()}

        for key, p_np in zip(state_dict_keys, global_params):
            if key in param_names:
                global_tensors.append(
                    torch.tensor(p_np, device=DEVICE, dtype=torch.float32)
                )

        # ── Optimizer with cosine annealing ──────────────────────────
        # Keep the optimizer identical for FedAvg and FedProx.
        base_lr = 1e-4
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.local_epochs, eta_min=1e-6
        )

        # Do not apply class reweighting twice: the sampler already balances
        # classes when it is enabled.
        criterion_weight = None if self.use_sampler else self.class_weights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=criterion_weight)

        self.model.train()
        for epoch_idx in range(self.local_epochs):
            for imgs, labels in self.train_loader:
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out     = self.model(imgs)
                    ce_loss = criterion(out, labels)

                    # ── FedProx proximal regularisation ──────────────
                    # ||w_local - w_global||^2 penalises local drift.
                    # Active when proximal_mu > 0 (i.e., FedProx mode).
                    if self.proximal_mu > 0.0:
                        prox = torch.tensor(0.0, device=DEVICE)
                        for lp, gt in zip(self.model.parameters(),
                                           global_tensors):
                            prox = prox + ((lp - gt) ** 2).sum()
                        # Standard FedProx uses the sum over parameter tensors;
                        # mu is tuned against this model and local objective.
                        loss = ce_loss + (self.proximal_mu / 2.0) * prox
                    else:
                        loss = ce_loss

                self.scaler.scale(loss).backward()

                # ── Gradient clipping prevents runaway updates ────────
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                self.scaler.step(optimizer)
                self.scaler.update()

            # Step cosine scheduler once per local epoch
            scheduler.step()

        updated = [v.cpu().numpy() for v in self.model.state_dict().values()]
        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated),
            num_examples=len(self.train_loader.dataset),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self._load_params(parameters_to_ndarrays(ins.parameters))
        criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(DEVICE))

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out  = self.model(imgs)
                    loss = criterion(out, labels)
                total_loss += loss.item() * imgs.size(0)
                preds       = out.argmax(1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss     = total_loss / total
        val_acc      = correct / total
        val_macro_f1 = f1_score(all_labels, all_preds,
                                 average="macro", zero_division=0)
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(val_loss),
            num_examples=total,
            metrics={"accuracy": float(val_acc),
                      "macro_f1": float(val_macro_f1)},
        )


# ── Custom strategy: FedAvg aggregation + per-round global eval + checkpoint ──
class FedProxStrategy(FedAvg):

    def __init__(self, save_path: str, proximal_mu: float,
                  local_epochs: int, **kwargs):
        super().__init__(**kwargs)
        self.save_path    = save_path
        self.proximal_mu  = proximal_mu
        self.local_epochs = local_epochs
        self.history      = []
        self.best_acc     = 0.0
        self.test_loader, self.class_names = get_test_loader(batch_size=32)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        agg_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if agg_params is not None:
            self._save_and_eval(server_round, agg_params)
        return agg_params, metrics

    def _save_and_eval(self, rnd: int, parameters: Parameters):
        ndarrays = parameters_to_ndarrays(parameters)
        model = get_model(num_classes=NUM_CLASSES, freeze_backbone=False).to(DEVICE)
        sd = OrderedDict(
            {k: torch.tensor(v)
             for k, v in zip(model.state_dict().keys(), ndarrays)}
        )
        model.load_state_dict(sd, strict=True)

        # Always save the latest checkpoint
        torch.save(sd, self.save_path)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs  = imgs.to(DEVICE, non_blocking=True)
                preds = model(imgs).argmax(1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())

        acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        self.history.append({"round": rnd, "test_acc": acc,
                              "test_macro_f1": macro_f1})

        # Also save best-accuracy checkpoint separately
        best_suffix = self.save_path.replace(".pth", "_best.pth")
        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(sd, best_suffix)

        mode = "FedAvg" if self.proximal_mu == 0 else f"FedProx(mu={self.proximal_mu})"
        marker = " ★ BEST" if acc == self.best_acc else ""
        print(f"\n  [Round {rnd:02d}] {mode} | "
              f"test_acc={acc:.4f}  macro-F1={macro_f1:.4f}{marker}")


# ── Client factory ────────────────────────────────────────────────────────────
def make_client_fn(proximal_mu, local_epochs, batch_size, use_sampler=True):
    def client_fn(context: Context) -> fl.client.Client:
        # context.node_config["partition-id"] gives the client index (0, 1, 2)
        cid = str(context.node_config.get("partition-id", 0))
        return HospitalClient(
            hospital_id  = int(cid) + 1,   # 0→H1, 1→H2, 2→H3
            proximal_mu  = proximal_mu,
            local_epochs = local_epochs,
            batch_size   = batch_size,
            use_sampler  = use_sampler,
        )
    return client_fn


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds",       type=int,   default=20)
    parser.add_argument("--local_epochs", type=int,   default=5)
    parser.add_argument("--proximal_mu",  type=float, default=0.0)
    parser.add_argument("--batch_size",   type=int,   default=16,
                        help="Lower to 8 if CUDA OOM on 4 GB GPU")
    parser.add_argument("--use_sampler",  type=str,   default="true",
                        help="WeightedRandomSampler on/off ('true'/'false')")
    parser.add_argument("--save_name",    type=str,
                        default="global_fedprox_model.pth")
    args = parser.parse_args()

    use_sampler = args.use_sampler.lower() in ["true", "1", "yes"]
    save_path   = os.path.join(CKPT_DIR, args.save_name)
    mode        = "FedAvg" if args.proximal_mu == 0 else "FedProx"

    print(f"\n{'='*66}")
    print(f"  {mode} Federated Simulation")
    print(f"  rounds={args.rounds}  local_epochs={args.local_epochs}")
    print(f"  proximal_mu={args.proximal_mu}  batch_size={args.batch_size}")
    print(f"  use_sampler={use_sampler}  device={DEVICE}")
    print(f"  save={save_path}")
    print("=" * 66)

    # ── Determine GPU fraction ────────────────────────────────────────
    # With 3 sequential virtual clients on 1 GPU we give each 1/3 of VRAM.
    num_gpus_per_client = (1.0 / 3.0) if DEVICE.type == "cuda" else 0.0

    init_model  = get_model(num_classes=NUM_CLASSES, freeze_backbone=False)
    init_params = ndarrays_to_parameters(
        [v.cpu().numpy() for v in init_model.state_dict().values()]
    )

    strategy = FedProxStrategy(
        save_path             = save_path,
        proximal_mu           = args.proximal_mu,
        local_epochs          = args.local_epochs,
        fraction_fit          = 1.0,
        fraction_evaluate     = 1.0,
        min_fit_clients       = 3,
        min_evaluate_clients  = 3,
        min_available_clients = 3,
        initial_parameters    = init_params,
    )

    fl.simulation.start_simulation(
        client_fn        = make_client_fn(args.proximal_mu,
                                           args.local_epochs,
                                           args.batch_size,
                                           use_sampler),
        num_clients      = 3,
        config           = fl.server.ServerConfig(num_rounds=args.rounds),
        strategy         = strategy,
        client_resources = {
            "num_cpus": 2,
            "num_gpus": num_gpus_per_client,
        },
    )

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  {mode} TRAINING SUMMARY")
    print(f"  {'Round':>5}  {'Test Acc':>10}  {'Macro-F1':>10}")
    print(f"  {'-'*40}")
    for row in strategy.history:
        marker = " ★" if row["test_acc"] == strategy.best_acc else ""
        print(f"  {row['round']:>5}  {row['test_acc']:>10.4f}"
              f"  {row['test_macro_f1']:>10.4f}{marker}")
    print("=" * 66)
    print(f"  Best test accuracy : {strategy.best_acc:.4f}")
    print(f"  Final model        → {save_path}")
    print(f"  Best model         → {save_path.replace('.pth', '_best.pth')}")


if __name__ == "__main__":
    main()
