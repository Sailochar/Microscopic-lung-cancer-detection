$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\sailo\AppData\Local\Programs\Python\Python38\python.exe"

Write-Host ""
Write-Host "======================================================================="
Write-Host "  FEDPROX vs FEDAVG EXPERIMENT — Proving FedProx Advantage on Non-IID"
Write-Host "======================================================================="
Write-Host ""
Write-Host "  Dataset layout (Non-IID confirmed):"
Write-Host "    Hospital 1: lung_aca=1687  lung_n=1286  lung_scc=1688  (balanced)"
Write-Host "    Hospital 2: lung_aca= 563  lung_n=2571  lung_scc= 562  (lung_n heavy)"
Write-Host "    Hospital 3: lung_aca=2250  lung_n= 643  lung_scc=2251  (aca/scc heavy)"
Write-Host ""
Write-Host "  Strategy:"
Write-Host "    FedAvg  → 20 rounds, 5 local epochs, mu=0.0"
Write-Host "    FedProx → 20 rounds, 5 local epochs, mu=0.01 : proximal term"
Write-Host ""

# ─── Step 1: FedAvg baseline ─────────────────────────────────────────────────
Write-Host '>>> [1/3] Running FedAvg (20 rounds, 5 local epochs, mu=0.0) ...'
Write-Host "          Estimated time: 10-15 minutes"
Write-Host ""
& $PYTHON src\fedprox_simulation.py `
    --rounds       20  `
    --local_epochs 5   `
    --proximal_mu  0.0 `
    --batch_size   16  `
    --save_name    global_fedavg_model.pth

Write-Host ""
Write-Host ">>> FedAvg COMPLETE"
Write-Host ""

# ─── Step 2: FedProx with matched settings ───────────────────────────────────
Write-Host '>>> [2/3] Running FedProx (20 rounds, 5 local epochs, mu=0.01) ...'
Write-Host "          Estimated time: 20-30 minutes"
Write-Host ""
& $PYTHON src\fedprox_simulation.py `
    --rounds       20  `
    --local_epochs 5   `
    --proximal_mu  0.01 `
    --batch_size   16  `
    --save_name    global_fedprox_model.pth

Write-Host ""
Write-Host ">>> FedProx COMPLETE"
Write-Host ""

# ─── Step 3: Generate comparison charts ──────────────────────────────────────
Write-Host ">>> [3/3] Generating final comparison charts ..."
& $PYTHON src\compare_models.py

Write-Host ""
Write-Host "======================================================================="
Write-Host "  EXPERIMENT COMPLETE!"
Write-Host "  Charts saved to: results\"
Write-Host "  Key files:"
Write-Host "    results\comparison_bar.png   ← main comparison chart"
Write-Host "    results\per_class_f1.png     ← per-class breakdown"
Write-Host "    results\fedprox_cm.png       ← FedProx confusion matrix"
Write-Host "    results\fedavg_cm.png        ← FedAvg confusion matrix"
Write-Host "======================================================================="
