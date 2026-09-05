$ErrorActionPreference = "Continue"
$PYTHON = "C:\Users\sailo\AppData\Local\Programs\Python\Python38\python.exe"

Write-Host "======================================================="
Write-Host " STARTING FULL FEDPROX PIPELINE EXECUTION"
Write-Host "======================================================="

Write-Host "`n>>> [1/6] Training Hospital 1 (20 epochs)..."
& $PYTHON src\train_local.py --hospital 1 --epochs 20 --batch_size 32

Write-Host "`n>>> [2/6] Training Hospital 2 (20 epochs)..."
& $PYTHON src\train_local.py --hospital 2 --epochs 20 --batch_size 32

Write-Host "`n>>> [3/6] Training Hospital 3 (20 epochs)..."
& $PYTHON src\train_local.py --hospital 3 --epochs 20 --batch_size 32

Write-Host "`n>>> [4/6] Evaluating Local Models..."
& $PYTHON src\evaluate.py --checkpoint checkpoints\hospital1_model.pth --name "Hospital 1"
& $PYTHON src\evaluate.py --checkpoint checkpoints\hospital2_model.pth --name "Hospital 2"
& $PYTHON src\evaluate.py --checkpoint checkpoints\hospital3_model.pth --name "Hospital 3"

Write-Host "`n>>> [5/6] Running Federated Simulations..."
Write-Host "--- FedAvg (20 rounds, 5 local epochs, mu=0.0) ---"
& $PYTHON src\fedprox_simulation.py `
    --rounds 20 --local_epochs 5 --proximal_mu 0.0 `
    --save_name global_fedavg_model.pth --batch_size 16

Write-Host "--- FedProx (20 rounds, 10 local epochs, mu=0.1) ---"
& $PYTHON src\fedprox_simulation.py `
    --rounds 20 --local_epochs 10 --proximal_mu 0.1 `
    --save_name global_fedprox_model.pth --batch_size 16

Write-Host "`n>>> [6/6] Generating Final Comparisons..."
& $PYTHON src\compare_models.py

Write-Host "`n======================================================="
Write-Host " PIPELINE COMPLETE. ALL CHARTS SAVED TO results\ FOLDER."
Write-Host "======================================================="
