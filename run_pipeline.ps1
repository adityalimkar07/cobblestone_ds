Write-Host "--- Starting Quantitative Trading Pipeline ---" -ForegroundColor Cyan

# 1. Setup Environment
Write-Host "[1/7] Installing dependencies..."
pip install -r requirements.txt

# 2. Create Workspace
Write-Host "[2/7] Creating data directory..."
if (!(Test-Path -Path "data")) { New-Item -ItemType Directory -Path "data" }

# 3. Data Ingestion
Write-Host "[3/7] Fetching market data..."
python src/data_ingestion.py

# 4. Feature Engineering
Write-Host "[4/7] Calculating alpha factors..."
python src/alpha_features.py

# 5. Training
Write-Host "[5/7] Training LSTM model..."
python src/quant_strategy.py

# 6. Simulation
Write-Host "[6/7] Running live simulation..."
python src/simulation_h1.py

# 7. Visualization
Write-Host "[7/7] Generating plots..."
python src/visualize_backtest.py
python src/simulation_h2.py

Write-Host "--- Pipeline Complete! ---" -ForegroundColor Green