#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- Starting Quantitative Trading Pipeline ---"

# 1. Setup Environment
echo "[1/7] Installing dependencies..."
pip install -r requirements.txt

# 2. Create Workspace
echo "[2/7] Creating data directory..."
mkdir -p data

# 3. Data Ingestion
echo "[3/7] Fetching market data from Energy-Charts..."
python src/data_ingestion.py

# 4. Feature Engineering
echo "[4/7] Calculating rolling alpha factors..."
python src/alpha_features.py

# 5. Training
echo "[5/7] Training LSTM model..."
python src/quant_strategy.py

# 6. Simulation
echo "[6/7] Running live simulation engine..."
python src/simulation_h1.py

# 7. Visualization
echo "[7/7] Generating performance curves and timelapse GIF..."
python src/visualize_backtest.py
python src/simulation_h2.py

echo "--- Pipeline Complete. Results available in output/ and data/ ---"