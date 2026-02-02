# European Power Trading Strategy (Quant + AI)

## Project Overview
I developed this project to build a robust, algorithmic trading strategy for the German Day-Ahead Power Market. The core objective was to move beyond simple forecasting and create an actionable **Alpha Strategy** that predicts price direction (Up/Down) with high conviction.

By fusing **Quantitative Finance** (101 Formulaic Alphas) with **Deep Learning** (PyTorch LSTM), I engineered a system that identifies profitable trading windows while strictly managing risk. A critical part of my work involved rigorously eliminating look-ahead bias and ensuring data integrity for realistic backtesting.


---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Data Ingestion
Fetch the latest fundamental data from Energy-Charts.
```bash
python src/data_ingestion.py
```

### 2. Feature Generation
Calculate the rolling alpha factors.
```bash
python src/alpha_features.py
```

### 3. Model Inference (Weights Provided)
The project includes pre-trained weights (`models/lstm_model_weights.pth`) for exact reproducibility.
Running the strategy will automatically load these weights instead of retraining.

```bash
python src/quant_strategy.py
```
*   **Note**: Using the shared weights ensures you get the exact stats reported in `report.tex`. Retraining (by deleting the `models/` folder) may lead to variances due to hardware non-determinism.

### 4. Running the Simulation
Execute the trading engine to verify performance on the test set.
```bash
python src/simulation_h1.py
python src/simulation_h2.py

```

### 5. Visualization (H2 Backtest)
Generate the Equity Curve and Timelapse GIF to visualize performance over time. (Corresponding graphs are automatically generated if you run simulation_h1.py and simulation_h2.py, respectively)
```bash
python src/visualize_backtest.py
```

---

## Repository Structure
```
├── data/               # Raw and Processed Data
├── output/             # Visualization, GIFs, and Predictions
├── output_2/           # 2024 Backtest Artifacts
├── src/                
│   ├── ai_component.py      # Groq + Google News Integration
│   ├── alpha_features.py    # feature_generator.py port (101 Alphas)
│   ├── data_ingestion.py    # Energy-Charts API Wrapper
│   ├── generate_2024_gif.py # Timelapse Generation (Sliding Window)
│   ├── live_simulation.py   # Trading Engine & Visualization
│   ├── quant_strategy.py    # PyTorch LSTM Classifier
│   ├── visualize_backtest.py# 2024 Equity Curve & Metrics
│   └── qa_checks.py         # Data Integrity Suite
├── requirements.txt
└── README.md
```

