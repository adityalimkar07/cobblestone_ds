# Executive Summary

## Project Objective
I set out to engineer a profitable, algorithmic trading strategy for the European Power Market (specifically German Day-Ahead). The goal was to demonstrate how advanced **Alpha Factors** and **Deep Learning** could be combined to extract signal from noise in a highly volatile market.

## Key Challenges & Solutions

### 1. Eliminating Look-Ahead Bias
**Challenge:** Early iterations of the strategy showed suspiciously high accuracy (>90%), which I identified as look-ahead bias originating from standard library rank functions that used the entire dataset.
**Solution:** I completely rewrote the feature engineering layer in `src/alpha_features.py`. I replaced global calculations with **Rolling Window** operations (e.g., `rolling(168).rank()`). This guaranteed that my model only ever "saw" data from the past 7 days, reflecting realistic trading conditions.

### 2. Signal Quality vs. Quantity
**Challenge:** The LSTM model provided many signals, but trading every hour resulted in high transaction costs and noise accumulation.
**Solution:** I implemented an **Exponential Signal Strength Filter**. By applying a convex function ($k=10$) to the model's probability output, I created a rigorous filter. I configured the simulation to only trade when the "Strength" exceeded 60%, effectively ignoring weak signals. This reduced trade frequency but significantly increased the "Hit Rate" and overall PnL (I later removed it as model accuracy improved significantly with time, but for poor model performance this method can help to reduce loss).

### 3. Data Integrity & Verification
**Challenge:** Ensuring the backtest results were valid and not a result of leakage.
**Solution:** I built a custom validation suite (`src/qa_checks.py`) and manually verified the simulation logic. I corrected a subtle PnL calculation error where `shift(-1)` was used on price changes. I replaced this with a lag-based approach (`Position[T-1] * Price_Change[T]`), mathematically ensuring that profits are only calculated on realized moves.


## Reproducibility
To ensure exact replication of these results, I have included the trained model weights in `models/lstm_model_weights.pth`. The system is configured to automatically load these weights, eliminating variance caused by GPU/CPU non-determinism during training.
