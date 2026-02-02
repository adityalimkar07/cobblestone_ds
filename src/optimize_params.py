import pandas as pd
import numpy as np
import itertools
from datetime import timedelta

def simulate_strategy(df, k, sl, tp, strength_thresh):
    # Vectorized checks? Hard with path dependence. We use optimized loop (numba would be better but std python is fine for 336 rows).
    # Wait, we need to run this on 2024 data (8760 rows) or Jan 2025 (336 rows)?
    # User said "strategy is working very poorly... minimize trades".
    # Usually we optimize on backtest data (2024), but user cares about current performance.
    # Let's run on Jan 2025 first as that's the "live" set. 
    # BUT 2 weeks is too short for stat sig.
    # Let's run on 2024 Backtest Data (full year) to find robust params.
    
    # Pre-calculate strength for all rows
    # strength_fn
    probs = df['quant_prob'].values
    prices = df['price_da'].values
    timestamps = df['timestamp'].values
    
    # Vectorized Strength Limits
    dist = np.abs(probs - 0.5)
    
    # 0.55/0.45 threshold is implied by dist > 0.05
    # calculate raw_strength
    # norm = (dist - 0.05) / 0.45
    # strength = (exp(k*norm) - 1) / (exp(k) - 1)
    
    # We can pre-calc this vector
    threshold_dist = 0.05
    
    with np.errstate(invalid='ignore'):
        norm = (dist - threshold_dist) / (0.5 - threshold_dist)
        strength_vec = (np.exp(k * norm) - 1) / (np.exp(k) - 1)
        
    strength_vec = np.where(dist <= threshold_dist, 0.0, strength_vec)
    strength_vec = np.nan_to_num(strength_vec)
    
    # Signals
    # Buy: prob > 0.55 & strength >= thresh
    # Sell: prob < 0.45 & strength >= thresh
    # Actually prob>0.55 is encoded in dist>0.05 + sign.
    
    buy_signals = (probs > 0.55) & (strength_vec >= strength_thresh)
    sell_signals = (probs < 0.45) & (strength_vec >= strength_thresh)
    
    # Simulation Loop
    position = 0
    entry_price = 0.0
    
    # Lists for metrics
    closed_pnls = []
    
    trades = 0
    sl_hits = 0
    tp_hits = 0
    
    cooldown = 0
    
    n = len(df)
    
    # Using simple array iteration for speed
    for i in range(n):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        p = prices[i]
        
        # Check Exit
        if position != 0:
            unrealized = position * (p - entry_price)
            if unrealized <= -sl:
                # SL Hit
                closed_pnls.append(-sl) # Approx execution at limit? Or P? Conservatively use SL limit.
                position = 0
                cooldown = 6
                sl_hits += 1
                continue
            elif unrealized >= tp:
                # TP Hit
                closed_pnls.append(tp)
                position = 0
                cooldown = 6
                tp_hits += 1
                continue
        
        # Check Entry
        if position == 0:
            if buy_signals[i]:
                position = 1
                entry_price = p
                trades += 1
            elif sell_signals[i]:
                position = -1
                entry_price = p
                trades += 1
        elif position == 1:
            if sell_signals[i]: # Flip
                # Close current
                pnl = (p - entry_price)
                closed_pnls.append(pnl)
                # Open new
                position = -1
                entry_price = p
                trades += 1
        elif position == -1:
             if buy_signals[i]: # Flip
                pnl = -(p - entry_price) # Short PnL
                closed_pnls.append(pnl)
                position = 1
                entry_price = p
                trades += 1
                
    # Metrics
    total_pnl = sum(closed_pnls)
    n_trades = len(closed_pnls)
    if n_trades > 0:
        win_rate = len([x for x in closed_pnls if x > 0]) / n_trades
        avg_ret = total_pnl / n_trades
    else:
        win_rate = 0
        avg_ret = 0
        
    # Approx Sharpe (using trade list, exact daily sharpe requires reconstructing hourly pnl)
    # Good enough proxy: Total PnL / Sqrt(Trades) ? 
    # Let's use Total PnL as primary objective, Filter by Win Rate > 45%.
    
    return {
        'k': k,
        'sl': sl,
        'tp': tp,
        'thresh': strength_thresh,
        'pnl': total_pnl,
        'trades': n_trades,
        'win_rate': win_rate,
        'sl_hits': sl_hits
    }

def main():
    print("Loading data...")
    try:
        df_pred = pd.read_csv("output/quant_predictions.csv")
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        
        df_power = pd.read_csv("data/german_power_data.csv")
        df_power['timestamp'] = pd.to_datetime(df_power['timestamp'])
        
        # Align
        df_pred['timestamp'] = df_pred['timestamp'] - timedelta(hours=1)
        df = pd.merge(df_power, df_pred, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Use 2024 Data for Optimization (Robustness)
        df_2024 = df[(df['timestamp'] >= '2024-01-01') & (df['timestamp'] <= '2024-12-31')].reset_index(drop=True)
        
        print(f"Optimizing on {len(df_2024)} hours of 2024 data.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Parameter Grid
    # k: User asked to vary k in e^k
    # SL/TP: User asked to find optimal
    # Threshold: User said "if strength say 75%", minimize trades
    
    k_vals = [3, 5, 8, 10] 
    sl_vals = [30, 45, 60] 
    tp_vals = [30, 45, 60, 90]
    thresh_vals = [0.5, 0.6, 0.7, 0.8] 
    
    results = []
    
    total_combs = len(k_vals) * len(sl_vals) * len(tp_vals) * len(thresh_vals)
    print(f"Testing {total_combs} combinations...")
    
    counter = 0
    for k, sl, tp, thresh in itertools.product(k_vals, sl_vals, tp_vals, thresh_vals):
        res = simulate_strategy(df_2024, k, sl, tp, thresh)
        results.append(res)
        counter += 1
        if counter % 50 == 0:
            print(f"Processed {counter}/{total_combs}...")
            
    # Convert to DF
    res_df = pd.DataFrame(results)
    
    # Filter for valid number of trades (e.g. > 50 trades in a year to be stat sig, but < 500 to minimize)
    # User said "minimize number of trades".
    # Let's sort by PnL first, then look at trades.
    
    print("\n--- Top 10 Configurations by Net PnL ---")
    top_pnl = res_df.sort_values('pnl', ascending=False).head(10)
    print(top_pnl.to_string(index=False))
    
    print("\n--- Top 10 Configurations by Win Rate (Min 50 Trades) ---")
    valid_trades = res_df[res_df['trades'] > 50]
    if not valid_trades.empty:
        top_wr = valid_trades.sort_values('win_rate', ascending=False).head(10)
        print(top_wr.to_string(index=False))
        
    # Heuristic for "Best"
    # High PnL, reasonable trades coverage, Win Rate > 50%
    best = res_df[(res_df['trades'] > 100) & (res_df['trades'] < 600)].sort_values('pnl', ascending=False).head(1)
    
    if not best.empty:
        print("\n--- Recommended Configuration (Balanced) ---")
        print(best.to_string(index=False))
        
        # Save to file for reading
        best.to_csv("output/optimal_params.csv", index=False)
    else:
        print("No balanced config found.")

if __name__ == "__main__":
    main()
