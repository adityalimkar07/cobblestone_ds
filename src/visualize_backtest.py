import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def visualize_2025_results():
    print("Visualizing 2025 Backtest Results (H1 & H2)...")
    
    # 1. Load Data
    pred_path = "output/quant_predictions.csv"
    power_path = "data/german_power_data.csv"
    
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} not found.")
        return

    df_pred = pd.read_csv(pred_path)
    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
    
    df_power = pd.read_csv(power_path)
    df_power['timestamp'] = pd.to_datetime(df_power['timestamp'])
    
    # Merge with Alignment Fix (Same as live_simulation.py)
    df_pred_aligned = df_pred.copy()
    df_pred_aligned['timestamp'] = df_pred_aligned['timestamp'] - pd.Timedelta(hours=1)
    df_pred_aligned = df_pred_aligned.rename(columns={'quant_prob': 'signal_aligned'})
    
    df_merged = pd.merge(df_power[['timestamp', 'price_da']], df_pred_aligned[['timestamp', 'signal_aligned']], on='timestamp', how='left')
    df_merged['signal_aligned'] = df_merged['signal_aligned'].fillna(0.5)
    
    # Define Periods
    periods = [
        ("2025 H1", "2025-01-01", "2025-06-01", "output/backtest_2025_h1_chart.png"),
        ("2025 H2", "2025-06-01", "2026-02-01", "output_2/backtest_2025_h2_chart.png")
    ]
    
    for name, start_date, end_date, out_path in periods:
        print(f"\nProcessing {name} ({start_date} to {end_date})...")
        mask = (df_merged['timestamp'] >= start_date) & (df_merged['timestamp'] <= end_date)
        df = df_merged.loc[mask].reset_index(drop=True)
        
        if len(df) == 0:
            print(f"No data for {name}")
            continue

        # --- Simulation Logic (Simplified Prob Threshold) ---
        STOP_LOSS_EUR = 45.0
        TAKE_PROFIT_EUR = 60.0
        
        position = 0
        entry_price = 0.0
        current_sl_val = STOP_LOSS_EUR
        current_tp_val = TAKE_PROFIT_EUR
        
        positions = np.zeros(len(df))
        cooldown = 0
        
        prices = df['price_da'].values
        probs = df['signal_aligned'].values
        timestamps = df['timestamp'].values
        
        for i in range(len(df)):
            price = prices[i]
            prob = probs[i]
            
            # Signal
            new_signal = 0
            ENTRY_THRESH_PROB = 0.10
            
            if prob > (0.5 + ENTRY_THRESH_PROB):
                new_signal = 1
            elif prob < (0.5 - ENTRY_THRESH_PROB):
                new_signal = -1
                
            if cooldown > 0:
                cooldown -= 1
                positions[i] = 0
                position = 0
                continue
            
            # Check Exit
            exited = False
            if position != 0:
                unrealized = position * (price - entry_price)
                if unrealized <= -current_sl_val:
                    position = 0
                    cooldown = 6
                    exited = True
                elif unrealized >= current_tp_val:
                    position = 0
                    cooldown = 6
                    exited = True
            
            # Check Entry
            if not exited:
                if new_signal != 0 and new_signal != position:
                    position = new_signal
                    entry_price = price
                    
                    # Risk Scale
                    dist = abs(prob - 0.5)
                    raw_strength = min(1.0, (dist - 0.05)/0.45) if dist > 0.05 else 0
                    risk_multiplier = 1.0 + raw_strength * 2.5
                    current_sl_val = STOP_LOSS_EUR * risk_multiplier
                    current_tp_val = TAKE_PROFIT_EUR * risk_multiplier
                elif new_signal == 0 and position != 0:
                     # Soft Exit on neutral
                     position = 0
            
            positions[i] = position
            
        df['position'] = positions
        df['price_change'] = df['price_da'].diff()
        df['hourly_pnl'] = df['position'].shift(1).fillna(0) * df['price_change']
        df['cumulative_pnl'] = df['hourly_pnl'].cumsum().fillna(0)
        
        # Metrics
        total_ret = df['cumulative_pnl'].iloc[-1]
        
        # Win Rate Calc
        trade_pnls = []
        current_trade_pnl = 0.0
        pos_arr = df['position'].values
        pnl_arr = df['hourly_pnl'].values
        
        for i in range(1, len(df)):
            prev_pos = pos_arr[i-1]
            curr_pos = pos_arr[i]
            pnl = pnl_arr[i]
            if prev_pos != 0:
                current_trade_pnl += pnl
                if curr_pos != prev_pos:
                    trade_pnls.append(current_trade_pnl)
                    current_trade_pnl = 0.0
                    
        wins = [p for p in trade_pnls if p > 0]
        total_trades_calc = len(trade_pnls)
        win_rate = (len(wins) / total_trades_calc) * 100 if total_trades_calc > 0 else 0.0
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['timestamp'], df['cumulative_pnl'], label='Cumulative PnL (€)', color='green')
        ax.set_title(f"{name} Results | Net PnL: €{total_ret:,.2f} | Win Rate: {win_rate:.1f}%")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL (€)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved chart to {out_path} (PnL: {total_ret:.2f})")

if __name__ == "__main__":
    visualize_2025_results()
