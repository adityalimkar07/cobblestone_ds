import pandas as pd
import numpy as np

def count_trades():
    print("Calculating Trade Counts and Metrics...")
    
    # 1. Load Data
    pred_path = "output/quant_predictions.csv"
    power_path = "data/german_power_data.csv"
    
    df_pred = pd.read_csv(pred_path)
    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
    
    df_power = pd.read_csv(power_path)
    df_power['timestamp'] = pd.to_datetime(df_power['timestamp'])
    
    # Alignment Fix
    df_pred_aligned = df_pred.copy()
    df_pred_aligned['timestamp'] = df_pred_aligned['timestamp'] - pd.Timedelta(hours=1)
    df_pred_aligned = df_pred_aligned.rename(columns={'quant_prob': 'signal_aligned'})
    
    df_base = pd.merge(df_power[['timestamp', 'price_da']], df_pred_aligned[['timestamp', 'signal_aligned']], on='timestamp', how='left')
    df_base['signal_aligned'] = df_base['signal_aligned'].fillna(0.5)
    
    periods = [
        ("H1 (Jan-Jun 2025)", "2025-01-01", "2025-06-01"),
        ("H2 (Jun-Feb 2026)", "2025-06-01", "2026-02-01")
    ]
    
    for name, start, end in periods:
        mask = (df_base['timestamp'] >= start) & (df_base['timestamp'] <= end)
        df = df_base.loc[mask].reset_index(drop=True)
        
        # Logic
        STOP_LOSS_EUR = 45.0
        TAKE_PROFIT_EUR = 60.0
        
        position = 0
        entry_price = 0.0
        current_sl_val = STOP_LOSS_EUR
        current_tp_val = TAKE_PROFIT_EUR
        
        positions = np.zeros(len(df))
        cooldown = 0
        trades_count = 0
        
        prices = df['price_da'].values
        probs = df['signal_aligned'].values
        
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
            
            exited_this_step = False
            if position != 0:
                unrealized = position * (price - entry_price)
                if unrealized <= -current_sl_val:
                    position = 0
                    cooldown = 6
                    exited_this_step = True
                elif unrealized >= current_tp_val:
                    position = 0
                    cooldown = 6
                    exited_this_step = True
                    
            if not exited_this_step:
                if new_signal != 0 and new_signal != position:
                    position = new_signal
                    entry_price = price
                    
                    # Risk Scale
                    dist = abs(prob - 0.5)
                    raw_strength = min(1.0, (dist - 0.05)/0.45) if dist > 0.05 else 0
                    risk_multiplier = 1.0 + raw_strength * 2.5
                    current_sl_val = STOP_LOSS_EUR * risk_multiplier
                    current_tp_val = TAKE_PROFIT_EUR * risk_multiplier
                    
                    trades_count += 1
                    
                elif new_signal == 0 and position != 0:
                    position = 0
            
            positions[i] = position
            
        # PnL Calc
        df['position'] = positions
        df['price_change'] = df['price_da'].diff()
        df['hourly_pnl'] = df['position'].shift(1).fillna(0) * df['price_change']
        total_pnl = df['hourly_pnl'].sum()
        
        print(f"--- {name} ---")
        print(f"Total Trades: {trades_count}")
        print(f"Total PnL: €{total_pnl:,.2f}")
        print("--------------------")

if __name__ == "__main__":
    count_trades()
