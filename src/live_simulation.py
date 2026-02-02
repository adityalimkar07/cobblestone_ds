import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os

def run_simulation():
    print("Preparing data for Live Simulation (Quant Alpha Strategy)...")
    
    # Risk Params (per MW)
    STOP_LOSS_EUR = 45.0
    TAKE_PROFIT_EUR = 60.0
    
    # 1. Load Data
    pred_path = "output/quant_predictions.csv"
    power_path = "data/german_power_data.csv"
    
    if not os.path.exists(pred_path):
        print("Predictions not found. Run src/quant_strategy.py first.")
        return

    df_pred = pd.read_csv(pred_path)
    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
    
    df_power = pd.read_csv(power_path)
    df_power['timestamp'] = pd.to_datetime(df_power['timestamp'])
    
    # Merge
    # Alignment: 
    # quant_prob at T is the prediction for T (Target).
    # This prediction was generated using data up to T-1.
    # So at Time T-1, I have the prediction for T.
    # I want to trade at T-1 based on this prediction.
    # Therefore, I map Prediction(T) to Timestamp T-1.
    
    df_pred_aligned = df_pred.copy()
    df_pred_aligned['timestamp'] = df_pred_aligned['timestamp'] - pd.Timedelta(hours=1)
    df_pred_aligned = df_pred_aligned.rename(columns={'quant_prob': 'signal_aligned'})
    
    df = pd.merge(df_power[['timestamp', 'price_da']], df_pred_aligned[['timestamp', 'signal_aligned']], on='timestamp', how='left')
    
    # Fill NaN signals (e.g. first hour) with 0.5 (Neutral)
    df['signal_aligned'] = df['signal_aligned'].fillna(0.5)
    
    # Filter for Jan-June 2025
    start_date = "2025-01-01"
    end_date = "2025-06-01"
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    df = df.loc[mask].reset_index(drop=True)
    
    print(f"Simulating {len(df)} hours...")
    
    
    # --- Config ---
    USE_Z_SCORE_THRESH = False # TOGGLE: Set True for Z-Score Logic
    Z_WINDOW = 168
    Z_THRESH = 1.0
    
    # Pre-calculate Rolling Stats if Dynamic
    if USE_Z_SCORE_THRESH:
        df['prob_rolling_mean'] = df['signal_aligned'].rolling(window=Z_WINDOW, min_periods=24).mean().bfill()
        df['prob_rolling_std'] = df['signal_aligned'].rolling(window=Z_WINDOW, min_periods=24).std().bfill()
    
    # --- Live Loop ---
    # Track hourly actions
    final_positions = np.zeros(len(df))
    final_actions = np.array([None] * len(df), dtype=object)
    final_probs = np.zeros(len(df)) # Store probabilities for viz
    pnl_hourly = []
    
    # State
    position = 0 # 1, -1, 0 (Hold until signal reversal or stop loss)
    entry_price = 0.0
    current_sl_val = STOP_LOSS_EUR
    current_tp_val = TAKE_PROFIT_EUR
    
    daily_pnl = 0.0
    current_day = df['timestamp'].iloc[0].date()
    
    # Markers for plotting
    output_markers = []
    
    sl_hits = 0
    tp_hits = 0
    cooldown = 0
    trades_count = 0
    
    cooldown_counter = 0 # Hours to wait after SL/TP
    
    for i, row in df.iterrows():
        ts = row['timestamp']
        price = row['price_da']
        prob = row['signal_aligned'] # Use Forward Forecast (Aligned)
        
        final_probs[i] = prob if not pd.isna(prob) else 0.5
        
        if pd.isna(prob): continue # Skip last row
        
        # Reset Daily PnL
        if ts.date() != current_day:
            current_day = ts.date()
            
        # --- 0. SIGNAL CALCULATION ---
        raw_strength = 0.0
        new_signal = 0
        
        if USE_Z_SCORE_THRESH:
            # Dynamic Logic
            mu = row['prob_rolling_mean']
            sigma = row['prob_rolling_std'] if row['prob_rolling_std'] > 0.001 else 1.0
            z_score = (prob - mu) / sigma
            
            if z_score > Z_THRESH:
                new_signal = 1
                raw_strength = min(1.0, 0.5 + (z_score - 1.0) * 0.25) 
            elif z_score < -Z_THRESH:
                new_signal = -1
                raw_strength = min(1.0, 0.5 + (abs(z_score) - 1.0) * 0.25)
        else:
            # Static Logic
            dist = abs(prob - 0.5)
            threshold = 0.05
            if dist > threshold:
                norm = (dist - threshold) / (0.5 - threshold)
                k = 10.0
                raw_strength = (np.exp(k * norm) - 1) / (np.exp(k) - 1)
            
            # Simplified Entry Logic (User Request: Fix Missing Trades)
            # Trade if Probability is strong (>60% or <40%)
            ENTRY_THRESH_PROB = 0.10 # dist from 0.5
            
            if prob > (0.5 + ENTRY_THRESH_PROB):
                new_signal = 1
            elif prob < (0.5 - ENTRY_THRESH_PROB):
                new_signal = -1
        
        # forced neutral if cooling down
        if cooldown > 0:
            cooldown -= 1
            position = 0
            final_positions[i] = 0
            continue # Skip logic this hour
            
        # Check Exits
        exited_this_step = False
        if position != 0:
            unrealized = position * (price - entry_price)
            
            # Dynamic Limits
            if unrealized <= -current_sl_val:
                position = 0
                cooldown = 6
                exited_this_step = True
                final_actions[i] = "SL_EXIT"
                output_markers.append({'ts': ts, 'price': price, 'marker': 'X', 'color': 'darkred', 'label': f'SL_{current_sl_val:.0f}'})
                sl_hits += 1
            elif unrealized >= current_tp_val:
                position = 0
                cooldown = 6
                exited_this_step = True
                final_actions[i] = "TP_EXIT"
                output_markers.append({'ts': ts, 'price': price, 'marker': '*', 'color': 'gold', 'label': f'TP_{current_tp_val:.0f}'})
                tp_hits += 1
        
        # Check Entries
        ENTRY_THRESH = 0.5 # Lower threshold if Z-Score acts as filter
        
        if not exited_this_step: 
            if new_signal != 0 and new_signal != position:
                # I Flip or Open
                position = new_signal
                entry_price = price
                
                # --- DYNAMIC RISK CALCULATION ---
                # Multiplier = 1 + (Strength - 0.5) * 2.5
                risk_multiplier = 1.0 + max(0, (raw_strength - 0.5)) * 2.5
                current_sl_val = STOP_LOSS_EUR * risk_multiplier
                current_tp_val = TAKE_PROFIT_EUR * risk_multiplier
                
                action_label = "BUY" if position == 1 else "SELL"
                final_actions[i] = action_label
                trades_count += 1
                
                color = 'lime' if position == 1 else 'red'
                marker = '^' if position == 1 else 'v'
                output_markers.append({'ts': ts, 'price': price, 'marker': marker, 'color': color, 'label': action_label})
            elif (USE_Z_SCORE_THRESH and new_signal == 0 and position != 0) or \
                 (not USE_Z_SCORE_THRESH and new_signal == 0 and position != 0): 
                # STRICT SIGNAL EXIT (Signal Fade)
                # If Z-Score drops back into neutral zone (|Z| < 1.0), we Exit.
                position = 0
                final_actions[i] = "FLAT_EXIT"
                trades_count += 1 
                output_markers.append({'ts': ts, 'price': price, 'marker': 'o', 'color': 'gray', 'label': 'Exit'})
            
        final_positions[i] = position
        
    # Calculate PnL Curve
    df['position'] = final_positions
    df['price_change'] = df['price_da'].diff()
    # PnL = Position[t-1] * PriceChange[t]
    df['pnl_hourly'] = df['position'].shift(1).fillna(0) * df['price_change']
    df['cumulative_pnl'] = df['pnl_hourly'].cumsum().fillna(0)
    
    total_return = df['cumulative_pnl'].iloc[-1]
    
    # 1. Reconstruct Closed Trades
    trade_pnls = []
    current_trade_pnl = 0.0
    pos_arr = df['position'].values
    pnl_arr = df['pnl_hourly'].values
    
    for i in range(1, len(df)):
        prev_pos = pos_arr[i-1]
        curr_pos = pos_arr[i]
        pnl = pnl_arr[i]
        
        if prev_pos != 0:
            current_trade_pnl += pnl
            if curr_pos != prev_pos: # Trade Closed/Flipped
                trade_pnls.append(current_trade_pnl)
                current_trade_pnl = 0.0
                
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    total_closed_trades = len(trade_pnls)
    
    win_rate = (len(wins) / total_closed_trades * 100) if total_closed_trades > 0 else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    
    # 2. Sharpe Ratio (Annualized)
    df['date'] = df['timestamp'].dt.date
    daily_pnl = df.groupby('date')['pnl_hourly'].sum()
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    print("-" * 30)
    print(f"JAN 2025 SIMULATION RESULTS")
    print(f"Total Return: €{total_return:.2f}")
    print(f"Closed Trades: {total_closed_trades}")
    print(f"Win Rate: {win_rate:.1f}% ({len(wins)}/{total_closed_trades})")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Avg Win: €{avg_win:.2f} | Avg Loss: €{avg_loss:.2f}")
    print(f"Risk:Reward: 1:{risk_reward:.2f}")
    print(f"SL Hits: {sl_hits} | TP Hits: {tp_hits}")
    print("-" * 30)
    
    # 3. Visualization
    SIM_HOURS = len(df)
    df_viz = df.head(SIM_HOURS).copy()
    viz_actions = final_actions[:SIM_HOURS]
    viz_probs = final_probs[:SIM_HOURS]
    
    fig = plt.figure(figsize=(14, 8)) # Adjusted height
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    # ax3 removed
    
    ax1.set_title(f'Quant Alpha Strategy | Net Return: €{total_return:.0f} | SL: {sl_hits} | TP: {tp_hits}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price EUR/MWh')
    ax2.set_ylabel('Position')
    # ax3 removed
    
    line_actual, = ax1.plot([], [], 'c-', label='Price', lw=2)
    
    line_pos, = ax2.step([], [], where='post', color='blue', lw=2)
    # line_pnl removed
    
    # Scatter Plots for markers
    scat_buy = ax1.scatter([], [], marker='^', c='lime', s=200, zorder=10)
    scat_sell = ax1.scatter([], [], marker='v', c='red', s=200, zorder=10)
    scat_sl = ax1.scatter([], [], marker='X', c='darkred', s=250, zorder=11) # Even bigger for SL
    scat_tp = ax1.scatter([], [], marker='*', c='gold', s=300, zorder=11) # Huge for TP
    
    # --- TABLE SETUP ---
    col_labels = ['Last Signal', 'Strength', 'Price', 'Action']
    table_data = [['Waiting...', '0.00%', '€0.00', '-']]
    the_table = ax1.table(cellText=table_data, colLabels=col_labels, 
                          loc='upper left', cellLoc='center', bbox=[0.02, 0.8, 0.3, 0.15])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    times = df_viz['timestamp'].values
    ax1.set_xlim(times[0], times[-1])
    ax1.set_ylim(df_viz['price_da'].min()-10, df_viz['price_da'].max()+10)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    
    # ax3 config removed
    
    def update(frame):
        curr = df_viz.iloc[:frame]
        ct = curr['timestamp'].values
        
        line_actual.set_data(ct, curr['price_da'].values)
        line_pos.set_data(ct, curr['position'].values)
        # line_pnl removed
        
        # Markers
        # Filter markers up to current time
        current_ts = ct[-1] if len(ct) > 0 else None
        if current_ts:
            valid_markers = [m for m in output_markers if m['ts'] <= current_ts and m['ts'] >= times[0]]
            
            buys = [m for m in valid_markers if m['label'] == 'BUY']
            sells = [m for m in valid_markers if m['label'] == 'SELL']
            sls = [m for m in valid_markers if str(m['label']).startswith('SL')]
            tps = [m for m in valid_markers if str(m['label']).startswith('TP')]
            
            if buys: scat_buy.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in buys]))
            if sells: scat_sell.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in sells]))
            if sls: scat_sl.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in sls]))
            if tps: scat_tp.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in tps]))
            
        if not curr.empty:
            cpnl = curr.iloc[-1]['cumulative_pnl']
            ts_str = curr.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')
            ax1.set_title(f'{ts_str} | PnL: €{cpnl:.2f} | SL Hits: {sl_hits} | TP Hits: {tp_hits}', fontsize=12, fontweight='bold')

            # --- UPDATE TABLE ---
            idx = frame - 1
            if idx >= 0:
                # Look back for last action
                last_action_idx = -1
                for k in range(idx, max(-1, idx-48), -1): 
                    if viz_actions[k] is not None:
                        last_action_idx = k
                        break
                
                if last_action_idx != -1:
                    last_act = viz_actions[last_action_idx]
                    last_pr = df_viz.iloc[last_action_idx]['price_da']
                    last_prob = viz_probs[last_action_idx]
                    
                    # Normalize Probability to Strength (0% to 100%)
                    # Neutral = 0.5 -> 0%
                    # Strong Buy (1.0) or Strong Sell (0.0) -> 100%
                    # Exponential Strength Calculation
                    # 0 near threshold (0.55/0.45), 1 at edge (1.0/0.0)
                    dist = abs(last_prob - 0.5)
                    threshold = 0.05
                    if dist <= threshold:
                        raw_strength = 0.0
                    else:
                        # Normalize [0.05, 0.5] -> [0, 1]
                        norm = (dist - threshold) / (0.5 - threshold)
                        
                        # True Exponential Function: (e^(kx) - 1) / (e^k - 1)
                        # k controls steepness. Higher k = more convex (stays low longer).
                        k = 5.0 
                        raw_strength = (np.exp(k * norm) - 1) / (np.exp(k) - 1)
                    strength_str = f"{raw_strength:.1%}"
                    
                    # Update cells: (row, col)
                    the_table.get_celld()[(1, 0)].get_text().set_text(str(last_act))
                    the_table.get_celld()[(1, 1)].get_text().set_text(strength_str)
                    the_table.get_celld()[(1, 2)].get_text().set_text(f"€{last_pr:.1f}")
                    the_table.get_celld()[(1, 3)].get_text().set_text("EXECUTED")

        return line_actual, line_pos, scat_buy, scat_sell, scat_sl, scat_tp, the_table

    # Optimized for Memory: Frame Skip 10x
    frames_indices = range(0, len(df_viz), 10)
    ani = animation.FuncAnimation(fig, update, frames=frames_indices, interval=100, blit=False)
    ani.save('output/simulation_replay_jan_jun.gif', writer='pillow', fps=15)
    print("Animation saved.")

if __name__ == "__main__":
    run_simulation()
