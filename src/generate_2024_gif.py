import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import os

def generate_timelapse():
    print("Generating 2025 H2 Timelapse GIF (Sliding Window - MATCHED SPEED)...")
    
    # 1. Load Data
    pred_path = "output/quant_predictions.csv"
    power_path = "data/german_power_data.csv"
    
    df_pred = pd.read_csv(pred_path)
    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
    
    df_power = pd.read_csv(power_path)
    df_power['timestamp'] = pd.to_datetime(df_power['timestamp'])
    
    # Check Resolution of Input for H2
    # H2 is 15-min data. Stride 12 = 3 Hours per Frame.
    
    # Alignment Fix
    df_pred_aligned = df_pred.copy()
    df_pred_aligned['timestamp'] = df_pred_aligned['timestamp'] - pd.Timedelta(hours=1)
    df_pred_aligned = df_pred_aligned.rename(columns={'quant_prob': 'signal_aligned'})
    
    df = pd.merge(df_power[['timestamp', 'price_da']], df_pred_aligned[['timestamp', 'signal_aligned']], on='timestamp', how='left')
    df['signal_aligned'] = df['signal_aligned'].fillna(0.5)
    
    # Filter 2025 H2
    start_date = "2025-06-01"
    end_date = "2026-02-01"
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    df = df.loc[mask].reset_index(drop=True)
    
    print(f"Simulating {len(df)} 15-min intervals (Approx {len(df)/4:.0f} hours)...")
    
    # 2. Trading Logic (Simplified Prob Threshold)
    STOP_LOSS_EUR = 45.0
    TAKE_PROFIT_EUR = 60.0
    
    position = 0
    entry_price = 0.0
    current_sl_val = STOP_LOSS_EUR
    current_tp_val = TAKE_PROFIT_EUR
    
    positions = np.zeros(len(df))
    cooldown = 0
    output_markers = []
    
    prices = df['price_da'].values
    probs = df['signal_aligned'].values
    timestamps = df['timestamp'].values
    
    trades_count = 0
    
    for i in range(len(df)):
        price = prices[i]
        prob = probs[i]
        ts = timestamps[i]
        
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
        exited_this_step = False
        if position != 0:
            unrealized = position * (price - entry_price)
            if unrealized <= -current_sl_val:
                position = 0
                cooldown = 6
                exited_this_step = True
                output_markers.append({'ts': ts, 'price': price, 'label': f'SL_{current_sl_val:.0f}'})
            elif unrealized >= current_tp_val:
                position = 0
                cooldown = 6
                exited_this_step = True
                output_markers.append({'ts': ts, 'price': price, 'label': f'TP_{current_tp_val:.0f}'})
                
        # Check Entry
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
                
                label = 'BUY' if position == 1 else 'SELL'
                output_markers.append({'ts': ts, 'price': price, 'label': label})
                trades_count += 1
                
            elif new_signal == 0 and position != 0:
                position = 0
                output_markers.append({'ts': ts, 'price': price, 'label': 'Exit'})
                
        positions[i] = position
        
    df['position'] = positions
    df['price_change'] = df['price_da'].diff()
    df['hourly_pnl'] = df['position'].shift(1).fillna(0) * df['price_change']
    df['cumulative_pnl'] = df['hourly_pnl'].cumsum().fillna(0)
    
    # Calculate Metrics for Title
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
    win_rate = (len(wins) / total_trades_calc * 100) if total_trades_calc > 0 else 0.0
    
    df['date'] = df['timestamp'].dt.date
    daily_pnl = df.groupby('date')['hourly_pnl'].sum()
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(365)
    else:
        sharpe = 0.0

    print(f"Metrics: WinRate={win_rate:.1f}%, Sharpe={sharpe:.2f}")

    # --- Visualization ---
    # Stride 12 on 15-min data = 3 Hours per Frame. MATCHING H1 SPEED.
    WINDOW_SIZE = 72 
    STRIDE = 12
    FRAMES_IDX = np.arange(WINDOW_SIZE, len(df), STRIDE)
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    line_actual, = ax1.plot([], [], 'c-', label='Price', lw=2)
    line_pos, = ax2.step([], [], where='post', color='blue', lw=2)
    
    scat_buy = ax1.scatter([], [], marker='^', c='lime', s=200, zorder=10)
    scat_sell = ax1.scatter([], [], marker='v', c='red', s=200, zorder=10)
    scat_sl = ax1.scatter([], [], marker='X', c='darkred', s=250, zorder=11)
    scat_tp = ax1.scatter([], [], marker='*', c='gold', s=300, zorder=11)
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Short', 'Flat', 'Long'])
    
    def update(frame_idx):
        start_idx = max(0, frame_idx - WINDOW_SIZE)
        end_idx = frame_idx
        
        curr = df.iloc[start_idx:end_idx]
        if curr.empty: return line_actual,
        
        ct = curr['timestamp'].values
        
        line_actual.set_data(ct, curr['price_da'].values)
        line_pos.set_data(ct, curr['position'].values)
        
        ax1.set_xlim(ct[0], ct[-1])
        min_p, max_p = curr['price_da'].min(), curr['price_da'].max()
        pad = (max_p - min_p) * 0.1 if max_p != min_p else 10
        ax1.set_ylim(min_p - pad, max_p + pad)
        
        t_start, t_end = ct[0], ct[-1]
        valid_markers = [m for m in output_markers if m['ts'] >= t_start and m['ts'] <= t_end]
        
        buys = [m for m in valid_markers if m['label'] == 'BUY']
        sells = [m for m in valid_markers if m['label'] == 'SELL']
        sls = [m for m in valid_markers if str(m['label']).startswith('SL')]
        tps = [m for m in valid_markers if str(m['label']).startswith('TP')]
        
        if buys: scat_buy.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in buys]))
        else: scat_buy.set_offsets(np.empty((0, 2)))
        
        if sells: scat_sell.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in sells]))
        else: scat_sell.set_offsets(np.empty((0, 2)))
 
        if sls: scat_sl.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in sls]))
        else: scat_sl.set_offsets(np.empty((0, 2)))
 
        if tps: scat_tp.set_offsets(np.array([[mdates.date2num(m['ts']), m['price']] for m in tps]))
        else: scat_tp.set_offsets(np.empty((0, 2)))
        
        full_slice = df.iloc[:end_idx]
        cpnl = full_slice['cumulative_pnl'].iloc[-1]
        ts_str = curr.iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M')
        
        title_top = f"2025 H2 Backtest | Sharpe: {sharpe:.2f} | Win Rate: {win_rate:.1f}%"
        title_bot = f"{ts_str} | Net PnL: €{cpnl:,.0f}"
        
        ax1.set_title(f"{title_top}\n{title_bot}", fontsize=12, fontweight='bold')
        
        return line_actual, line_pos, scat_buy, scat_sell, scat_sl, scat_tp
 
    # Interval 100ms = 10fps
    ani = animation.FuncAnimation(fig, update, frames=FRAMES_IDX, interval=100, blit=False)
    
    out_gif = "output_2/backtest_2025_h2_matched.gif"
    if not os.path.exists("output_2"): os.makedirs("output_2")
    
    print(f"Saving Sliding GIF to {out_gif} (Frames: {len(FRAMES_IDX)})...")
    ani.save(out_gif, writer='pillow', fps=10) # FORCE 10 FPS
    print("Done.")

if __name__ == "__main__":
    generate_timelapse()
