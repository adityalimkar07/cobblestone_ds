import pandas as pd
import numpy as np
import os

def ts_argmax(series, window):
    return series.rolling(window).apply(lambda x: np.argmax(x), raw=True)

def delta(series, periods):
    return series.diff(periods)

def scale(series):
    return (series - series.mean()) / series.std()

def vwap(data):
    # VWAP usually requires Intraday Volume. We use Load as volume proxy.
    return (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

def calculate_alphas(df):
    print("Generating 101 Alphas (Adapted for Power Market)...")
    
    # Map Power Data to OHLCV format
    # Price = Close
    # Load = Volume
    # Open, High, Low are approximated from Close for this exercise
    data = df.copy()
    data['close'] = data['price_da']
    data['volume'] = data['load'] # Proxy
    data['open'] = data['close'].shift(1).ffill() # Fixed: ffill only
    data['high'] = data[['open', 'close']].max(axis=1) 
    data['low'] = data[['open', 'close']].min(axis=1)  
    data['returns'] = data['close'].pct_change()
    data['vwap'] = vwap(data)
    data['adv20'] = data['volume'].rolling(window=20).mean()
    
    # helper for rolling rank
    def rolling_rank(series, window=168):
        # 168 hours = 1 week rolling rank
        return series.rolling(window).rank(pct=True)

    try:
        # A1: Volume change vs Return
        v_diff = np.log(data['volume']).diff()
        ret = (data['close'] - data['open']) / data['open']
        data['Alpha_1'] = rolling_rank(-1 * v_diff.rolling(6).corr(ret))
        
        # A2: 
        data['Alpha_2'] = rolling_rank(-1 * data['open'].rolling(10).corr(data['volume']))
        
        # A3: FIXED - Use proper rolling rank with min_periods
        # Inner rank: rank within rolling 9-period window
        inner_rank = data['low'].rolling(9, min_periods=2).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 1 else 0.5,
            raw=False
        )
        data['Alpha_3'] = rolling_rank(-1 * inner_rank)
        
        # A4: 
        open_vwap_diff = data['open'] - (data['vwap'].rolling(10).sum() / 10)
        close_vwap_abs = np.abs(data['close'] - data['vwap'])
        data['Alpha_4'] = open_vwap_diff * rolling_rank(-1 * close_vwap_abs)
        
        # A5: 
        data['Alpha_5'] = rolling_rank(-1 * data['open'].rolling(10).corr(data['volume']))
        
        # A6: 
        data['Alpha_6'] = rolling_rank(-1 * (data['open'].rolling(5).sum() * data['returns'].rolling(5).sum()))
        
        # A9: Close delta vs Volume delta (Already safe, no rank)
        data['Alpha_9'] = np.sign(delta(data['volume'], 1)) * -1 * delta(data['close'], 1)
        
        # A12: 
        data['Alpha_12'] = rolling_rank(-1 * data['high'].rolling(5).corr(data['volume']))
        
        # A28: Target (Direction)
        data['Alpha_28'] = np.where(data['close'].diff() > 0, 1, 0)
        
        # Filling gaps - FIXED: Use ffill to preserve more data
        # Replace inf/-inf with NaN first
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill to propagate last valid values
        data = data.ffill()
        
        # For any remaining NaNs at the start, fill with 0
        data = data.fillna(0)
        
        # Select Alphas
        cols = [c for c in data.columns if 'Alpha_' in c]
        print(f"Index Columns: {cols}")
        
        # We need the Target Price for Fair Value modeling
        # 'price_da' is the original column
        return data[['timestamp', 'price_da'] + cols]
        
    except Exception as e:
        print(f"Error in Alpha Gen: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

if __name__ == "__main__":
    input_path = "data/german_power_data.csv"
    if os.path.exists(input_path):
        df = pd.read_csv(input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        alpha_df = calculate_alphas(df)
        alpha_df.to_csv("data/alpha_features.csv", index=False)
        print("Saved data/alpha_features.csv")
    else:
        print("No input data.")
