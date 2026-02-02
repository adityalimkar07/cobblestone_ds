import yfinance as yf
import pandas as pd
import os

def fetch_ticker_data(ticker, name, start_date, end_date):
    print(f"  Downloading {name} ({ticker}) OHLCV...")
    try:
        # Fetch OHLCV
        data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        if data.empty:
            print(f"    Warning: No data for {ticker}")
            return None
            
        # Handle MultiIndex columns if present (yfinance update)
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data.columns = data.columns.droplevel(1)
            except:
                pass
                
        # Keep only Close/Volume
        if 'Close' in data.columns:
            return data[['Close']].rename(columns={'Close': name})
        elif 'Adj Close' in data.columns:
            return data[['Adj Close']].rename(columns={'Adj Close': name})
        else:
            return None
            
    except Exception as e:
        print(f"    Error fetching {ticker}: {e}")
        return None

def fetch_all_commodities(start_date="2021-01-01", end_date="2026-01-01"):
    print("Fetching Commodities Data (Oil, Gas, Coal, Carbon)...")
    
    # Tickers
    tickers = {
        'Brent_Oil': 'BZ=F',
        'Natural_Gas': 'NG=F',
        'Coal_API2': 'MTF=F', # Rotterdam Coal Futures often tricky on Yahoo, trying proxy or ETF? Using Futures.
        'Carbon_EUA': 'KEUA.F' # KraneShares European Carbon Allowance Strategy ETF as proxy for EUA? Or actual futures 'CFI2Z24.NYM'? Yahoo is bad for Carbon.
        # Simplification: Use Brent and NatGas as main proxies for now.
    }
    
    dfs = []
    for name, ticker in tickers.items():
        df = fetch_ticker_data(ticker, name, start_date, end_date)
        if df is not None:
            dfs.append(df)
            
    if not dfs:
        print("No commodities fetched.")
        return
        
    # Merge all
    full_df = pd.concat(dfs, axis=1)
    full_df = full_df.ffill().fillna(method='bfill') # Daily data needs filling weekends if merged with hourly later
    
    # Resample to Hourly to match Power Data? 
    # Or just save daily.
    # User requirement was hourly. We upsample.
    full_df.index = pd.to_datetime(full_df.index)
    full_df = full_df.resample('1H').ffill()
    
    output_path = "data/commodities_data.csv"
    full_df.to_csv(output_path)
    print(f"Commodities saved to {output_path}")

if __name__ == "__main__":
    fetch_all_commodities()
