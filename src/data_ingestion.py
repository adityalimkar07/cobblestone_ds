import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

class EnergyChartsFetcher:
    def __init__(self, bzn='DE-LU'):
        self.bzn = bzn
        self.base_url = "https://api.energy-charts.info"
        self.session = requests.Session()

    def fetch_prices(self, start_date, end_date):
        """Fetch Day-Ahead Prices."""
        url = f"{self.base_url}/price"
        params = {
            "bzn": self.bzn,
            "start": start_date,
            "end": end_date
        }
        print(f"Fetching prices from {start_date} to {end_date}...")
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['unix_seconds'], unit='s'),
            'price_da': data['price']
        })
        return df

    def fetch_generation_and_load(self, start_date, end_date):
        """Fetch Load and Generation (Wind/Solar) data."""
        # Note: 'public_power' endpoint often contains Load and Generation By Type
        url = f"{self.base_url}/public_power"
        params = {
            "bzn": self.bzn,
            "start": start_date,
            "end": end_date
        }
        print(f"Fetching public power data from {start_date} to {end_date}...")
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Create base DF with timestamps
        df = pd.DataFrame({'timestamp': pd.to_datetime(data['unix_seconds'], unit='s')})
        
        # Parse the production_types list
        # We need: Load, Wind Onshore, Wind Offshore, Solar
        target_map = {
            'Load': 'load',
            'Wind onshore': 'wind_onshore',
            'Wind offshore': 'wind_offshore',
            'Solar': 'solar'
        }
        
        for item in data['production_types']:
            name = item['name']
            if name in target_map:
                col_name = target_map[name]
                # Handle potential length mismatch if API returns incomplete arrays
                # Usually Energy-Charts aligns them, but good to be safe.
                # Here we assume alignment with 'unix_seconds'
                df[col_name] = item['data']
        
        return df

    def get_dataset(self, start_year, end_year):
        """
        Main runner to get data for full years.
        Splits by year to avoid API timeouts if valid ranges are small.
        Energy-Charts handles yearly ranges well.
        """
        all_prices = []
        all_fundamentals = []
        
        # Iterate over years
        for year in range(start_year, end_year + 1):
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            
            try:
                p_df = self.fetch_prices(start, end)
                all_prices.append(p_df)
                
                f_df = self.fetch_generation_and_load(start, end)
                all_fundamentals.append(f_df)
                
            except Exception as e:
                print(f"Error fetching {year}: {e}")
        
        if not all_prices or not all_fundamentals:
            raise ValueError("No data fetched.")

        prices_df = pd.concat(all_prices, ignore_index=True)
        fundamentals_df = pd.concat(all_fundamentals, ignore_index=True)
        
        # Merge
        # Inner join on timestamp to ensure alignment
        full_df = pd.merge(prices_df, fundamentals_df, on='timestamp', how='inner')
        
        # Clean up
        # Sum Wind
        # Some responses might miss 'wind_offshore' if landlocked, but DE has it.
        if 'wind_offshore' not in full_df.columns:
            full_df['wind_offshore'] = 0
            
        full_df['wind_total'] = full_df.get('wind_onshore', 0) + full_df.get('wind_offshore', 0)
        
        # Set likely columns needed
        final_cols = ['timestamp', 'price_da', 'load', 'solar', 'wind_total', 'wind_onshore', 'wind_offshore']
        # Filter only existing cols
        final_cols = [c for c in final_cols if c in full_df.columns]
        
        return full_df[final_cols]

if __name__ == "__main__":
    fetcher = EnergyChartsFetcher(bzn='DE-LU')
    # Fetch 5 Years (2021-2026)
    start_year = 2021
    end_year = 2026
    
    print(f"Starting data ingestion from {start_year} to {end_year}...")
    df = fetcher.get_dataset(start_year, end_year)
    
    # Remove duplicates
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    
    # Save
    output_path = "data/german_power_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Total Rows: {len(df)}")
    print(df.head())
    print(df.tail())
    
    print("\n[INFO] Data Generation Complete.")
    print("       Run 'python src/qa_checks.py' to verify data integrity.")
