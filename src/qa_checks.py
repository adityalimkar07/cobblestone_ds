import pandas as pd
import numpy as np
import os

def run_qa_checks():
    print("--- DATA QUALITY ASSURANCE REPORT ---\n")
    
    file_path = "data/german_power_data.csv"
    if not os.path.exists(file_path):
        print(f"CRITICAL: Data file {file_path} NOT found.")
        return

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Source Verification (Metadata Check)
    print(f"1. FILE METADATA")
    print(f"   - Path: {file_path}")
    print(f"   - Rows: {len(df)}")
    print(f"   - Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   - Source Logic: 'src/data_ingestion.py' -> 'api.energy-charts.info' (Fraunhofer ISE)")
    print("")

    # 2. Completeness (Missing Values)
    print("2. COMPLETENESS CHECK")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   [PASS] No missing values found in any column.")
    else:
        print("   [FAIL] Missing values detected:")
        print(missing[missing > 0])
    print("")

    # 3. Time Continuity (Gaps)
    print("3. TEMPORAL CONTINUITY")
    time_diffs = df['timestamp'].diff().dropna()
    expected_diff = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs != expected_diff]
    
    if len(gaps) == 0:
        print("   [PASS] Time series is continuous (Hourly). No gaps.")
    else:
        print(f"   [FAIL] {len(gaps)} non-hourly gaps detected.")
        print(gaps.head())
    print("")

    # 4. Value Validity (Outliers/bounds)
    print("4. VALUE VALIDITY CHECKS")
    # Price: Can be negative, but usually > -500 and < 4000
    min_price = df['price_da'].min()
    max_price = df['price_da'].max()
    print(f"   - Price Range: [{min_price:.2f}, {max_price:.2f}] EUR/MWh")
    
    if min_price < -500 or max_price > 4000:
        print("   [WARN] Extreme price outliers detected.")
    else:
        print("   [PASS] Prices within expected European Spot bounds.")

    # Fundamentals: Must be >= 0
    neg_load = df[df['load'] < 0]
    neg_solar = df[df['solar'] < 0]
    
    if len(neg_load) == 0 and len(neg_solar) == 0:
        print("   [PASS] Fundamental drivers (Load/Solar) are non-negative.")
    else:
        print(f"   [FAIL] Negative physical values detected (Load: {len(neg_load)}, Solar: {len(neg_solar)})")
        
    print("\n--- END OF REPORT ---")

if __name__ == "__main__":
    run_qa_checks()
