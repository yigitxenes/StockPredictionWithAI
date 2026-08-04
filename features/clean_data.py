import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config


def load_raw(ticker, raw_path):
    file_path = os.path.join(raw_path, f"{ticker}.csv")
    df = pd.read_csv(file_path)
    return df


def check_missing(df, ticker):
    print(f"{ticker}, row : {len(df)}")
    
    na_counts = df.isna().sum()
    if na_counts.sum() > 0:
        print("missing columns")
        print(na_counts[na_counts > 0])
    
    else:
        print("no missing value")
        
    all_business_days = pd.bdate_range(df.index.min(), df.index.max())
    missing_days = all_business_days.difference(df.index)
    
    if len(missing_days) > 0: 
        print(f"missing business days :{len(missing_days)}" )
        print(list(missing_days)[:5])
    else:
        print("no missing business day")
        


def clean(df):
    before = len(df)
    
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing_cols = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset = existing_cols)
    
    after = len(df)
    dropped = before - after
    
    if dropped > 0:
        print(f"{dropped} dropped missing row")
        
    return df

def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    raw_path = cfg["data"]["raw_data_path"]
    processed_path = cfg["data"]["processed_data_path"]

    os.makedirs(processed_path, exist_ok=True)

    for ticker in tickers:
        df = load_raw(ticker, raw_path)
        check_missing(df, ticker)
        df_clean = clean(df)

        out_file = os.path.join(processed_path, f"{ticker}.csv")
        df_clean.to_csv(out_file)
        print(f"  Saved: {out_file}  ({len(df_clean)} row)")


if __name__ == "__main__":
    main()