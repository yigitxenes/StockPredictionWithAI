import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config


def finalize(df, ticker):
    before = len(df)
    
    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0]
    
    if len(na_counts) > 0:
        print(f"NaN rows before cleaning")
        print(f"  {na_counts.to_dict()}")
        
        
    df_clean = df.dropna()
    after = len(df_clean)
    
    print(f"{ticker}: {before} -> {after}, {before - after} rows removed")
    
    return df_clean


def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    processed_path = cfg["data"]["processed_data_path"]
    
    for ticker in tickers:
        file_path = os.path.join(processed_path, f"{ticker}.csv")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True,
                         date_format="%Y-%m-%d")
        
        df_final = finalize(df,ticker)
        
        df_final = df_final.sort_index()
        
        df_final.to_csv(file_path)
    

if __name__ == "__main__":
    main()