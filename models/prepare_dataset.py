import os
import sys

import pandas as pd 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config

def combine_datasets(tickers, processed_path):
    alldfs = []
    
    for ticker in tickers:
        file_path = os.path.join(processed_path, f"{ticker}.csv")
        df = pd.read_csv(file_path, parse_dates=["Date"]
                         ,date_format= "%Y-%m-%d")
        
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        
        df["ticker"] = ticker
        alldfs.append(df)
    
    combined = pd.concat(alldfs, axis=0)
    combined = combined.sort_values("Date").reset_index(drop=True)
    
    return combined

def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    processed_path = cfg["data"]["processed_data_path"]
    
    combined = combine_datasets(tickers, processed_path)
    
    out_file = os.path.join(processed_path, "combined_datasets.csv")
    combined.to_csv(out_file, index = False)
    
    print(f"Out File Created Successfully : {out_file}")
    print(f"Total number of rows : {len(combined)}")
    print(f"Ticker Distribution : {combined["ticker"].value_counts()}")
    
if __name__ == "__main__":
    main()