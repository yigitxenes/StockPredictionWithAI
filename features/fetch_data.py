import os
import sys

import pandas as pd
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config

def fetch_ohlcv(ticker : str, start_date : str, end_date : str, interval : str):
    df = yf.download(ticker, start= start_date, end= end_date, interval= interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df

def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]
    interval = cfg["data"]["interval"]
    raw_path = cfg["data"]["raw_data_path"]
    
    os.makedirs(raw_path, exist_ok=True)
    
    for ticker in tickers:
        print(f"{ticker}, ({start_date} -> {end_date})")
        df = fetch_ohlcv(ticker, start_date, end_date, interval)
        
        if df.empty:
            print(f"{ticker} couldn't retrieve data.")
            continue
        
        out_file = os.path.join(raw_path, f"{ticker}.csv")
        df.to_csv(out_file)
        
        print(f"{out_file} saved.")
        

if __name__ == "__main__":
    main()