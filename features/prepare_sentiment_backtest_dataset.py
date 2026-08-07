import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from features.fetch_data import fetch_ohlcv
from features.clean_data import clean
from features.indicators import add_indicators
from features.targets import add_direction_target, add_volatility_target
from features.finalize_dataset import finalize

RAW_PATH = "data/raw_sentiment_backtest"
PROCESSED_PATH = "data/processed_sentiment_backtest"


def main():
    cfg = load_config()
    tickers = cfg["data"]["sentiment_backtest_tickers"]
    start_date = cfg["data"]["sentiment_backtest_start"]
    end_date = cfg["data"]["sentiment_backtest_end"]
    interval = cfg["data"]["interval"]
    
    direction_horizon = cfg["targets"]["direction"]["horizon_days"]
    volatility_horizon = cfg["targets"]["volatility"]["horizon_days"]
    volatility_method = cfg["targets"]["volatility"]["method"]
    
    os.makedirs(RAW_PATH, exist_ok = True)
    os.makedirs(PROCESSED_PATH, exist_ok = True)
    
    for ticker in tickers:
        print(f"------------{ticker}-----------")
        
        #FETCHING
        df = fetch_ohlcv(ticker, start_date, end_date, interval)
        if df.empty:
            print(f"No data for {ticker}")
            continue
        
        raw_file = os.path.join(RAW_PATH, f"{ticker}.csv")
        df.to_csv(raw_file)
        print(f"Raw data saved -> {raw_file} ::: {len(df)} Rows")
        
        #CLEANING
        df = pd.read_csv(raw_file)
        df_clean = clean(df)
        print(f"Number of rows after cleaning : {len(df_clean)}")
        
        #INDICATORS
        df_clean = df_clean.set_index(df_clean.columns[0])
        df_clean.index.name = "Date"
        df_ind = add_indicators(df_clean, cfg)
        
        #TARGETS
        df_ind = add_direction_target(df_ind, direction_horizon)
        df_ind = add_volatility_target(df_ind, volatility_horizon, volatility_method)
        
        #FINALIZE
        df_final = finalize(df_ind, ticker)
        df_final = df_final.sort_index()
        
        processed_file = os.path.join(PROCESSED_PATH, f"{ticker}.csv")
        df_final.to_csv(processed_file)
        
        print(f"Saved {processed_file} ::::: Number of Rows -> {len(df_final)}")
        

if __name__ == "__main__":
    main()