##Lookahead Bias Test
"""The value of a technical indicator at time t must remain the same
whether or not the dataset is cut off after t.
Otherwise, the indicator uses future data (data leakage).
Just testing with the RSI pandas-ta is expected to behave same with other indicators"""

import os
import sys

import pandas as pd
import pandas_ta as ta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config


def compute_rsi_full(df, length):
    ##Rsi value over full dataset
    return ta.rsi(df["Close"], length = length)

def compute_rsi_truncated(df, length, cutoff_idx):
    ##Cuts the dataset till the cutoff index then calculates rsi
    truncated = df.iloc[: cutoff_idx + 1]
    rsi_series = ta.rsi(truncated["Close"], length = length)
    return rsi_series.iloc[-1] # just the last value


def test_rsi_no_lookahead(ticker, raw_or_processed_path, rsi_length,
                          n_checkpoints : int = 5):
    
    file_path = os.path.join(raw_or_processed_path, f"{ticker}.csv")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True,
                     date_format="%Y-%m-%d")
    
    full_rsi = compute_rsi_full(df, rsi_length)
    
    valid_start = rsi_length + 5
    valid_end = len(df) - 1
    step = max((valid_end - valid_start) // n_checkpoints, 1)
    checkpoints = range(valid_start, valid_end, step)
    
    all_passed = True
    
    for idx in checkpoints:
        full_value = full_rsi.iloc[idx]
        truncated_value = compute_rsi_truncated(df, rsi_length, idx)
        
        #A tolerance of 1e-6 prevents false alarms 
        #caused by float calculation differences (not actual leakage).
        if abs(full_value - truncated_value) > 1e-6:
            print(f"FAILED IDX = {idx}: full rsi = {full_value}, truncated = {truncated_value}")
            all_passed = False
    
    return all_passed



def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    rsi_length = cfg["indicators"]["rsi"]["length"]
    
    
    processed_path = cfg["data"]["processed_data_path"]
    
    print(f"LookAhead Bias Testing")
    all_tickers_passed = True
    
    for ticker in tickers:
        print(f"{ticker} : ")
        passed = test_rsi_no_lookahead(ticker, processed_path, rsi_length)
        status = "PASSED" if passed else "FAILED"
        print(f"result : {status}")
        all_tickers_passed = all_tickers_passed and passed
    
    if all_tickers_passed:
        print("All tickers passed test is succesfull")
        
    else:
        print("There is lookahead bias in some tickers")
    
if __name__ == "__main__":
    main()