import os 
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config


def add_direction_target(df, horizon_days):
    ##if t+horizon close bigger than t close 1, else 0 
    future_close = df["Close"].shift(-horizon_days)
    
    direction = pd.Series(index = df.index, dtype = "float")
    valid_mask = future_close.notna()
    direction[valid_mask] = (future_close[valid_mask] > df["Close"][valid_mask]).astype(int)
    
    df["target_direction"] = direction
    return df


def add_volatility_target(df, horizon_days, method):
    ##volatility until n days after (rolling std)
    daily_returns = df["Close"].pct_change()##change between current and prior
    
    if method == "rolling_std":
        ##std of daily returns between t and horizon days
        future_std = daily_returns.shift(-horizon_days).rolling(
            window = horizon_days).std()
        
        df["target_volatility"] = future_std
    
    else:
        raise ValueError(f"Unknown volatility method {method}")
    
    return df


def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    processed_path = cfg["data"]["processed_data_path"]
    
    direction_horizon = cfg["targets"]["direction"]["horizon_days"]
    volatility_horizon = cfg["targets"]["volatility"]["horizon_days"]
    volatility_method = cfg["targets"]["volatility"]["method"]
    
    
    for ticker in tickers:
        file_path = os.path.join(processed_path, f"{ticker}.csv")
        df = pd.read_csv(file_path, index_col= 0, parse_dates=True, 
                         date_format="%Y-%m-%d")
        
        df = add_direction_target(df,direction_horizon)
        df = add_volatility_target(df, volatility_horizon, volatility_method)
        
        ## in the last lines of the df can't calculate target because no future value
        ##we mark these rows right now
        n_missing_target = df["target_direction"].isna().sum()
        n_missing_vol = df["target_volatility"].isna().sum()
        
        df.to_csv(file_path)
        print(f"{ticker} added target_direction and target_volatility")
        print(f"missing direction number : {n_missing_target}")
        print(f"missing volatility number : {n_missing_vol}")

if __name__ == "__main__":
    main()