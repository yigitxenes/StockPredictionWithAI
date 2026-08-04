import os
import sys

import pandas as pd
import pandas_ta as ta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config


def add_indicators(df, cfg):
    ind_cfg = cfg["indicators"]
    
    #RSI Relative Strength Index
    df.ta.rsi(length=ind_cfg["rsi"]["length"], append=True)
    
    #MACD Moving Average Convergence Divergence
    df.ta.macd(fast = ind_cfg["macd"]["fast"],
                         slow = ind_cfg["macd"]["slow"],
                         signal = ind_cfg["macd"]["signal"],append = True)
    
    #Bollinger Bands
    df.ta.bbands(length = ind_cfg["bollinger"]["length"], 
                 std = ind_cfg["bollinger"]["std"],
                 append = True)
    
    
    #ATR Volatility
    df.ta.atr(length = ind_cfg["atr"]["length"], append = True)
    
    #ADX Trend strength
    df.ta.adx(length = ind_cfg["adx"]["length"], append = True)
    
    #OBV Volume
    df.ta.obv(append = True)
    
    return df
    
    
def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    processed_path = cfg["data"]["processed_data_path"]
    
    for ticker in tickers:
        file_path = os.path.join(processed_path, f"{ticker}.csv")
        df = pd.read_csv(file_path, index_col= 0, parse_dates=True, date_format="%Y-%m-%d")
        
        before_cols = set(df.columns)
        df = add_indicators(df, cfg)
        new_cols = set(df.columns) - before_cols
        
        df.to_csv(file_path)
        print(f"{ticker} : {len(new_cols)} added new column -> {file_path}")
        print(f"New Columns {sorted(new_cols)}")
        

if __name__ == "__main__":
    main()
        
        
        