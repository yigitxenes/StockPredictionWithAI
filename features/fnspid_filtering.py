import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config


def get_fnspid_tickers(cfg):
    tickers = cfg["data"]["tickers"]
    fnspid_map = cfg["data"].get("fnspid_ticker_map", {})
    
    return [fnspid_map.get(t, t) for t in tickers]


def filter_large_csv(file_path, fnspid_tickers, ticker_col,
                     keep_cols, chunksize : int = 100000):

    tickers_set = set(fnspid_tickers)
    
    filtered_chunks = []
    total_rows_read = 0
    total_rows_kept = 0
    
    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory= False,
                             usecols= keep_cols, on_bad_lines="skip"):
        
        total_rows_read += len(chunk)
        
        mask = chunk[ticker_col].isin(tickers_set)
        filtered = chunk[mask]
        
        if not filtered.empty:
            filtered_chunks.append(filtered)
            total_rows_kept += len(filtered)
            
        print(f"Processed Rows : {total_rows_read:,}, Kept : {total_rows_kept:,}")
    print()
    result = pd.concat(filtered_chunks, ignore_index= True)
    return result


def main():
    cfg = load_config()
    fnspid_map = cfg["data"].get("fnspid_ticker_map", {})
    fnspid_tickers = get_fnspid_tickers(cfg)
    
    fnspid_path = "data/raw/All_external.csv"
    ticker_col = "Stock_symbol"
    keep_cols = ["Date", "Article_title", "Stock_symbol", "Publisher", "Article"]
    
    print(f"Looking for these symbols on FNSPID : {fnspid_tickers}")
    
    filtered = filter_large_csv(fnspid_path, fnspid_tickers, ticker_col, keep_cols)
    
    #FB -> Meta transition fix
    reverse_map = {v: k for k, v in fnspid_map.items()}
    filtered["Stock_symbol"] = filtered["Stock_symbol"].replace(reverse_map)
    
    out_path = "data/raw/fnspid_filtered.csv"
    filtered.to_csv(out_path, index = False)
    print(f"Saved File : {out_path}, number of rows : {len(filtered)}")

    print("Ticker Distribution : ")
    print(filtered["Stock_symbol"].value_counts())
    
if __name__ == "__main__":
    main()