import os
import sys

import numpy as np
import pandas as pd 
import yfinance as yf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from features.fetch_data import fetch_ohlcv

def compute_return_features(df, prefix):
    out = pd.DataFrame(index= df.index)
    out[f"{prefix}_level"] = df["Close"]
    out[f"{prefix}_return_1d"] = df["Close"].pct_change()
    out[f"{prefix}_return_5d"] = df["Close"].pct_change(5)
    return out


def fetch_all_cross_asset_series(cfg):
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]
    cross_cfg = cfg["data"]["cross_asset_tickers"]
    
    all_tickers = (
        cross_cfg["market"] + cross_cfg["volatility"] + cross_cfg["macro"] + 
        list(set(cfg["data"]["sector_map"].values()))
    )
    
    all_tickers = sorted(set(all_tickers))
    
    series = {}
    for ticker in all_tickers:
        print(f"Fetching : {ticker}")
        df = fetch_ohlcv(ticker, start_date, end_date, "1d")
        
        if df.empty:
            print(f"WARNING : No data found for {ticker}")
            continue
        
        #Taking out special characters from the ticker name for safe column names
        safe_name = ticker.replace("^", "").replace("-", "_").replace(".","_").lower()
        series[safe_name] = compute_return_features(df, safe_name)
    
    return series

def build_cross_asset_features(cfg):
    series = fetch_all_cross_asset_series(cfg)
    
    combined = None
    for name, df in series.items():
        if combined is None:
            combined = df
        else:
            combined = combined.join(df, how = "outer")
    
    combined.index.name = "Date"
    combined = combined.reset_index()
    return combined


def build_sector_features(cfg):
    sector_map = cfg["data"]["sector_map"]
    start_date = cfg["data"]["start_date"]
    end_date = cfg["data"]["end_date"]
    
    unique_sectors = sorted(set(sector_map.values()))
    sector_data = {}
    
    for sector_etf in unique_sectors:
        print(f"Fetching sector ETF {sector_etf}")
        df = fetch_ohlcv(sector_etf, start_date, end_date, "1d")
        if df.empty:
            print(f"WARNING : No data fetched for {sector_etf}")
            continue
        
        feats = compute_return_features(df, "sector")
        feats.index.name = "Date"
        sector_data[sector_etf] = feats.reset_index()
    
    rows = []
    for ticker, sector_etf in sector_map.items():
        if sector_etf not in sector_data:
            continue
        df = sector_data[sector_etf].copy()
        df["ticker"] = ticker
        rows.append(df)
    
    return pd.concat(rows, ignore_index= True)


def merge_cross_asset_and_sector(technical_df, cross_asset_df, sector_df):
    merged = technical_df.merge(cross_asset_df, on = "Date", how = "left")
    merged = merged.merge(sector_df, on = ["ticker", "Date"], how = "left")
    return merged


def main():
    cfg = load_config()
    processed_path = cfg["data"]["processed_data_path"]
    
    technical_path = os.path.join(processed_path, "combined_datasets_with_sentiments.csv")
    technical_df = pd.read_csv(technical_path, parse_dates= ["Date"])
    print(f"Technical + sentiment dataset : {len(technical_df)} rows")
    
    cross_asset_df = build_cross_asset_features(cfg)
    print(f"Cross-asset Feature dataset : {len(cross_asset_df)} rows"
        f"{cross_asset_df['Date'].min().date()} -> {cross_asset_df['Date'].max().date()}")
    
    sector_df = build_sector_features(cfg)
    print(f"Sector dataset : {len(sector_df)} rows")
    
    merged= merge_cross_asset_and_sector(technical_df, cross_asset_df, sector_df)
    
    new_cols = [c for c in merged.columns if c not in technical_df]
    still_missing = merged[new_cols].isna().sum()
    print(f"\nNewly added columns: {new_cols}")
    print(f"Missing columns:\n{still_missing}")

    out_file = os.path.join(processed_path, "combined_datasets_with_sentiments_and_crossasset.csv")
    merged.to_csv(out_file, index=False)

    print(f"\nSaved: {out_file}")
    print(f"Total rows: {len(merged)} (Is it same with technical: {len(merged) == len(technical_df)})")
    
    
if __name__ == "__main__":
    main()