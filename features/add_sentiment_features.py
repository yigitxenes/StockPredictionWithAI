import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from models.prepare_dataset import combine_datasets

SENTIMENT_COLS = ["sentiment_score", "sentiment_confidence", "sentiment_std", "news_count_daily"]

def load_sentiment(cfg):
    path = cfg["data"]["sentiment_output_path"]
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns= {"date": "Date"})
    return df


def reindex_sentiment_to_technical_dates(sentiment_df, technical_dates_by_ticker):
    result_parts = []
    
    for ticker, dates in technical_dates_by_ticker.items():
        ticker_sent = sentiment_df[sentiment_df["ticker"] == ticker].set_index("Date")
        
        full_index = pd.DatetimeIndex(sorted(dates))
        reindexed = ticker_sent.reindex(full_index)
        
        reindexed["ticker"] = ticker
        reindexed["news_count_daily"] = reindexed["news_count_daily"].fillna(0)
        reindexed["sentiment_score"] = reindexed["sentiment_score"].fillna(0.0)
        reindexed["sentiment_confidence"] = reindexed["sentiment_confidence"].fillna(0.0)
        reindexed["sentiment_std"] = reindexed["sentiment_std"].fillna(0.0)
        
        reindexed.index.name = "Date"
        result_parts.append(reindexed.reset_index())
    
    return pd.concat(result_parts, ignore_index= True)


def add_ema_momentum(df, ema_span):
    df = df.sort_values(["ticker", "Date"]).reset_index(drop = True)
    
    df["sentiment_ema"] = (
        df.groupby("ticker")["sentiment_score"]
        .transform(lambda s: s.ewm(span = ema_span, adjust = False).mean())
    )
    
    df["sentiment_momentum"] = (
        df.groupby("ticker")["sentiment_score"]
        .transform(lambda s: s.diff().fillna(0))
    )
    
    return df

def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    processed_path = cfg["data"]["processed_data_path"]
    ema_span = cfg["sentiment_features"]["ema_span"]
    
    technical_df = combine_datasets(tickers, processed_path)
    print(f"Technical dataset : {len(technical_df)} rows, {technical_df["ticker"].nunique()} tickers")
    
    sentiment_df = load_sentiment(cfg)
    print(f"Sentiment dataset : {len(sentiment_df)} rows")
    
    missing_in_sentiment = set(tickers) - set(sentiment_df["ticker"].unique())
    if missing_in_sentiment:
        print(f"Warning : Missing in sentiment tickers ==== {missing_in_sentiment}")
    
    technical_dates_by_ticker = {
        ticker : technical_df[technical_df["ticker"] == ticker]["Date"].tolist()
        for ticker in tickers
    }
    
    reindexed_sentiment = reindex_sentiment_to_technical_dates(sentiment_df, technical_dates_by_ticker)
    reindexed_sentiment = add_ema_momentum(reindexed_sentiment, ema_span)
    
    merge_cols = ["ticker", "Date"] + SENTIMENT_COLS + ["sentiment_ema", "sentiment_momentum"]
    merged = technical_df.merge(reindexed_sentiment[merge_cols], on = ["ticker", "Date"], how = "left")
    
    before = len(merged)
    still_missing = merged[SENTIMENT_COLS].isna().any(axis = 1).sum()
    if still_missing > 0:
        print(f"WARNING : After merging there is {still_missing} rows missing")
    
    out_file = os.path.join(processed_path, "combined_datasets_with_sentiments.csv")
    merged.to_csv(out_file, index = False)
    
    print(f"Saved : {out_file}")
    print(f"Total rows: {len(merged)} (Is it same with technical: {len(merged) == before})")

    
if __name__ == "__main__":
    main()