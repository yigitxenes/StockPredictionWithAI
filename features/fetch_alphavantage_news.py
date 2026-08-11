import os
import sys
import time
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.alphavantage.co/query"
RAW_OUT_PATH = "data/raw/alphavantage_filtered.csv"

RPM_LIMIT = 5
RPD_LIMIT = 25
SLEEP_BETWEEN_REQUESTS = 60 / RPM_LIMIT + 1
FINAL_END = "20250101T0000"

def fetch_news_window(ticker, time_from, time_to, api_key, limit = 1000):
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "limit": limit,
        "sort": "EARLIEST",
        "apikey": api_key,
    }
    
    response = requests.get(BASE_URL, params= params)
    response.raise_for_status()
    data = response.json()
    
    if "Information" in data or "Note" in data:
        raise RuntimeError(data.get("Information") or data.get("Note"))
    
    return data.get("feed", [])

def to_fnspid_schema(articles, ticker):
    rows = []
    for art in articles:
        rows.append({
            "Date" : art["time_published"][:8],
            "Article_title" : art.get("title", ""),
            "Stock_symbol" :  ticker,
            "Publisher" : art.get("source", "")
            ,"Article" : art.get("summary", "")
        })
    
    return rows

def load_existing():
    if not os.path.exists(RAW_OUT_PATH):
        return pd.DataFrame(columns=["Date", "Article_title", "Stock_symbol", "Publisher", "Article"])
    return pd.read_csv(
        RAW_OUT_PATH, header=None,
        names=["Date", "Article_title", "Stock_symbol", "Publisher", "Article"],
    )

def get_resume_point(existing_df, ticker, default_start):
    ticker_rows = existing_df[existing_df["Stock_symbol"] == ticker]
    if ticker_rows.empty:
        return default_start
    max_date = ticker_rows["Date"].astype(str).max()
    return f"{max_date}T2359"



def main():
    cfg = load_config()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHAVANTAGE API KEY NOT FOUND")
    
    tickers = cfg["data"]["sentiment_backtest_tickers"]
    gap_start_default = "20200609T0000"

    existing_df = load_existing()
    existing_df = existing_df.drop_duplicates(subset=["Date", "Article_title", "Stock_symbol"])
    existing_keys = set(zip(existing_df["Date"].astype(str), existing_df["Article_title"], existing_df["Stock_symbol"]))


    request_count = 0
    new_rows_total = 0


    for ticker in tickers:
        if request_count >= RPD_LIMIT:
            print("REACHED THE DAILY REQUEST LIMIT")
            break
        time_from = get_resume_point(existing_df, ticker, gap_start_default)

        if time_from >= FINAL_END:
            print(f"{ticker}: already completed.")
            continue

        print(f"{ticker} : {gap_start_default} -> {FINAL_END}")

        try:
            articles = fetch_news_window(ticker, time_from, FINAL_END, api_key)
            request_count += 1
            new_records = to_fnspid_schema(articles, ticker)

            fresh = [
                r for r in new_records
                if (r["Date"], r["Article_title"], r["Stock_symbol"]) not in existing_keys
            ]


            print(f"  {len(articles)} makale donduruldu, {len(fresh)} tanesi yeni")

            if fresh:
                df_new = pd.DataFrame(fresh)
                df_new.to_csv(RAW_OUT_PATH, mode="a", header=False, index=False)
                new_rows_total += len(fresh)
                for r in fresh:
                    existing_keys.add((r["Date"], r["Article_title"], r["Stock_symbol"]))

            if len(articles) == 1000:
                print(f"  UYARI: {ticker} hala 1000 tavanina takiliyor, sonraki calistirmada devam edecek")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except RuntimeError as e:
            print(f"  Hata/limit: {e}")
            break

    print(f"\nBu oturumda eklenen yeni satir: {new_rows_total}")


if __name__ == "__main__":
    main()