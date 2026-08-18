import os
import sys
import json
from datetime import datetime

import pandas as pd
from google import genai
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from agent.write_report import load_model_bundle
from agent.live_data_fetch import get_live_technical_row, get_live_sentiment, build_feature_row

load_dotenv()

LOG_FILE = "data/processed/prospective_predictions.csv"

def get_previous_logged_date(ticker):
    if not os.path.exists(LOG_FILE):
        return None
    existing = pd.read_csv(LOG_FILE)
    ticker_rows = existing[existing["ticker"] == ticker]
    if ticker_rows.empty:
        return None
    return ticker_rows["prediction_date"].max()

def log_prediction_for_ticker(ticker, cfg, gemini_client):
    company_name = cfg["data"]["company_names"].get(ticker, ticker)
    direction_bundle = load_model_bundle("final_direction_model")

    technical_row, as_of_date = get_live_technical_row(ticker, cfg)

    model_name = cfg["agent"]["prototype_model"]
    sentiment_result = get_live_sentiment(ticker, company_name, gemini_client, model_name,
                                           reference_date=as_of_date)

    X = build_feature_row(technical_row, sentiment_result, ticker,
                           direction_bundle["feature_cols"], direction_bundle["tickers"])
    direction_proba = direction_bundle["model"].predict_proba(X)[0][1]
    predicted_up = int(direction_proba >= 0.5)

    return {
        "ticker": ticker,
        "prediction_date": str(as_of_date.date()),   # tahminin yapildigi gun (t)
        "logged_at": datetime.now().isoformat(),
        "close_price_at_prediction": float(technical_row["Close"]),
        "predicted_proba_up": float(direction_proba),
        "predicted_direction": predicted_up,          # 1 = yukselis bekleniyor
        "actual_next_close": None,                     # sonradan doldurulacak
        "actual_direction": None,                       # sonradan doldurulacak
        "correct": None,                                 # sonradan doldurulacak
    }


def append_to_log(row):
    file_exists = os.path.exists(LOG_FILE)
    df_row = pd.DataFrame([row])
    df_row.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)


def already_logged_today(ticker, as_of_date_str):
    if not os.path.exists(LOG_FILE):
        return False
    existing = pd.read_csv(LOG_FILE)
    return ((existing["ticker"] == ticker) & (existing["prediction_date"] == as_of_date_str)).any()


def main():
    cfg = load_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    tickers = cfg["data"]["tickers"]
    if "TSLA" in tickers:
        tickers = [t for t in tickers if t != "TSLA"]  # sentiment kapsami disinda, tutarlilik icin atla

    for ticker in tickers:
        try:
            technical_row, as_of_date = get_live_technical_row(ticker, cfg)
            as_of_date_str = str(as_of_date.date())

            prev_date = get_previous_logged_date(ticker)
            if prev_date is not None and as_of_date_str == prev_date:
                print(f"UYARI: {ticker} icin tarih ilerlemedi ({as_of_date_str}) -- "
                  f"yfinance verisi gecikmis olabilir, bugun atlaniyor")
                continue

            if already_logged_today(ticker, str(as_of_date.date())):
                print(f"{ticker}: {as_of_date.date()} icin zaten kayitli, atlaniyor")
                continue

            print(f"{ticker}: tahmin uretiliyor...")
            row = log_prediction_for_ticker(ticker, cfg, client)
            append_to_log(row)
            print(f"  Kaydedildi: {row['prediction_date']}, proba_up={row['predicted_proba_up']:.3f}")

        except Exception as e:
            print(f"{ticker}: HATA {type(e).__name__}: {e}")
            continue


if __name__ == "__main__":
    main()