import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from features.fetch_data import fetch_ohlcv

LOG_FILE = "data/processed/prospective_predictions.csv"


def main():
    cfg = load_config()
    df = pd.read_csv(LOG_FILE, parse_dates=["prediction_date"])

    incomplete = df[df["actual_direction"].isna()]
    print(f"Doldurulacak {len(incomplete)} satir var")

    for idx, row in incomplete.iterrows():
        ticker = row["ticker"]
        pred_date = row["prediction_date"]

        end_date = (pred_date + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        price_df = fetch_ohlcv(ticker, pred_date.strftime("%Y-%m-%d"), end_date, "1d")

        future_rows = price_df[price_df.index > pred_date]
        if future_rows.empty:
            continue  # henuz bir sonraki islem gunu olusmamis

        next_close = future_rows.iloc[0]["Close"]
        actual_direction = int(next_close > row["close_price_at_prediction"])
        correct = int(actual_direction == row["predicted_direction"])

        df.loc[idx, "actual_next_close"] = next_close
        df.loc[idx, "actual_direction"] = actual_direction
        df.loc[idx, "correct"] = correct

        print(f"{ticker} {pred_date.date()}: tahmin={row['predicted_direction']}, "
              f"gercek={actual_direction}, {'DOGRU' if correct else 'YANLIS'}")

    df.to_csv(LOG_FILE, index=False)
    print(f"\nGuncellendi: {LOG_FILE}")

    completed = df.dropna(subset=["correct"])
    if len(completed) > 0:
        accuracy = completed["correct"].mean()
        print(f"\nSu ana kadarki prospektif accuracy: {accuracy:.1%} ({len(completed)} tahmin)")


if __name__ == "__main__":
    main()