import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from features.fetch_alphavantage_news import (
    RAW_OUT_PATH, FINAL_END, load_existing, get_resume_point,
)

def is_effectively_complete(resume_point, final_end, tolerance_days=2):
    """Resume point ile final_end arasinda sadece birkac gunluk fark varsa
    ve bu fark tekrar tekrar kapanmiyorsa (yani gercekte haber yoksa)
    complete sayilir."""
    rp_date = datetime.strptime(resume_point[:8], "%Y%m%d")
    fe_date = datetime.strptime(final_end[:8], "%Y%m%d")
    return (fe_date - rp_date).days <= tolerance_days

def audit_coverage():
    cfg = load_config()
    tickers = cfg["data"]["sentiment_backtest_tickers"]
    gap_start_default = "20200609T0000"

    existing_df = load_existing()
    existing_df = existing_df.drop_duplicates(
        subset=["Date", "Article_title", "Stock_symbol"]
    )

    print(f"{'Ticker':8s} {'Makale':>8s} {'Min Tarih':>10s} {'Max Tarih':>10s} "
          f"{'Resume Point':>14s} {'Durum':>12s}")
    print("-" * 70)

    incomplete_tickers = []

    for ticker in tickers:
        ticker_rows = existing_df[existing_df["Stock_symbol"] == ticker]
        n_articles = len(ticker_rows)

        if ticker_rows.empty:
            print(f"{ticker:8s} {0:8d} {'-':>10s} {'-':>10s} {'-':>14s} {'HIC VERI YOK':>12s}")
            incomplete_tickers.append(ticker)
            continue

        min_date = ticker_rows["Date"].astype(str).min()
        max_date = ticker_rows["Date"].astype(str).max()
        resume_point = get_resume_point(existing_df, ticker, gap_start_default)

        is_complete = is_effectively_complete(resume_point, FINAL_END)
        status = "TAMAM" if is_complete else "EKSIK"
        if not is_complete:
            incomplete_tickers.append(ticker)

        print(f"{ticker:8s} {n_articles:8d} {min_date:>10s} {max_date:>10s} "
              f"{resume_point:>14s} {status:>12s}")

    print("-" * 70)
    if incomplete_tickers:
        print(f"\nEksik ticker'lar ({len(incomplete_tickers)}): {incomplete_tickers}")
        print("-> features/fetch_alphavantage_news.py'yi tekrar calistir.")
    else:
        print("\nTum ticker'lar 2025-01-01'e kadar tamamlanmis gorunuyor.")
        print("(Not: bu sadece resume-point kontrolu; asagidaki ay-bazli")
        print(" bosluk kontrolunu de calistirip supheli 'sessiz bosluklari' incele.)")

    # Ay bazli haber sayisi - suphesiz uzun bosluklari yakalamak icin
    print("\n=== Ay bazli haber sayisi (0 veya cok dusuk olan aylara dikkat) ===")
    existing_df["date_dt"] = pd.to_datetime(existing_df["Date"], format="%Y%m%d", errors="coerce")
    existing_df["year_month"] = existing_df["date_dt"].dt.to_period("M")

    pivot = existing_df.pivot_table(
        index="year_month", columns="Stock_symbol",
        values="Article_title", aggfunc="count", fill_value=0,
    )
    print(pivot.to_string())


if __name__ == "__main__":
    audit_coverage()