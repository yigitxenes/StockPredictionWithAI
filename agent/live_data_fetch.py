import os
import sys
import json
from datetime import datetime, timedelta

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from features.fetch_data import fetch_ohlcv
from features.indicators import add_indicators
from agent.fetch_news import fetch_company_news

def get_live_technical_data(ticker, cfg):
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

    df = fetch_ohlcv(ticker, start_date, end_date, "1d")
    if df.empty:
        raise ValueError(f"{ticker} icin canli OHLCV verisi alinamadi")

    print(f"[DEBUG] {ticker}: dropna ONCESI son 3 satir:")
    print(df.tail(3))

    df = add_indicators(df, cfg)
    df = df.dropna()

    print(f"[DEBUG] {ticker}: dropna SONRASI son 3 satir:")
    print(df.tail(3))

    if df.empty:
        raise ValueError(f"{ticker} icin yeterli gecmis veri yok (gosterge warmup basarisiz)")

    return df



def get_live_technical_row(ticker, cfg):
    """Sadece en son satiri (ve tarihini) istenen yerler icin -- write_report.py
    hala bunu kullaniyor, davranisi degismedi."""
    df = get_live_technical_data(ticker, cfg)
    latest_row = df.iloc[-1].copy()
    latest_date = df.index[-1]
    return latest_row, latest_date


LIVE_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "number", "minimum": -1, "maximum": 1},
        "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
        "article_scores": {"type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1}},
    },
    "required": ["overall_score", "overall_confidence", "reasoning", "article_scores"],
}


def get_live_sentiment(ticker, company_name, gemini_client, model_name,
                        reference_date, lookback_days=3, max_articles=15):
    """Son N gunun haberlerini cekip guncel sentiment uretir.

    ONEMLI: 'son N gun', datetime.now() DEGIL reference_date'e gore
    hesaplanir -- bu, teknik verinin ait oldugu tarihle sentiment
    penceresinin AYNI zaman noktasina ankorlanmasini saglar. Aksi halde
    (datetime.now() kullanilsaydi) teknik veri X gunune ait olurken
    sentiment X+birkac gununu yansitabilir, bu da point-in-time
    tutarsizligina yol acar (rapor v4, bolum 5.2'deki ilkeyle ayni
    mantik -- sadece backtest degil, canli sistemde de gecerli)."""
    to_date = reference_date.strftime("%Y-%m-%d")
    from_date = (reference_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        news_list = fetch_company_news(ticker, company_name, from_date, to_date)
    except Exception as e:
        print(f"Haber cekilemedi: {e}")
        news_list = []

    if not news_list:
        return {
            "overall_score": 0.0, "overall_confidence": 0.0,
            "reasoning": "Bu tarih araliginda hisseyle ilgili haber bulunamadi.",
            "article_scores": [], "news_count": 0, "news_count_total_found": 0,
        }

    articles_to_score = news_list[:max_articles]
    articles_text = "\n\n".join(
        f"{i}. Baslik: {n['headline']}\n   Ozet: {n.get('summary', '')[:400]}"
        for i, n in enumerate(articles_to_score, 1)
    )

    prompt = f"""Sen bir finansal haber analistisin. Asagida {company_name} ({ticker})
hissesi ile ilgili {from_date} - {to_date} araliginda yayinlanmis haberler var.

{articles_text}

Bu haberlere dayanarak genel bir sentiment skoru, guven skoru, kisa bir
gerekce ve her haber icin ayri bir skor uret. SADECE JSON dondur."""

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={"response_mime_type": "application/json", "response_schema": LIVE_SENTIMENT_SCHEMA},
    )

    result = json.loads(response.text)
    result["news_count"] = len(articles_to_score)
    result["news_count_total_found"] = len(news_list)
    return result


def build_feature_row(technical_row, sentiment_result, ticker, feature_cols, all_tickers):
    """Model'in beklendigi feature_cols sirasina/setine BIREBIR uyan bir
    satir olusturur -- egitimde kullanilan kolon isimleriyle tam eslesme
    saglanmazsa model sessizce yanlis sonuc uretir, bu yuzden feature_cols
    listesi tek dogruluk kaynagi olarak kullaniliyor."""
    row = {}

    for col in feature_cols:
        if col.startswith("ticker_"):
            row[col] = 1 if col == f"ticker_{ticker}" else 0
        elif col == "sentiment_score":
            row[col] = sentiment_result["overall_score"]
        elif col == "sentiment_confidence":
            row[col] = sentiment_result["overall_confidence"]
        elif col == "sentiment_std":
            scores = sentiment_result.get("article_scores", [])
            row[col] = pd.Series(scores).std() if len(scores) > 1 else 0.0
        elif col == "news_count_daily":
            row[col] = sentiment_result.get("news_count", 0)
        elif col in ("sentiment_ema", "sentiment_momentum"):
            # canli tek-nokta veride gecmis seri olmadigi icin, o anki
            # skoru yaklasik deger olarak kullaniyoruz
            row[col] = sentiment_result["overall_score"]
        elif col in technical_row.index:
            row[col] = technical_row[col]
        else:
            row[col] = 0.0

    return pd.DataFrame([row])[feature_cols]