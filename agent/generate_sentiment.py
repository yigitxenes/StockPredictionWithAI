import os
import sys
import json
import pandas as pd
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config

load_dotenv()



RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tickers": {
            "type": "array",
            "description": "Her hisse icin ayri bir sonuc nesnesi, girdideki hisse sirasiyla ayni sirada",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Hisse sembolu, girdideki etiketle birebir ayni",
                    },
                    "overall_score": {
                        "type": "number",
                        "minimum": -1,
                        "maximum": 1,
                        "description": "Bu hisse icin gunun geneli icin -1 ile +1 arasi sentiment skoru",
                    },
                    "overall_confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Bu skora ne kadar emin olundugu, 0 ile 1 arasi",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Skorun kisa gerekcesi, 1-2 cumle",
                    },
                    "article_scores": {
                        "type": "array",
                        "items": {"type": "number", "minimum": -1, "maximum": 1},
                        "description": "Bu hisseye ait her haberin AYRI skoru, girdi sirasiyla ayni sirada",
                    },
                },
                "required": ["ticker", "overall_score", "overall_confidence", "reasoning", "article_scores"],
            },
        }
    },
    "required": ["tickers"],
}


PROMPT_TEMPLATE = """Sen bir finansal haber analistisin. Asagida {date} tarihinde
yayinlanmis, {n_tickers} farkli hisseye ait haberler var.

ONEMLI KURAL: Her hisse icin SADECE bu tarihte verilen haberleri kullan.
Bu haberlerden SONRA piyasada ne oldugunu bilmiyormus gibi davran, gelecege
dair hicbir cikarim yapma - sadece haberin kendi icerigine dayanarak
sentiment belirle. Bir hissenin haberini yorumlarken diger hisselerin
haberlerini kullanma, her hisseyi bagimsiz degerlendir.

{tickers_text}

Her hisse icin: kendi haberlerinin ayri skorunu ve o hisse icin toplu bir
skor uret. Cikan "tickers" array'inin sirasi ve ticker etiketleri girdiyle
BIREBIR ayni olmali."""

TICKER_BLOCK_TEMPLATE = """### Hisse {i}: {ticker}
{articles_text}"""


OUTPUT_FILE = "data/processed_sentiment_backtest/sentiment_scores.csv"
OUTPUT_COLUMNS = [
    "ticker", "date", "sentiment_score", "sentiment_confidence",
    "news_count_daily", "sentiment_std", "reasoning",
]

RPM_LIMIT = 15
RPD_LIMIT = 500
SAFETY_MARGIN = 20  
SLEEP_BETWEEN_REQUESTS = 60 / RPM_LIMIT + 0.5   # ~4.5 seconds
MAX_REQUESTS_TODAY = RPD_LIMIT - SAFETY_MARGIN   # 480

MAX_TICKERS_PER_BATCH = 8
MAX_ARTICLES_PER_BATCH = 50



def build_articles_text(articles: list) -> str:
    lines = []
    for i, art in enumerate(articles, 1):
        title = art.get("Article_title", "")
        summary = str(art.get("Article", ""))[:500]
        lines.append(f"{i}. Baslik: {title}\n   Icerik: {summary}")
    return "\n\n".join(lines)

def build_batches(date_tickers, max_tickers, max_articles):
    batches = []
    current = []
    current_article_count = 0
    
    for ticker, articles in date_tickers:
        would_exceed_tickers = len(current) + 1 > max_tickers
        would_exceed_articles = current_article_count + len(articles) > max_articles
        
        if current and (would_exceed_tickers or would_exceed_articles):
            batches.append(current)
            current = []
            current_article_count = 0
        
        current.append((ticker, articles))
        current_article_count += len(articles)
    
    if current:
        batches.append(current)
    
    return batches 
        


def generate_crossticker_sentiment(client, date_str, ticker_batch,model_name):
    ticker_text_parts = []
    for i, (ticker, articles) in enumerate(ticker_batch, 1):
        ticker_text_parts.append(
            TICKER_BLOCK_TEMPLATE.format(i = i, ticker = ticker, articles_text = build_articles_text(articles))
        )
    
    prompt = PROMPT_TEMPLATE.format(
        date=date_str,
        n_tickers=len(ticker_batch),
        tickers_text="\n\n".join(ticker_text_parts),
    )
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            http_options=types.HttpOptions(timeout=30_000)
        )
    )
    result = json.loads(response.text)
    
    return result

def call_with_timeout(client, date_str, ticker_batch, model_name, timeout_seconds=30):
    """Process generate_crossticker_sentiment on different thread when times up 
    throws TimeoutError"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            generate_crossticker_sentiment, client, date_str, ticker_batch, model_name
        )
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            raise TimeoutError(f"Request failed to respond in {timeout_seconds} seconds.")
        
        
def load_already_processed():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    existing = pd.read_csv(OUTPUT_FILE)
    return set(zip(existing["ticker"], existing["date"]))

def append_result(row):
    file_exists = os.path.exists(OUTPUT_FILE)
    df_row = pd.DataFrame([row])
    df_row.to_csv(OUTPUT_FILE, mode="a", header= not file_exists, index= False)
    


def main():
    cfg = load_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = cfg["agent"]["prototype_model"]
    client = genai.Client(api_key= api_key)
    
    news_df = pd.read_csv("data/raw/fnspid_filtered.csv")
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors= "coerce")
    backtest_start = pd.Timestamp(cfg["data"]["sentiment_backtest_start"], tz="UTC")
    backtest_end = pd.Timestamp(cfg["data"]["sentiment_backtest_end"], tz="UTC")
    news_df = news_df[(news_df["Date"] >= backtest_start) & (news_df["Date"] <= backtest_end)]

    news_df["date_str"] = news_df["Date"].dt.strftime("%Y-%m-%d")
    
    already_processed = load_already_processed()
    print(f"Number of already processed groups : {len(already_processed)}")
    
    groups = news_df.groupby(["Stock_symbol", "date_str"])
    total_groups = len(groups)
    
    by_date = {}
    for (ticker, date_str), group in groups:
        if (ticker, date_str) in already_processed:
            continue
        by_date.setdefault(date_str, []).append((ticker, group.to_dict("records")))
    
    all_batches = []
    for date_str in sorted(by_date.keys()):
        for batch in build_batches(by_date[date_str], MAX_TICKERS_PER_BATCH, MAX_ARTICLES_PER_BATCH):
            all_batches.append((date_str, batch))
    
    
    remaining = total_groups - len(already_processed)
    print(f"Total Groups : {total_groups}----------Remaining : {remaining}")
    
    processed_batches = 0
    processed_groups = 0
    error_count = 0
    
    for date_str, ticker_batch in all_batches:
        if processed_batches >= MAX_REQUESTS_TODAY:
            print("Reached Daily Request Limit")
            break

        try:
            result = call_with_timeout(client, date_str, ticker_batch, model_name, timeout_seconds=45)
            tickers_result = result.get("tickers", [])

            if len(tickers_result) != len(ticker_batch):
                raise ValueError(
                    f"Expected {len(ticker_batch)} ticker, returning{len(tickers_result)} ticker -skip batch"
                )

            for (expected_ticker, articles), t_res in zip(ticker_batch, tickers_result):
                if t_res.get("ticker") != expected_ticker:
                    raise ValueError(
                        f"Expected {expected_ticker}, Result {t_res.get('ticker')}"
                    )

                article_scores = t_res.get("article_scores", [])
                sentiment_std = pd.Series(article_scores).std() if len(article_scores) > 1 else 0.0

                row = {
                    "ticker": expected_ticker,
                    "date": date_str,
                    "sentiment_score": t_res["overall_score"],
                    "sentiment_confidence": t_res["overall_confidence"],
                    "news_count_daily": len(articles),
                    "sentiment_std": sentiment_std,
                    "reasoning": t_res["reasoning"],
                }
                append_result(row)
                processed_groups += 1

            processed_batches += 1

            if processed_batches % 10 == 0:
                print(f"  Processed (this session): {processed_batches} batches / {processed_groups} groups, "
                      f"Error: {error_count}")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            error_count += 1
            print(f"Error date={date_str}, tickers={[t for t, _ in ticker_batch]} : {type(e).__name__}: {e}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("Rate Limit Error Detected Continue next day")
                break

            if error_count > 15:
                print("Too many consecutive errors. Script is shutting.")
                break

            time.sleep(5)
            continue

    print(f"\nProcessed This Session: {processed_batches} batches / {processed_groups} groups, Error: {error_count}")
    print(f"Total progress: {len(already_processed) + processed_groups}/{total_groups}")


if __name__ == "__main__":
    main()