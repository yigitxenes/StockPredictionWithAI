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
        "overall_score": {
            "type": "number",
            "description": "Gunun geneli icin -1 (cok negatif) ile +1 (cok pozitif) arasi sentiment skoru",
        },
        "overall_confidence": {
            "type": "number",
            "description": "Bu skora ne kadar emin olundugu, 0 ile 1 arasi",
        },
        "reasoning": {
            "type": "string",
            "description": "Skorun kisa gerekcesi, 1-2 cumle",
        },
        "article_scores": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Her haberin AYRI skoru, -1 ile +1 arasi, girdi sirasiyla ayni sirada",
        },
    },
    "required": ["overall_score", "overall_confidence", "reasoning", "article_scores"],
}

PROMPT_TEMPLATE = """Sen bir finansal haber analistisin. Asagida {ticker} hissesi ile
ilgili {date} tarihinde yayinlanmis {n_articles} haber var.

ONEMLI KURAL: Sadece bu haberlerin verildigi bilgiyi kullan. Bu haberlerden
SONRA piyasada ne oldugunu bilmiyormus gibi davran, gelecege dair hicbir
cikarim yapma - sadece haberin kendi icerigine dayanarak sentiment belirle.

Haberler:
{articles_text}

Her haber icin ayri bir sentiment skoru ve genel gun icin toplu bir skor uret."""


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


def build_articles_text(articles: list) -> str:
    lines = []
    for i, art in enumerate(articles, 1):
        title = art.get("Article_title", "")
        summary = str(art.get("Article", ""))[:500]
        lines.append(f"{i}. Baslik: {title}\n   Icerik: {summary}")
    return "\n\n".join(lines)

def generate_daily_statements(client, ticker, date, articles, model_name):
    
    prompt = PROMPT_TEMPLATE.format(
        ticker = ticker,
        date = date,
        n_articles = len(articles),
        articles_text = build_articles_text(articles)
    )
    
    response = client.models.generate_content(
        model = model_name,
        contents = prompt,
        config = types.GenerateContentConfig(
            response_mime_type = "application/json",
            response_schema = RESPONSE_SCHEMA,
            http_options=types.HttpOptions(timeout=30_000)
        )
    )
    result = json.loads(response.text)
    
    return result

def call_with_timeout(client, ticker, date_str, articles, model_name, timeout_seconds=30):
    """generate_daily_statements'i ayri bir thread'de calistirir, sure dolunca
    TimeoutError firlatir - kutuphanenin kendi timeout hatasina guvenmiyoruz."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            generate_daily_statements, client, ticker, date_str, articles, model_name
        )
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            raise TimeoutError(f"Istek {timeout_seconds} saniyede yanit vermedi (SDK askida kaldi)")
        
        
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
    remaining = total_groups - len(already_processed)
    print(f"Total Groups : {total_groups}----------Remaining : {remaining}")
    
    processed_count = 0
    error_count = 0
    
    for (ticker, date_str), group in groups:
        if (ticker, date_str) in already_processed:
            continue
        
        if processed_count >= MAX_REQUESTS_TODAY:
            print(f"Reached Daily Request Limit ")
            break
        
        articles = group.to_dict("records")
        
        try:
            result = call_with_timeout(client, ticker, date_str, articles, model_name, timeout_seconds=30)
            
            article_scores = result.get("article_scores", [])
            sentiment_std = pd.Series(article_scores).std() if len(article_scores) > 1 else 0.0
            row = {
                "ticker": ticker,
                "date": date_str,
                "sentiment_score": result["overall_score"],
                "sentiment_confidence": result["overall_confidence"],
                "news_count_daily": len(articles),
                "sentiment_std": sentiment_std,
                "reasoning": result["reasoning"]
            }
            append_result(row)
            processed_count += 1
            
            if processed_count % 20 == 0:
                print(f"  Processed (this session): {processed_count}/{MAX_REQUESTS_TODAY}, "
                      f"Error: {error_count}, Remaining: ~{remaining - processed_count}\n")
            
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        
        except Exception as e:
            error_count += 1
            print(f"Error {ticker}, {date_str} : {e}")
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("Rate Limit Error Detected Continue next day")
                break
            
            if error_count > 15:
                print("Too many consecutive errors. Script is shutting.")
                break

            time.sleep(5)
            continue
        
    print(f"\nProcessed This Session: {processed_count}, Error: {error_count}")
    print(f"Total progress: {len(already_processed) + processed_count}/{total_groups}")

if __name__ == "__main__":
    main()