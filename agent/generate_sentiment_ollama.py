import os
import sys
import json
import time
import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2:3b"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {
            "type": "number", "minimum": -1, "maximum": 1,
            "description": "SADECE -1.0 ile 1.0 arasinda ondalikli sayi. Ornekler: -0.8, -0.3, 0.0, 0.4, 0.9. ASLA 1'den buyuk, ASLA tam sayi (1,2..10) kullanma.",
        },
        "overall_confidence": {
            "type": "number", "minimum": 0, "maximum": 1,
            "description": "SADECE 0.0 ile 1.0 arasinda ondalikli sayi. Ornekler: 0.3, 0.6, 0.85. ASLA 1'den buyuk (6, 60 gibi) bir deger kullanma.",
        },
        "reasoning": {"type": "string"},
        "article_scores": {
            "type": "array",
            "items": {
                "type": "number", "minimum": -1, "maximum": 1,
                "description": "Her biri -1.0 ile 1.0 arasinda ondalikli sayi, ORNEK: 0.2 (asla 8, 9 gibi tam sayi degil)",
            },
        },
    },
    "required": ["overall_score", "overall_confidence", "reasoning", "article_scores"],
}

PROMPT_TEMPLATE = """Sen bir finansal haber analistisin. Asagida {ticker} hissesi
ile ilgili {date} tarihinde yayinlanmis haberler var.

ONEMLI KURAL: SADECE bu tarihte verilen haberleri kullan. Bu haberlerden
SONRA piyasada ne oldugunu bilmiyormus gibi davran, gelecege dair hicbir
cikarim yapma - sadece haberin kendi icerigine dayanarak sentiment belirle.

SKOR OLCEGI: Tum skorlar -1.0 (cok negatif) ile +1.0 (cok pozitif) arasinda
ONDALIKLI sayilar olmali. ORNEK dogru degerler: -0.7, -0.2, 0.0, 0.35, 0.9.
1-10 veya 0-100 gibi baska bir olcek KULLANMA.

{articles_text}

Bu haberlere dayanarak {ticker} icin genel bir sentiment skoru, guven skoru,
kisa bir gerekce ve her haber icin ayri bir skor uret. SADECE JSON dondur,
baska hicbir metin ekleme."""

RETRY_PROMPT_SUFFIX = "\n\nUYARI: Onceki cevabinda skorlari YANLIS olcekte vermis olabilirsin (orn. 1-10 arasi tam sayilar). Skorlar SADECE -1.0 ile 1.0 arasinda ONDALIKLI olmali. Dogru ornekler: 0.6, -0.4, 0.15, -0.9. Haberi tekrar oku ve DOGRU olcekte skorla."
OUTPUT_FILE = "data/processed_sentiment_backtest/sentiment_scores.csv"
DEBUG_FAILURES_FILE = "data/processed_sentiment_backtest/ollama_failures_debug.jsonl"

NUM_CTX = 8192
MAX_ARTICLES_PER_REQUEST = 15
RETRY_TEMPERATURE = 0.4


def build_articles_text(articles: list) -> str:
    lines = []
    for i, art in enumerate(articles, 1):
        title = art.get("Article_title", "")
        summary = str(art.get("Article", ""))[:400]
        lines.append(f"{i}. Baslik: {title}\n   Icerik: {summary}")
    return "\n\n".join(lines)


def generate_single_ticker_sentiment(ticker, date_str, articles, temperature=0.1, extra_instruction=""):
    """Basarili olursa (parsed_json, raw_content) donderir."""
    articles_text = build_articles_text(articles[:MAX_ARTICLES_PER_REQUEST])

    prompt = PROMPT_TEMPLATE.format(
        ticker=ticker,
        date=date_str,
        articles_text=articles_text,
    )
    prompt += extra_instruction

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": NUM_CTX,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    return json.loads(content), content


def is_degenerate(t_res):
    reasoning = (t_res.get("reasoning") or "").strip()
    score = t_res.get("overall_score", 0)
    confidence = t_res.get("overall_confidence", 0)
    article_scores = t_res.get("article_scores", [])

    reasoning_missing = len(reasoning) < 15
    zero_signal = (score == 0 and confidence == 0)

    try:
        out_of_range = (
            not (-1 <= float(score) <= 1)
            or not (0 <= float(confidence) <= 1)
            or any(not (-1 <= float(s) <= 1) for s in article_scores)
        )
    except (TypeError, ValueError):
        out_of_range = True

    return reasoning_missing or zero_signal or out_of_range


def log_failure(ticker, date_str, raw_content, reason):
    with open(DEBUG_FAILURES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ticker": ticker,
            "date": date_str,
            "reason": reason,
            "raw_content": raw_content,
        }, ensure_ascii=False) + "\n")


def load_already_processed():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    existing = pd.read_csv(OUTPUT_FILE)
    return set(zip(existing["ticker"], existing["date"]))


def append_result(row):
    file_exists = os.path.exists(OUTPUT_FILE)
    df_row = pd.DataFrame([row])
    df_row.to_csv(OUTPUT_FILE, mode="a", header=not file_exists, index=False)


def load_combined_news(cfg):
    fnspid_df = pd.read_csv("data/raw/fnspid_filtered.csv")
    fnspid_df["Date"] = pd.to_datetime(fnspid_df["Date"], errors="coerce", utc=True)

    av_path = "data/raw/alphavantage_filtered.csv"
    if os.path.exists(av_path):
        av_df = pd.read_csv(
            av_path, header=None,
            names=["Date", "Article_title", "Stock_symbol", "Publisher", "Article"],
        )
        av_df["Date"] = pd.to_datetime(av_df["Date"], format="%Y%m%d", errors="coerce", utc=True)
        combined = pd.concat([fnspid_df, av_df], ignore_index=True)
    else:
        combined = fnspid_df

    before = len(combined)
    combined = combined.drop_duplicates(subset=["Date", "Article_title", "Stock_symbol"])
    if len(combined) < before:
        print(f"Birlesirken {before - len(combined)} tekrar temizlendi (FNSPID/AV sinir bolgesi)")

    return combined


def _write_row(ticker, date_str, articles, t_res):
    article_scores = t_res.get("article_scores", [])
    sentiment_std = pd.Series(article_scores).std() if len(article_scores) > 1 else 0.0

    row = {
        "ticker": ticker,
        "date": date_str,
        "sentiment_score": t_res["overall_score"],
        "sentiment_confidence": t_res["overall_confidence"],
        "news_count_daily": len(articles),
        "sentiment_std": sentiment_std,
        "reasoning": t_res["reasoning"],
    }
    append_result(row)


def process_one_group(ticker, date_str, articles):
    """Bir (ticker, gun) grubunu isler. Basarili olursa CSV'ye yazar ve
    (True, None) doner. Basarisiz olursa (iki denemeden sonra) debug
    log'a yazar ve (False, sebep) doner."""

    # 1. deneme -- dusuk sicaklik, standart prompt
    try:
        t_res, raw = generate_single_ticker_sentiment(ticker, date_str, articles, temperature=0.1)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        log_failure(ticker, date_str, None, f"istek/parse hatasi: {type(e).__name__}: {e}")
        return False, "istek_hatasi"

    if not is_degenerate(t_res):
        _write_row(ticker, date_str, articles, t_res)
        return True, None

    # 2. deneme -- dejenereyse, hem sicakligi artir hem promptu netlestir
    # (sadece sicaklik degistirmenin ayni takilmayi tekrarladigi gozlemlendi)
    log_failure(ticker, date_str, raw, "dejenere (1. deneme, temp=0.1)")
    try:
        t_res_retry, raw_retry = generate_single_ticker_sentiment(
            ticker, date_str, articles,
            temperature=RETRY_TEMPERATURE,
            extra_instruction=RETRY_PROMPT_SUFFIX,
        )
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        log_failure(ticker, date_str, None, f"retry istek/parse hatasi: {type(e).__name__}: {e}")
        return False, "istek_hatasi"

    if not is_degenerate(t_res_retry):
        _write_row(ticker, date_str, articles, t_res_retry)
        return True, None

    log_failure(ticker, date_str, raw_retry, f"dejenere (2. deneme, temp={RETRY_TEMPERATURE}, retry-prompt)")
    return False, "dejenere"


def main():
    cfg = load_config()

    news_df = load_combined_news(cfg)
    backtest_start = pd.Timestamp(cfg["data"]["sentiment_backtest_start"], tz="UTC")
    backtest_end = pd.Timestamp(cfg["data"]["sentiment_backtest_end"], tz="UTC")
    news_df = news_df[(news_df["Date"] >= backtest_start) & (news_df["Date"] <= backtest_end)]
    news_df["date_str"] = news_df["Date"].dt.strftime("%Y-%m-%d")

    already_processed = load_already_processed()
    print(f"Number of already processed groups : {len(already_processed)}")

    groups = news_df.groupby(["Stock_symbol", "date_str"])
    total_groups = len(groups)

    remaining_groups = [
        (ticker, date_str, group.to_dict("records"))
        for (ticker, date_str), group in groups
        if (ticker, date_str) not in already_processed
    ]
    remaining_groups.sort(key=lambda x: x[1])

    remaining = total_groups - len(already_processed)
    print(f"Total Groups : {total_groups} ---- Remaining : {remaining}")

    processed_groups = 0
    error_count = 0
    degenerate_count = 0
    request_error_count = 0
    consecutive_errors = 0
    start_time = time.time()

    for ticker, date_str, articles in remaining_groups:
        t0 = time.time()
        success, fail_reason = process_one_group(ticker, date_str, articles)
        elapsed = time.time() - t0

        if success:
            processed_groups += 1
            consecutive_errors = 0
        else:
            error_count += 1
            consecutive_errors += 1
            if fail_reason == "dejenere":
                degenerate_count += 1
            else:
                request_error_count += 1
            print(f"Hata ticker={ticker}, date={date_str} : {fail_reason} ({elapsed:.1f}sn)")

            if consecutive_errors > 30:
                print("Cok fazla ARDISIK hata, script durduruluyor (Ollama askida olabilir).")
                break

        total_processed = processed_groups + error_count
        if total_processed % 10 == 0:
            total_elapsed = time.time() - start_time
            rate = total_processed / total_elapsed * 60
            print(f"  {processed_groups} basarili / {total_processed} denenen, "
                  f"hiz ~{rate:.1f} grup/dk, "
                  f"Hata: {error_count} (dejenere: {degenerate_count}, istek: {request_error_count})")

    print(f"\nBu oturumda: {processed_groups} basarili, {error_count} hata "
          f"(dejenere: {degenerate_count}, istek: {request_error_count})")
    print(f"Toplam ilerleme: {len(already_processed) + processed_groups}/{total_groups}")


if __name__ == "__main__":
    main()
