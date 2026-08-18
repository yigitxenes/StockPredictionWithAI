import os
import sys
import json

import joblib
from google import genai
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config
from agent.live_data_fetch import get_live_technical_row, get_live_sentiment, build_feature_row

load_dotenv()

MODEL_DIR = "models"

REPORT_PROMPT_TEMPLATE = """Sen dürüst ve temkinli bir finansal analiz asistanisin.
Asagidaki bilgilere dayanarak {ticker} ({company_name}) hissesi icin GUNLUK
BILGILENDIRME RAPORU yaz.

KESIN KURALLAR:
- Bu bir yatirim tavsiyesi DEGILDIR, bunu raporun basinda acikca belirt.
- Modelin tarihsel performansini ASAGIDA verilen gercek sayilarla anlat,
  kendi tahminini uydurma. Model AUC'si {direction_auc} -- bu sans seviyesine
  ({chance_level}) yakin, bunu SESLI VE ACIK sekilde belirt.
- "Kesin", "garanti", "yuksek ihtimalle kazandirir" gibi ifadeler KULLANMA.
- Olasiliklari sun ama hemen ardindan bunun ne kadar guvenilmez oldugunu acikla.
- Sentiment/haber bulunamadiysa bunu gizleme, acikca "bu tarih araliginda haber
  bulunamadi" de.
- Kac haber TARANDIGINI (degerlendirilen sayi) belirt, "bulunan toplam" ile
  karistirma -- bunlar farkli olabilir.

VERILER:
- Bu rapor su tarihe ait en guncel piyasa kapanisina dayanmaktadir: {as_of_date}
  (NOT: gercek zamanli degildir, en son mevcut kapanis verisidir)
- Teknik gostergeler: RSI={rsi:.1f}, MACD histogram={macd_hist:.4f}, ADX={adx:.1f}
- Model yon tahmini: {direction_proba:.1%} yukselis olasiligi (Tarihsel AUC: {direction_auc}, {direction_note})
- Model volatilite tahmini (5 gun): {volatility_pred:.4f} (Naive baseline: {naive_vol:.4f}, {volatility_note})
- Sentiment skoru: {sentiment_score:.2f} (guven: {sentiment_confidence:.2f})
- Degerlendirilen haber sayisi: {news_count} (bulunan toplam: {news_count_total})
- Sentiment gerekcesi: {sentiment_reasoning}

Rapor Turkce, 150-250 kelime, baslik + kisa paragraflar halinde olsun."""


def load_model_bundle(name):
    return joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))


def load_performance_summary():
    with open(os.path.join(MODEL_DIR, "model_performance_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def generate_report(ticker, cfg, gemini_client):
    company_name = cfg["data"]["company_names"].get(ticker, ticker)

    direction_bundle = load_model_bundle("final_direction_model")
    volatility_bundle = load_model_bundle("final_volatility_model")
    perf = load_performance_summary()

    print(f"[{ticker}] Canli teknik veri cekiliyor...")
    technical_row, as_of_date = get_live_technical_row(ticker, cfg)

    print(f"[{ticker}] Guncel haber/sentiment cekiliyor (referans tarih: {as_of_date.date()})...")
    model_name = cfg["agent"]["prototype_model"]
    sentiment_result = get_live_sentiment(
        ticker, company_name, gemini_client, model_name,
        reference_date=as_of_date,
    )

    print(f"[{ticker}] Tahmin uretiliyor...")
    X_direction = build_feature_row(technical_row, sentiment_result, ticker,
                                     direction_bundle["feature_cols"], direction_bundle["tickers"])
    X_volatility = build_feature_row(technical_row, sentiment_result, ticker,
                                      volatility_bundle["feature_cols"], volatility_bundle["tickers"])

    direction_proba = direction_bundle["model"].predict_proba(X_direction)[0][1]
    volatility_pred = volatility_bundle["model"].predict(X_volatility)[0]

    direction_auc = perf["direction"]["value"]
    direction_note = perf["direction"]["note"]
    volatility_note = perf["volatility"]["interpretation"]
    naive_vol = perf["volatility"]["naive_baseline_rmse"]

    prompt = REPORT_PROMPT_TEMPLATE.format(
        ticker=ticker, company_name=company_name, as_of_date=as_of_date.strftime("%Y-%m-%d"),
        rsi=technical_row.get("RSI_14", float("nan")),
        macd_hist=technical_row.get("MACDh_12_26_9", float("nan")),
        adx=technical_row.get("ADX_14", float("nan")),
        direction_proba=direction_proba, direction_auc=direction_auc,
        chance_level="0.50", direction_note=direction_note,
        volatility_pred=volatility_pred, naive_vol=naive_vol, volatility_note=volatility_note,
        sentiment_score=sentiment_result["overall_score"],
        sentiment_confidence=sentiment_result["overall_confidence"],
        news_count=sentiment_result.get("news_count", 0),
        news_count_total=sentiment_result.get("news_count_total_found", 0),
        sentiment_reasoning=sentiment_result["reasoning"],
    )

    response = gemini_client.models.generate_content(model=model_name, contents=prompt)

    return {
        "ticker": ticker,
        "as_of_date": str(as_of_date.date()),
        "direction_probability_up": float(direction_proba),
        "volatility_forecast": float(volatility_pred),
        "sentiment": sentiment_result,
        "report_text": response.text,
        "model_reliability_note": f"Modelin tarihsel AUC'si {direction_auc} (sans seviyesi: 0.50)",
    }


def main():
    cfg = load_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    result = generate_report(ticker, cfg, client)

    print("\n" + "=" * 60)
    print(f"(as_of_date: {result['as_of_date']} -- gercek zamanli degil, en son kapanisa dayali)")
    print(result["report_text"])
    print("=" * 60)


if __name__ == "__main__":
    main()