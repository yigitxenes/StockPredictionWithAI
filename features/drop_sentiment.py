import pandas as pd
import re

INPUT_PATH = "sentiment_scores.csv"
OUTPUT_PATH = "sentiment_scores.csv"

def has_cjk(text):
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', str(text)))

df = pd.read_csv(INPUT_PATH)
before = len(df)

bad_std = (df["news_count_daily"] == 1) & (df["sentiment_std"] != 0)
bad_conf = (df["sentiment_confidence"] < 0) | (df["sentiment_confidence"] > 1)
bad_reasoning = df["reasoning"].apply(has_cjk)

bad_mask = bad_std | bad_conf | bad_reasoning

print(f"Toplam satır: {before}")
print(f"news_count=1 ama std!=0: {bad_std.sum()}")
print(f"confidence [0,1] dışında: {bad_conf.sum()}")
print(f"reasoning'de yabancı script (CJK): {bad_reasoning.sum()}")
print(f"Union (drop edilecek): {bad_mask.sum()}")

df_clean = df[~bad_mask].copy()
print(f"Temizlik sonrası: {len(df_clean)} ({before - len(df_clean)} satır silindi)")

df_clean.to_csv(OUTPUT_PATH, index=False)