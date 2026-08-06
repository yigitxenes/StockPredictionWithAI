import os
import sys

import requests
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import load_config

load_dotenv() #load the env file to this enviroment

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

def is_likely_relevant(news_item, ticker, company_name):
    text = (news_item.get("headline", "") + " " + news_item.get("summary", "")).lower()
    return ticker.lower() in text or company_name.lower() in text

def fetch_company_news(ticker,company_name, from_date, to_date):
    #Date Format YYYY-MM-DD
    api_key = os.getenv("FINNHUB_API_KEY")
    
    if not api_key:
        raise ValueError("FINNHUB API KEY NOT FOUND.")
    
    url = f"{FINNHUB_BASE_URL}/company-news"
    
    params = {
        "symbol" : ticker,
        "from" : from_date,
        "to" : to_date,
        "token" : api_key
    }
    
    response = requests.get(url, params)
    response.raise_for_status() #Execption for error 4xx/5xx
    all_news = response.json()
    
    relevant_news = [n for n in all_news if is_likely_relevant(n, ticker, company_name)]
    
    return relevant_news


def main():
    cfg = load_config()
    tickers = cfg["data"]["tickers"]
    company_names = cfg["data"]["company_names"]

    from_date = "2026-07-25"
    to_date = "2026-08-05"

    for ticker in tickers:
        company_name = company_names.get(ticker, ticker)  
        news_list = fetch_company_news(ticker, company_name, from_date, to_date)
        print(f"For {ticker} ({company_name}) : {from_date} -> {to_date} :::: {len(news_list)} news found.\n")
    
    for item in news_list[:5]:
        print(f"Caption : {item["headline"]}")
        print(f"Source : {item["source"]}")
        print(f"Date : {item["datetime"]}")
        print(f"Summary : {item["summary"]}")
        print(f"Link : {item["url"]}")
    

if __name__ == "__main__":
    main()