import requests
import pandas as pd
import io
from bs4 import BeautifulSoup
import sqlite3
import yfinance as yf
import time
from datetime import datetime, date, time
import pytz  # Make sure to import pytz for timezone handling
from openai import OpenAI

client = OpenAI(api_key="")
import json

# Set your OpenAI API key (or set it in your environment variable OPENAI_API_KEY)

# --- Remove or comment out FinBERT code ---
# from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# 
# # Load FinBERT Model
# tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
# model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
# finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
# --- End FinBERT removal ---


# Your API Token
API_TOKEN = "89501805-b73b-4ed5-baea-1748ee898a73"
BASE_URL = "https://elite.finviz.com/news_export.ashx"

# Headers to mimic a real browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Referer": "https://elite.finviz.com/",
    "Upgrade-Insecure-Requests": "1"
}

# Replace with your Finviz session cookies (get them from browser DevTools)
COOKIES = {
    ".ASPXAUTH": "184E291C2CC06150882CC3ED0A70C16BA13D2E497E70F67B78ED178D74FCAE6150C771862FD8218CD6E346DFB6A273077098AA67DD1351FBAD15C3E6CE1F001FFEC95945E6BBEC3DD19E346AF00E9C707AF9D5931516E8035A8F916D3958DD3D037D45C1E2E54E9EBD0DE4B9B6C3D9CFA627C7664235850EBED8AF9AA64195E2091643C6FCAC4D7EF2FF3E2A01C9BD4645572465E27B4EC4E0209D7F8328F576C5F846AC1E01A8444E98B493828F0AF7256B3FE08D0448AE26B864794E22EA0E80551EC6C44D7259B1B405462CE1823897D7C22E",
    "EliteSession": "YOUR_ELITESESSION_COOKIE_HERE",
    "panoramaId": "a4b535a871c6fc19d99fbc27bf39a9fb927aabb609df1f8fcae8016252571d7b",
    "_ga": "GA1.1.623035271.1739240112",
}

# Connect to SQLite Database
conn = sqlite3.connect("stock_sentiment.db")
cursor = conn.cursor()

# Create Table if Not Exists (Updated to store SentimentScore, ConfidenceScore, and PriceChange)
cursor.execute("""
    CREATE TABLE IF NOT EXISTS StockNews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Title TEXT,
        Source TEXT,
        Date TEXT,
        URL TEXT,
        Category TEXT,
        Ticker TEXT,
        Full_Text TEXT,
        SentimentScore REAL,
        ConfidenceScore REAL,
        PriceChange REAL
    )
""")

conn.commit()


def classify_sentiment_chatgpt(text):
    """
    Use OpenAI's ChatGPT to perform sentiment analysis.
    Returns:
       sentiment (float): a value between -1 (very negative) and 1 (very positive)
       confidence (float): a value between 0 and 1.
    """
    try:
        prompt = (
            "Please analyze the sentiment of the following text. "
            "Return your result as a JSON object with two keys: 'sentiment' and 'confidence'. "
            "'sentiment' should be a float between -1 (very negative) and 1 (very positive), "
            "and 'confidence' should be a float between 0 and 1. "
            "Here is the text:\n\n"
            f"'''{text}'''"
        )
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=60,
        temperature=0.0)
        reply = response.choices[0].message.content.strip()
        result = json.loads(reply)
        sentiment = result.get("sentiment", 0.0)
        confidence = result.get("confidence", 0.0)
        return round(sentiment, 4), round(confidence, 4)
    except Exception as e:
        print(f"⚠️ Error analyzing sentiment via ChatGPT: {e}")
        return 0.0, 0.0


def fetch_finviz_news(filters=""):
    url = f"{BASE_URL}?v=3&auth={API_TOKEN}&{filters}"
    with requests.Session() as session:
        response = session.get(url, headers=HEADERS, cookies=COOKIES)

    if response.status_code == 200:
        raw_text = response.text.strip()
        # Check if the response is HTML rather than CSV
        if raw_text.startswith("<!DOCTYPE html>"):
            print("❌ Received HTML response instead of CSV. Check your API token, cookies, and URL.")
            return None

        try:
            df = pd.read_csv(io.StringIO(raw_text), delimiter=",", quotechar='"', 
                             on_bad_lines="skip", header=None)
            if df.shape[1] == 1:
                df = df[0].apply(lambda row: row.split(",")).apply(pd.Series)
                if df.shape[1] < 6:
                    print(f"❌ After manual split, unexpected CSV format: got {df.shape[1]} columns")
                    print("Raw text for debugging:\n", raw_text)
                    return None
            if df.shape[1] < 6:
                print(f"❌ Unexpected CSV format: got {df.shape[1]} columns")
                return None

            df.columns = ["Title", "Source", "Date", "URL", "Category", "Ticker"]
            df = df.dropna(subset=["Title", "Source", "Date", "URL", "Ticker"])

            # Scrape full article text
            df["Full_Text"] = df["URL"].apply(scrape_article_text)

            # Perform sentiment analysis using ChatGPT
            df[["SentimentScore", "ConfidenceScore"]] = df["Full_Text"].apply(lambda text: pd.Series(classify_sentiment_chatgpt(text)))

            # Insert multiple rows for articles with multiple tickers
            for _, row in df.iterrows():
                tickers = row["Ticker"].split(",")  # Split multiple tickers
                for ticker in tickers:
                    price_change = get_price_change(ticker.strip(), row["Date"])
                    # Use ChatGPT for sentiment analysis again in case you need per-ticker context
                    sentiment_score, confidence_score = classify_sentiment_chatgpt(row["Full_Text"])
                    if price_change is None:
                        price_change = 0.0
                    if sentiment_score is None:
                        sentiment_score = 0.0
                    cursor.execute("""
                        INSERT INTO StockNews (Title, Source, Date, URL, Category, Ticker, Full_Text, SentimentScore, ConfidenceScore, PriceChange)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (row["Title"], row["Source"], row["Date"], row["URL"], row["Category"], ticker.strip(), row["Full_Text"], sentiment_score, confidence_score, price_change))

            conn.commit()
            print("✅ Successfully stored Stock News with Sentiment, Confidence Scores, and Price Changes in Database")
            return True
        except Exception as e:
            print(f"❌ Error parsing response: {e}")
            return None
    else:
        print(f"❌ Failed to fetch news. Status Code: {response.status_code}")
        return None

def scrape_article_text(url):
    """
    Scrape the full text of a news article from its URL while skipping problematic domains.
    """
    blocked_domains = ["businesswire.com"]

    if any(domain in url for domain in blocked_domains):
        print(f"⚠️ Skipping {url} - Known to block scrapers.")
        return "Skipped due to website restrictions."

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        article_text = " ".join([p.text for p in paragraphs])

        return article_text[:5000]  # Limit text size
    except Exception as e:
        print(f"⚠️ Failed to scrape article from {url}: {e}")

    return "Error fetching article text."


def get_price_change(ticker, article_datetime_str):
    """
    Calculates price change from article timestamp to the latest price using 1-minute data.
    Gracefully handles non-trading hours by falling back to price before the article.
    """
    try:
        import pytz
        article_dt = datetime.strptime(article_datetime_str.strip(), "%Y-%m-%d %H:%M:%S")
        eastern = pytz.timezone("America/New_York")
        article_dt = eastern.localize(article_dt)

        now = datetime.now(eastern)
        if (now - article_dt).days > 7:
            print(f"⚠️ Article too old for 1-minute data: {ticker}")
            return None

        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d", interval="1m", prepost=False)

        if hist.empty:
            return None

        hist.index = pd.to_datetime(hist.index).tz_convert("America/New_York")

        # Try price at or after article time
        after_article = hist[hist.index >= article_dt]

        if not after_article.empty:
            article_price = after_article.iloc[0]["Close"]
        else:
            # No data after article time — fallback to price just before
            before_article = hist[hist.index < article_dt]
            if not before_article.empty:
                article_price = before_article.iloc[-1]["Close"]
                print(f"⚠️ Using price before article time for {ticker}")
            else:
                print(f"⚠️ No data before or after article time for {ticker}")
                return None

        latest_price = hist["Close"].iloc[-1]
        price_change = ((latest_price - article_price) / article_price) * 100
        return round(price_change, 2)

    except Exception as e:
        print(f"⚠️ Error calculating price change for {ticker}: {e}")
        return None


# Run the function to fetch, scrape, analyze, and store news
fetch_finviz_news()
