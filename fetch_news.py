from pyfinviz.news import News
import requests
import pandas as pd
import io
from bs4 import BeautifulSoup
import sqlite3
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import time
from datetime import datetime
import pytz
import yfinance as yf
import re


# A set of known tickers to validate against (you can preload a static list or fetch dynamically)
# Optional: cache or limit to US tickers from major indices
import yfinance as yf

# Load all valid tickers (one-time fetch — cache or store this if needed)
VALID_TICKERS = set(ticker.strip().upper() for ticker in yf.tickers_sp500())  # or use a larger static list

def extract_tickers_from_text(text):
    potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
    return [ticker for ticker in potential_tickers if ticker in VALID_TICKERS]


# Load FinBERT Model
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

def fetch_finviz_news():
    print("📥 Fetching news from Finviz (via pyfinviz)...")

    conn = sqlite3.connect("stock_sentiment.db")
    cursor = conn.cursor()

    # Create table with unique (Title, URL, Ticker) if not exists
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
            PriceChange REAL,
            UNIQUE(URL, Ticker)
        )
    """)

    try:
        finviz_news = News()
        df = finviz_news.news_df  # DataFrame with 'Title', 'Link', 'Date'
        df.rename(columns={"Headline": "Title", "URL": "Link", "Time": "Date"}, inplace=True)


        for _, row in df.iterrows():
            title = row['Title']
            url = row['Link']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_text = scrape_article_text(url)
            sentiment_score, confidence_score = classify_sentiment_finbert(full_text)

            # Extract tickers using a simple heuristic (update logic if needed)
            tickers = extract_tickers_from_text(title)
            for ticker in tickers:
                price_change = get_price_change(ticker, timestamp)

                try:
                    cursor.execute("""
                        INSERT INTO StockNews (
                            Title, Source, Date, URL, Category, Ticker, Full_Text,
                            SentimentScore, ConfidenceScore, PriceChange
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        title, "Finviz", timestamp, url, "News", ticker,
                        full_text, sentiment_score, confidence_score, price_change
                    ))
                except sqlite3.IntegrityError:
                    print(f"⚠️ Skipping duplicate: ({url}, {ticker})")
        
        conn.commit()
        print("✅ News fetched and stored from pyfinviz.")
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
    finally:
        conn.close()

def scrape_article_text(url):
    blocked_domains = ["businesswire.com"]

    # ✅ NEW: Fix for relative Finviz paths
    if url.startswith("/news/"):
        url = "https://finviz.com" + url

    if any(domain in url for domain in blocked_domains):
        print(f"⚠️ Skipping {url} - Known to block scrapers.")
        return "Skipped due to website restrictions."

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")
        article_text = " ".join([p.text for p in paragraphs])

        return article_text[:5000]
    except Exception as e:
        print(f"⚠️ Failed to scrape article from {url}: {e}")

    return "Error fetching article text."


def classify_sentiment_finbert(text):
    """
    Use FinBERT to classify sentiment and return both sentiment score and confidence score.
    Positive: Closer to +1, Negative: Closer to -1, Neutral: Around 0.
    Confidence Score: The highest probability from the model.
    """
    try:
        result = finbert_pipeline(text[:512], top_k=None)  # Get all sentiment scores

        # Ensure result is a list of dictionaries
        if isinstance(result, list) and isinstance(result[0], list):
            result = result[0]  # Extract first element (list of label-score dictionaries)

        # Mapping labels to sentiment scores
        score_mapping = {"positive": 1, "negative": -1, "neutral": 0}

        # Compute weighted sentiment score
        sentiment_score = sum(score_mapping[item["label"].lower()] * item["score"] for item in result)

        # Get the highest confidence score
        confidence_score = max(item["score"] for item in result)

        return round(sentiment_score, 4), round(confidence_score, 4)  # Return both
    except Exception as e:
        print(f"⚠️ Error analyzing sentiment: {e}")
        return 0.0, 0.0  # Default to neutral with 0 confidence


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
