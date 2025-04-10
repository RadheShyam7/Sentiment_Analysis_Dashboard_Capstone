import requests
import pandas as pd
import io
from bs4 import BeautifulSoup
import sqlite3
import yfinance as yf
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import time
from datetime import datetime, date, time
import pytz  # Make sure to import pytz for timezone handling


# Load FinBERT Model
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Your API Token
API_TOKEN = "6f0ab75a-803d-49e7-91ed-f8c818db85e5"
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

def fetch_finviz_news(filters=""):
    """
    Fetch stock news headlines from Finviz.
    Scrape full article text, perform sentiment analysis, fetch stock price change, and store in SQLite.
    """
    url = f"{BASE_URL}?v=3&auth={API_TOKEN}&{filters}"

    with requests.Session() as session:
        response = session.get(url, headers=HEADERS, cookies=COOKIES)

    if response.status_code == 200:
        try:
            raw_text = response.text.strip()
            df = pd.read_csv(io.StringIO(raw_text), delimiter=",", quotechar='"', on_bad_lines="skip")

            # Ensure proper column naming
            df.columns = ["Title", "Source", "Date", "URL", "Category", "Ticker"]
            df = df.dropna(subset=["Title", "Source", "Date", "URL", "Ticker"])

            # Scrape full article text
            df["Full_Text"] = df["URL"].apply(scrape_article_text)

            # Perform sentiment analysis
            df[["SentimentScore", "ConfidenceScore"]] = df["Full_Text"].apply(lambda text: pd.Series(classify_sentiment_finbert(text)))

            # Insert multiple rows for articles with multiple tickers
            for _, row in df.iterrows():
                tickers = row["Ticker"].split(",")  # Split multiple tickers
                for ticker in tickers:
                    price_change = get_price_change(ticker.strip(), row["Date"])
  # Get price change
                    sentiment_score, confidence_score = classify_sentiment_finbert(row["Full_Text"])

                    # Ensure defaults if values are missing
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
