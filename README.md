# Stock Sentiment Analysis Dashboard

A Flask-based dashboard that pulls live stock news, scores sentiment with FinBERT, and correlates it against real price and volume movement — visualized through an interactive 3D chart and a sortable news table.

## Overview

This project automates the pipeline from raw financial news to actionable signal:

1. **Collect** — pulls stock news headlines and articles from Finviz on a recurring schedule.
2. **Analyze** — scores each article's sentiment using a finance-tuned BERT model (FinBERT).
3. **Correlate** — pulls historical price data via Yahoo Finance to measure how price moved after each article's publish time, plus relative trading volume.
4. **Visualize** — serves the results through a Flask web app with a sortable news table and a 3D scatter plot (sentiment vs. price change vs. relative volume).

## Features

- **Automated news ingestion**: background thread refreshes stock news every 5 minutes without blocking the web server.
- **FinBERT sentiment scoring**: each article gets a sentiment score (-1 to +1) and a confidence score.
- **Article scraping**: full article text is pulled and stored (not just headlines) for more accurate sentiment analysis.
- **Price impact tracking**: uses `yfinance` minute-level data to calculate the percentage price change following each article.
- **Relative volume**: compares each ticker's latest trading volume against its 10-day average.
- **Interactive dashboard**:
  - Sortable news table (by sentiment score or price change).
  - 3D scatter plot (Plotly) of sentiment vs. price change vs. relative volume, with ticker search/highlight.
- **Persistent storage**: all data is stored in a local SQLite database with duplicate protection.

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Flask, Python |
| NLP / Sentiment | Hugging Face `transformers`, FinBERT (`yiyanghkust/finbert-tone`), PyTorch |
| Data | `pandas`, SQLite |
| Market Data | `yfinance` |
| Scraping | `requests`, `BeautifulSoup4` |
| Visualization | Plotly (3D scatter), HTML/CSS/JS |

## Project Structure

```
.
├── app.py              # Flask app: routes, chart data endpoint, background news updater
├── fetch_news.py       # News scraping, FinBERT sentiment scoring, price change calculation
├── index.html          # Dashboard template (news table + 3D chart)
├── requirements.txt    # Python dependencies
├── install.sh           # Environment setup script
└── stock_sentiment.db  # SQLite database (created on first run)
```

## Setup

### Prerequisites

- Python 3.9+
- A Finviz Elite account (for the news export API)

### Installation

```bash
git clone <repo-url>
cd <repo-name>
bash install.sh
```

This creates a virtual environment, activates it, and installs all dependencies from `requirements.txt`.

### Configuration

Before running the app, set your Finviz credentials as environment variables rather than hardcoding them:

```bash
export FINVIZ_API_TOKEN="your_token_here"
export FINVIZ_AUTH_COOKIE="your_cookie_here"
```

> **Note:** Earlier versions of this project had credentials hardcoded in `fetch_news.py`. If you're forking or reusing this code, make sure to move any tokens/cookies out of source control and into environment variables or a `.env` file (excluded via `.gitignore`).

### Running the app

```bash
source venv/bin/activate
python app.py
```

The dashboard will be available at `http://127.0.0.1:5000`. On startup, a background thread begins fetching and scoring news every 5 minutes.

## How It Works

1. `fetch_finviz_news()` pulls the latest news export from Finviz, scrapes each linked article's full text, and runs it through FinBERT for sentiment scoring.
2. For each ticker mentioned, `get_price_change()` pulls minute-level historical price data and computes the percentage change from the article's publish time to now.
3. Results are stored in `stock_sentiment.db`, deduplicated on `(URL, Ticker)`.
4. The Flask app queries this database to render the sortable news table and to compute relative volume for the 3D chart.

## Roadmap / Possible Improvements

- Move credentials to environment variables / `.env` (see note above).
- Add caching for `yfinance` calls to reduce redundant API requests.
- Add unit tests for sentiment scoring and price change calculations.
- Deploy behind a production WSGI server (e.g., Gunicorn) instead of Flask's dev server.
