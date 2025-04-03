from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import json
import yfinance as yf

app = Flask(__name__)

def get_stock_news(sort_by=None, order="desc"):
    conn = sqlite3.connect("stock_sentiment.db")
    cursor = conn.cursor()
    query = ("SELECT Title, Source, Date, URL, Category, Ticker, "
             "SentimentScore, ConfidenceScore, PriceChange FROM StockNews")
    order_query = "DESC" if order == "desc" else "ASC"
    if sort_by == "sentiment":
        query += f" ORDER BY SentimentScore {order_query}"
    elif sort_by == "price":
        query += f" ORDER BY PriceChange {order_query}"
    else:
        query += " ORDER BY Date DESC"
    cursor.execute(query)
    news = cursor.fetchall()
    conn.close()
    return news

@app.route("/")
def index():
    sort_by = request.args.get("sort_by")
    order = request.args.get("order", "desc")
    news = get_stock_news(sort_by, order)
    return render_template("index.html", news=news, sort_by=sort_by, order=order)

@app.route("/stock/<ticker>")
def stock_detail(ticker):
    # Trim any extra spaces from the ticker
    ticker = ticker.strip()
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
    except Exception as e:
        info = {}
        print(f"Error fetching info for {ticker}: {e}")
    try:
        history = stock.history(period="1y")
        history = history.reset_index().to_dict(orient="records")
    except Exception as e:
        history = []
        print(f"Error fetching history for {ticker}: {e}")
    return render_template("stock_detail.html", ticker=ticker, info=info, history=history)

if __name__ == "__main__":
    app.run(debug=True)