from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import json
import plotly
import plotly.express as px

app = Flask(__name__)

def get_stock_news(sort_by=None, order="desc"):
    conn = sqlite3.connect("stock_sentiment.db")
    cursor = conn.cursor()

    # Default sorting by date
    query = "SELECT Title, Source, Date, URL, Category, Ticker, SentimentScore, ConfidenceScore, PriceChange FROM StockNews"
    
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
    order = request.args.get("order", "desc")  # Default to descending order
    news = get_stock_news(sort_by, order)
    return render_template("index.html", news=news, sort_by=sort_by, order=order)

@app.route("/chart-data")
def chart_data():
    conn = sqlite3.connect("stock_sentiment.db")
    df = pd.read_sql_query("SELECT Ticker, SentimentScore, PriceChange FROM StockNews", conn)
    conn.close()

    df = df.dropna(subset=["SentimentScore", "PriceChange"])

    if df.empty:
        return json.dumps({"data": [], "layout": {"title": "No Data Available"}})

    # Manually construct the Plotly scatter plot with hover labels
    data = [{
        "x": df["SentimentScore"].tolist(),
        "y": df["PriceChange"].tolist(),
        "mode": "markers",
        "marker": {"size": 10},
        "hoverinfo": "text",  # Show only on hover
        "text": [f"{ticker}<br>({score}, {change}%)" for ticker, score, change in zip(df["Ticker"], df["SentimentScore"], df["PriceChange"])],
        "type": "scatter"
    }]

    layout = {
        "title": "Sentiment Score vs. Price Change",
        "xaxis": {"title": "Sentiment Score"},
        "yaxis": {"title": "Price Change (%)"},
        "template": "plotly_white"
    }

    return json.dumps({"data": data, "layout": layout})

if __name__ == "__main__":
    app.run(debug=True)
