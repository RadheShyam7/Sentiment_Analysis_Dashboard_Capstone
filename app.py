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
    
    # Get basic data from StockNews
    df_news = pd.read_sql_query("SELECT Ticker, SentimentScore, PriceChange FROM StockNews", conn)
    
    # Get volume data for each ticker
    tickers = list(set(df_news['Ticker'].tolist()))
    volume_data = {}
    
    # Suppress yfinance download progress output
    import yfinance as yf
    import io
    import sys
    
    # Track successful volume calculations
    successful_volume_calcs = 0
    
    # Redirect stdout during yfinance calls to suppress progress bars
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        for ticker in tickers:
            try:
                # Get data for last 10 days
                ticker_data = yf.download(ticker, period="10d", progress=False)
                
                # Check if we have enough data properly
                if not ticker_data.empty and len(ticker_data) > 1:
                    # Calculate average volume and relative volume
                    volumes = ticker_data['Volume'].values  # Get as numpy array
                    avg_volume = volumes[:-1].mean()  # Average excluding latest day
                    latest_volume = volumes[-1]  # Latest day's volume
                    
                    if avg_volume > 0:
                        rel_volume = float(latest_volume / avg_volume)
                        # Cap extreme values to prevent outliers
                        rel_volume = min(max(rel_volume, 0.1), 10.0)
                        volume_data[ticker] = rel_volume
                        successful_volume_calcs += 1
                    else:
                        volume_data[ticker] = 1.0
                else:
                    volume_data[ticker] = 1.0
            except Exception:
                volume_data[ticker] = 1.0
    finally:
        # Restore stdout
        sys.stdout = old_stdout
    
    # Log summary instead of individual errors
    print(f"Volume data: processed {len(tickers)} tickers, {successful_volume_calcs} successful calculations")
    
    # Add relative volume to dataframe
    df_news['RelativeVolume'] = df_news['Ticker'].map(volume_data)
    
    # Fill missing values with 1.0
    df_news['RelativeVolume'] = df_news['RelativeVolume'].fillna(1.0)
    
    # Drop rows with missing values
    df = df_news.dropna(subset=["SentimentScore", "PriceChange"])
    
    if df.empty:
        return json.dumps({"data": [], "layout": {"title": "No Data Available"}})

    # Create a 3D scatter plot - converting all Series to lists explicitly
    data = [{
        "x": df["SentimentScore"].tolist(),
        "y": df["PriceChange"].tolist(),
        "z": df["RelativeVolume"].tolist(),
        "mode": "markers",
        "type": "scatter3d", 
        "marker": {
            "size": 10,
            "color": df["SentimentScore"].tolist(),  # Convert to list
            "colorscale": "RdBu", 
            "colorbar": {"title": "Sentiment"},
            "opacity": 0.8
        },
        "text": [f"{ticker}<br>Sentiment: {score:.4f}<br>Price: {change:.2f}%<br>Vol: {vol:.2f}x" 
                for ticker, score, change, vol in zip(
                    df["Ticker"].tolist(), 
                    df["SentimentScore"].tolist(), 
                    df["PriceChange"].tolist(), 
                    df["RelativeVolume"].tolist()
                )],
        "hoverinfo": "text"
    }]

    layout = {
        "title": "Stock Sentiment vs Price Change vs Volume",
        "scene": {
            "xaxis": {"title": "Sentiment Score"},
            "yaxis": {"title": "Price Change (%)"},
            "zaxis": {"title": "Relative Volume (x avg)"}
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 50},
        "template": "plotly_white"
    }

    return json.dumps({"data": data, "layout": layout})

if __name__ == "__main__":
    app.run(debug=True)
