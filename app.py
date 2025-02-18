from flask import Flask, render_template
import sqlite3

app = Flask(__name__)

def get_stock_news():
    conn = sqlite3.connect("stock_sentiment.db")
    cursor = conn.cursor()
    
    # Fetch SentimentScore and ConfidenceScore
    cursor.execute("SELECT Title, Source, Date, URL, Category, Ticker, SentimentScore, ConfidenceScore FROM StockNews ORDER BY Date DESC LIMIT 50")
    news = cursor.fetchall()
    
    conn.close()
    return news



@app.route("/")
def index():
    news = get_stock_news()
    return render_template("index.html", news=news)

if __name__ == "__main__":
    app.run(debug=True)
