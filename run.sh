#!/bin/bash
# filepath: /Users/aryan/DS440-Rev1/Sentiment_Analysis_Dashboard_Capstone/run.sh
# Run the fetch_news.py script in the background
python3 fetch_news.py &

# Run the Flask app
python3 app.py