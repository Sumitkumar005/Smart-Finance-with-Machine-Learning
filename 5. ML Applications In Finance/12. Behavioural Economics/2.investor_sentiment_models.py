# Sentiment analysis and correlation with stock price movements

# Import required libraries
import pandas as pd
import numpy as np
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Initialize sentiment analyzer
import nltk
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to fetch social media/news data (mock example)
def fetch_sentiment_data():
    """
    Simulate fetching social media/news data with date and content. 
    In production, integrate APIs like Twitter or NewsAPI.
    """
    data = {
        "date": [
            "2024-11-20", "2024-11-21", "2024-11-21", "2024-11-22", 
            "2024-11-22", "2024-11-23", "2024-11-24"
        ],
        "content": [
            "The stock market is on fire today, so many opportunities!",
            "Tech stocks are falling; investors are worried.",
            "Market stability observed; growth potential for biotech stocks.",
            "Huge potential in green energy stocks. Great time to invest!",
            "Major drop in oil prices has spooked some investors.",
            "Strong quarterly results for some big players in the market.",
            "Investors optimistic about upcoming economic data."
        ]
    }
    return pd.DataFrame(data)

# Perform sentiment analysis
def perform_sentiment_analysis(data):
    """
    Analyze sentiment from textual data using VADER.
    """
    sentiments = []
    for text in data["content"]:
        sentiment = sia.polarity_scores(text)["compound"]
        sentiments.append(sentiment)
    data["sentiment_score"] = sentiments
    return data

# Fetch stock price data using Alpha Vantage API
def fetch_stock_data(stock_symbol, api_key):
    """
    Retrieve historical stock data from Alpha Vantage API.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={stock_symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    if "Time Series (Daily)" in data:
        prices = []
        for date, stats in data["Time Series (Daily)"].items():
            prices.append({
                "date": date,
                "close": float(stats["4. close"])
            })
        stock_data = pd.DataFrame(prices)
        stock_data["date"] = pd.to_datetime(stock_data["date"])
        stock_data.sort_values("date", inplace=True)
        return stock_data
    else:
        raise ValueError("Error fetching stock data from Alpha Vantage")

# Correlate sentiment data with stock prices
def correlate_sentiment_with_prices(sentiment_data, stock_data):
    """
    Merge sentiment data with stock prices and calculate correlation.
    """
    # Convert dates to datetime for alignment
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    
    # Merge sentiment scores with stock prices
    combined_data = pd.merge(sentiment_data, stock_data, on="date", how="inner")
    
    # Simple linear regression to study the relationship
    X = combined_data[["sentiment_score"]].values
    y = combined_data["close"].values
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    
    # Plot the data
    plt.scatter(combined_data["sentiment_score"], combined_data["close"], color="blue", alpha=0.6)
    plt.plot(combined_data["sentiment_score"], model.predict(X), color="red")
    plt.title("Sentiment Score vs. Stock Price")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Stock Price")
    plt.show()
    
    return combined_data, r_squared

# Main function
def main():
    # API Key for Alpha Vantage (Replace with your key)
    api_key = "your_api_key_here"  # <-- Replace with your Alpha Vantage API key
    stock_symbol = "AAPL"  # Example stock symbol (Apple Inc.)

    # Step 1: Fetch sentiment data
    sentiment_data = fetch_sentiment_data()
    print("Sentiment Data:")
    print(sentiment_data)

    # Step 2: Perform sentiment analysis
    sentiment_data = perform_sentiment_analysis(sentiment_data)
    print("\nSentiment Analysis Results:")
    print(sentiment_data)

    # Step 3: Fetch stock price data
    try:
        stock_data = fetch_stock_data(stock_symbol, api_key)
        print("\nStock Price Data:")
        print(stock_data.head())
    except ValueError as e:
        print(e)
        return

    # Step 4: Correlate sentiment with stock prices
    combined_data, r_squared = correlate_sentiment_with_prices(sentiment_data, stock_data)
    print("\nCombined Data (Sentiment + Stock Prices):")
    print(combined_data)

    print(f"\nR-squared value of the model: {r_squared}")

# Run the script
if __name__ == "__main__":
    main()
