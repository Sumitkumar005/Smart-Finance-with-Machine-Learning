# Scrape Reddit data to perform sentiment analysis and correlate this with stock or cryptocurrency trends.
'''
Python script to scrape Reddit data, analyze sentiments, and correlate them with stock or cryptocurrency trends.

Make sure to set up a Reddit API key from https://www.reddit.com/prefs/apps.
'''

# Import necessary libraries
import praw
import pandas as pd
from textblob import TextBlob
import yfinance as yf
import matplotlib.pyplot as plt

# Reddit API credentials
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',  # Replace with your Reddit client ID
    client_secret='YOUR_CLIENT_SECRET',  # Replace with your Reddit client secret
    user_agent='YOUR_APP_NAME'  # Replace with your app name
)

# Function to scrape Reddit posts
def scrape_reddit(subreddit_name, keyword, limit=100):
    '''
    Scrapes Reddit posts from a specific subreddit based on a keyword.

    Parameters:
    subreddit_name (str): The subreddit to scrape.
    keyword (str): The keyword to search for in posts.
    limit (int): The number of posts to scrape.

    Returns:
    pd.DataFrame: DataFrame containing scraped post titles and sentiments.
    '''
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for submission in subreddit.search(keyword, limit=limit):
        posts.append({
            'title': submission.title,
            'created': pd.to_datetime(submission.created_utc, unit='s')
        })
    return pd.DataFrame(posts)

# Function to analyze sentiment of Reddit posts
def analyze_sentiments(df):
    '''
    Analyzes sentiment polarity of post titles using TextBlob.

    Parameters:
    df (pd.DataFrame): DataFrame containing Reddit post titles.

    Returns:
    pd.DataFrame: DataFrame with an additional 'sentiment' column.
    '''
    sentiments = []
    for title in df['title']:
        sentiment = TextBlob(title).sentiment.polarity
        sentiments.append(sentiment)
    df['sentiment'] = sentiments
    return df

# Function to fetch stock or cryptocurrency data
def fetch_stock_data(ticker, start_date, end_date):
    '''
    Fetches stock or cryptocurrency data using Yahoo Finance.

    Parameters:
    ticker (str): The stock or cryptocurrency ticker symbol.
    start_date (str): Start date for fetching data (YYYY-MM-DD format).
    end_date (str): End date for fetching data (YYYY-MM-DD format).

    Returns:
    pd.DataFrame: DataFrame containing stock/cryptocurrency price data.
    '''
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Main execution
if __name__ == "__main__":
    # Scrape Reddit data
    subreddit_name = 'CryptoCurrency'  # Example subreddit
    keyword = 'Bitcoin'
    reddit_data = scrape_reddit(subreddit_name, keyword, limit=100)

    # Analyze sentiments
    reddit_data = analyze_sentiments(reddit_data)

    # Display sentiment analysis results
    print("Sample Sentiment Analysis Data:")
    print(reddit_data.head())

    # Plot sentiment trends
    plt.figure(figsize=(10, 5))
    reddit_data.set_index('created')['sentiment'].plot(title='Sentiment Over Time')
    plt.ylabel('Sentiment Polarity')
    plt.xlabel('Date')
    plt.grid()
    plt.show()

    # Fetch stock/cryptocurrency data
    ticker = 'BTC-USD'  # Bitcoin ticker
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    stock_data = fetch_stock_data(ticker, start_date, end_date)

    # Correlation between sentiment and stock price
    if not stock_data.empty:
        # Merge datasets
        stock_data['Date'] = stock_data.index.date
        reddit_data['Date'] = reddit_data['created'].dt.date
        merged_data = pd.merge(
            stock_data[['Date', 'Close']], reddit_data[['Date', 'sentiment']],
            on='Date', how='inner'
        )
        
        # Calculate correlation
        correlation = merged_data['Close'].corr(merged_data['sentiment'])
        print(f"Correlation between sentiment and {ticker} price: {correlation:.2f}")

        # Plot correlation
        plt.figure(figsize=(10, 5))
        plt.scatter(merged_data['sentiment'], merged_data['Close'], alpha=0.7)
        plt.title(f'Sentiment vs {ticker} Close Price')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel(f'{ticker} Close Price')
        plt.grid()
        plt.show()
    else:
        print(f"No stock/cryptocurrency data available for ticker {ticker}.")
