# Create a dashboard that aggregates information from various sources like Crunchbase, Twitter, and academic journals
# to identify promising startups for investment. You could use web scraping and NLP techniques to get this data.
pip install tweepy requests beautifulsoup4 nltk pandas matplotlib
# Import necessary libraries
import tweepy
import requests
import json
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Initialize NLTK and Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Twitter API Setup
TWITTER_API_KEY = 'your_twitter_api_key'
TWITTER_API_SECRET_KEY = 'your_twitter_api_secret_key'
TWITTER_ACCESS_TOKEN = 'your_twitter_access_token'
TWITTER_ACCESS_TOKEN_SECRET = 'your_twitter_access_token_secret'

auth = tweepy.OAuth1UserHandler(consumer_key=TWITTER_API_KEY,
                                consumer_secret=TWITTER_API_SECRET_KEY,
                                access_token=TWITTER_ACCESS_TOKEN,
                                access_token_secret=TWITTER_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# Crunchbase API Setup (API Key)
CRUNCHBASE_API_KEY = 'your_crunchbase_api_key'
CRUNCHBASE_API_URL = 'https://api.crunchbase.com/v3.1/organizations'

# Function to fetch startups from Crunchbase using Crunchbase API
def fetch_startups_from_crunchbase(query="startup", location="San Francisco"):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    params = {
        'user_key': CRUNCHBASE_API_KEY,
        'query': query,
        'location': location
    }
    response = requests.get(CRUNCHBASE_API_URL, headers=headers, params=params)
    data = response.json()
    startups = []

    if data['data']['items']:
        for item in data['data']['items']:
            startup = {
                'name': item['properties']['name'],
                'description': item['properties']['short_description'],
                'category': item['properties']['category_list'],
                'funding': item['properties']['funding_rounds'],
                'location': item['properties']['location']
            }
            startups.append(startup)
    
    return startups

# Function to fetch tweets related to startups
def fetch_tweets(query="startup investment"):
    tweets = api.search_tweets(q=query, lang="en", count=100)
    tweet_data = []

    for tweet in tweets:
        tweet_data.append({
            'username': tweet.user.screen_name,
            'tweet': tweet.text,
            'created_at': tweet.created_at,
            'likes': tweet.favorite_count,
            'retweets': tweet.retweet_count
        })

    return tweet_data

# Web scraping function to extract academic journal articles related to startups
def fetch_academic_journals(query="startup investment site:scholar.google.com"):
    url = f'https://scholar.google.com/scholar?q={query}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = []
    for result in soup.find_all('h3', class_='gs_rt'):
        title = result.get_text()
        link = result.find('a')['href'] if result.find('a') else None
        articles.append({'title': title, 'link': link})
    
    return articles

# Sentiment Analysis on Tweets
def analyze_sentiment(tweets):
    sentiment_scores = []
    for tweet in tweets:
        sentiment = sia.polarity_scores(tweet['tweet'])
        sentiment_scores.append({
            'tweet': tweet['tweet'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu'],
            'compound': sentiment['compound']
        })

    return sentiment_scores

# Main function to aggregate data
def aggregate_dashboard_data():
    # Fetch data from Crunchbase, Twitter, and Academic Journals
    crunchbase_startups = fetch_startups_from_crunchbase()
    twitter_tweets = fetch_tweets()
    academic_journals = fetch_academic_journals()

    # Sentiment analysis on Twitter data
    sentiment_data = analyze_sentiment(twitter_tweets)

    # Convert sentiment data into a DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df['sentiment'] = sentiment_df['compound'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

    # Print and visualize results
    print("Crunchbase Startups Data:", crunchbase_startups)
    print("Twitter Sentiment Analysis Data:")
    print(sentiment_df)

    # Plotting sentiment distribution
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    sentiment_counts.plot(kind='bar', title="Sentiment Analysis of Twitter Mentions")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Mentions")
    plt.show()

# Run the dashboard aggregation
if __name__ == '__main__':
    aggregate_dashboard_data()
