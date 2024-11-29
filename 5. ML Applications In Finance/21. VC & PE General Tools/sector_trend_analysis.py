# Create an algorithm that uses machine learning to identify emerging sectors or trends based on news articles, 
# patent filings, academic papers, or market data.

"""
Python script for identifying emerging sectors or trends using machine learning and natural language processing (NLP).
This script fetches data from news APIs, preprocesses it, and applies topic modeling to extract trends.
"""

# Import necessary libraries
import requests
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from datetime import datetime, timedelta

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# News API key (replace with your own API key)
NEWS_API_KEY = "your_news_api_key"

# Step 1: Fetch Data from News API
def fetch_news_data(query, from_date, to_date, api_key=NEWS_API_KEY):
    """
    Fetch news articles from the NewsAPI based on a query and date range.
    """
    url = (
        f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}"
        f"&sortBy=popularity&language=en&apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return pd.DataFrame([{'title': a['title'], 'description': a['description'], 'content': a['content']} for a in articles])
    else:
        print(f"Failed to fetch news: {response.status_code}, {response.json()}")
        return pd.DataFrame()

# Step 2: Text Preprocessing
def preprocess_text(text):
    """
    Preprocess text by removing special characters, stopwords, and converting to lowercase.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    tokens = text.split()  # Tokenize text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Step 3: Apply NLP Techniques - Topic Modeling
def extract_topics(text_data, num_topics=5, num_words=10):
    """
    Apply LDA topic modeling to extract topics from the text data.
    """
    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(text_data)

    # Apply Latent Dirichlet Allocation (LDA)
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(X)

    # Extract topics
    words = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words = [words[i] for i in topic.argsort()[-num_words:]]
        topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
    return topics, lda_model

# Step 4: Generate Word Cloud for Visualizing Topics
def generate_word_cloud(text_data):
    """
    Generate a word cloud from the text data.
    """
    text = ' '.join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Main Function
if __name__ == "__main__":
    # Define parameters
    query = "emerging technologies"
    today = datetime.now()
    from_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')

    # Fetch data
    print("Fetching news articles...")
    news_data = fetch_news_data(query, from_date, to_date)

    if not news_data.empty:
        # Preprocess text data
        print("Preprocessing text...")
        news_data['cleaned_content'] = news_data['content'].apply(preprocess_text)

        # Extract topics using NLP
        print("Extracting topics...")
        topics, lda_model = extract_topics(news_data['cleaned_content'].dropna())
        print("\nIdentified Topics:")
        for topic in topics:
            print(topic)

        # Visualize data with a word cloud
        print("Generating word cloud...")
        generate_word_cloud(news_data['cleaned_content'])

        # Save results to CSV
        news_data.to_csv("emerging_trends_news_data.csv", index=False)
        print("Data saved to 'emerging_trends_news_data.csv'.")
    else:
        print("No articles found for the given query and date range.")
