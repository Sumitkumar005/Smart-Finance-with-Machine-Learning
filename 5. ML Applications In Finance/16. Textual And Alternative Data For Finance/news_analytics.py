# Create a real-time dashboard that uses NLP to analyze financial news for keywords and sentiments that could be trading signals.
# Import necessary libraries
import requests
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import streamlit as st
from datetime import datetime

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set up the News API key
NEWS_API_KEY = "your_newsapi_key_here"  # Replace with your actual API key

# Function to fetch news articles from NewsAPI
def fetch_news(api_key, query="finance", language="en", sort_by="publishedAt"):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&sortBy={sort_by}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch news: {response.status_code}")
        return None

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    keywords = [word for word in words if word.isalpha() and word.lower() not in stop_words]
    return keywords

# Streamlit app
st.title("Real-Time Financial News Sentiment Dashboard")
st.markdown("""
Analyze financial news in real-time for trading signals using NLP. The app fetches news articles, 
extracts keywords, and performs sentiment analysis.
""")

# User input for the news query
query = st.text_input("Enter the topic to search for:", "finance")

# Fetch news articles
if st.button("Fetch News"):
    with st.spinner("Fetching news articles..."):
        news_data = fetch_news(NEWS_API_KEY, query=query)
        if news_data:
            articles = news_data.get("articles", [])
            if articles:
                st.success(f"Fetched {len(articles)} articles.")
                
                # Process and display articles
                news_df = pd.DataFrame(articles)
                news_df = news_df[["title", "description", "publishedAt", "url"]]

                # Perform NLP analysis
                results = []
                for _, row in news_df.iterrows():
                    title = row["title"]
                    description = row["description"] if row["description"] else ""
                    combined_text = f"{title} {description}"

                    # Sentiment Analysis
                    polarity, subjectivity = analyze_sentiment(combined_text)

                    # Keyword Extraction
                    keywords = extract_keywords(combined_text)

                    results.append({
                        "Title": title,
                        "Description": description,
                        "Published At": row["publishedAt"],
                        "URL": row["url"],
                        "Sentiment (Polarity)": polarity,
                        "Sentiment (Subjectivity)": subjectivity,
                        "Keywords": ", ".join(keywords[:10])  # Show top 10 keywords
                    })

                # Convert to DataFrame for display
                result_df = pd.DataFrame(results)

                # Display in Streamlit
                st.dataframe(result_df)
                
                # Provide download option
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No articles found.")
