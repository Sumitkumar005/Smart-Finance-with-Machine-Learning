# Use Natural Language Processing (NLP) to analyze a company's sustainability reports and practices.
# Quantify metrics and present them in a user-friendly dashboard.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
from wordcloud import WordCloud
import spacy

# Load pre-trained spaCy model for NLP tasks
nlp = spacy.load('en_core_web_sm')

# Sample sustainability report data (In a real-world scenario, this would be loaded from actual reports)
reports = [
    "The company aims to reduce its carbon footprint by 20% over the next five years. Sustainability is at the core of our operations.",
    "We focus on renewable energy sources and have switched 50% of our energy consumption to solar power. Green energy is important for long-term sustainability.",
    "We are committed to diversity and inclusion, with various initiatives to increase the representation of women in leadership positions.",
    "The company has reduced waste by 30% in the last year and aims to reach zero waste by 2030.",
    "Sustainable practices in water management have helped us reduce water consumption by 15% this year."
]

# Convert the reports into a DataFrame for easy analysis
df_reports = pd.DataFrame(reports, columns=['report'])

# Text Preprocessing: Remove stopwords, punctuation, and lemmatize the text using spaCy
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert text to lowercase
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Apply text preprocessing to each report
df_reports['processed_report'] = df_reports['report'].apply(preprocess_text)

# TF-IDF Vectorization: Convert text to numerical data using Term Frequency-Inverse Document Frequency
tfidf_vectorizer = TfidfVectorizer(max_features=100)
X_tfidf = tfidf_vectorizer.fit_transform(df_reports['processed_report'])

# Latent Dirichlet Allocation (LDA) for Topic Modeling
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda_topics = lda.fit_transform(X_tfidf)

# Display the top words for each topic
def get_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

get_top_words(lda, tfidf_vectorizer.get_feature_names_out(), 10)

# Word Cloud Visualization
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df_reports['processed_report']))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Extracting some simple sustainability metrics (word frequencies)
sustainability_keywords = ['carbon', 'renewable', 'energy', 'diversity', 'inclusion', 'waste', 'water', 'sustainability']
keyword_counts = {keyword: df_reports['processed_report'].str.contains(keyword).sum() for keyword in sustainability_keywords}

# Visualize keyword frequencies in a bar chart
fig = px.bar(x=list(keyword_counts.keys()), y=list(keyword_counts.values()), labels={'x': 'Sustainability Keywords', 'y': 'Frequency'}, title='Sustainability Metrics Based on Reports')
fig.show()

# Quantify sentiment (positive or negative) in the reports using spaCy's Sentiment Analysis
def analyze_sentiment(text):
    doc = nlp(text)
    return doc.sentiment

df_reports['sentiment'] = df_reports['report'].apply(analyze_sentiment)

# Plot sentiment over time (or over reports)
plt.figure(figsize=(10, 5))
plt.plot(df_reports['sentiment'], marker='o', linestyle='-', color='b')
plt.title('Sentiment Analysis of Sustainability Reports')
plt.xlabel('Report Index')
plt.ylabel('Sentiment Score')
plt.show()

# Final thoughts: Now we have a basic framework for analyzing sustainability reports using NLP.
# We extracted topics, word frequencies, and sentiment to quantify sustainability efforts.
# This can be expanded further by using larger datasets, more advanced NLP techniques,
# and creating a full-fledged interactive dashboard.

# Save the processed data for future use
df_reports.to_csv('processed_sustainability_reports.csv', index=False)

# Example of how to use processed data for future predictions or analysis
# Example: Let's say we want to predict if a report is "highly sustainable" based on sentiment and keywords
df_reports['high_sustainability'] = df_reports['sentiment'].apply(lambda x: 1 if x > 0 else 0)
df_reports.to_csv('sustainability_predictions.csv', index=False)

