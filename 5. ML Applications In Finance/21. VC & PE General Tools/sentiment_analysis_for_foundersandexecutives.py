# Use Natural Language Processing to analyze interviews, podcasts, or social media interactions involving company founders
# or executives to gauge leadership quality and public perception..
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import nltk

nltk.download('punkt')

# Simulated Dataset
data = {
    "Source": ["Interview", "Podcast", "Twitter Post", "Blog Article", "Facebook Post"],
    "Content": [
        "Our CEO spoke about the future of technology and innovation.",
        "The founder shared insightful thoughts on company culture and growth.",
        "People seem unhappy with recent decisions made by the leadership.",
        "The leadership team's new policy on employee benefits is widely appreciated.",
        "This executive has a knack for inspiring change and leading effectively."
    ]
}

df = pd.DataFrame(data)

# Sentiment Analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Sentiment polarity (-1 to 1)
    return sentiment

df["Sentiment"] = df["Content"].apply(analyze_sentiment)
df["Sentiment_Label"] = df["Sentiment"].apply(
    lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
)

# Key Topic Extraction using Latent Dirichlet Allocation (LDA)
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df["Content"])
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Display topics
def display_topics(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(topic_words)}")
    return topics

topics = display_topics(lda, vectorizer.get_feature_names_out(), 5)

# Word Cloud
all_text = " ".join(df["Content"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

# Visualization
plt.figure(figsize=(15, 8))

# Sentiment Distribution
plt.subplot(2, 1, 1)
df["Sentiment_Label"].value_counts().plot(kind="bar", color=["green", "red", "blue"], alpha=0.7)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Frequency")

# Word Cloud
plt.subplot(2, 1, 2)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Content")

plt.tight_layout()
plt.show()

# Print Topics
print("Key Topics Identified from Content:")
for topic in topics:
    print(topic)
