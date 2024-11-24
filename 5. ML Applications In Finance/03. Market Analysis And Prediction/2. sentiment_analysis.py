# Analyzing news articles or social media to gauge market sentiment

'''
In this Python script, we use sentiment analysis to analyze text data from news articles or social media. 
Sentiment analysis can provide insights into market sentiment, which is an essential aspect of financial decision-making.
'''

# Importing necessary libraries
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample data: News articles or social media posts (for illustration purposes)
data = {
    'text': [
        'The stock market is booming today with strong gains.',
        'The economy is facing a downturn due to rising inflation.',
        'Tech stocks are experiencing a sharp decline.',
        'Investors are optimistic about the upcoming earnings reports.',
        'There is a lot of uncertainty in the market right now.'
    ]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Function to calculate sentiment polarity using TextBlob
def get_sentiment(text):
    # Create a TextBlob object for the given text
    blob = TextBlob(text)
    # Return the polarity score (-1 to 1) where negative values indicate negative sentiment and positive values indicate positive sentiment
    return blob.sentiment.polarity

# Apply sentiment analysis to each article/post
df['sentiment'] = df['text'].apply(get_sentiment)

# Display the results
print("Sentiment Analysis Results:")
print(df[['text', 'sentiment']])

# Plotting the sentiment distribution
plt.figure(figsize=(10, 6))
plt.hist(df['sentiment'], bins=20, color='skyblue', edgecolor='black')
plt.title('Sentiment Distribution of News Articles/Social Media Posts')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Example: Predicting market sentiment based on the average polarity score
average_sentiment = df['sentiment'].mean()

print(f'Average Sentiment: {average_sentiment:.2f}')

# Interpreting the result
if average_sentiment > 0:
    print("The overall market sentiment is positive.")
elif average_sentiment < 0:
    print("The overall market sentiment is negative.")
else:
    print("The overall market sentiment is neutral.")
