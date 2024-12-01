# An algorithm that matches startups with potential investors based on investor prekference, startup sector, stage, and other factors.

"""
Algorithm to match startups with potential investors based on preferences and factors like sector, stage, and other attributes.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load the data (Startups and Investors data)
# Replace these file paths with your actual data paths.
startups_data_path = "startups.csv"  # A CSV file containing startups data
investors_data_path = "investors.csv"  # A CSV file containing investors data

# Example CSV structure:
# Startups: ['startup_name', 'sector', 'stage', 'location', 'funding_amount']
# Investors: ['investor_name', 'preferred_sectors', 'preferred_stages', 'location', 'investment_capacity']

startups_df = pd.read_csv(startups_data_path)
investors_df = pd.read_csv(investors_data_path)

# Step 2: Preprocess the data
def preprocess_data(df, feature_columns):
    """
    Combines feature columns into a single string for each entry for easier comparison.
    """
    df['combined_features'] = df[feature_columns].fillna('').apply(lambda x: ' '.join(x), axis=1)
    return df

startups_df = preprocess_data(startups_df, ['sector', 'stage', 'location'])
investors_df = preprocess_data(investors_df, ['preferred_sectors', 'preferred_stages', 'location'])

# Step 3: Compute similarity scores
def compute_similarity(startup_features, investor_features):
    """
    Calculates similarity between startups and investors using cosine similarity.
    """
    vectorizer = CountVectorizer()
    combined_data = startup_features + investor_features
    feature_matrix = vectorizer.fit_transform(combined_data)

    # Extract similarity for startups and investors
    startup_matrix = feature_matrix[:len(startup_features)]
    investor_matrix = feature_matrix[len(startup_features):]

    similarity_scores = cosine_similarity(startup_matrix, investor_matrix)
    return similarity_scores

# Calculate similarity
similarity_scores = compute_similarity(startups_df['combined_features'], investors_df['combined_features'])

# Step 4: Match startups with investors
def match_startups_investors(similarity_matrix, startups, investors, top_n=3):
    """
    Matches startups with investors based on similarity scores.
    Returns top N matches for each startup.
    """
    matches = {}
    for i, startup in enumerate(startups['startup_name']):
        investor_indices = similarity_matrix[i].argsort()[::-1][:top_n]
        matches[startup] = investors.iloc[investor_indices]['investor_name'].tolist()
    return matches

matches = match_startups_investors(similarity_scores, startups_df, investors_df)

# Step 5: Display results
print("Top matches for startups:")
for startup, top_investors in matches.items():
    print(f"Startup '{startup}' is matched with investors: {', '.join(top_investors)}")

# Step 6: Optional - API integration for enrichment
# Replace 'your_api_key_here' with your actual API key if you use an external service.
def enrich_data_with_api(data, api_key):
    """
    Example function to enrich startup or investor data using an external API.
    """
    import requests
    enriched_data = []
    for entry in data:
        response = requests.get(
            f"https://api.example.com/enrich?query={entry}",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        if response.status_code == 200:
            enriched_data.append(response.json())
        else:
            enriched_data.append({"error": "Failed to fetch data"})
    return enriched_data

# Uncomment the following lines if you have an API key to enrich data
# api_key = "your_api_key_here"
# enriched_startups = enrich_data_with_api(startups_df['startup_name'], api_key)
# print("Enriched data for startups:", enriched_startups)

# Step 7: Save the matches (optional)
matches_df = pd.DataFrame([(k, v) for k, lst in matches.items() for v in lst], columns=['Startup', 'Investor'])
matches_df.to_csv("startup_investor_matches.csv", index=False)

print("Matching process completed. Results saved to 'startup_investor_matches.csv'.")
