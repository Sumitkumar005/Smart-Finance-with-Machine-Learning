'''
# Recommender systems for personalized financial planning and product recommendations.

This script implements a recommender system to suggest personalized financial products and plans based on user preferences, behavior, and historical data. The system uses collaborative filtering and content-based filtering techniques.
'''

# Import libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Sample dataset
data = {
    'user_id': [1, 2, 3, 4, 5],
    'investment_interest': ['mutual_funds,stocks', 'real_estate,stocks', 'insurance,mutual_funds', 'cryptocurrency,stocks', 'mutual_funds,real_estate'],
    'risk_tolerance': [0.8, 0.6, 0.5, 0.9, 0.4],  # 1.0: high risk tolerance, 0.0: low risk tolerance
    'financial_goals': ['retirement', 'wealth_growth', 'insurance', 'crypto_growth', 'real_estate'],
    'product_ratings': [
        {'mutual_funds': 4, 'stocks': 5},
        {'real_estate': 3, 'stocks': 4},
        {'insurance': 5, 'mutual_funds': 3},
        {'cryptocurrency': 5, 'stocks': 4},
        {'mutual_funds': 3, 'real_estate': 4}
    ]
}

# Convert the data into a DataFrame
user_df = pd.DataFrame(data)

# Helper function to create a product matrix for ratings
def create_product_matrix(data):
    all_products = set([item for sublist in data['product_ratings'] for item in sublist.keys()])
    product_matrix = []
    for ratings in data['product_ratings']:
        row = [ratings.get(product, 0) for product in all_products]
        product_matrix.append(row)
    return pd.DataFrame(product_matrix, columns=all_products, index=data['user_id'])

# Create product rating matrix
product_matrix = create_product_matrix(user_df)

# Normalize the product ratings using MinMaxScaler
scaler = MinMaxScaler()
normalized_product_matrix = pd.DataFrame(
    scaler.fit_transform(product_matrix),
    columns=product_matrix.columns,
    index=product_matrix.index
)

# Compute similarity matrix
similarity_matrix = cosine_similarity(normalized_product_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=user_df['user_id'], columns=user_df['user_id'])

# Recommend products based on collaborative filtering
def recommend_products(user_id, similarity_df, product_matrix, top_n=3):
    similar_users = similarity_df.loc[user_id].sort_values(ascending=False).index[1:]  # Exclude self
    recommended_products = {}
    
    for sim_user in similar_users:
        user_products = product_matrix.loc[sim_user]
        for product, rating in user_products.items():
            if product_matrix.loc[user_id, product] == 0:  # User hasn't rated this product
                if product not in recommended_products:
                    recommended_products[product] = 0
                recommended_products[product] += rating * similarity_df.loc[user_id, sim_user]
    
    # Sort by weighted scores
    recommended_products = sorted(recommended_products.items(), key=lambda x: x[1], reverse=True)
    return [product for product, _ in recommended_products[:top_n]]

# Example usage: Recommend products for user 1
recommended_products = recommend_products(user_id=1, similarity_df=similarity_df, product_matrix=product_matrix)
print(f"Recommended products for user 1: {recommended_products}")

# Content-based filtering: Recommend financial plans based on features
def recommend_financial_plan(user_id, user_df, top_n=3):
    user_row = user_df.loc[user_df['user_id'] == user_id]
    features = ['investment_interest', 'financial_goals']
    feature_matrix = pd.get_dummies(user_df[features].apply(lambda x: ','.join(x), axis=1))
    
    user_features = feature_matrix.loc[user_df['user_id'] == user_id]
    similarities = cosine_similarity(user_features, feature_matrix)[0]
    
    user_df['similarity'] = similarities
    recommendations = user_df.sort_values(by='similarity', ascending=False)[1:top_n+1]  # Exclude self
    return recommendations[['user_id', 'investment_interest', 'financial_goals']]

# Example usage: Recommend financial plans for user 1
recommended_plans = recommend_financial_plan(user_id=1, user_df=user_df)
print("\nRecommended financial plans for user 1:")
print(recommended_plans)

# Visualize similarity matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("User Similarity Matrix")
plt.xlabel("User ID")
plt.ylabel("User ID")
plt.show()

# Save model and results (optional)
user_df.to_csv("user_data_with_recommendations.csv", index=False)
print("\nUser data with recommendations saved to 'user_data_with_recommendations.csv'.")
