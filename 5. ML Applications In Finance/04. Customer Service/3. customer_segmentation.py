"""
Python script for customer segmentation using K-Means clustering.
This script identifies distinct customer groups based on behavioral and demographic features,
enabling tailored product offerings.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Sample customer data: [Age, Income, Spending Score]
data = {
    'age': [25, 45, 30, 40, 35, 22, 48, 50, 23, 41],
    'income': [30000, 80000, 45000, 75000, 50000, 32000, 85000, 90000, 31000, 70000],
    'spending_score': [60, 40, 70, 45, 65, 80, 35, 30, 85, 50]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print("Sample Customer Data:")
print(df)

# Feature selection
features = ['age', 'income', 'spending_score']
X = df[features]

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding the optimal number of clusters using the Elbow Method
inertia = []
range_n_clusters = range(1, 11)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()

# Based on the elbow curve, let's choose 3 clusters
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels to the original dataset
df['cluster'] = kmeans.labels_

# Evaluate the clustering using the silhouette score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score for {optimal_clusters} clusters: {silhouette_avg:.2f}')

# Visualizing the clusters
plt.figure(figsize=(10, 6))

# Scatter plot for age vs. spending score
plt.subplot(1, 2, 1)
for cluster in range(optimal_clusters):
    clustered_data = df[df['cluster'] == cluster]
    plt.scatter(clustered_data['age'], clustered_data['spending_score'], label=f'Cluster {cluster}')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Age vs Spending Score')
plt.legend()

# Scatter plot for income vs. spending score
plt.subplot(1, 2, 2)
for cluster in range(optimal_clusters):
    clustered_data = df[df['cluster'] == cluster]
    plt.scatter(clustered_data['income'], clustered_data['spending_score'], label=f'Cluster {cluster}')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Income vs Spending Score')
plt.legend()

plt.tight_layout()
plt.show()

# Display the final dataset with cluster assignments
print("\nCustomer Segments with Cluster Labels:")
print(df)

# Example usage: Filtering customers in a specific cluster for tailored marketing
target_cluster = 1
target_customers = df[df['cluster'] == target_cluster]
print(f"\nCustomers in Cluster {target_cluster}:")
print(target_customers)

# Save the results to a CSV file
df.to_csv('customer_segments.csv', index=False)
print("\nClustered data has been saved to 'customer_segments.csv'.")
