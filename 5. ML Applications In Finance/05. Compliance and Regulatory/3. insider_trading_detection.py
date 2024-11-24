# Identifying suspicious trading patterns using anomaly detection

"""
This script demonstrates the use of anomaly detection to identify suspicious trading patterns.
The dataset includes features such as trading volume, price changes, and other metrics that could 
indicate unusual behavior. We use Isolation Forest as the anomaly detection algorithm.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Sample dataset
data = {
    'trading_volume': [1000, 1500, 2000, 2500, 3000, 50000, 1200, 1800, 2400, 100000],
    'price_change': [1.5, 1.2, 1.8, 2.0, 2.5, 20.0, 1.1, 1.4, 2.1, 50.0],
    'number_of_trades': [150, 200, 250, 300, 400, 5000, 180, 220, 260, 7000]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df['transaction_id'] = range(1, len(df) + 1)  # Add transaction IDs for easy tracking

# Display initial dataset
print("Initial Dataset:")
print(df)

# Features for anomaly detection
features = ['trading_volume', 'price_change', 'number_of_trades']

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Initialize Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
df['anomaly'] = iso_forest.fit_predict(df_scaled)

# Add anomaly column: -1 indicates an anomaly, 1 indicates normal
df['anomaly'] = df['anomaly'].map({-1: 'Suspicious', 1: 'Normal'})

# Visualize anomalies
plt.figure(figsize=(10, 6))
for feature in features:
    plt.scatter(df['transaction_id'], df[feature], c=(df['anomaly'] == 'Suspicious'), cmap='coolwarm', label=feature)
    plt.xlabel('Transaction ID')
    plt.ylabel(feature)
    plt.title(f"{feature} with Anomaly Detection")
    plt.legend()
    plt.show()

# Display results
print("\nAnomaly Detection Results:")
print(df)

# Evaluation (For real-world data, you need labeled examples for this step)
# Assume we have true labels for evaluation (0 for normal, 1 for suspicious)
true_labels = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]  # Example true labels
predicted_labels = (df['anomaly'] == 'Suspicious').astype(int)  # Convert Suspicious to 1, Normal to 0

print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels))

# Function to identify suspicious transactions
def identify_suspicious_transactions(df):
    suspicious = df[df['anomaly'] == 'Suspicious']
    return suspicious

# Get suspicious transactions
suspicious_transactions = identify_suspicious_transactions(df)

print("\nSuspicious Transactions Identified:")
print(suspicious_transactions)

# Save the results to a CSV file
df.to_csv('trading_patterns_analysis.csv', index=False)
print("\nResults saved to 'trading_patterns_analysis.csv'.")
