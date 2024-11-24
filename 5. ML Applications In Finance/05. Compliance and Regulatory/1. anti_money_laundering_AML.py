"""
Python script for detecting potentially illegal actions through transaction monitoring.
This script uses a supervised learning approach to flag suspicious transactions based
on patterns in historical transaction data.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the dataset (example data format)
"""
Assume the dataset has the following columns:
    - transaction_id: Unique ID for the transaction
    - amount: Transaction amount
    - location: Categorical data for transaction location
    - time_of_day: Categorical data for time (e.g., morning, afternoon, night)
    - account_age: Age of the account in days
    - is_suspicious: Target variable (1 if suspicious, 0 otherwise)
"""
data = pd.read_csv('transaction_data.csv')

# Data preview
print("First few rows of the dataset:")
print(data.head())

# Data preprocessing
# Convert categorical columns to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['location', 'time_of_day'], drop_first=True)

# Separate features and target variable
X = data.drop(columns=['transaction_id', 'is_suspicious'])
y = data['is_suspicious']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Visualize the feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Transaction Monitoring Model')
plt.gca().invert_yaxis()
plt.show()

# Function to predict suspicious transactions
def predict_suspicious_transaction(transaction_data):
    """
    Predict whether a transaction is suspicious.
    transaction_data: dictionary containing transaction details
    """
    transaction_df = pd.DataFrame([transaction_data])
    transaction_df = pd.get_dummies(transaction_df, columns=['location', 'time_of_day'], drop_first=True)
    missing_cols = set(X.columns) - set(transaction_df.columns)
    for col in missing_cols:
        transaction_df[col] = 0
    transaction_df = transaction_df[X.columns]  # Ensure column order matches training data

    prediction = model.predict(transaction_df)[0]
    if prediction == 1:
        return "The transaction is suspicious."
    else:
        return "The transaction is not suspicious."

# Example usage of the prediction function
example_transaction = {
    'amount': 10000,
    'location': 'City_A',
    'time_of_day': 'night',
    'account_age': 365
}
print("\nPrediction for example transaction:")
print(predict_suspicious_transaction(example_transaction))

# Save the model for deployment
import joblib
joblib.dump(model, 'transaction_monitoring_model.pkl')

"""
Summary:
1. The script loads transaction data and preprocesses it for machine learning.
2. A Random Forest Classifier is trained to classify transactions as suspicious or not.
3. Evaluation metrics like accuracy and feature importance are provided.
4. A custom function allows predicting whether a new transaction is suspicious.
5. The trained model is saved for future use.
"""
