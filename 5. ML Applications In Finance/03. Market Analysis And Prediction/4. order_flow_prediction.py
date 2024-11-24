# Predicting the future order flow (buy/sell) based on existing order books

'''
Python script to predict the future order flow (buy/sell) based on the existing order book. We use historical market data
such as the price and volume of orders to train a model that can predict the next order as a buy or sell.
'''

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Sample data: [price, volume, order_type (buy/sell)]
# The target variable is 'future_order_flow', where 1 means buy and 0 means sell
data = {
    'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    'volume': [200, 150, 300, 250, 400, 350, 500, 450, 600, 550],
    'order_type': ['buy', 'sell', 'buy', 'sell', 'buy', 'buy', 'sell', 'sell', 'buy', 'sell'],
    'future_order_flow': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]  # 1 = buy, 0 = sell
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Convert the categorical 'order_type' column into a numerical value (buy = 1, sell = 0)
df['order_type'] = df['order_type'].apply(lambda x: 1 if x == 'buy' else 0)

# Feature selection (X) and target variable (y)
X = df[['price', 'volume', 'order_type']]  # Features (Price, Volume, Order Type)
y = df['future_order_flow']  # Target variable (future order flow)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')

# Function to predict the future order flow (buy/sell)
def predict_future_order_flow(price, volume, order_type):
    prediction = clf.predict([[price, volume, order_type]])[0]
    
    if prediction == 1:
        return "The predicted future order flow is a BUY."
    else:
        return "The predicted future order flow is a SELL."

# Example usage of the prediction function
print(predict_future_order_flow(106, 400, 1))  # Example with a BUY order type
print(predict_future_order_flow(108, 600, 0))  # Example with a SELL order type

'''
Here's what each part of the code does:

1. **Import Libraries:** The necessary Python libraries for data manipulation and machine learning are imported.

2. **Sample Data:** A DataFrame is created from a dictionary where each entry corresponds to an order with attributes such as price, volume, and order type (buy/sell). The target variable is the future order flow, where 1 indicates a future buy and 0 indicates a future sell.

3. **Data Preprocessing:** The categorical column 'order_type' is converted to numerical values, where 'buy' becomes 1 and 'sell' becomes 0.

4. **Data Splitting:** The dataset is split into training and testing sets using `train_test_split()`.

5. **Model Training:** A Random Forest Classifier is initialized and trained on the training set.

6. **Evaluation:** The model is evaluated using the test set, and accuracy along with the confusion matrix are printed.

7. **Prediction Function:** The `predict_future_order_flow()` function takes a price, volume, and order type as inputs and returns whether the predicted future order flow is a buy or sell.

8. **Example Usage:** The `predict_future_order_flow()` function is called with sample data to demonstrate how to use it.

This is a basic example for educational purposes, and in real-world scenarios, you would work with much larger datasets, perform feature engineering, and tune the model for better accuracy.
'''

