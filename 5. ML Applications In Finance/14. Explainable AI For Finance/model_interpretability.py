'''
Machine Learning Model for Predicting Stock Prices with SHAP for Interpretability
'''

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import yfinance as yf  # For fetching stock data

# Fetch stock data using Yahoo Finance API
stock_symbol = 'AAPL'  # Example stock symbol (Apple Inc.)
start_date = '2020-01-01'
end_date = '2023-01-01'

print("Fetching stock data...")
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Feature engineering
data['Return'] = data['Adj Close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=5).std()
data['MA_10'] = data['Adj Close'].rolling(window=10).mean()
data['MA_50'] = data['Adj Close'].rolling(window=50).mean()
data['Target'] = data['Adj Close'].shift(-1)  # Predicting next day's closing price

# Drop rows with NaN values due to rolling calculations
data.dropna(inplace=True)

# Selecting features and target
features = ['Adj Close', 'Return', 'Volatility', 'MA_10', 'MA_50']
X = data[features]
y = data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training the model...")
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# SHAP for model interpretability
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize SHAP summary plot
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_test)

# Save the model for later use
import joblib
joblib.dump(model, 'stock_price_prediction_model.pkl')

'''
API Key Note:
No API key is required for yfinance, as it is a free library that does not require authentication for basic operations.
If a paid API (like Alpha Vantage or Quandl) is used, you'd need to include the API key and adapt the data fetching code accordingly.
'''

# Flask API for predictions
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data, index=[0])  # Convert JSON to DataFrame
        
        # Predict stock price
        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

'''
Explanations:

1. **Data Fetching**: Stock data is retrieved using yfinance. You can replace this with other APIs if required.
2. **Feature Engineering**: Features like returns, volatility, and moving averages are computed.
3. **Model Training**: A RandomForestRegressor is trained to predict the next day's closing price.
4. **SHAP Integration**: SHAP values explain the contribution of each feature to the model's predictions.
5. **Flask API**: A simple Flask API endpoint allows users to make predictions using POST requests.

Replace the stock symbol and dates with your desired values. Ensure all libraries are installed (`pip install pandas numpy scikit-learn shap flask matplotlib yfinance`).
'''
