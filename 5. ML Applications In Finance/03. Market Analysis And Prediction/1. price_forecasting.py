# Using time-series models to predict future stock prices or market trends.

'''
This script demonstrates how to use ARIMA (AutoRegressive Integrated Moving Average)
model for predicting stock prices based on historical data.
'''

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the stock price data (for example, Apple stock data)
# You can replace this with any time-series data, such as stock prices
data = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')

# Visualize the stock data
plt.figure(figsize=(10, 6))
plt.plot(data['Close'])
plt.title('Stock Price History')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# Prepare the data (using the closing price for forecasting)
stock_prices = data['Close']

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(stock_prices) * 0.8)
train, test = stock_prices[:train_size], stock_prices[train_size:]

# Fit the ARIMA model
# ARIMA(p,d,q) where p = number of lag observations, d = number of times differenced, q = size of moving average window
model = ARIMA(train, order=(5, 1, 0))  # You can adjust these parameters for better performance
model_fit = model.fit()

# Make predictions on the test set
predictions = model_fit.forecast(steps=len(test))

# Evaluate the model's performance using Mean Squared Error
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize the predictions vs actual stock prices
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, color='blue', label='Actual Stock Price')
plt.plot(test.index, predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction Using ARIMA')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Optionally, save the trained model for future use
import joblib
joblib.dump(model_fit, 'stock_price_model.pkl')

'''
Here's a breakdown of what each part of the code does:

1. Import Libraries: We import the necessary libraries for data handling, plotting, and machine learning.
2. Load Data: The stock price data is loaded using pandas. You can replace it with your dataset.
3. Data Preprocessing: We extract the 'Close' price from the dataset as our target variable.
4. Train-Test Split: The data is split into training and testing sets (80% for training, 20% for testing).
5. ARIMA Model: An ARIMA model is initialized and fitted to the training data. You can modify the parameters (p, d, q) to fine-tune the model.
6. Predictions: We forecast the future stock prices using the trained model and evaluate the model's performance using Mean Squared Error (MSE).
7. Visualization: The actual vs predicted stock prices are plotted to visually assess the model's performance.
8. Save Model: The trained model is saved for future use using the `joblib` library.
'''

# Example usage of the model (predicting next 5 days)
future_predictions = model_fit.forecast(steps=5)
print("Future stock price predictions:", future_predictions)
