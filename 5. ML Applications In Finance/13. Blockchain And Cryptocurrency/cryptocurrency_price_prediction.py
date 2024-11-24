"""
Script for predicting cryptocurrency prices using LSTM.
We use the CoinGecko API for fetching historical cryptocurrency price data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
import datetime

# Function to fetch historical cryptocurrency prices using the CoinGecko API
def fetch_crypto_data(crypto_id="bitcoin", currency="usd", days="365"):
    """
    Fetch historical cryptocurrency data for the given parameters.
    Args:
        crypto_id (str): Cryptocurrency identifier (e.g., "bitcoin", "ethereum").
        currency (str): The currency for price data (e.g., "usd").
        days (str): Number of days to fetch historical data ("max" for all data).
    Returns:
        pandas.DataFrame: DataFrame with 'timestamp' and 'price' columns.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {"vs_currency": currency, "days": days, "interval": "daily"}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']  # Extract price data
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
        return df
    else:
        raise Exception(f"Failed to fetch data. Error code: {response.status_code}")

# Fetch data
crypto_id = "bitcoin"  # Change to any cryptocurrency ID supported by CoinGecko
currency = "usd"
days = "365"  # Fetch 1 year of data
data = fetch_crypto_data(crypto_id, currency, days)
data.set_index('timestamp', inplace=True)
print(f"Data fetched for {crypto_id}:")
print(data.head())

# Plot the historical price data
plt.figure(figsize=(12, 6))
plt.plot(data['price'], label=f'{crypto_id.capitalize()} Price')
plt.title(f'{crypto_id.capitalize()} Historical Price Data')
plt.xlabel('Date')
plt.ylabel(f'Price ({currency.upper()})')
plt.legend()
plt.show()

# Preprocessing data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['price']].values)

# Create training and testing datasets
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # Use the past 60 days to predict the next day's price
X, y = create_dataset(data_scaled, time_step)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input data to [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
epochs = 50
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict and inverse transform the scaled data
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Prices', color='blue')
plt.plot(predicted, label='Predicted Prices', color='red')
plt.title(f'{crypto_id.capitalize()} Price Prediction')
plt.xlabel('Days')
plt.ylabel(f'Price ({currency.upper()})')
plt.legend()
plt.show()

# Save the model
model.save('crypto_price_prediction_lstm.h5')
print("Model saved as 'crypto_price_prediction_lstm.h5'")
