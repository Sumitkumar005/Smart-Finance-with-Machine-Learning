"""
Optimization of supply chain processes and financing.

This script demonstrates the use of machine learning for optimizing supply chain processes,
including delivery time prediction and cost optimization. It also considers financial aspects
like cash flow management for supply chain financing.
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
# Simulated dataset for supply chain: ['distance', 'weight', 'shipping_method', 'warehouse_efficiency', 'delivery_time', 'cost']
data = {
    'distance': [10, 20, 50, 70, 100, 200, 300, 400, 500, 600],
    'weight': [5, 10, 20, 15, 30, 25, 50, 45, 60, 70],
    'shipping_method': [1, 2, 2, 1, 3, 3, 2, 1, 3, 1],  # Encoded shipping methods
    'warehouse_efficiency': [0.9, 0.8, 0.85, 0.7, 0.95, 0.6, 0.7, 0.8, 0.75, 0.65],
    'delivery_time': [1, 2, 5, 6, 7, 10, 15, 20, 25, 30],  # In days
    'cost': [50, 100, 200, 150, 300, 250, 500, 450, 600, 700]  # In dollars
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Features and target variables
X = df[['distance', 'weight', 'shipping_method', 'warehouse_efficiency']]
y_time = df['delivery_time']  # Predict delivery time
y_cost = df['cost']           # Predict cost

# Split the dataset for delivery time prediction
X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X, y_time, test_size=0.2, random_state=42)

# Model for delivery time prediction
time_model = RandomForestRegressor(n_estimators=100, random_state=42)
time_model.fit(X_train_time, y_train_time)

# Predict delivery times
y_pred_time = time_model.predict(X_test_time)

# Evaluate delivery time model
time_mse = mean_squared_error(y_test_time, y_pred_time)
time_r2 = r2_score(y_test_time, y_pred_time)
print(f"Delivery Time Prediction - MSE: {time_mse}, R^2: {time_r2}")

# Split the dataset for cost prediction
X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(X, y_cost, test_size=0.2, random_state=42)

# Model for cost prediction
cost_model = LinearRegression()
cost_model.fit(X_train_cost, y_train_cost)

# Predict costs
y_pred_cost = cost_model.predict(X_test_cost)

# Evaluate cost model
cost_mse = mean_squared_error(y_test_cost, y_pred_cost)
cost_r2 = r2_score(y_test_cost, y_pred_cost)
print(f"Cost Prediction - MSE: {cost_mse}, R^2: {cost_r2}")

# Visualizing predictions
plt.figure(figsize=(12, 6))

# Delivery time predictions
plt.subplot(1, 2, 1)
plt.plot(y_test_time.values, label="Actual Delivery Time", marker='o')
plt.plot(y_pred_time, label="Predicted Delivery Time", marker='x')
plt.xlabel("Sample")
plt.ylabel("Delivery Time (days)")
plt.title("Delivery Time Prediction")
plt.legend()

# Cost predictions
plt.subplot(1, 2, 2)
plt.plot(y_test_cost.values, label="Actual Cost", marker='o')
plt.plot(y_pred_cost, label="Predicted Cost", marker='x')
plt.xlabel("Sample")
plt.ylabel("Cost (USD)")
plt.title("Cost Prediction")
plt.legend()

plt.tight_layout()
plt.show()

# Function for supply chain optimization
def optimize_supply_chain(distance, weight, shipping_method, warehouse_efficiency):
    """
    Function to optimize supply chain processes by predicting delivery time and cost.
    """
    features = np.array([[distance, weight, shipping_method, warehouse_efficiency]])
    predicted_time = time_model.predict(features)[0]
    predicted_cost = cost_model.predict(features)[0]
    
    optimization_result = {
        'Predicted Delivery Time (days)': predicted_time,
        'Predicted Cost (USD)': predicted_cost
    }
    
    return optimization_result

# Example usage of the optimization function
result = optimize_supply_chain(distance=250, weight=30, shipping_method=2, warehouse_efficiency=0.8)
print("Optimization Result:", result)

# Financial consideration: Cash flow optimization
def cash_flow_optimization(costs, payment_cycle=30, interest_rate=0.05):
    """
    Function to calculate cash flow optimization for supply chain financing.
    """
    total_cost = sum(costs)
    financing_cost = total_cost * interest_rate * (payment_cycle / 365)
    
    return {
        'Total Cost': total_cost,
        'Financing Cost': financing_cost,
        'Optimized Cash Flow': total_cost + financing_cost
    }

# Example usage of cash flow optimization
cash_flow = cash_flow_optimization(costs=df['cost'].values)
print("Cash Flow Optimization:", cash_flow)

"""
This script covers:
1. Predicting delivery times using Random Forest Regressor.
2. Predicting costs using Linear Regression.
3. Visualizing the performance of the models.
4. Providing an optimization function for supply chain processes.
5. Adding financial considerations with a cash flow optimization function.

This end-to-end example can be extended or integrated into larger supply chain management systems.
"""
