# Investment analysis using machine learning

import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load the dataset (e.g., investment data)
# Ensure you replace 'investment_data.csv' with the actual file path to your dataset
data = pd.read_csv('investment_data.csv')

# EDA: Visualizing the distribution of the target variable (returns))
sns.histplot(data['returns'], kde=True)
plt.title('Returns Distribution')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()

# Check for correlation between features and target (returns)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Data Preprocessing

# Handle missing values (if any)
# Here we drop rows with missing values, but you can also impute them
data = data.dropna()  # Alternatively, use data.fillna() for imputation

# If 'sector' or other columns are categorical, apply one-hot encoding
data = pd.get_dummies(data, columns=['sector'])

# Feature Engineering - selecting relevant features
features = ['investment_amount', 'market_trend', 'volatility', 'sector', 'historical_performance']
X = data[features]  # Features used for predicting returns
y = data['returns']   # Target variable (investment returns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Linear Regression)
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the returns on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse}')

# R-squared (R²) score
r2 = r2_score(y_test, y_pred)
print(f'R-squared (R²): {r2}')

# Optionally, perform cross-validation to check the stability of the model
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Cross-validation MSE scores: {cv_scores}')

# Save the trained model for future use
joblib.dump(model, 'investment_analysis_model.pkl')

# Example of loading the model and making predictions with new data
# Uncomment below if you want to test it

# loaded_model = joblib.load('investment_analysis_model.pkl')
# new_data = pd.DataFrame({'investment_amount': [10000], 'market_trend': [0.5], 'volatility': [0.2], 'sector': ['Tech'], 'historical_performance': [0.1]})
# new_data = pd.get_dummies(new_data, columns=['sector'])
# predicted_return = loaded_model.predict(new_data)
# print(f'Predicted return for new data: {predicted_return}')
