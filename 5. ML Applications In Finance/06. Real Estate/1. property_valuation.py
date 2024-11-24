# Automated valuation models for real estate pricing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load the dataset (e.g., real estate data)
# Make sure to replace 'real_estate_data.csv' with the correct path to your dataset
data = pd.read_csv('real_estate_data.csv')

# EDA: Visualizing the distribution of the target variable (price)
sns.histplot(data['price'], kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Check for correlation between features and target (price)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Data Preprocessing

# Handle missing values (if any)
# Here we drop rows with missing values, but you can also impute them
data = data.dropna()  # Alternatively, use data.fillna() for imputation

# If 'location' is categorical, apply one-hot encoding to convert it to numerical data
data = pd.get_dummies(data, columns=['location'])

# Feature Engineering - selecting relevant features
features = ['location', 'size', 'age', 'number_of_bedrooms', 'distance_to_city_center']
X = data[features]  # Features used for predicting the price
y = data['price']   # Target variable (price of the property)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the property prices on the test set
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
joblib.dump(model, 'property_valuation_model.pkl')

# Example of loading the model and making predictions with new data
# Uncomment below if you want to test it

# loaded_model = joblib.load('property_valuation_model.pkl')
# new_data = pd.DataFrame({'location': ['location_A'], 'size': [2000], 'age': [10], 'number_of_bedrooms': [3], 'distance_to_city_center': [5]})
# new_data = pd.get_dummies(new_data, columns=['location'])
# predicted_price = loaded_model.predict(new_data)
# print(f'Predicted price for new data: {predicted_price}')

