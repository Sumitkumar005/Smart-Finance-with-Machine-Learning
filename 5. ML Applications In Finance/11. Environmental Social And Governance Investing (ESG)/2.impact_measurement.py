# Environmental and Social Impact Prediction Model for a Company

'''
Python script to calculate the environmental and social impact of a company based on publicly available data
such as emissions reports, community involvement, and other corporate social responsibility (CSR) data.
We'll use machine learning (Random Forest Regressor) to predict the company's environmental and social score.
'''

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample Data: [emissions, community_involvement, csr_spending, number_of_employees]
# Target variable: 'impact_score' represents the combined environmental and social impact score of the company
data = {
    'emissions': [1000, 2500, 1500, 3000, 500, 700, 1200, 2200, 1800, 800],  # Measured in metric tons of CO2
    'community_involvement': [50, 40, 60, 30, 90, 80, 70, 55, 45, 65],  # Number of community programs supported
    'csr_spending': [100000, 150000, 120000, 160000, 80000, 95000, 110000, 105000, 130000, 115000],  # In USD
    'number_of_employees': [5000, 10000, 7000, 15000, 3000, 4500, 6000, 12000, 9500, 7000],  # Total number of employees
    'impact_score': [70, 55, 65, 50, 90, 85, 75, 60, 68, 80]  # Environmental and social impact score
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Feature Engineering: Selecting relevant features
X = df[['emissions', 'community_involvement', 'csr_spending', 'number_of_employees']]  # Features
y = df['impact_score']  # Target variable (impact score)

# Plotting data points
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['emissions'], df['impact_score'], c=df['impact_score'], cmap='viridis')
plt.xlabel('Emissions (in metric tons of CO2)')
plt.ylabel('Impact Score')
plt.title('Emissions vs Impact Score')

plt.subplot(1, 2, 2)
plt.scatter(df['community_involvement'], df['impact_score'], c=df['impact_score'], cmap='viridis')
plt.xlabel('Community Involvement (Number of Programs)')
plt.ylabel('Impact Score')
plt.title('Community Involvement vs Impact Score')

plt.tight_layout()
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the impact score on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to predict environmental and social impact score for a company based on given features
def predict_impact(emissions, community_involvement, csr_spending, number_of_employees):
    prediction = model.predict([[emissions, community_involvement, csr_spending, number_of_employees]])[0]
    return prediction

# Example usage of the prediction function
print("Predicted Impact Score for Company A:", predict_impact(1500, 60, 120000, 8000))  # Example prediction
print("Predicted Impact Score for Company B:", predict_impact(2000, 50, 95000, 7000))  # Example prediction

# Save the trained model for future use (Optional)
import joblib
joblib.dump(model, 'company_impact_model.pkl')

'''
Here's what each part of the code does:

1. **Data Preparation**: We create a DataFrame with sample data that includes company features such as emissions, community involvement, CSR spending, and the number of employees. The target variable is the 'impact_score', which reflects the environmental and social impact of the company.

2. **Feature Engineering**: We select the relevant features (`emissions`, `community_involvement`, `csr_spending`, `number_of_employees`) to predict the target variable, `impact_score`.

3. **Model Training**: We train a Random Forest Regressor model using scikit-learn to predict the impact score based on the features. We split the data into a training set and a test set to evaluate the model.

4. **Model Evaluation**: We use the Mean Squared Error (MSE) metric to evaluate the model's performance on the test data.

5. **Prediction**: The `predict_impact()` function is defined to predict the impact score for new companies based on their feature values (such as emissions, community involvement, etc.).

6. **Visualization**: We plot the relationships between the emissions, community involvement, and impact score using scatter plots to visualize the data and gain insights.

7. **Model Persistence**: Finally, the trained model is saved using `joblib` to make predictions in the future without retraining the model.

In a real-world scenario, you would use real-time data, integrate more features, and improve the model by using more advanced techniques such as feature scaling, hyperparameter tuning, and cross-validation.
'''
