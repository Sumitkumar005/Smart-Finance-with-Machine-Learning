# Analyze satellite images for signs of crop health and size. These metrics can provide insights into future commodity prices.
# Import libraries
import ee
import folium
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initialize the Earth Engine API (if you haven't already authenticated, it will ask you to authenticate)
ee.Initialize()

# Define the region of interest (ROI) (example coordinates: lat/lon for a specific location)
region = ee.Geometry.Polygon(
        [[[-85.0, 33.0],
          [-85.0, 32.5],
          [-84.5, 32.5],
          [-84.5, 33.0]]])

# Function to compute NDVI from satellite imagery
def calculate_ndvi(image):
    # Calculate NDVI from the RED and NIR bands
    ndvi = image.normalizedDifference(['B4', 'B5']).rename('NDVI')
    return ndvi

# Load satellite imagery (e.g., Sentinel 2 or Landsat) for a specific time range
# Example: Sentinel-2 data from 2022
image_collection = ee.ImageCollection('COPERNICUS/S2') \
                    .filterBounds(region) \
                    .filterDate('2022-01-01', '2022-12-31') \
                    .map(calculate_ndvi)

# Select the NDVI band
ndvi_image = image_collection.mean()  # Take the mean NDVI over the year

# Visualize NDVI on the map
ndvi_map = folium.Map(location=[32.75, -84.75], zoom_start=10)
ndvi_map.add_ee_layer(ndvi_image, {'min': -0.1, 'max': 0.8, 'palette': ['blue', 'white', 'green']}, 'NDVI')

# Function to add Earth Engine layers to folium
def add_ee_layer(self, ee_object, vis_params, name):
    map_id_dict = ee_object.getMapId(vis_params)
    folium.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr="Map data &copy; <a href='http://www.google.com/earth/'>Google Earth Engine</a>",
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Add the layer to the map
folium.Map.add_ee_layer = add_ee_layer
ndvi_map

# Extract NDVI data for training a machine learning model
# Convert the NDVI image to a pandas DataFrame for analysis
def image_to_dataframe(image):
    # Sample a region (you can define your region of interest here)
    sample = image.sample(region=region, scale=30, numPixels=500)
    df = pd.DataFrame(sample.getInfo()['features'])
    return df

# Convert NDVI image to dataframe
df = image_to_dataframe(ndvi_image)

# Feature Engineering: Use NDVI as the feature
X = df[['properties']['NDVI']]  # NDVI values as features
y = df['properties']['crop_size']  # Target variable (crop size, or a custom target you define)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict crop size on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Example prediction
def predict_crop_size(ndvi_value):
    predicted_size = model.predict([[ndvi_value]])[0]
    return predicted_size

# Example usage of prediction function
print(f"Predicted crop size for NDVI value 0.4: {predict_crop_size(0.4)}")

# Visualizing the NDVI against crop size
plt.scatter(X_test, y_test, color='blue', label='True values')
plt.scatter(X_test, y_pred, color='red', label='Predicted values')
plt.title('Crop Size Prediction')
plt.xlabel('NDVI')
plt.ylabel('Crop Size')
plt.legend()
plt.show()

# Optionally, save the trained model for future use
import joblib
joblib.dump(model, 'crop_size_prediction_model.pkl')

