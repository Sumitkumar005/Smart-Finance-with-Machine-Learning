# Use satellite images pre and post-natural disasters to assess the impact on infrastructure, agriculture, and local economies,
# which can significantly affect market conditions.
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Google Earth Engine API Key (replace with your own key)
API_KEY = 'your-google-earth-engine-api-key'

# Define the URL for fetching satellite images (You can use different services like NASA, Google Earth Engine, etc.)
SATELLITE_API_URL = 'https://earthengine.googleapis.com/v1alpha/satellite_images'

# Function to fetch satellite image based on a location and disaster event (coordinates)
def fetch_satellite_image(lat, lon, date_start, date_end):
    """
    Fetch satellite images from Google Earth Engine or other APIs based on disaster area.
    """
    # Construct the API request URL
    params = {
        'lat': lat,
        'lon': lon,
        'date_start': date_start,
        'date_end': date_end,
        'api_key': API_KEY
    }

    # Send the request to the satellite API
    response = requests.get(SATELLITE_API_URL, params=params)

    if response.status_code == 200:
        print("Satellite image fetched successfully.")
        return Image.open(BytesIO(response.content))  # Return the image as a PIL object
    else:
        print(f"Failed to fetch image. Status code: {response.status_code}")
        return None

# Function to analyze changes in the images (pre vs post disaster)
def analyze_impact(pre_image, post_image):
    """
    Analyzes the impact of a natural disaster on infrastructure, agriculture, etc.
    (e.g., by comparing pre and post-disaster satellite images).
    """
    # Convert images to numpy arrays for processing
    pre_array = np.array(pre_image)
    post_array = np.array(post_image)

    # Compute the difference between pre and post-disaster images (simple approach)
    diff = np.abs(post_array - pre_array)

    # Calculate the percentage change in the affected area (this can be refined further)
    diff_percentage = np.mean(diff) / 255  # Normalizing the diff for visualization

    print(f"Change Percentage: {diff_percentage * 100:.2f}%")

    # Visualize the difference
    plt.figure(figsize=(10, 5))

    # Display Pre and Post images side by side
    plt.subplot(1, 2, 1)
    plt.imshow(pre_array)
    plt.title("Pre-Disaster Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(post_array)
    plt.title("Post-Disaster Image")
    plt.axis('off')

    plt.show()

    # Display the change image (difference)
    plt.imshow(diff)
    plt.title("Change (Pre vs Post)")
    plt.axis('off')
    plt.show()

    return diff_percentage

# Example usage
if __name__ == '__main__':
    # Define coordinates of the area affected by the natural disaster (e.g., Hurricane in a region)
    latitude = 12.9716   # Example latitude
    longitude = 77.5946  # Example longitude (adjust for the affected area)
    
    # Define start and end dates for the satellite images (adjust based on disaster dates)
    date_start = '2023-01-01'
    date_end = '2023-01-10'

    # Fetch the pre-disaster image
    pre_disaster_image = fetch_satellite_image(latitude, longitude, date_start, date_end)

    # Fetch the post-disaster image (after the event)
    post_disaster_image = fetch_satellite_image(latitude, longitude, '2023-01-15', '2023-01-20')

    if pre_disaster_image and post_disaster_image:
        # Analyze the impact by comparing the pre and post images
        impact = analyze_impact(pre_disaster_image, post_disaster_image)
        print(f"Disaster Impact Percentage: {impact * 100:.2f}%")
    else:
        print("Failed to fetch necessary satellite images.")
