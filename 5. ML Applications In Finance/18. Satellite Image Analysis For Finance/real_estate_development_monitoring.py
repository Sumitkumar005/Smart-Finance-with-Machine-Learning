# Use time-lapsed satellite images to monitor construction and development activity in specific geographical regions.
# This can provide insights into real estate markets and housing prices.
pip install earthengine-api geemap
earthengine authenticate
# Use time-lapsed satellite images to monitor construction and development activity in specific geographical regions.
# This can provide insights into real estate markets and housing prices.

# Import necessary libraries
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np

# Initialize the Earth Engine API
ee.Initialize()

# Define the region of interest (ROI)
# Example: Coordinates for a location (change to the area you're interested in)
roi = ee.Geometry.Polygon(
        [[[78.9629, 20.5937], [78.9629, 15.0], [85.0, 15.0], [85.0, 20.5937]]])  # Define bounding box

# Define the time period for the analysis
start_date = '2015-01-01'
end_date = '2020-01-01'

# Fetch satellite data (Landsat 8 imagery for this example)
# Using Landsat 8 surface reflectance data
dataset = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
    .filterBounds(roi) \
    .filterDate(ee.Date(start_date), ee.Date(end_date)) \
    .sort('system:time_start')

# Visualize the first image in the collection
first_image = dataset.first()

# Define visualization parameters for the satellite image
vis_params = {
    'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue bands for natural color
    'min': 0,
    'max': 3000,
    'gamma': 1.4
}

# Center the map on the region of interest
Map = geemap.Map()
Map.centerObject(roi, 10)

# Add the first satellite image to the map
Map.addLayer(first_image, vis_params, 'First Image')

# Plotting the time-lapsed imagery (using a composite)
def create_composite(image_collection):
    # Compute median composite for the time period
    composite = image_collection.median()
    return composite

# Create a time-lapse composite of the satellite images
composite_image = create_composite(dataset)

# Add the composite image to the map
Map.addLayer(composite_image, vis_params, 'Time-lapsed Image Composite')

# Display the map
Map

# Convert the image to a NumPy array for analysis
# We will use the red band to track construction activity (change this logic depending on your specific requirements)
red_band = composite_image.select('B4')

# Convert image to an array for further processing
array = red_band.toArray().getInfo()
print(array)

# Plot the time-lapsed satellite images to visualize changes over time
# Create a simple visualization with matplotlib (you can customize this part based on your data and needs)
plt.figure(figsize=(12, 6))
plt.imshow(array, cmap='gray')
plt.title("Time-lapsed Satellite Image Composite")
plt.colorbar(label='Red Band Intensity')
plt.show()

# Analysis of changes over time: compare images from different years
# You can use a similar approach to track changes by comparing specific time points
def analyze_change(image1, image2):
    # Subtract the two images to highlight changes (construction activity)
    change_image = image2.subtract(image1)
    return change_image

# Example: Compare 2015 and 2019 images (change to your desired years)
image_2015 = dataset.filterDate('2015-01-01', '2015-12-31').mean()
image_2019 = dataset.filterDate('2019-01-01', '2019-12-31').mean()

# Analyze changes
change_image = analyze_change(image_2015, image_2019)

# Add the change image to the map to visualize the construction activity
Map.addLayer(change_image, {'min': -3000, 'max': 3000, 'palette': ['blue', 'white', 'red']}, 'Construction Activity')

# Display the map with changes over time
Map

# Optionally, export the change detection image to Google Drive or other platforms
# This can be useful for further processing and analysis
export_task = ee.batch.Export.image.toDrive(
    image=change_image,
    description='ConstructionActivityExport',
    folder='RealEstateAnalysis',
    region=roi,
    scale=30
)

# Start the export task
export_task.start()
