# Analyze satellite images to identify new or depleted natural resources like forests, water bodies, or mineral deposits,
# which can be crucial information for investing in relevant sectors.
# Importing necessary libraries
import ee
import folium
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from folium import plugins

# Authenticate and initialize Earth Engine API
ee.Authenticate()
ee.Initialize()

# Define a region of interest (ROI) (For example, selecting an area of interest over a forest region)
roi = ee.Geometry.Rectangle([-122.6, 37.0, -122.3, 37.4])

# Fetch satellite image data (using Sentinel-2 for vegetation analysis)
# We use Sentinel-2 for forest analysis, water bodies, or mineral deposits as it's high-resolution
image = ee.ImageCollection("COPERNICUS/S2") \
    .filterBounds(roi) \
    .filterDate('2023-01-01', '2023-12-31') \
    .mean() \
    .clip(roi)

# Function to select relevant bands for analysis (e.g., vegetation indices for forest detection)
# Using NDVI (Normalized Difference Vegetation Index) for forest detection
def compute_ndvi(image):
    red = image.select('B4')  # Red band
    nir = image.select('B8')  # Near Infrared band
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return image.addBands(ndvi)

# Compute NDVI for the image
ndvi_image = compute_ndvi(image)

# Visualization of the NDVI image
# Setting visualization parameters for NDVI
ndvi_params = {
    'min': -0.5,
    'max': 0.8,
    'palette': ['blue', 'white', 'green']
}

# Display NDVI using folium map
Map = folium.Map(location=[37.3, -122.5], zoom_start=10)
Map.add_ee_layer(ndvi_image.select('NDVI'), ndvi_params, 'NDVI')

# Function to add Earth Engine layer to folium map
def add_ee_layer(self, ee_object, vis_params, name):
    map_id_dict = ee_object.getMapId(vis_params)
    folium.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Add the layer to the map
folium.Map.add_ee_layer = add_ee_layer
Map

# KMeans clustering to classify the image into different resource categories
# Resampling the image to a smaller scale for KMeans clustering
image_resized = ndvi_image.select('NDVI').reduceResolution(
    reducer=ee.Reducer.mean(),
    maxPixels=1024
)

# Sample pixels for clustering
sample = image_resized.sample(region=roi, scale=30, numPixels=5000)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3)  # Assume 3 clusters: forest, water bodies, and barren land
samples_array = np.array(sample.getInfo()['features'])
features = [x['properties']['NDVI'] for x in samples_array]

# Fit the KMeans model
kmeans.fit(features)

# Classify the image based on KMeans
clustered_image = image_resized.updateMask(image_resized.gt(0)) \
    .addBands(ee.Image(kmeans.predict(features)).rename('Cluster'))

# Display the classified image
clustered_params = {
    'min': 0,
    'max': 3,
    'palette': ['blue', 'green', 'yellow']
}
Map.add_ee_layer(clustered_image.select('Cluster'), clustered_params, 'Clustered Resources')

# Display the map
Map

# Save and download the output if needed
export_task = ee.batch.Export.image.toDrive(
    image=clustered_image,
    description='Clustered_Image_Export',
    fileFormat='GeoTIFF',
    region=roi,
    scale=30
)
export_task.start()

