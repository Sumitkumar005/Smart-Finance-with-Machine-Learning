# Monitor shipping lanes and ports to gauge activity levels, which could be indicative of economic health 
# or trade flow between countries.
pip install requests pandas matplotlib
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Your API key (replace 'YOUR_API_KEY' with your actual key from MarineTraffic or another API provider)
API_KEY = 'YOUR_API_KEY'

# Define the base URL for MarineTraffic API
BASE_URL = 'https://api.marinetraffic.com/api/v1/'

# Function to get vessels at a port or within a specific geographical area
def get_vessel_data(lat_min, lat_max, lon_min, lon_max):
    endpoint = f"VesselPositions.json?lat_min={lat_min}&lat_max={lat_max}&lon_min={lon_min}&lon_max={lon_max}&apikey={API_KEY}"
    response = requests.get(BASE_URL + endpoint)
    return response.json()

# Example: Define the bounding box for a region (e.g., Near the coast of the US, North America)
lat_min = 24.396308
lat_max = 49.384358
lon_min = -125.0
lon_max = -66.93457

# Fetch vessel data in the region
data = get_vessel_data(lat_min, lat_max, lon_min, lon_max)

# Process the data (example: extracting relevant info about vessels)
vessel_data = []
for vessel in data['data']:
    vessel_info = {
        'ship_id': vessel['MMSI'],
        'ship_name': vessel['SHIPNAME'],
        'lat': vessel['LAT'],
        'lon': vessel['LON'],
        'status': vessel['STATUS']
    }
    vessel_data.append(vessel_info)

# Convert the data into a Pandas DataFrame for easy manipulation
df = pd.DataFrame(vessel_data)

# Display basic statistics and the first few records
print("Number of vessels detected:", len(df))
print("Sample data:", df.head())

# Plot vessel locations on a map (simple scatter plot)
plt.figure(figsize=(10, 8))
plt.scatter(df['lon'], df['lat'], c='blue', marker='o', alpha=0.5)
plt.title('Vessel Positions in the Selected Region')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Now let's check activity levels at a specific port by getting data on ships visiting a port
def get_ports_data():
    # Define the endpoint to get ports information
    endpoint = f"Ports.json?apikey={API_KEY}"
    response = requests.get(BASE_URL + endpoint)
    return response.json()

# Fetch the port data (list of major ports)
ports_data = get_ports_data()

# Extract port information and put it into a DataFrame
ports = []
for port in ports_data['data']:
    port_info = {
        'port_id': port['PORTID'],
        'port_name': port['PORTNAME'],
        'country': port['COUNTRY'],
        'latitude': port['LAT'],
        'longitude': port['LON']
    }
    ports.append(port_info)

# Convert ports data into a DataFrame
ports_df = pd.DataFrame(ports)

# Display the first few ports
print("List of major ports:", ports_df.head())

# Plot the locations of ports on a map
plt.figure(figsize=(10, 8))
plt.scatter(ports_df['longitude'], ports_df['latitude'], c='red', marker='x', alpha=0.7)
plt.title('Major Ports Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Function to analyze trade flow based on vessel activity
def analyze_trade_flow(vessel_data):
    trade_flow_by_country = vessel_data['status'].value_counts()
    print("Trade flow activity by vessel status:")
    print(trade_flow_by_country)

# Analyze the trade flow by checking vessel status in the selected region
analyze_trade_flow(df)

# Note: For real-time, dynamic updates, you might want to set this script to run at regular intervals or on demand.
