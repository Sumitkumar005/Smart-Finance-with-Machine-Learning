# Build a tool to compare past M&A deals based on various metrics like deal size, industry, and financial ratios,
# to gauge the attractiveness of a potential new deal.
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Alpha Vantage API Key (use your own API key here)
API_KEY = 'YOUR_ALPHA_VANTAGE_API_KEY'

# Function to get stock data from Alpha Vantage
def get_stock_data(symbol):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print(f"Error fetching data for {symbol}")
        return None
    
    # Extracting the most recent closing price
    latest_data = list(data["Time Series (Daily)"].values())[0]
    close_price = float(latest_data["4. close"])

    return close_price

# Function to fetch company metrics
def get_company_metrics(symbol):
    url = f'https://www.alphavantage.co/query'
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol,
        'apikey': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "Error Message" in data:
        print(f"Error fetching overview for {symbol}")
        return None
    
    # Extracting relevant financial data
    market_cap = float(data['MarketCapitalization']) if data.get('MarketCapitalization') else 0
    pe_ratio = float(data['PERatio']) if data.get('PERatio') else 0
    return market_cap, pe_ratio

# Function to calculate the M&A attractiveness score
def calculate_attractiveness(deal_size, industry, market_cap, pe_ratio):
    # Basic scoring algorithm (you can modify this based on the specific industry and ratios)
    score = 0
    # Larger deals may be considered more attractive
    score += deal_size / 1000000000  # Scale down deal size (in billions)
    
    # P/E ratio - generally, lower P/E is more attractive for value-based investing
    if pe_ratio > 0:
        score += 1 / pe_ratio

    # Market cap - larger market cap companies may be less risky
    if market_cap > 10000000000:  # 10 Billion cap
        score += 0.5

    # Add industry weighting (based on personal analysis or benchmarks)
    industry_weights = {
        'Technology': 1.2,
        'Healthcare': 1.1,
        'Finance': 1.0,
        'Energy': 0.8,
        'Consumer': 1.0,
    }

    score *= industry_weights.get(industry, 1)

    return score

# Function to plot comparison between deals
def plot_comparison(deals_df):
    deals_df.plot(kind='bar', x='Deal Name', y='Attractiveness Score', legend=False, color='skyblue')
    plt.title("M&A Deal Attractiveness Comparison")
    plt.xlabel("Deal Name")
    plt.ylabel("Attractiveness Score")
    plt.tight_layout()
    plt.show()

# Example list of past M&A deals with hypothetical data
deals = [
    {"deal_name": "Company A acquires Company B", "deal_size": 2500000000, "industry": "Technology", "symbol": "AAPL"},
    {"deal_name": "Company C merges with Company D", "deal_size": 1500000000, "industry": "Healthcare", "symbol": "JNJ"},
    {"deal_name": "Company E acquires Company F", "deal_size": 5000000000, "industry": "Finance", "symbol": "JPM"},
    {"deal_name": "Company G merges with Company H", "deal_size": 3000000000, "industry": "Energy", "symbol": "XOM"},
]

# Create a DataFrame to store deal information
deal_data = []

# Process each deal
for deal in deals:
    symbol = deal["symbol"]
    market_cap, pe_ratio = get_company_metrics(symbol)
    if market_cap and pe_ratio:
        attractiveness_score = calculate_attractiveness(
            deal["deal_size"], deal["industry"], market_cap, pe_ratio
        )
        deal_data.append({
            "Deal Name": deal["deal_name"],
            "Deal Size (USD)": deal["deal_size"],
            "Industry": deal["industry"],
            "Attractiveness Score": attractiveness_score
        })

# Convert to DataFrame for easy manipulation and visualization
deals_df = pd.DataFrame(deal_data)

# Print the DataFrame
print(deals_df)

# Plot the comparison chart
plot_comparison(deals_df)
