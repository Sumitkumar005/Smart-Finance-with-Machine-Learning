# Develop a script that scans financial databases to identify companies that meet specific M&A criteria,
# such as EBITDA margins, revenue growth, or market cap..
pip install yfinance
# Import libraries
import yfinance as yf
import pandas as pd

# List of companies (tickers) to scan for M&A criteria
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']  # Add more tickers as needed

# Define the criteria for M&A
def meet_criteria(stock_info):
    """
    Function to check if the company meets specific M&A criteria:
    - EBITDA margin > 20%
    - Revenue growth > 10%
    - Market cap > 10 billion USD
    """
    try:
        # Get stock details from Yahoo Finance API
        ebitda_margin = stock_info.get('EBITDA', 0) / stock_info.get('revenue', 1)  # Example formula for EBITDA margin
        revenue_growth = stock_info.get('revenueGrowth', 0)  # Example formula for revenue growth
        market_cap = stock_info.get('marketCap', 0)

        # Check if the company meets the criteria
        if ebitda_margin > 0.20 and revenue_growth > 0.10 and market_cap > 10e9:
            return True
        else:
            return False
    except KeyError as e:
        print(f"Key error: {e}")
        return False

# Initialize an empty list to store companies that meet the criteria
companies_meeting_criteria = []

# Loop through the tickers and fetch the financial data
for ticker in tickers:
    try:
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        
        # Check if the company meets the M&A criteria
        if meet_criteria(stock_info):
            companies_meeting_criteria.append(ticker)
            print(f"{ticker} meets M&A criteria.")
        else:
            print(f"{ticker} does not meet M&A criteria.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")

# Output the results
print("\nCompanies that meet M&A criteria:")
for company in companies_meeting_criteria:
    print(company)
