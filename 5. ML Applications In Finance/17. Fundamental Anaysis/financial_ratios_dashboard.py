# Develop a dashboard that displays crucial financial ratios, automatically calculated from a company's balance sheet,
# income statement, and cash flow statement.
pip install yfinance dash pandas plotly
import yfinance as yf
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Fetch financial data for the company using yfinance (as an example, using Apple - AAPL)
def fetch_financial_data(ticker):
    company = yf.Ticker(ticker)
    
    # Fetch balance sheet, income statement, and cash flow statement
    balance_sheet = company.balance_sheet
    income_statement = company.financials
    cash_flow = company.cashflow
    
    # Extract relevant data
    total_assets = balance_sheet.loc['Total Assets']
    total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest']
    equity = balance_sheet.loc['Total Stockholder Equity']
    
    net_income = income_statement.loc['Net Income']
    revenue = income_statement.loc['Total Revenue']
    
    operating_cash_flow = cash_flow.loc['Operating Cash Flow']
    
    return total_assets, total_liabilities, equity, net_income, revenue, operating_cash_flow

# Calculate financial ratios
def calculate_ratios(total_assets, total_liabilities, equity, net_income, revenue, operating_cash_flow):
    # Financial Ratios
    current_ratio = total_assets / total_liabilities
    quick_ratio = (total_assets - equity) / total_liabilities
    return_on_equity = net_income / equity
    profit_margin = net_income / revenue
    operating_cash_flow_ratio = operating_cash_flow / total_liabilities
    
    return current_ratio, quick_ratio, return_on_equity, profit_margin, operating_cash_flow_ratio

# Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout for the app
app.layout = html.Div([
    html.H1('Company Financial Ratios Dashboard'),
    html.Div([
        html.Label('Enter Company Ticker:'),
        dcc.Input(id='ticker-input', value='AAPL', type='text', debounce=True),
        html.Button('Submit', id='submit-button', n_clicks=0),
    ]),
    
    html.Div(id='financial-ratios', children=[
        html.H3('Financial Ratios'),
        html.P('Loading...'),
    ]),
    
    # Graph for visual representation of financial ratios
    dcc.Graph(id='ratios-graph')
])

# Define the callback to update the dashboard
@app.callback(
    [Output('financial-ratios', 'children'),
     Output('ratios-graph', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def update_dashboard(n_clicks, ticker):
    if n_clicks > 0:
        # Fetch the financial data
        total_assets, total_liabilities, equity, net_income, revenue, operating_cash_flow = fetch_financial_data(ticker)
        
        # Calculate the financial ratios
        current_ratio, quick_ratio, return_on_equity, profit_margin, operating_cash_flow_ratio = calculate_ratios(
            total_assets, total_liabilities, equity, net_income, revenue, operating_cash_flow)
        
        # Prepare the output text
        ratios_text = [
            html.H4(f'Ticker: {ticker}'),
            html.P(f'Current Ratio: {current_ratio:.2f}'),
            html.P(f'Quick Ratio: {quick_ratio:.2f}'),
            html.P(f'Return on Equity: {return_on_equity:.2f}'),
            html.P(f'Profit Margin: {profit_margin:.2f}'),
            html.P(f'Operating Cash Flow Ratio: {operating_cash_flow_ratio:.2f}')
        ]
        
        # Create the plot
        figure = {
            'data': [
                go.Bar(
                    x=['Current Ratio', 'Quick Ratio', 'Return on Equity', 'Profit Margin', 'Operating Cash Flow Ratio'],
                    y=[current_ratio, quick_ratio, return_on_equity, profit_margin, operating_cash_flow_ratio],
                    name='Financial Ratios'
                )
            ],
            'layout': go.Layout(
                title='Key Financial Ratios',
                yaxis={'title': 'Ratio Value'},
                xaxis={'title': 'Financial Ratio'},
            )
        }
        
        return ratios_text, figure
    else:
        return [], {}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
