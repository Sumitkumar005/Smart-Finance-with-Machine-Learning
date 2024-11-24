# Build an application that tracks key performance indicators (KPIs) of a venture capital portfolio,
# such as customer growth rate, churn, and lifetime value. You can use Python libraries like Dash or Streamlit
# for interactive dashboards.
pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Sample Data for VC Portfolio
data = {
    'customer_id': range(1, 11),
    'signup_date': pd.to_datetime([
        '2021-01-15', '2021-02-20', '2021-03-25', '2021-04-10', '2021-05-30',
        '2021-06-14', '2021-07-18', '2021-08-22', '2021-09-10', '2021-10-05'
    ]),
    'last_purchase_date': pd.to_datetime([
        '2022-01-12', '2022-02-15', '2022-03-20', '2022-04-05', '2022-06-01',
        '2022-06-20', '2022-07-30', '2022-08-18', '2022-09-12', '2022-10-10'
    ]),
    'total_spent': [500, 800, 350, 1200, 950, 400, 500, 1100, 450, 300],
    'churn': [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]  # 1 indicates churned customer
}

# Convert the data into DataFrame
df = pd.DataFrame(data)

# Customer Growth Rate Calculation
def customer_growth_rate(start_date, end_date):
    total_customers_start = df[df['signup_date'] <= start_date].shape[0]
    total_customers_end = df[df['signup_date'] <= end_date].shape[0]
    growth_rate = (total_customers_end - total_customers_start) / total_customers_start * 100
    return growth_rate

# Customer Churn Rate Calculation
def churn_rate():
    churned = df['churn'].sum()
    total_customers = df.shape[0]
    churn_rate = churned / total_customers * 100
    return churn_rate

# Customer Lifetime Value (LTV) Calculation
def calculate_ltv():
    total_revenue = df['total_spent'].sum()
    total_customers = df.shape[0]
    ltv = total_revenue / total_customers
    return ltv

# Streamlit Application
st.title("Venture Capital Portfolio KPI Dashboard")
st.markdown("### Track the Key Performance Indicators (KPIs) of your Venture Capital Portfolio")

# Select Date Range for Customer Growth Calculation
start_date = st.date_input('Select Start Date for Customer Growth Rate', datetime(2021, 1, 1))
end_date = st.date_input('Select End Date for Customer Growth Rate', datetime(2022, 12, 31))

# Calculate KPIs
growth_rate = customer_growth_rate(start_date, end_date)
churn = churn_rate()
ltv = calculate_ltv()

# Display KPIs
st.subheader("Customer Growth Rate (%)")
st.write(f"{growth_rate:.2f}%")
st.subheader("Customer Churn Rate (%)")
st.write(f"{churn:.2f}%")
st.subheader("Customer Lifetime Value (LTV) ($)")
st.write(f"${ltv:.2f}")

# Visualize Customer Growth Over Time
st.subheader("Customer Growth Over Time")

# Create a chart for customer signups over time
signup_data = df.groupby(df['signup_date'].dt.to_period('M')).size().reset_index(name='customer_count')
signup_data['signup_date'] = signup_data['signup_date'].dt.to_timestamp()

plt.figure(figsize=(10, 6))
sns.lineplot(x='signup_date', y='customer_count', data=signup_data)
plt.title('Customer Signups Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
st.pyplot(plt)

# Visualize Churn Distribution
st.subheader("Customer Churn Distribution")

plt.figure(figsize=(6, 6))
sns.countplot(x='churn', data=df)
plt.title('Customer Churn Distribution')
plt.xlabel('Churn Status')
plt.ylabel('Count')
st.pyplot(plt)

# Visualize LTV Distribution
st.subheader("Customer Lifetime Value Distribution")

plt.figure(figsize=(6, 6))
sns.histplot(df['total_spent'], kde=True, bins=5)
plt.title('Customer Lifetime Value Distribution')
plt.xlabel('Total Spent ($)')
plt.ylabel('Count')
st.pyplot(plt)

# Adding a predictive model (Optional for more advanced functionality)
# Here we could use a machine learning model to predict future KPIs or customer churn

# Example code to add predictive analytics model if required:
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(df[['total_spent', 'age']], df['churn'])
# st.write(f"Model Score: {model.score(X_test, y_test)}")

# Display an interactive table of customer data
st.subheader("Customer Data")
st.write(df)

# Sidebar for additional controls
st.sidebar.title("Additional Settings")
st.sidebar.markdown("""
You can change the time range or adjust other parameters in the sidebar to see different trends.
""")
