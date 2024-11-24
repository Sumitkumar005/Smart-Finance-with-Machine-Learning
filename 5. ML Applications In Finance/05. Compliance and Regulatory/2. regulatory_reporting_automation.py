"""
Automated systems for generating regulatory reports.

This script processes financial transaction data, calculates key metrics, generates visual summaries,
and produces a structured report that aligns with regulatory compliance requirements.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF  # Library to create PDF reports

# Generate sample financial transaction data
data = {
    'Transaction_ID': range(1, 101),
    'Transaction_Date': pd.date_range(start="2024-01-01", periods=100, freq='D'),
    'Transaction_Amount': np.random.uniform(100, 10000, size=100),
    'Customer_ID': np.random.randint(1000, 1100, size=100),
    'Transaction_Type': np.random.choice(['Credit', 'Debit'], size=100),
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Add a column for Regulatory Category (e.g., High-value transactions)
df['Regulatory_Category'] = np.where(df['Transaction_Amount'] > 5000, 'High-Value', 'Standard')

# Function to calculate summary statistics
def calculate_statistics(dataframe):
    summary = {
        'Total_Transactions': len(dataframe),
        'Total_Amount': dataframe['Transaction_Amount'].sum(),
        'Average_Amount': dataframe['Transaction_Amount'].mean(),
        'High_Value_Transactions': len(dataframe[dataframe['Regulatory_Category'] == 'High-Value']),
        'Standard_Transactions': len(dataframe[dataframe['Regulatory_Category'] == 'Standard']),
    }
    return summary

# Calculate the summary statistics
statistics = calculate_statistics(df)

# Visualize transaction amounts by type
plt.figure(figsize=(10, 6))
df.groupby('Transaction_Type')['Transaction_Amount'].sum().plot(kind='bar', color=['blue', 'orange'])
plt.title('Total Transaction Amount by Type')
plt.ylabel('Total Amount')
plt.xlabel('Transaction Type')
plt.savefig('transaction_type_summary.png')
plt.close()  # Close the plot to prevent overlap in future plots

# Visualize transaction categories
plt.figure(figsize=(10, 6))
df['Regulatory_Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
plt.title('Transaction Categories')
plt.ylabel('')
plt.savefig('transaction_category_summary.png')
plt.close()

# Create a regulatory report in PDF format
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Regulatory Report', align='C', ln=1)
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=1)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

# Initialize the PDF
pdf = PDF()
pdf.add_page()

# Add report title
pdf.set_font('Arial', 'B', 14)
pdf.cell(0, 10, 'Regulatory Compliance Report', align='C', ln=1)
pdf.ln(10)

# Add metadata
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Report Generated On: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=1)
pdf.cell(0, 10, 'Generated by: Automated System', ln=1)
pdf.ln(10)

# Add statistics
pdf.chapter_title('Summary Statistics')
stats_text = '\n'.join([f"{key}: {value}" for key, value in statistics.items()])
pdf.chapter_body(stats_text)

# Add visual summaries
pdf.chapter_title('Visual Summaries')
pdf.image('transaction_type_summary.png', x=10, y=None, w=190)
pdf.add_page()
pdf.image('transaction_category_summary.png', x=10, y=None, w=190)

# Save the PDF
pdf.output('Regulatory_Report.pdf')

# Print completion message
print("Regulatory report has been generated successfully as 'Regulatory_Report.pdf'.")

'''
This script automates the process of generating regulatory reports. Here's how it works:
1. It processes sample financial transaction data.
2. Calculates key metrics, such as total transaction amounts, average values, and counts of high-value transactions.
3. Visualizes the data to provide insights into transaction types and categories.
4. Generates a structured report in PDF format that includes both the numerical summaries and visualizations.

You can adapt this code for real-world datasets and tailor it to meet specific regulatory requirements.
'''
