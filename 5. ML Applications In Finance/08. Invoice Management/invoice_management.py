'''
Automated processing and management of invoices.
This script demonstrates how to automate invoice data handling, process outstanding payments, and generate summary reports.
'''

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Sample data: Creating an example dataset
data = {
    'Invoice_ID': [1001, 1002, 1003, 1004, 1005],
    'Client': ['Client A', 'Client B', 'Client C', 'Client D', 'Client E'],
    'Amount': [2500, 4500, 3200, 1500, 2800],
    'Due_Date': ['2024-11-01', '2024-11-15', '2024-10-30', '2024-11-20', '2024-10-25'],
    'Status': ['Paid', 'Pending', 'Pending', 'Paid', 'Pending']
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Convert 'Due_Date' to datetime format
df['Due_Date'] = pd.to_datetime(df['Due_Date'])

# Today's date
today = dt.datetime.now()

# Function to identify overdue invoices
def check_overdue(df, today):
    df['Overdue'] = df.apply(lambda row: 'Yes' if row['Status'] == 'Pending' and row['Due_Date'] < today else 'No', axis=1)
    return df

# Update DataFrame with overdue status
df = check_overdue(df, today)

# Generate a summary of pending and overdue invoices
def generate_summary(df):
    pending_invoices = df[df['Status'] == 'Pending']
    overdue_invoices = pending_invoices[pending_invoices['Overdue'] == 'Yes']
    summary = {
        'Total Invoices': len(df),
        'Paid Invoices': len(df[df['Status'] == 'Paid']),
        'Pending Invoices': len(pending_invoices),
        'Overdue Invoices': len(overdue_invoices),
        'Total Amount Pending': pending_invoices['Amount'].sum(),
        'Total Overdue Amount': overdue_invoices['Amount'].sum()
    }
    return summary

# Get summary report
summary_report = generate_summary(df)
print("\nInvoice Summary Report:")
for key, value in summary_report.items():
    print(f"{key}: {value}")

# Plotting the data
def plot_invoice_data(df):
    status_counts = df['Status'].value_counts()
    overdue_counts = df['Overdue'].value_counts()
    
    plt.figure(figsize=(12, 6))
    
    # Plot status counts
    plt.subplot(1, 2, 1)
    status_counts.plot(kind='bar', color=['green', 'red'], alpha=0.7)
    plt.title('Invoice Status')
    plt.xlabel('Status')
    plt.ylabel('Count')
    
    # Plot overdue counts
    plt.subplot(1, 2, 2)
    overdue_counts.plot(kind='bar', color=['blue', 'orange'], alpha=0.7)
    plt.title('Overdue Status')
    plt.xlabel('Overdue')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Visualize the data
plot_invoice_data(df)

# Save processed invoice data to a CSV file
output_file = 'processed_invoices.csv'
df.to_csv(output_file, index=False)
print(f"\nProcessed invoice data saved to {output_file}.")

# Function to send reminders for overdue invoices
def send_reminders(df):
    overdue_invoices = df[(df['Status'] == 'Pending') & (df['Overdue'] == 'Yes')]
    print("\nSending reminders for overdue invoices:")
    for _, invoice in overdue_invoices.iterrows():
        print(f"Reminder sent to {invoice['Client']} for Invoice ID: {invoice['Invoice_ID']} - Amount: {invoice['Amount']}")

# Send reminders
send_reminders(df)
