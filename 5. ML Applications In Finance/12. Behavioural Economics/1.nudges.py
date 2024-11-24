"""
Script for integrating with a personal finance app to provide real-time nudges for better financial habits.
Encourages saving and responsible spending based on user behavior.
"""

import requests
import json
import time
import random
from datetime import datetime

# Constants
API_BASE_URL = "https://api.personalfinanceapp.com/v1"  # Replace with the actual API URL
API_KEY = "YOUR_API_KEY_HERE"  # Replace with the actual API key

# Example nudges
NUDGES = {
    "spending_alert": "You've spent more than usual this week. Consider limiting discretionary spending.",
    "savings_goal": "Great job on savings so far! You're only ${remaining} away from your goal.",
    "spending_category_alert": "Your dining expenses have increased. Consider cooking at home to save money.",
    "general_tip": "Tip: Automate your savings to reach your goals faster!",
    "budget_reminder": "Reminder: You’ve reached 80% of your monthly budget. Time to review your spending habits.",
}

# Function to make API calls
def api_request(endpoint, method="GET", data=None):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{API_BASE_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError("Unsupported HTTP method")

        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Fetch user financial data
def get_user_financial_data(user_id):
    endpoint = f"users/{user_id}/financial_data"
    return api_request(endpoint)

# Fetch user savings goal data
def get_user_savings_goals(user_id):
    endpoint = f"users/{user_id}/savings_goals"
    return api_request(endpoint)

# Analyze spending behavior
def analyze_spending(data):
    high_spending_categories = []
    total_spent = 0

    for category, amount in data.get("spending_by_category", {}).items():
        total_spent += amount
        if amount > data["average_spending"].get(category, 0) * 1.2:
            high_spending_categories.append(category)

    return total_spent, high_spending_categories

# Send a nudge to the user
def send_nudge(user_id, message):
    endpoint = f"users/{user_id}/send_nudge"
    data = {"message": message}
    response = api_request(endpoint, method="POST", data=data)
    if response:
        print(f"Nudge sent successfully: {message}")

# Main function to integrate the features
def main():
    user_id = "12345"  # Replace with the actual user ID
    print(f"Fetching financial data for user {user_id}...")
    
    # Fetch user data
    user_data = get_user_financial_data(user_id)
    if not user_data:
        print("Failed to fetch user financial data.")
        return

    savings_data = get_user_savings_goals(user_id)
    if not savings_data:
        print("Failed to fetch user savings goals.")
        return

    # Analyze spending
    total_spent, high_spending_categories = analyze_spending(user_data)
    print(f"Total spent this month: ${total_spent:.2f}")
    print(f"High spending categories: {', '.join(high_spending_categories) if high_spending_categories else 'None'}")

    # Select appropriate nudges
    if total_spent > user_data["monthly_budget"] * 0.8:
        send_nudge(user_id, NUDGES["budget_reminder"])
    elif high_spending_categories:
        for category in high_spending_categories:
            send_nudge(user_id, NUDGES["spending_category_alert"].replace("dining", category))

    # Savings nudge
    for goal in savings_data.get("goals", []):
        remaining = goal["target_amount"] - goal["current_amount"]
        if remaining > 0:
            send_nudge(user_id, NUDGES["savings_goal"].replace("${remaining}", f"${remaining:.2f}"))

    # General tip
    if random.random() < 0.5:  # Send a random tip occasionally
        send_nudge(user_id, NUDGES["general_tip"])

# Run the script
if __name__ == "__main__":
    main()
