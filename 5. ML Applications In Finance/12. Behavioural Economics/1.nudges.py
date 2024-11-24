# Script to provide real-time nudges for saving or responsible spending by integrating with personal finance apps

# Import necessary libraries
import requests
import time
from datetime import datetime

# Placeholder function to integrate with personal finance API
def fetch_user_finance_data(api_key, user_id):
    """
    Simulates fetching user financial data from a personal finance app.
    In practice, replace with actual API calls.
    """
    # Example API endpoint (replace with actual endpoint)
    endpoint = f"https://api.financeapp.com/user/{user_id}/data"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(endpoint, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching data: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Analyze user spending patterns
def analyze_spending(data):
    """
    Analyzes spending data to identify overspending trends or saving opportunities.
    """
    categories = data.get('spending_categories', {})
    high_spending_category = max(categories, key=categories.get, default=None)
    total_spent = sum(categories.values())
    budget = data.get('monthly_budget', 0)
    
    if total_spent > budget:
        return f"Alert: You've exceeded your monthly budget by ${total_spent - budget:.2f}. Consider cutting back on {high_spending_category}."
    elif total_spent > 0.8 * budget:
        return f"Warning: You've spent 80% of your monthly budget. Monitor your expenses in {high_spending_category}."
    else:
        return f"Good job staying within budget! Keep saving."

# Provide saving tips based on user data
def provide_saving_tips(data):
    """
    Generates tailored saving tips based on user behavior.
    """
    recurring_expenses = data.get('recurring_expenses', [])
    high_expenses = sorted(recurring_expenses, key=lambda x: x['amount'], reverse=True)[:3]
    
    tips = ["Consider reducing or renegotiating high recurring expenses like subscriptions or memberships:"]
    for expense in high_expenses:
        tips.append(f"- {expense['name']}: ${expense['amount']:.2f} per month.")
    
    tips.append("Automate your savings by setting up a transfer to your savings account after each paycheck.")
    return "\n".join(tips)

# Send a nudge to the user
def send_nudge(nudge_message, user_contact):
    """
    Sends a nudge message to the user.
    """
    print(f"Sending nudge to {user_contact}: {nudge_message}")
    # Placeholder for actual messaging logic (e.g., SMS, email, app notification)

# Main function to monitor user finance and provide nudges
def monitor_finances(api_key, user_id, user_contact):
    """
    Periodically fetches user data, analyzes behavior, and sends nudges.
    """
    while True:
        print(f"Fetching data for user {user_id} at {datetime.now()}...")
        data = fetch_user_finance_data(api_key, user_id)
        
        if data:
            # Analyze spending and provide nudges
            spending_alert = analyze_spending(data)
            saving_tips = provide_saving_tips(data)
            
            # Combine nudges
            nudge_message = f"{spending_alert}\n\nSaving Tips:\n{saving_tips}"
            
            # Send nudge
            send_nudge(nudge_message, user_contact)
        
        # Wait for a specific interval before the next check (e.g., 1 hour)
        time.sleep(3600)

# Example usage
if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    USER_ID = "user_123"
    USER_CONTACT = "user_email@example.com"
    
    monitor_finances(API_KEY, USER_ID, USER_CONTACT)
