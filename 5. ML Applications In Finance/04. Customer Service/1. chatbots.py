# Automated conversational agents for customer service..

'''
Python script for building a basic conversational agent (chatbot) for customer service.
This example uses Natural Language Processing (NLP) techniques to understand and respond to user queries.
'''

# Import necessary libraries
import random
import nltk
from nltk.chat.util import Chat, reflections

# Ensure NLTK components are available
nltk.download('punkt')

# Define pairs of patterns and responses for the chatbot
# These are simple rule-based responses for demonstration purposes
pairs = [
    [r"hi|hello|hey", ["Hello! How can I assist you today?", "Hi there! How can I help?"]],
    [r"how can I (.*)", ["I can help you with %1. Could you provide more details?"]],
    [r"(.*) your name?", ["I am your customer service assistant, here to help you!"]],
    [r"(.*) product (.*)", ["Can you please provide the product name or details?", 
                             "I'd be happy to assist with product inquiries. Could you elaborate?"]],
    [r"(.*) refund (.*)", ["To process a refund, I'll need your order details. Please provide them.",
                            "Refunds usually take 5-7 business days. Can I have your order ID?"]],
    [r"(.*) not working", ["I apologize for the inconvenience. Could you describe the issue in more detail?",
                            "I'm here to help with troubleshooting. What seems to be the problem?"]],
    [r"bye|exit|quit", ["Goodbye! Have a great day!", "Thank you for contacting us. Bye!"]],
]

# Reflections are used for simple pronoun replacements in responses
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}

# Initialize the chatbot with the pairs and reflections
chatbot = Chat(pairs, reflections)

# Function to start the chatbot
def start_chat():
    print("Welcome to the customer service chatbot! Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot.respond(user_input)
        print(f"Chatbot: {response}")

# Start the chatbot
if __name__ == "__main__":
    start_chat()
