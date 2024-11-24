'''
Python script for Optical Character Recognition (OCR) to read invoices and input data into a database or accounting software.
'''

# Import necessary libraries
import os
import pytesseract
from PIL import Image
import sqlite3
import re
import json
import requests

# Initialize Tesseract executable path (update this with your local path to Tesseract)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Google Vision API key (if using Google Vision API, replace with your own key)
GOOGLE_VISION_API_KEY = "YOUR_API_KEY_HERE"

# Function to process invoices with Tesseract OCR
def process_invoice_tesseract(image_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(img)
            print(f"Extracted Text from Tesseract OCR:\n{text}")
            return text
    except Exception as e:
        print(f"Error processing the image with Tesseract: {e}")
        return None

# Function to process invoices with Google Vision API
def process_invoice_google_vision(image_path):
    try:
        # Load the image file
        with open(image_path, 'rb') as img_file:
            image_content = img_file.read()
        
        # Define the request payload
        url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        payload = {
            "requests": [
                {
                    "image": {"content": image_content.decode('latin1')},
                    "features": [{"type": "TEXT_DETECTION"}],
                }
            ]
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()
        
        # Parse the text from the response
        text = result['responses'][0].get('fullTextAnnotation', {}).get('text', '')
        print(f"Extracted Text from Google Vision API:\n{text}")
        return text
    except Exception as e:
        print(f"Error processing the image with Google Vision API: {e}")
        return None

# Function to parse extracted text for invoice details
def parse_invoice_details(text):
    # Define regex patterns to extract specific details (example: invoice number, date, total)
    invoice_number_pattern = r"Invoice Number[:\s]*([A-Za-z0-9-]+)"
    date_pattern = r"Date[:\s]*([0-9/.-]+)"
    total_pattern = r"Total[:\s]*([0-9.,]+)"

    details = {
        "invoice_number": re.search(invoice_number_pattern, text).group(1) if re.search(invoice_number_pattern, text) else None,
        "date": re.search(date_pattern, text).group(1) if re.search(date_pattern, text) else None,
        "total": re.search(total_pattern, text).group(1) if re.search(total_pattern, text) else None,
    }
    print(f"Parsed Invoice Details: {details}")
    return details

# Function to store invoice details into a database
def store_invoice_details(details):
    try:
        # Connect to SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect('invoices.db')
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_number TEXT,
            date TEXT,
            total REAL
        )
        ''')

        # Insert the parsed invoice details into the database
        cursor.execute('''
        INSERT INTO invoices (invoice_number, date, total)
        VALUES (?, ?, ?)
        ''', (details['invoice_number'], details['date'], float(details['total'].replace(',', '')) if details['total'] else None))

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()
        print("Invoice details stored in the database successfully.")
    except Exception as e:
        print(f"Error storing invoice details into the database: {e}")

# Main function
def main():
    # Path to the invoice image
    image_path = 'invoice_sample.jpg'  # Replace with the path to your invoice image
    
    # Choose OCR method
    use_google_vision = False  # Set to True to use Google Vision API

    if use_google_vision:
        text = process_invoice_google_vision(image_path)
    else:
        text = process_invoice_tesseract(image_path)

    if text:
        # Parse and store the extracted details
        invoice_details = parse_invoice_details(text)
        store_invoice_details(invoice_details)
    else:
        print("No text extracted from the invoice.")

if __name__ == '__main__':
    main()
