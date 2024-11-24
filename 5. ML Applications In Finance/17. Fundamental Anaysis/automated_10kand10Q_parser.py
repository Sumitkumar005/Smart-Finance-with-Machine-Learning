# Write a script that uses NLP to automatically extract and summarize key financial metrics and textual insights from companies'
# 10-K and 10-Q reports.
# Financial Metrics and Textual Insights Extraction Script

# Import required libraries
import requests
from bs4 import BeautifulSoup
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Optional: Import OpenAI API for advanced summarization (requires API key)
try:
    import openai
    USE_OPENAI = True
except ImportError:
    USE_OPENAI = False

# Optional: Set up OpenAI API key (if applicable)
# Replace with your OpenAI API key
OPENAI_API_KEY = "your-api-key-here"
if USE_OPENAI:
    openai.api_key = OPENAI_API_KEY

# Function to fetch 10-K or 10-Q reports from EDGAR
def fetch_sec_report(cik, report_type="10-K"):
    """
    Fetches the latest 10-K or 10-Q report for a given company from the SEC EDGAR system.

    Args:
        cik (str): The Central Index Key of the company.
        report_type (str): The type of report to fetch ('10-K' or '10-Q').

    Returns:
        str: The plain text content of the report.
    """
    base_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={report_type}&count=10"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(base_url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch data for CIK {cik}. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    document_links = soup.find_all('a', href=True, text='Documents')

    if not document_links:
        print(f"No {report_type} reports found for CIK {cik}.")
        return None

    # Get the first document link
    document_url = "https://www.sec.gov" + document_links[0]['href']
    doc_response = requests.get(document_url, headers=headers)
    doc_soup = BeautifulSoup(doc_response.content, 'html.parser')

    # Find the link to the full text
    report_link = doc_soup.find('a', href=True, text=re.compile(r'\.htm$'))
    if not report_link:
        print("Full text link not found.")
        return None

    full_report_url = "https://www.sec.gov" + report_link['href']
    report_response = requests.get(full_report_url, headers=headers)
    report_soup = BeautifulSoup(report_response.content, 'html.parser')

    # Extract plain text from the report
    return report_soup.get_text()

# Function to extract financial metrics using regular expressions
def extract_financial_metrics(report_text):
    """
    Extracts key financial metrics from the report text using regular expressions.

    Args:
        report_text (str): The plain text of the financial report.

    Returns:
        dict: A dictionary of extracted financial metrics.
    """
    metrics = {}

    # Patterns for extracting financial metrics
    revenue_pattern = r"(?:Revenue|Total Revenue|Sales):?\s*\$?([\d,\.]+)"
    profit_pattern = r"(?:Net Income|Profit|Net Profit):?\s*\$?([\d,\.]+)"
    assets_pattern = r"(?:Total Assets):?\s*\$?([\d,\.]+)"
    liabilities_pattern = r"(?:Total Liabilities):?\s*\$?([\d,\.]+)"

    # Apply patterns
    metrics['Revenue'] = re.findall(revenue_pattern, report_text, re.IGNORECASE)
    metrics['Net Profit'] = re.findall(profit_pattern, report_text, re.IGNORECASE)
    metrics['Total Assets'] = re.findall(assets_pattern, report_text, re.IGNORECASE)
    metrics['Total Liabilities'] = re.findall(liabilities_pattern, report_text, re.IGNORECASE)

    # Clean up results (convert to numbers where possible)
    for key, value in metrics.items():
        if value:
            metrics[key] = [float(v.replace(',', '').replace('$', '')) for v in value]

    return metrics

# Function to summarize textual insights using NLP
def summarize_text_nlp(report_text):
    """
    Summarizes key insights from the report using NLP.

    Args:
        report_text (str): The plain text of the financial report.

    Returns:
        str: A summarized text of key insights.
    """
    # Load spaCy NLP model
    nlp = spacy.load('en_core_web_sm')

    # Process the text
    doc = nlp(report_text)

    # Extract key sentences based on named entities and TF-IDF scores
    sentences = [sent.text for sent in doc.sents]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    X = vectorizer.fit_transform(sentences)

    # Select top sentences based on TF-IDF scores
    scores = np.asarray(X.sum(axis=1)).flatten()
    top_indices = scores.argsort()[-5:][::-1]
    summary = "\n".join([sentences[i] for i in top_indices])

    return summary

# Function to summarize insights using OpenAI GPT (optional)
def summarize_text_gpt(report_text):
    """
    Summarizes key insights from the report using OpenAI GPT.

    Args:
        report_text (str): The plain text of the financial report.

    Returns:
        str: A summarized text of key insights.
    """
    if not USE_OPENAI:
        raise ImportError("OpenAI library is not installed. Install it using 'pip install openai'.")

    # GPT prompt
    prompt = (
        "Extract and summarize key financial metrics and textual insights from the following report:\n\n"
        f"{report_text[:4000]}..."  # GPT has a token limit; truncate the text
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        temperature=0.5
    )

    return response.choices[0].text.strip()

# Main function
if __name__ == "__main__":
    # Example: Fetch and analyze a report for Apple Inc. (CIK: 0000320193)
    cik = "0000320193"
    report_text = fetch_sec_report(cik, report_type="10-K")

    if report_text:
        print("Extracting financial metrics...")
        metrics = extract_financial_metrics(report_text)
        print(f"Extracted Metrics: {metrics}")

        print("\nSummarizing textual insights using NLP...")
        nlp_summary = summarize_text_nlp(report_text)
        print(f"NLP Summary:\n{nlp_summary}")

        if USE_OPENAI:
            print("\nSummarizing textual insights using OpenAI GPT...")
            gpt_summary = summarize_text_gpt(report_text)
            print(f"GPT Summary:\n{gpt_summary}")
