# Use NLP to analyze a large corpus of successful investment theses and presentations to generate a template or 
# even a first draft for a new investment thesis.
pip install openai pandas nltk
# Import necessary libraries
import openai
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Download the punkt tokenizer from NLTK
nltk.download('punkt')

# Set your OpenAI API key (make sure to replace with your own API key)
openai.api_key = 'YOUR_API_KEY'

# Load the corpus of investment theses or presentations (you can upload your data as a .csv, .txt, or other formats)
# For this example, we're assuming that you have a CSV file with investment thesis data
# You can also use text files with investment theses.
data = pd.read_csv('investment_theses.csv')  # Replace with your own corpus file

# Assuming the CSV has a column 'thesis' which contains the text of each investment thesis
corpus = data['thesis'].tolist()

# Tokenizing the corpus into sentences for NLP processing
corpus_sentences = []
for text in corpus:
    sentences = sent_tokenize(text)
    corpus_sentences.extend(sentences)

# Preprocess corpus - For simplicity, let's just concatenate all sentences
# but you can preprocess it further for better results
text_corpus = " ".join(corpus_sentences)

# Function to call the OpenAI GPT API for generating the investment thesis
def generate_investment_thesis(corpus_text):
    prompt = f"Analyze the following text and generate a first draft of a new investment thesis:\n\n{corpus_text}\n\nDraft the investment thesis:"

    # Using the OpenAI API to generate a new investment thesis
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use different engines like 'gpt-3.5-turbo'
        prompt=prompt,
        max_tokens=1000,  # You can adjust this depending on how long you want the thesis to be
        n=1,  # Generate 1 result
        stop=None,  # You can define a stop sequence if needed
        temperature=0.7  # Controls the creativity of the response (between 0 and 1)
    )

    # Extract the generated thesis from the response
    generated_thesis = response.choices[0].text.strip()
    return generated_thesis

# Generate the investment thesis
new_investment_thesis = generate_investment_thesis(text_corpus)

# Output the generated investment thesis
print("Generated Investment Thesis:\n")
print(new_investment_thesis)

# Optionally, save the generated thesis to a text file
with open('generated_investment_thesis.txt', 'w') as file:
    file.write(new_investment_thesis)
