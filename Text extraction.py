import requests
from bs4 import BeautifulSoup
import os

def extract_text_from_file(file_path):
    """Extract text from a plain text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_webpage(url):
    """Extract text from a web page using BeautifulSoup."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from the body of the page
        text = soup.get_text(separator=' ', strip=True)
        return text
    else:
        raise Exception(f"Failed to retrieve webpage. Status code: {response.status_code}")

def process_document(input_data):
    """Process the input and extract clean text."""
    # Check if the input is a URL or a file path
    if input_data.startswith('http') or input_data.startswith('www'):
        # If it's a URL, extract text from the webpage
        return extract_text_from_webpage(input_data)
    elif os.path.isfile(input_data):
        # If it's a file path, extract text from the text file
        return extract_text_from_file(input_data)
    else:
        raise ValueError("Invalid input. Please provide a valid URL or file path.")

# Example usage:
# You can provide a URL or file path to the `process_document` function
document_input = "https://example.com"  # Replace with your URL or file path
cleaned_text = process_document(document_input)

# Print the cleaned text
print(cleaned_text[:500])  # Print the first 500 characters for preview
