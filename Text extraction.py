import requests
from bs4 import BeautifulSoup
import re

def fetch_webpage_content(url):
    """Fetch the content of a webpage."""
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    return response.text

def extract_text_from_html(html_content):
    """Extract and clean text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean and normalize whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text

def read_text_file(file_path):
    """Read and return the content of a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_text(text):
    """Further clean text by removing special characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    return text.strip()

def get_document_content(source):
    """Determine if the source is a URL or a file path and fetch content accordingly."""
    if source.startswith('http://') or source.startswith('https://'):
        html_content = fetch_webpage_content(source)
        text = extract_text_from_html(html_content)
    else:
        text = read_text_file(source)
    
    return clean_text(text)

# Example usage
source = 'https://example.com'  # or 'path/to/your/file.txt'
cleaned_text = get_document_content(source)
print(cleaned_text)
