import re
import string
import sys
import nltk

# Set UTF-8 encoding for stdout to prevent Windows console encoding errors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Programmatically check and download required NLTK resources
def download_nltk_resources():
    resources = {
        'corpora/stopwords': 'stopwords',
        'corpora/wordnet': 'wordnet',
        'tokenizers/punkt': 'punkt',
        'tokenizers/punkt_tab': 'punkt_tab'
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download nltk resource '{name}': {e}")

# Run the downloader
download_nltk_resources()

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Define English stopwords
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

def clean_text(text):
    """
    Cleans raw tweet text by removing:
    - HTML tags
    - URLs
    - Mentions (@user)
    - Numbers
    - Emojis (non-ASCII characters)
    - Punctuation (keeping hashtag words but removing '#' sign)
    - Extra spaces
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Remove Mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # 5. Remove Emojis (Encode as ASCII and ignore non-ASCII characters)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 7. Handle Hashtags: Remove '#' symbol but keep the keyword word
    text = text.replace('#', '')
    
    # 8. Remove Punctuation
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    # 9. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text, method='lemmatize'):
    """
    Applies tokenization, stop-word removal, and stemming/lemmatization to the cleaned text.
    
    Parameters:
    - text (str): The raw text to process.
    - method (str): 'lemmatize' for WordNet Lemmatization, 'stem' for Porter Stemming.
    
    Returns:
    - str: Preprocessed, space-separated tokens.
    """
    # First clean the text
    cleaned = clean_text(text)
    
    if not cleaned:
        return ""
    
    # Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        # Fallback split if word_tokenize fails
        tokens = cleaned.split()
        
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Apply Lemmatization or Stemming
    if method == 'lemmatize':
        processed_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    elif method == 'stem':
        processed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    else:
        processed_tokens = filtered_tokens
        
    return " ".join(processed_tokens)

if __name__ == "__main__":
    # Test cases
    test_tweets = [
        "@amazon I love this phone 😍🔥 https://abc.com",
        "Worst service ever! #disappointed #waste 123",
        "It's just an average day in Seattle... http://weather.com",
        "I hate the new software update! It keeps freezing... 😠"
    ]
    
    print("Testing Preprocessing Module:")
    for t in test_tweets:
        cleaned = clean_text(t)
        preprocessed = preprocess_text(t, method='lemmatize')
        print(f"\nOriginal: {t}")
        print(f"Cleaned:  {cleaned}")
        print(f"Preproc:  {preprocessed}")
