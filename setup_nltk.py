import nltk
import os
import sys

def setup_nltk():
    """Download required NLTK data"""
    required_packages = [
        'punkt',      # Tokenizer
        'stopwords',  # Stopwords
        'wordnet',    # Lemmatizer
        'averaged_perceptron_tagger'  # POS tagger
    ]
    
    print("Setting up NLTK...")
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
            print(f"✓ {package} already installed")
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package)
            print(f"✓ {package} installed")

if __name__ == "__main__":
    setup_nltk() 