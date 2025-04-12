import re
import string
import nltk
import pandas as pd
import contractions
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Dictionary of common slang and abbreviations
slang_dict = {
    "btw": "by the way",
    "tbh": "to be honest",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "idk": "i don't know",
    "lol": "laughing out loud",
    "rofl": "rolling on floor laughing",
    "afaik": "as far as i know",
    "brb": "be right back",
    "lmk": "let me know",
    "omg": "oh my god",
    "wtf": "what the f*ck",
    "bff": "best friends forever",
    "fyi": "for your information",
    "thx": "thanks",
    "w/": "with",
    "w/o": "without",
    "u": "you",
    "r": "are",
    "ur": "your",
    "b4": "before",
    "ppl": "people",
    "2day": "today",
    "2morrow": "tomorrow",
    "y'all": "you all",
    "gonna": "going to",
    "wanna": "want to",
    "dunno": "don't know",
    "gotta": "got to"
}

def preprocess_text(text, max_length=100, remove_stopwords=True, 
                   use_lemmatization=True, use_stemming=False, 
                   expand_contractions=True, convert_slang=True):
    """
    Clean and preprocess text data with advanced options
    
    Args:
        text (str): Input text to preprocess
        max_length (int): Maximum number of tokens to keep
        remove_stopwords (bool): Whether to remove stopwords
        use_lemmatization (bool): Whether to apply lemmatization
        use_stemming (bool): Whether to apply stemming (overrides lemmatization)
        expand_contractions (bool): Whether to expand contractions (e.g., don't -> do not)
        convert_slang (bool): Whether to convert common slang and abbreviations
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Handle empty or non-string inputs
        if not isinstance(text, str) or pd.isna(text):
            print(f"Warning: Invalid text input type: {type(text)}")
            return ""
        
        print(f"Processing text: {text[:100]}...")
        
        # Convert to lowercase
        text = text.lower()
        print("Converted to lowercase")
        
        # Expand contractions if enabled (e.g., "don't" -> "do not")
        if expand_contractions:
            try:
                text = contractions.fix(text)
                print("Expanded contractions")
            except Exception as e:
                print(f"Warning: Error expanding contractions: {e}")
                # Fall back to basic handling if contractions library fails
                text = text.replace("n't", " not")
                text = text.replace("'ve", " have")
                text = text.replace("'ll", " will")
                text = text.replace("'m", " am")
                text = text.replace("'re", " are")
                text = text.replace("'s", " is")
                text = text.replace("'d", " would")
                print("Used fallback contraction expansion")
        
        # Convert emoji to text
        try:
            text = emoji.demojize(text)
            text = text.replace(":", "").replace("_", " ")
            print("Converted emojis")
        except Exception as e:
            print(f"Warning: Error converting emojis: {e}")
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
        print("Removed URLs")
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        print("Removed HTML tags")
        
        # Replace email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        print("Replaced email addresses")
        
        # Replace phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' PHONE ', text)
        print("Replaced phone numbers")
        
        # Replace numbers
        text = re.sub(r'\d+', ' NUM ', text)
        print("Replaced numbers")
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        print("Normalized spaces")
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        print("Removed punctuation")
        
        try:
            # Tokenize
            tokens = word_tokenize(text)
            print(f"Tokenized text into {len(tokens)} tokens")
            
            # Convert slang and abbreviations
            if convert_slang:
                tokens = [slang_dict.get(token, token) for token in tokens]
                # Retokenize after slang conversion as some slang maps to multiple words
                tokens = [word for token in tokens for word in (word_tokenize(token) if isinstance(token, str) and ' ' in token else [token])]
                print("Converted slang")
            
            # Remove stopwords if enabled
            if remove_stopwords:
                original_length = len(tokens)
                tokens = [token for token in tokens if token not in stop_words]
                print(f"Removed {original_length - len(tokens)} stopwords")
            
            # Apply stemming or lemmatization
            if use_stemming:
                tokens = [stemmer.stem(token) for token in tokens]
                print("Applied stemming")
            elif use_lemmatization:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                print("Applied lemmatization")
            
            # Truncate to max_length
            if max_length > 0:
                original_length = len(tokens)
                tokens = tokens[:max_length]
                if original_length > max_length:
                    print(f"Truncated from {original_length} to {max_length} tokens")
            
            # Join tokens back to text
            processed_text = ' '.join(tokens)
            print("Joined tokens back to text")
            
            # Final cleanup of any remaining multiple spaces
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            print("Final cleanup complete")
            
            print(f"Final processed text: {processed_text[:100]}...")
            return processed_text
            
        except Exception as e:
            print(f"Error during tokenization/processing: {e}")
            return text  # Return cleaned text without tokenization if it fails
    
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        if isinstance(text, str):
            return text  # Return original text if preprocessing fails
        return ""  # Return empty string for non-string inputs
    
def compute_text_features(text):
    """
    Compute additional text features for enhanced analysis
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of text features
    """
    if not isinstance(text, str) or pd.isna(text):
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0,
            'has_urls': False,
            'has_numbers': False,
            'has_email': False
        }
    
    try:
        # Basic counts
        char_count = len(text)
        word_count = len(re.findall(r'\w+', text))
        sentence_count = max(1, len(re.split(r'[.!?]+', text)))
        
        # Average lengths
        avg_word_length = char_count / max(1, word_count)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Special character counts
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Uppercase ratio
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / max(1, char_count)
        
        # Content indicators
        has_urls = bool(re.search(r'http\S+|www\S+|https\S+', text))
        has_numbers = bool(re.search(r'\d+', text))
        has_email = bool(re.search(r'\S+@\S+', text))
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'uppercase_ratio': uppercase_ratio,
            'has_urls': has_urls,
            'has_numbers': has_numbers,
            'has_email': has_email
        }
    
    except Exception as e:
        print(f"Error computing text features: {e}")
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0,
            'has_urls': False,
            'has_numbers': False,
            'has_email': False
        } 