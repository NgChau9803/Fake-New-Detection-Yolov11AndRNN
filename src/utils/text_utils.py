import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import contractions
import emoji
import yaml
import os

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Load slang/abbreviation dictionary from config
def load_slang_dict():
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'config', 'slang_dict.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        return {}
    except Exception as e:
        print(f"Warning: Could not load slang dictionary: {e}")
        return {}

SLANG_DICT = load_slang_dict()

class TextProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def normalize_text(self, text, method='advanced'):
        """
        Normalize text using different methods:
        - 'basic': minimal preprocessing
        - 'standard': remove stopwords, lowercase, etc.
        - 'advanced': includes slang handling and lemmatization
        - 'bert': minimal preprocessing suitable for BERT models
        """
        if not isinstance(text, str):
            return ""
            
        # Basic preprocessing for all methods
        text = self._expand_contractions(text)
        text = self._replace_emojis(text)
        text = self._replace_slang(text)
        
        if method == 'basic':
            return text.lower()
            
        elif method == 'standard':
            text = text.lower()
            text = self._remove_punctuation(text)
            text = self._remove_special_chars(text)
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            return ' '.join(tokens)
            
        elif method == 'advanced':
            text = text.lower()
            text = self._remove_punctuation(text)
            text = self._remove_special_chars(text)
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            return ' '.join(tokens)
            
        elif method == 'bert':
            # Minimal preprocessing for BERT
            text = text.lower()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        return text
    
    def tokenize(self, text, method='word'):
        """
        Tokenize text using different methods:
        - 'word': standard word tokenization
        - 'char': character-level tokenization
        - 'subword': subword tokenization (simplified)
        """
        if method == 'word':
            return word_tokenize(text)
        elif method == 'char':
            return list(text)
        elif method == 'subword':
            # Simplified subword tokenization
            words = word_tokenize(text)
            subwords = []
            for word in words:
                if len(word) > 5:
                    # Split long words into subwords
                    subwords.extend([word[:3], word[3:]])
                else:
                    subwords.append(word)
            return subwords
        return word_tokenize(text)
    
    def _expand_contractions(self, text):
        """Expand contractions like don't to do not"""
        return contractions.fix(text)
    
    def _replace_emojis(self, text):
        """Replace emojis with their textual description"""
        return emoji.demojize(text, delimiters=(" ", " "))
    
    def _replace_slang(self, text):
        """Replace internet slang and abbreviations with their meanings"""
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in SLANG_DICT:
                words[i] = SLANG_DICT[word.lower()]
        return ' '.join(words)
    
    def _remove_punctuation(self, text):
        """Remove punctuation from text"""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def _remove_special_chars(self, text):
        """Remove special characters and numbers"""
        return re.sub(r'[^a-zA-Z\s]', '', text)
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(t) for t in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(t) for t in tokens] 