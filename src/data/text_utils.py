import re
import string
import nltk
import pandas as pd
import contractions
import emoji
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, TweetTokenizer, MWETokenizer
from collections import Counter
from textblob import TextBlob

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
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
tweet_tokenizer = TweetTokenizer()

# Dictionary of common slang and abbreviations (expanded)
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
    "gotta": "got to",
    # Additional slang and social media abbreviations
    "abt": "about",
    "af": "as f*ck",
    "asap": "as soon as possible",
    "bae": "before anyone else",
    "bf": "boyfriend",
    "dm": "direct message",
    "fb": "facebook",
    "fomo": "fear of missing out",
    "ftw": "for the win",
    "gf": "girlfriend",
    "gr8": "great",
    "ht": "hat tip",
    "icymi": "in case you missed it",
    "ig": "instagram",
    "ikr": "i know right",
    "jk": "just kidding",
    "lmao": "laughing my a** off",
    "lmfao": "laughing my f*cking a** off",
    "nbd": "no big deal",
    "nsfw": "not safe for work",
    "nvm": "never mind",
    "omw": "on my way",
    "ootd": "outfit of the day",
    "pm": "private message",
    "rn": "right now",
    "rt": "retweet",
    "smh": "shaking my head",
    "sus": "suspicious",
    "tbt": "throwback thursday",
    "tfw": "that feeling when",
    "tmi": "too much information",
    "ttyl": "talk to you later",
    "ty": "thank you",
    "yolo": "you only live once",
    "yt": "youtube",
    "tl;dr": "too long didn't read",
    "fwiw": "for what it's worth",
    "hmu": "hit me up",
    "stfu": "shut the f*ck up",
    "tldr": "too long didn't read"
}

# Multi-word expressions to keep together during tokenization
multiword_expressions = [
    ('fake', 'news'),
    ('climate', 'change'),
    ('artificial', 'intelligence'),
    ('machine', 'learning'),
    ('deep', 'learning'),
    ('united', 'states'),
    ('covid', '19'),
    ('social', 'media'),
    ('new', 'york'),
    ('los', 'angeles'),
    ('black', 'lives', 'matter'),
    ('fake', 'media'),
    ('white', 'house'),
    ('supreme', 'court'),
    ('donald', 'trump'),
    ('joe', 'biden'),
    ('elon', 'musk'),
    ('facebook', 'meta')
]

# Initialize multi-word tokenizer
mwe_tokenizer = MWETokenizer(multiword_expressions)

def extract_named_entities(text):
    """
    Extract named entities from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with counts of different entity types
    """
    try:
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        named_entities = nltk.ne_chunk(pos_tags)
        
        entity_counts = {
            'PERSON': 0,
            'ORGANIZATION': 0,
            'LOCATION': 0,
            'DATE': 0,
            'TIME': 0,
            'MONEY': 0,
            'PERCENT': 0,
            'OTHER': 0
        }
        
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                label = chunk.label()
                if label in entity_counts:
                    entity_counts[label] += 1
                else:
                    entity_counts['OTHER'] += 1
        
        return entity_counts
    except Exception as e:
        return {
            'PERSON': 0,
            'ORGANIZATION': 0,
            'LOCATION': 0,
            'DATE': 0,
            'TIME': 0,
            'MONEY': 0,
            'PERCENT': 0,
            'OTHER': 0
        }

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with sentiment polarity and subjectivity
    """
    try:
        analysis = TextBlob(text)
        return {
            'polarity': float(analysis.sentiment.polarity),
            'subjectivity': float(analysis.sentiment.subjectivity)
        }
    except Exception as e:
        return {
            'polarity': 0.0,
            'subjectivity': 0.0
        }

def extract_pos_tags(text):
    """
    Extract part-of-speech distribution from text
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with POS tag counts
    """
    try:
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        tag_counts = Counter([tag for _, tag in pos_tags])
        return {
            'noun_count': sum(tag_counts.get(tag, 0) for tag in ['NN', 'NNS', 'NNP', 'NNPS']),
            'verb_count': sum(tag_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']),
            'adj_count': sum(tag_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS']),
            'adv_count': sum(tag_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS']),
            'pron_count': sum(tag_counts.get(tag, 0) for tag in ['PRP', 'PRP$', 'WP', 'WP$'])
        }
    except Exception as e:
        return {
            'noun_count': 0,
            'verb_count': 0,
            'adj_count': 0,
            'adv_count': 0,
            'pron_count': 0
        }

def preprocess_text(text, max_length=512, remove_stopwords=True, 
                   use_lemmatization=True, use_stemming=False, 
                   expand_contractions=True, convert_slang=True,
                   use_mwe=True, use_twitter_tokenizer=False):
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
        use_mwe (bool): Whether to use multi-word expression tokenization
        use_twitter_tokenizer (bool): Whether to use Twitter-optimized tokenizer
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Handle empty or non-string inputs
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions if enabled (e.g., "don't" -> "do not")
        if expand_contractions:
            try:
                text = contractions.fix(text)
            except Exception as e:
                # Fall back to basic handling if contractions library fails
                text = text.replace("n't", " not")
                text = text.replace("'ve", " have")
                text = text.replace("'ll", " will")
                text = text.replace("'m", " am")
                text = text.replace("'re", " are")
                text = text.replace("'s", " is")
                text = text.replace("'d", " would")
        
        # Convert emoji to text
        try:
            text = emoji.demojize(text)
            text = text.replace(":", "").replace("_", " ")
        except Exception as e:
            pass
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Replace email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Replace phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' PHONE ', text)
        
        # Replace numbers
        text = re.sub(r'\d+', ' NUM ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        try:
            # Tokenize based on selected method
            if use_twitter_tokenizer:
                tokens = tweet_tokenizer.tokenize(text)
            else:
                tokens = word_tokenize(text)
            
            # Apply multi-word expressions if enabled
            if use_mwe:
                tokens = mwe_tokenizer.tokenize(tokens)
            
            # Convert slang and abbreviations
            if convert_slang:
                tokens = [slang_dict.get(token, token) for token in tokens]
                # Retokenize after slang conversion as some slang maps to multiple words
                tokens = [word for token in tokens for word in (word_tokenize(token) if isinstance(token, str) and ' ' in token else [token])]
            
            # Remove stopwords if enabled
            if remove_stopwords:
                tokens = [token for token in tokens if token not in stop_words]
            
            # Apply stemming or lemmatization
            if use_stemming:
                tokens = [stemmer.stem(token) for token in tokens]
            elif use_lemmatization:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            # Truncate to max_length
            if max_length > 0:
                tokens = tokens[:max_length]
            
            # Join tokens back to text
            processed_text = ' '.join(tokens)
            
            # Final cleanup of any remaining multiple spaces
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            return processed_text
            
        except Exception as e:
            return text  # Return cleaned text without tokenization if it fails
    
    except Exception as e:
        if isinstance(text, str):
            return text  # Return original text if preprocessing fails
        return ""  # Return empty string for non-string inputs
    
def compute_text_features(text):
    """
    Compute comprehensive text features for enhanced analysis
    
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
            'has_email': False,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'noun_count': 0,
            'verb_count': 0,
            'adj_count': 0,
            'adv_count': 0,
            'pron_count': 0,
            'entity_person_count': 0,
            'entity_org_count': 0,
            'entity_loc_count': 0,
            'lexical_diversity': 0.0
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
        
        # Advanced NLP features
        sentiment = analyze_sentiment(text)
        pos_counts = extract_pos_tags(text)
        named_entities = extract_named_entities(text)
        
        # Lexical diversity (unique words / total words)
        words = re.findall(r'\w+', text.lower())
        unique_words = set(words)
        lexical_diversity = len(unique_words) / max(1, len(words))
        
        # Combine all features
        features = {
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
            'has_email': has_email,
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
            'noun_count': pos_counts['noun_count'],
            'verb_count': pos_counts['verb_count'],
            'adj_count': pos_counts['adj_count'],
            'adv_count': pos_counts['adv_count'],
            'pron_count': pos_counts['pron_count'],
            'entity_person_count': named_entities['PERSON'],
            'entity_org_count': named_entities['ORGANIZATION'],
            'entity_loc_count': named_entities['LOCATION'],
            'lexical_diversity': lexical_diversity
        }
        
        return features
        
    except Exception as e:
        # Return default values if feature extraction fails
        return {
            'char_count': len(text) if isinstance(text, str) else 0,
            'word_count': len(text.split()) if isinstance(text, str) else 0,
            'sentence_count': 1,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0,
            'has_urls': False,
            'has_numbers': False,
            'has_email': False,
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'noun_count': 0,
            'verb_count': 0,
            'adj_count': 0,
            'adv_count': 0,
            'pron_count': 0,
            'entity_person_count': 0,
            'entity_org_count': 0,
            'entity_loc_count': 0,
            'lexical_diversity': 0.0
        }

def create_text_embeddings(text, method='tf-idf', vocabulary=None, max_features=10000):
    """
    Create vector embeddings for text using different methods
    
    Args:
        text (str): Input text
        method (str): Embedding method ('tf-idf', 'count', 'binary')
        vocabulary (dict): Optional pre-defined vocabulary
        max_features (int): Maximum number of features for the vectorizer
        
    Returns:
        numpy.ndarray: Text embedding vector
    """
    if not isinstance(text, str) or pd.isna(text) or not text.strip():
        # Return zero vector of appropriate size
        if method == 'tf-idf' or method == 'count' or method == 'binary':
            return np.zeros(max_features)
        return np.zeros(300)  # Default for word2vec and other methods
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        
        if method == 'tf-idf':
            vectorizer = TfidfVectorizer(max_features=max_features, vocabulary=vocabulary)
            if vocabulary is None:
                # Fit and transform if no vocabulary provided
                vector = vectorizer.fit_transform([text]).toarray()[0]
            else:
                # Just transform if vocabulary is provided
                vector = vectorizer.transform([text]).toarray()[0]
            return vector
            
        elif method == 'count':
            vectorizer = CountVectorizer(max_features=max_features, vocabulary=vocabulary)
            if vocabulary is None:
                vector = vectorizer.fit_transform([text]).toarray()[0]
            else:
                vector = vectorizer.transform([text]).toarray()[0]
            return vector
            
        elif method == 'binary':
            vectorizer = CountVectorizer(max_features=max_features, vocabulary=vocabulary, binary=True)
            if vocabulary is None:
                vector = vectorizer.fit_transform([text]).toarray()[0]
            else:
                vector = vectorizer.transform([text]).toarray()[0]
            return vector
            
        else:
            # Default to zeros if method not supported
            return np.zeros(max_features)
            
    except Exception as e:
        # Return zero vector on error
        return np.zeros(max_features) 