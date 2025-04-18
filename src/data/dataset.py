import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import pickle
import hashlib
import shutil
from tqdm import tqdm
from sklearn.utils import resample
from src.data.image_utils import preprocess_image, augment_image
from src.data.text_utils import preprocess_text, compute_text_features
from typing import List, Dict, Any, Tuple
from .dataset_utils import get_image_paths, standardize_metadata, create_standardized_df, validate_dataset
import re
import random
import logging
from glob import glob
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

# Import utility functions
from src.utils.text_utils import TextProcessor
from src.utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)

def get_fakeddit_image_path(article_id, images_dir):
    """
    Get the image path for a Fakeddit article based on the article ID
    
    Args:
        article_id: ID of the article
        images_dir: Directory containing Fakeddit images
        
    Returns:
        list: List of possible image paths
    """
    # In Fakeddit, the image name is the article ID with jpg extension
    image_paths = []
    
    # Standard path
    standard_path = os.path.join(images_dir, f"{article_id}.jpg")
    if os.path.exists(standard_path):
        image_paths.append(standard_path)
        
    # Sometimes there might be different extensions
    for ext in ['png', 'jpeg', 'gif', 'webp']:
        alt_path = os.path.join(images_dir, f"{article_id}.{ext}")
        if os.path.exists(alt_path):
            image_paths.append(alt_path)
    
    return image_paths

def get_fakenewsnet_image_paths(article_id, source, label, images_dir):
    """
    Get image paths for a FakeNewsNet article
    
    Args:
        article_id: ID of the article (e.g., gossipcop-1344)
        source: Source dataset (gossipcop or politifact)
        label: Article label (fake or real)
        images_dir: Base directory containing FakeNewsNet images
        
    Returns:
        list: List of image paths for the article
    """
    # Construct the directory path for the article's images
    # Based on instruction.md, the structure is:
    # images/fakenewsnet/[source]/[label]/[article_id]/[image files]
    article_dir = os.path.join(images_dir, source, label, article_id)
    
    # Check if directory exists
    if not os.path.exists(article_dir):
        return []
    
    # Get all image files in the directory
    image_paths = []
    for ext in ['jpg', 'png', 'jpeg', 'gif', 'webp', 'svg']:
        image_paths.extend(glob(os.path.join(article_dir, f"*.{ext}")))
    
    return image_paths

class DatasetProcessor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize dataset processor with configuration"""
        self.config = config
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor(config.get('image_processing', {}))
        self.raw_dir = config['data']['raw_dir']
        self.processed_dir = os.path.join(os.getcwd(), config['data']['processed_dir'])
        self.images_dir = config['data']['images_dir']
        self.cache_dir = os.path.join(os.getcwd(), config['data']['cache_dir'])
        
        print(f"Initializing DatasetProcessor with:")
        print(f"  Raw data dir: {self.raw_dir}")
        print(f"  Processed dir: {self.processed_dir}")
        print(f"  Images dir: {self.images_dir}")
        print(f"  Cache dir: {self.cache_dir}")
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up caching configuration
        self.use_cache = config['data'].get('cache_features', False)
        
        # Set up balanced sampling configuration
        self.balanced_sampling = config['data'].get('balanced_sampling', False)
        
        # Set dataset parameters
        self.max_text_length = config['data'].get('max_text_length', 128)
        self.max_vocab_size = config['data'].get('max_vocab_size', 20000)
        self.batch_size = config['training'].get('batch_size', 32)
        self.val_split = config['data'].get('val_split', 0.15)
        self.test_split = config['data'].get('test_split', 0.15)
        self.random_seed = config['data'].get('random_seed', 42)
        
        # Initialize tokenizer information
        self.word_index = None
        
    def process_datasets(self):
        """Process all datasets and save standardized data"""
        print("Processing datasets...")
        
        # Process Fakeddit dataset
        fakeddit_df = self.load_fakeddit()
        if not fakeddit_df.empty:
            self._save_processed_data(fakeddit_df, 'fakeddit')
        
        # Process FakeNewsNet dataset
        fakenewsnet_df = self.load_fakenewsnet()
        if not fakenewsnet_df.empty:
            self._save_processed_data(fakenewsnet_df, 'fakenewsnet')
        
        # Combine datasets if both are available
        if not fakeddit_df.empty and not fakenewsnet_df.empty:
            combined_df = pd.concat([fakeddit_df, fakenewsnet_df], ignore_index=True)
            self._save_processed_data(combined_df, 'combined')
            
        print("Dataset processing complete!")
    
    def _save_processed_data(self, df: pd.DataFrame, dataset_name: str):
        """Save processed dataset to CSV file"""
        output_path = os.path.join(self.processed_dir, f"{dataset_name}_processed.csv")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try to save the CSV file
        try:
            df.to_csv(output_path, index=False)
            print(f"Saved processed {dataset_name} dataset to {output_path}")
        except PermissionError as pe:
            print(f"Permission error saving {dataset_name} dataset to {output_path}: {pe}")
            print(f"Will continue without saving {dataset_name} dataset...")
            return
        
        # Save statistics
        stats_path = os.path.join(self.processed_dir, f"{dataset_name}_stats.txt")
        try:
            with open(stats_path, 'w') as f:
                f.write(f"{dataset_name.upper()} Dataset Statistics\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total samples: {len(df)}\n")
                f.write(f"Label distribution:\n{df['label'].value_counts()}\n")
                f.write(f"Average text length: {df['text'].str.len().mean():.2f}\n")
                f.write(f"Images available: {df['has_image'].sum()} ({df['has_image'].sum()/len(df)*100:.2f}%)\n")
                
                if 'metadata' in df.columns:
                    f.write("\nMetadata Statistics:\n")
                    sources = df['metadata'].apply(lambda x: x.get('source', '')).unique()
                    f.write(f"Number of unique sources: {len(sources)}\n")
                    
                    authors = set()
                    for authors_list in df['metadata'].apply(lambda x: x.get('authors', [])):
                        if isinstance(authors_list, list):
                            authors.update(authors_list)
                    f.write(f"Number of unique authors: {len(authors)}\n")
            
            print(f"Saved {dataset_name} statistics to {stats_path}")
        except PermissionError as pe:
            print(f"Permission error saving {dataset_name} statistics to {stats_path}: {pe}")
            print(f"Will continue without saving {dataset_name} statistics...")

    def load_fakeddit(self) -> pd.DataFrame:
        """Load and process Fakeddit dataset"""
        data = []
        
        # Check if files are specified in config
        if 'files' not in self.config['data']['fakeddit']:
            print("No Fakeddit files specified in config.")
            return pd.DataFrame()
        
        for file_path in self.config['data']['fakeddit']['files']:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            try:
                # Read TSV file with explicit column names
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
                
                # Print column names for debugging
                print(f"\nColumns in {os.path.basename(file_path)}:")
                print(df.columns.tolist())
                
                for _, row in df.iterrows():
                    try:
                        article_id = str(row['id'])
                        # Get image paths based on the article ID
                        image_paths = get_fakeddit_image_path(article_id, self.config['data']['fakeddit']['images_dir'])
                        
                        # Check if image exists
                        has_image = row.get('hasImage', False)
                        if has_image and not image_paths:
                            # Try to get image URL and download if needed
                            image_url = row.get('image_url', '')
                            if image_url:
                                print(f"Image for {article_id} not found locally but URL available: {image_url}")
                        
                        # Create title + selftext combination (if available)
                        title = str(row.get('clean_title', row.get('title', '')))
                        selftext = str(row.get('selftext', ''))
                        text = title + ' ' + selftext if selftext.strip() else title
                        
                        # Create standardized data entry with error handling
                        entry = {
                            'id': article_id,
                            'text': text,
                            'clean_text': text,  # Will be preprocessed later
                            'image_paths': image_paths,
                            'has_image': bool(image_paths) or has_image,
                            'label': 1 if row.get('2_way_label', 0) == 1 else 0,  # Convert to binary classification
                            'metadata': standardize_metadata({
                                'subreddit': row.get('subreddit', ''),
                                'author': row.get('author', ''),
                                'score': row.get('score', 0),
                                'num_comments': row.get('num_comments', 0),
                                'upvote_ratio': row.get('upvote_ratio', 0.0),
                                'created_utc': row.get('created_utc', ''),
                                'domain': row.get('domain', '')
                            }, 'fakeddit'),
                            'dataset_source': 'fakeddit',
                            'file_source': os.path.basename(file_path)
                        }
                        data.append(entry)
                    except Exception as e:
                        print(f"Error processing row in {file_path}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

        if not data:
            print("No valid Fakeddit data found")
            return pd.DataFrame()
            
        # Create standardized DataFrame
        df = create_standardized_df(data, 'fakeddit')
        
        # Print dataset statistics
        stats = validate_dataset(df, 'fakeddit')
        print("\nFakeddit Dataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Text length - Mean: {stats['text_length_stats']['mean']:.2f}, "
              f"Min: {stats['text_length_stats']['min']}, "
              f"Max: {stats['text_length_stats']['max']}")
        print(f"Images available: {stats['image_stats']['total_with_images']} "
              f"({stats['image_stats']['percentage_with_images']:.2f}%)")
        
        return df

    def load_fakenewsnet(self) -> pd.DataFrame:
        """Load and process FakeNewsNet dataset"""
        data = []
        
        # Check if base_dir is in config
        if 'fakenewsnet' not in self.config['data'] or 'base_dir' not in self.config['data']['fakenewsnet']:
            print("FakeNewsNet base_dir not specified in config. Skipping FakeNewsNet dataset.")
            return pd.DataFrame()
        
        base_dir = self.config['data']['fakenewsnet']['base_dir']
        images_dir = self.config['data']['fakenewsnet']['images_dir']
        news_content_file = self.config['data']['fakenewsnet'].get('news_content_file', 'news content.json')
        
        # Validate base directory exists
        if not os.path.exists(base_dir):
            print(f"Error: FakeNewsNet base directory not found: {base_dir}")
            return pd.DataFrame()
        
        # Set default sources and labels if not in config
        if 'sources' not in self.config['data']['fakenewsnet']:
            print("No sources specified for FakeNewsNet, using defaults ['gossipcop', 'politifact']")
            self.config['data']['fakenewsnet']['sources'] = ['gossipcop', 'politifact']
        
        if 'labels' not in self.config['data']['fakenewsnet']:
            print("No labels specified for FakeNewsNet, using defaults ['fake', 'real']")
            self.config['data']['fakenewsnet']['labels'] = ['fake', 'real']
        
        # Process each source (gossipcop and politifact)
        for source in self.config['data']['fakenewsnet']['sources']:
            source_dir = os.path.join(base_dir, source)
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory not found: {source_dir}")
                continue
            
            # Process each label (fake and real)
            for label in self.config['data']['fakenewsnet']['labels']:
                label_dir = os.path.join(source_dir, label)
                if not os.path.exists(label_dir):
                    print(f"Warning: Label directory not found: {label_dir}")
                    continue
                
                # Get all article directories
                article_dirs = [d for d in os.listdir(label_dir) 
                              if os.path.isdir(os.path.join(label_dir, d))]
                
                print(f"Processing {source}/{label}: {len(article_dirs)} articles")
                
                # Process each article
                for article_dir in article_dirs:
                    try:
                        # Construct path to news content JSON
                        json_path = os.path.join(label_dir, article_dir, news_content_file)
                        if not os.path.exists(json_path):
                            print(f"Warning: JSON file not found for {article_dir}: {json_path}")
                            continue
                        
                        # Load article data
                        with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                            article = json.load(f)
                            
                        # Get image paths for this article
                        image_paths = get_fakenewsnet_image_paths(article_dir, source, label, images_dir)
                        
                        # Combine title and text for full text
                        title = article.get('title', '')
                        text = article.get('text', '')
                        full_text = f"{title} {text}" if title else text
                        
                        # Create standardized data entry
                        data.append({
                            'id': f"{source}_{article_dir}",
                            'text': full_text,
                            'clean_text': full_text,  # Will be preprocessed later
                            'image_paths': image_paths,
                            'has_image': bool(image_paths),
                            'label': 1 if label == 'fake' else 0,
                            'metadata': standardize_metadata({
                                'url': article.get('url', ''),
                                'title': article.get('title', ''),
                                'authors': article.get('authors', []),
                                'keywords': article.get('keywords', []),
                                'publish_date': article.get('publish_date', ''),
                                'source': article.get('source', source),
                                'summary': article.get('summary', ''),
                                'top_img': article.get('top_img', '')
                            }, 'fakenewsnet'),
                            'dataset_source': 'fakenewsnet',
                            'file_source': f"{source}/{label}/{article_dir}"
                        })
                        
                    except Exception as e:
                        print(f"Error processing article {article_dir}: {e}")
                        continue
                        
        if not data:
            print("No valid FakeNewsNet data found")
            return pd.DataFrame()
            
        # Create standardized DataFrame
        df = create_standardized_df(data, 'fakenewsnet')
        
        # Validate dataset and print statistics
        stats = validate_dataset(df, 'fakenewsnet')
        print("\nFakeNewsNet Dataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Text length - Mean: {stats['text_length_stats']['mean']:.2f}, "
              f"Min: {stats['text_length_stats']['min']}, "
              f"Max: {stats['text_length_stats']['max']}")
        print(f"Images available: {stats['image_stats']['total_with_images']} "
              f"({stats['image_stats']['percentage_with_images']:.2f}%)")
        
        return df

    def preprocess_dataset(self, df=None):
        """Preprocess the combined dataset for training"""
        if df is None:
            # Load from saved file if not provided
            combined_path = os.path.join(self.processed_dir, 'combined_processed.csv')
            if os.path.exists(combined_path):
                df = pd.read_csv(combined_path)
            else:
                df = self.combine_datasets()
        
        # Apply balanced sampling if enabled
        df = self.apply_balanced_sampling(df)
        
        # Process text data
        print("Processing text data...")
        df['processed_text'] = df['clean_text'].apply(
            lambda x: preprocess_text(
                x,
                max_length=self.max_text_length,
                remove_stopwords=True,
                use_lemmatization=True,
                use_stemming=False,
                expand_contractions=True,
                convert_slang=True
            )
        )
        
        # Extract text features for the model
        print("Extracting text features...")
        df['text_features'] = df['processed_text'].apply(compute_text_features)
        
        # Process image data
        print("Processing image data...")
        def process_image(row):
            image_paths = row['image_paths']
            if not image_paths:
                return None
                
            # Try each image path until we find a valid one
            for img_path in image_paths:
                if os.path.exists(img_path):
                    try:
                        # Return the first valid image path
                        # The actual preprocessing will be done during training
                        return img_path
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")
                        continue
            return None
        
        df['processed_image_path'] = df.apply(process_image, axis=1)
        
        # Create a sample of preprocessed images to validate the process
        print("Creating sample preprocessed images...")
        sample_images_dir = os.path.join(self.processed_dir, 'sample_images')
        try:
            os.makedirs(sample_images_dir, exist_ok=True)
            
            # Process a small sample of images to verify preprocessing
            sample_size = min(10, len(df[df['processed_image_path'].notna()]))
            sample_df = df[df['processed_image_path'].notna()].sample(sample_size)
            
            for _, row in sample_df.iterrows():
                try:
                    img_path = row['processed_image_path']
                    if img_path and os.path.exists(img_path):
                        # Check file permissions before proceeding
                        if not os.access(img_path, os.R_OK):
                            print(f"Warning: No read permission for image {img_path}")
                            continue
                            
                        # Preprocess image with configuration
                        img_array = preprocess_image(
                            img_path,
                            target_size=self.config['model']['image']['input_shape'][:2]
                        )
                        
                        # Create sample path and check write permissions for directory
                        sample_path = os.path.join(sample_images_dir, f"{row['id']}_preprocessed.npy")
                        sample_dir = os.path.dirname(sample_path)
                        if not os.access(sample_dir, os.W_OK):
                            print(f"Warning: No write permission for directory {sample_dir}")
                            continue
                            
                        # Save preprocessed image as numpy array
                        try:
                            np.save(sample_path, img_array)
                        except PermissionError as pe:
                            print(f"Permission error saving sample to {sample_path}: {pe}")
                            continue
                        
                        # Apply augmentation (if enabled) and save
                        if self.config['data'].get('augment_images', False):
                            aug_config = {
                                'flip': True,
                                'rotation': 0.1,
                                'zoom': 0.1,
                                'contrast': 0.1,
                                'brightness': 0.1
                            }
                            aug_img = augment_image(img_array, aug_config)
                            aug_path = os.path.join(sample_images_dir, f"{row['id']}_augmented.npy")
                            try:
                                np.save(aug_path, aug_img)
                            except PermissionError as pe:
                                print(f"Permission error saving augmented sample to {aug_path}: {pe}")
                except Exception as e:
                    print(f"Error saving sample image for ID {row['id']}: {e}")
        except PermissionError as e:
            print(f"Permission error creating sample images directory: {e}")
            print("Skipping sample image creation due to permission issues")
        
        # Save preprocessed dataset
        preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
        try:
            df.to_csv(preprocessed_path, index=False)
            print(f"Saved preprocessed dataset to {preprocessed_path}")
        except PermissionError as pe:
            print(f"Permission error saving preprocessed dataset to {preprocessed_path}: {pe}")
            print("Will continue without saving preprocessed dataset...")
        
        # Print statistics
        print(f"Preprocessed dataset size: {len(df)} records")
        print(f"Records with valid text: {df['processed_text'].notna().sum()} ({df['processed_text'].notna().sum()/len(df)*100:.2f}%)")
        print(f"Records with valid images: {df['processed_image_path'].notna().sum()} ({df['processed_image_path'].notna().sum()/len(df)*100:.2f}%)")
        
        return df
    
    def combine_datasets(self):
        """Combine standardized datasets"""
        fakeddit_path = os.path.join(self.processed_dir, 'fakeddit_processed.csv')
        fakenewsnet_path = os.path.join(self.processed_dir, 'fakenewsnet_processed.csv')
        
        dfs_to_combine = []
        
        # Load Fakeddit data if available
        if os.path.exists(fakeddit_path):
            fakeddit_df = pd.read_csv(fakeddit_path)
            if not fakeddit_df.empty:
                dfs_to_combine.append(fakeddit_df)
        else:
            fakeddit_df = self.load_fakeddit()
            if not fakeddit_df.empty:
                dfs_to_combine.append(fakeddit_df)
                
        # Load FakeNewsNet data if available        
        if os.path.exists(fakenewsnet_path):
            fakenewsnet_df = pd.read_csv(fakenewsnet_path)
            if not fakenewsnet_df.empty:
                dfs_to_combine.append(fakenewsnet_df)
        else:
            fakenewsnet_df = self.load_fakenewsnet()
            if not fakenewsnet_df.empty:
                dfs_to_combine.append(fakenewsnet_df)
            
        # Check if datasets are empty
        if not dfs_to_combine:
            raise ValueError("No valid data files found. Please check file paths and formats.")
            
        # Combine datasets
        combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        
        # Save combined dataset
        combined_path = os.path.join(self.processed_dir, 'combined_processed.csv')
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(combined_path), exist_ok=True)
            combined_df.to_csv(combined_path, index=False)
            print(f"Saved combined dataset to {combined_path}")
        except PermissionError as pe:
            print(f"Permission error saving combined dataset to {combined_path}: {pe}")
            print("Will continue without saving combined dataset...")
        
        print(f"Combined dataset created with {len(combined_df)} records")
        
        # Print label distribution
        label_counts = combined_df['label'].value_counts()
        print("Label distribution in combined dataset:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} records ({count/len(combined_df)*100:.2f}%)")
        
        return combined_df
    
    def apply_balanced_sampling(self, df):
        """Apply balanced sampling to handle class imbalance"""
        if not self.balanced_sampling:
            return df
        
        print("Applying balanced sampling...")
        
        # Get counts for each class
        label_counts = df['label'].value_counts()
        
        # Determine the number of samples to use (use the size of the smallest class for balance)
        min_class_size = label_counts.min()
        balanced_samples = []
        
        for label, count in label_counts.items():
            # If this class has more samples than we need, downsample
            class_df = df[df['label'] == label]
            if count > min_class_size:
                # Downsample this class
                downsampled = resample(
                    class_df,
                    replace=False,
                    n_samples=min_class_size,
                    random_state=42
                )
                balanced_samples.append(downsampled)
            else:
                # Keep all samples for this class
                balanced_samples.append(class_df)
        
        # Combine all balanced classes
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        
        print(f"Balanced dataset size: {len(balanced_df)} records")
        # Print label distribution
        label_counts = balanced_df['label'].value_counts()
        print("Label distribution after balancing:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} records ({count/len(balanced_df)*100:.2f}%)")
        
        return balanced_df
    
    def _get_cache_path(self, dataset_hash):
        """Generate a cache file path based on dataset hash"""
        return os.path.join(self.cache_dir, f"features_{dataset_hash}.pkl")
    
    def _compute_dataset_hash(self, df):
        """Compute a hash of the dataset for caching purposes"""
        # Use a subset of columns to compute the hash
        hash_columns = ['id', 'processed_text', 'processed_image_path', 'label']
        hash_df = df[hash_columns].copy()
        
        # Convert to string and hash
        df_str = hash_df.to_string()
        return hashlib.md5(df_str.encode()).hexdigest()
    
    def create_tf_datasets(self, df=None):
        """Create TensorFlow datasets for training, validation, and testing"""
        if df is None:
            preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
            if os.path.exists(preprocessed_path):
                df = pd.read_csv(preprocessed_path)
            else:
                df = self.preprocess_dataset()
        
        # Compute dataset hash for caching
        dataset_hash = self._compute_dataset_hash(df)
        cache_path = self._get_cache_path(dataset_hash)
        
        # Check if cached features exist
        if self.use_cache and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                features_list = cache_data['features']
                labels_list = cache_data['labels']
                tokenizer = cache_data['tokenizer']
        else:
            # Create a vocabulary from text data
            from tensorflow.keras.preprocessing.text import Tokenizer
            tokenizer = Tokenizer(num_words=self.max_vocab_size)
            tokenizer.fit_on_texts(df['processed_text'].fillna(''))
            
            # Save the tokenizer
            tokenizer_path = os.path.join(self.processed_dir, 'tokenizer.pickle')
            try:
                with open(tokenizer_path, 'wb') as handle:
                    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except PermissionError as pe:
                print(f"Permission error saving tokenizer to {tokenizer_path}: {pe}")
                print("Continuing without saving tokenizer...")
            
            # Function to create features and labels
            def create_features(row):
                # Text features
                text = row['processed_text'] if pd.notna(row['processed_text']) else ''
                text_sequence = tokenizer.texts_to_sequences([text])[0]
                # Pad sequence to fixed length
                if len(text_sequence) > self.max_text_length:
                    text_sequence = text_sequence[:self.max_text_length]
                else:
                    text_sequence = text_sequence + [0] * (self.max_text_length - len(text_sequence))
                
                # Image features - load and preprocess image if available
                image_features = np.zeros(self.config['model']['image']['input_shape'])
                if pd.notna(row['processed_image_path']) and os.path.exists(str(row['processed_image_path'])):
                    # Check file permissions before processing
                    img_path = str(row['processed_image_path'])
                    if not os.access(img_path, os.R_OK):
                        print(f"Warning: No read permission for image {img_path}")
                    else:
                        try:
                            image_features = preprocess_image(
                                img_path,
                                target_size=self.config['model']['image']['input_shape'][:2]
                            )
                        except Exception as e:
                            print(f"Error preprocessing image {img_path}: {e}")
                
                # Metadata features - extract key metadata and convert to numerical format
                metadata_features = np.zeros(10)  # Fixed size for metadata features
                
                # Handle metadata which could be a string (from CSV) or dict
                metadata = row['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                # Extract numerical metadata if available
                if isinstance(metadata, dict):
                    # Example metadata extraction - customize based on your needs
                    if 'score' in metadata:
                        try:
                            metadata_features[0] = float(metadata['score'])
                        except:
                            pass
                    if 'upvote_ratio' in metadata:
                        try:
                            metadata_features[1] = float(metadata['upvote_ratio'])
                        except:
                            pass
                    if 'num_comments' in metadata:
                        try:
                            metadata_features[2] = float(metadata['num_comments'])
                        except:
                            pass
                
                return {
                    'text': text_sequence,
                    'image': image_features,
                    'metadata': metadata_features
                }, row['label']
            
            # Apply the function to create features for each row
            features_list = []
            labels_list = []
            
            print("Creating features...")
            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    features, label = create_features(row)
                    features_list.append(features)
                    labels_list.append(int(label))
                except Exception as e:
                    print(f"Error creating features for row with ID {row.get('id', 'unknown')}: {e}")
            
            # Cache the features if caching is enabled
            if self.use_cache:
                print(f"Caching features to {cache_path}")
                cache_dir = os.path.dirname(cache_path)
                
                # Check if cache directory exists and is writable
                if not os.path.exists(cache_dir):
                    try:
                        os.makedirs(cache_dir, exist_ok=True)
                    except PermissionError as pe:
                        print(f"Permission error creating cache directory {cache_dir}: {pe}")
                        print("Continuing without caching...")
                        self.use_cache = False
                
                # Check if we have write access to the cache directory
                if self.use_cache and not os.access(cache_dir, os.W_OK):
                    print(f"No write permission for cache directory {cache_dir}")
                    print("Continuing without caching...")
                    self.use_cache = False
                
                # Try to save the cache
                if self.use_cache:
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump({
                                'features': features_list,
                                'labels': labels_list,
                                'tokenizer': tokenizer
                            }, f)
                    except PermissionError as pe:
                        print(f"Permission error saving cache to {cache_path}: {pe}")
                        print("Continuing without caching...")
        
        # Split into train, validation, and test sets
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        
        # First split: separate test set
        train_val_features, test_features, train_val_labels, test_labels = train_test_split(
            features_list, labels_list, test_size=1-train_ratio-val_ratio, random_state=42, stratify=labels_list
        )
        
        # Second split: separate train and validation sets
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_val_features, train_val_labels, 
            test_size=val_ratio/(train_ratio+val_ratio), 
            random_state=42,
            stratify=train_val_labels
        )
        
        print(f"Train set: {len(train_features)} samples")
        print(f"Validation set: {len(val_features)} samples")
        print(f"Test set: {len(test_features)} samples")
        
        # Create TensorFlow datasets
        def create_tf_dataset(features, labels, is_training=False):
            def gen():
                for f, l in zip(features, labels):
                    yield f, l
            
            dataset = tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    {
                        'text': tf.TensorSpec(shape=(self.max_text_length,), dtype=tf.int32),
                        'image': tf.TensorSpec(shape=self.config['model']['image']['input_shape'], dtype=tf.float32),
                        'metadata': tf.TensorSpec(shape=(10,), dtype=tf.float32)
                    },
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=min(len(features), 10000))
            
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        train_dataset = create_tf_dataset(train_features, train_labels, is_training=True)
        val_dataset = create_tf_dataset(val_features, val_labels)
        test_dataset = create_tf_dataset(test_features, test_labels)
        
        return train_dataset, val_dataset, test_dataset, tokenizer.word_index
    
    def create_cross_dataset_validation_set(self, df=None):
        """Create a validation set from a different dataset source than training"""
        if df is None:
            preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
            if os.path.exists(preprocessed_path):
                df = pd.read_csv(preprocessed_path)
            else:
                df = self.preprocess_dataset()
        
        # Check if there are multiple dataset sources
        sources = df['dataset_source'].unique()
        if len(sources) <= 1:
            print("Warning: Only one dataset source found, cannot create cross-dataset validation")
            return None, None
        
        # Choose one source for validation and the rest for training
        val_source = sources[0]
        train_sources = sources[1:]
        
        print(f"Creating cross-dataset validation set: Using {val_source} for validation and {train_sources} for training")
        
        train_df = df[df['dataset_source'].isin(train_sources)]
        val_df = df[df['dataset_source'] == val_source]
        
        print(f"Training set size: {len(train_df)} samples")
        print(f"Cross-dataset validation set size: {len(val_df)} samples")
        
        return train_df, val_df
