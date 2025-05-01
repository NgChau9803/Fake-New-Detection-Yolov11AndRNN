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
        
        # Convert all paths to absolute paths for consistency
        base_path = os.getcwd()
        self.raw_dir = os.path.join(base_path, config['data']['raw_dir'])
        self.processed_dir = os.path.join(base_path, config['data']['processed_dir'])
        self.images_dir = os.path.join(base_path, config['data']['images_dir'])
        self.cache_dir = os.path.join(base_path, config['data']['cache_dir'])
        self.force_reprocess = config['data'].get('force_reprocess', False)
        
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
        self.train_ratio = config['data'].get('train_ratio', 0.7)
        self.val_split = config['data'].get('val_split', 0.15)
        self.test_split = config['data'].get('test_split', 0.15)
        self.random_seed = config['data'].get('random_seed', 42)
        
        # Text preprocessing options
        self.text_preproc_config = config['data'].get('text_preprocessing', {
            'remove_stopwords': True,
            'use_lemmatization': True,
            'use_stemming': False,
            'expand_contractions': True,
            'convert_slang': True
        })
        
        # Image preprocessing options
        self.image_preproc_config = config['data'].get('image_preprocessing', {
            'normalization': 'standard',
            'resize_method': 'bilinear',
            'apply_clahe': False
        })
        
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
                    
                    # Function to extract source from metadata (which might be string or dict)
                    def get_source(metadata):
                        if isinstance(metadata, dict):
                            return metadata.get('source', '')
                        elif isinstance(metadata, str):
                            try:
                                # Try to parse as JSON
                                metadata_dict = json.loads(metadata)
                                return metadata_dict.get('source', '')
                            except:
                                return ''
                        return ''
                    
                    # Get unique sources
                    try:
                        sources = df['metadata'].apply(get_source).unique()
                        f.write(f"Number of unique sources: {len(sources)}\n")
                    except Exception as e:
                        f.write(f"Error analyzing sources: {str(e)}\n")
                    
                    # Function to extract authors from metadata
                    def get_authors(metadata):
                        if isinstance(metadata, dict):
                            return metadata.get('authors', [])
                        elif isinstance(metadata, str):
                            try:
                                # Try to parse as JSON
                                metadata_dict = json.loads(metadata)
                                return metadata_dict.get('authors', [])
                            except:
                                return []
                        return []
                    
                    # Get unique authors
                    try:
                        authors = set()
                        for authors_list in df['metadata'].apply(get_authors):
                            if isinstance(authors_list, list):
                                authors.update(authors_list)
                        f.write(f"Number of unique authors: {len(authors)}\n")
                    except Exception as e:
                        f.write(f"Error analyzing authors: {str(e)}\n")
            
            print(f"Saved {dataset_name} statistics to {stats_path}")
        except PermissionError as pe:
            print(f"Permission error saving {dataset_name} statistics to {stats_path}: {pe}")
            print(f"Will continue without saving {dataset_name} statistics...")

    def load_fakeddit(self, subset: str = 'train', sample_size: int = None) -> pd.DataFrame:
        """
        Process Fakeddit dataset
        subset: train, val, or test
        """
        output_path = os.path.join(self.processed_dir, f"fakeddit_{subset}_processed.csv")
        
        # Check if processed file exists
        if os.path.exists(output_path) and not self.force_reprocess:
            print(f"Loading processed Fakeddit {subset} dataset from {output_path}")
            try:
                df = pd.read_csv(output_path)
                print(f"Fakeddit {subset} dataset loaded: {len(df)} samples")
                if sample_size and sample_size < len(df):
                    df = df.sample(sample_size, random_state=42)
                    print(f"Sampled {sample_size} examples from Fakeddit {subset}")
                return df
            except Exception as e:
                print(f"Error loading processed file {output_path}: {e}")
                print("Will reprocess the dataset...")
        
        # Map the subset to the corresponding file in the config
        subset_to_file = {
            'train': 'multimodal_train.tsv',
            'val': 'multimodal_validate.tsv',
            'test': 'multimodal_test_public.tsv'
        }
        
        tsv_file = subset_to_file.get(subset, 'multimodal_train.tsv')
        tsv_path = os.path.join(self.raw_dir, 'fakeddit', tsv_file)
        
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Fakeddit {subset} file not found at {tsv_path}")
        
        # Load TSV data
        print(f"Loading Fakeddit {subset} dataset from {tsv_path}")
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Standardize column names and content
        standardized_df = pd.DataFrame()
        standardized_df['id'] = df['id']
        standardized_df['text'] = df['clean_title'].fillna('')
        if 'title' in df.columns:
            standardized_df['text'] = df['title'].fillna('') + ' ' + standardized_df['text']
        
        # Handle image paths
        standardized_df['image_path'] = df.apply(
            lambda row: os.path.join(os.path.join(self.images_dir, 'fakeddit/public_image_set'), 
                                   f"{row['id']}.jpg") if row.get('hasImage', False) else None,
            axis=1
        )
        standardized_df['has_image'] = df.get('hasImage', False)
        
        # Create label based on 2way_label (0: real, 1: fake)
        standardized_df['label'] = df['2_way_label']
        
        # Add metadata
        standardized_df['metadata'] = df.apply(
            lambda row: {
                'source': row.get('domain', ''),
                'authors': [],
                'publish_date': '',
                'subreddit': row.get('subreddit', ''),
                'upvote_ratio': row.get('upvote_ratio', 0),
                'score': row.get('score', 0),
                'num_comments': row.get('num_comments', 0)
            }, axis=1
        )
        
        # Add dataset source information
        standardized_df['dataset_source'] = 'fakeddit'
        
        # Print dataset statistics
        print(f"Fakeddit {subset} dataset:")
        print(f"Total samples: {len(standardized_df)}")
        print(f"Label distribution: {standardized_df['label'].value_counts()}")
        print(f"Images available: {standardized_df['has_image'].sum()} ({standardized_df['has_image'].sum()/len(standardized_df)*100:.2f}%)")
        
        # Sample if needed
        if sample_size and sample_size < len(standardized_df):
            standardized_df = standardized_df.sample(sample_size, random_state=42)
            print(f"Sampled {sample_size} examples from Fakeddit {subset}")
        
        # Save processed data
        self._save_processed_data(standardized_df, f"fakeddit_{subset}")
        
        return standardized_df

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
                remove_stopwords=self.text_preproc_config['remove_stopwords'],
                use_lemmatization=self.text_preproc_config['use_lemmatization'],
                use_stemming=self.text_preproc_config['use_stemming'],
                expand_contractions=self.text_preproc_config['expand_contractions'],
                convert_slang=self.text_preproc_config['convert_slang']
            )
        )
        
        # Extract text features for the model
        print("Extracting text features...")
        df['text_features'] = df['processed_text'].apply(compute_text_features)
        
        # Process image data
        print("Processing image data...")
        def process_image(row):
            # First check if we have a single image_path (from Fakeddit)
            if 'image_path' in row and pd.notna(row['image_path']):
                img_path = row['image_path']
                if os.path.exists(img_path):
                    return img_path
            
            # If no valid image_path, try image_paths (from FakeNewsNet)
            if 'image_paths' not in row or pd.isna(row['image_paths']):
                return None
            
            image_paths = row['image_paths']
            
            # Handle if image_paths is a string (from CSV loading)
            if isinstance(image_paths, str):
                try:
                    # Try to load as JSON if it's a string representation of a list
                    if image_paths.startswith('[') and image_paths.endswith(']'):
                        image_paths = json.loads(image_paths)
                        
                        # Now we have a list, check each path
                        for img_path in image_paths:
                            if isinstance(img_path, str) and os.path.exists(img_path):
                                return img_path
                    else:
                        # Single path as string
                        if os.path.exists(image_paths):
                            return image_paths
                except:
                    # If it's not valid JSON but a single path
                    if os.path.exists(image_paths):
                        return image_paths
            
            # Handle if image_paths is already a list
            elif isinstance(image_paths, list):
                for img_path in image_paths:
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        return img_path
            
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
                print(f"Loaded preprocessed dataset with {len(df)} samples")
            else:
                df = self.preprocess_dataset()
        
        # Compute dataset hash for caching
        dataset_hash = self._compute_dataset_hash(df)
        cache_path = self._get_cache_path(dataset_hash)
        
        # Check if cached features exist
        if self.use_cache and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    features_list = cache_data['features']
                    labels_list = cache_data['labels']
                    tokenizer = cache_data['tokenizer']
                    print(f"Successfully loaded cached features: {len(features_list)} samples")
            except Exception as e:
                print(f"Error loading cached features: {e}")
                print("Proceeding with feature extraction...")
                tokenizer, features_list, labels_list = self._extract_features(df)
        else:
            print("No cache found or cache disabled. Extracting features...")
            tokenizer, features_list, labels_list = self._extract_features(df)
        
        # Determine train/val/test split ratios
        train_ratio = self.train_ratio
        val_ratio = self.val_split
        
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
        
    def _extract_features(self, df):
        """Extract features from dataset in batches to reduce memory usage"""
        # Create a vocabulary from text data
        from tensorflow.keras.preprocessing.text import Tokenizer
        print("Creating tokenizer...")
        tokenizer = Tokenizer(num_words=self.max_vocab_size)
        tokenizer.fit_on_texts(df['processed_text'].fillna(''))
        
        # Save the tokenizer
        tokenizer_path = os.path.join(self.processed_dir, 'tokenizer.pickle')
        try:
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved tokenizer to {tokenizer_path}")
        except PermissionError as pe:
            print(f"Permission error saving tokenizer to {tokenizer_path}: {pe}")
            print("Continuing without saving tokenizer...")
        
        # Process in smaller batches to reduce memory pressure
        features_list = []
        labels_list = []
        # Use smaller batch size to reduce memory pressure
        batch_size = 300  # Reduced from 1000 to 300
        
        print(f"Processing {len(df)} samples in batches of {batch_size}...")
        
        # Get system memory info for monitoring
        try:
            import psutil
            def get_memory_usage():
                process = psutil.Process()
                memory_info = process.memory_info()
                return f"Memory usage: {memory_info.rss / (1024 * 1024):.1f} MB"
        except ImportError:
            def get_memory_usage():
                return "Memory usage tracking not available"
        
        for i in range(0, len(df), batch_size):
            # Print memory usage before processing batch
            print(f"Before batch {i//batch_size + 1}: {get_memory_usage()}")
            
            batch_df = df.iloc[i:i+batch_size]
            batch_features = []
            batch_labels = []
            
            print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            # Process each row in the batch
            success_count = 0
            error_count = 0
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {i//batch_size + 1}"):
                try:
                    # Extract features for this sample
                    features, label = self._create_features_for_sample(row, tokenizer)
                    batch_features.append(features)
                    batch_labels.append(label)
                    success_count += 1
                except Exception as e:
                    # Handle errors gracefully - log but continue
                    print(f"Error processing row {row.get('id', 'unknown')}: {e}")
                    error_count += 1
                    continue
            
            # Report batch processing results
            print(f"Batch {i//batch_size + 1} complete: {success_count} successful, {error_count} errors")
            
            # Extend main lists with batch results
            features_list.extend(batch_features)
            batch_features = None  # Help garbage collector
            
            labels_list.extend(batch_labels)
            batch_labels = None  # Help garbage collector
            
            batch_df = None  # Help garbage collector
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            # Print memory usage after processing batch
            print(f"After batch {i//batch_size + 1}: {get_memory_usage()}")
            
            # If we're at a multiple of 5 batches, save progress
            if (i//batch_size + 1) % 5 == 0:
                print(f"Intermediate progress - processed {len(features_list)} samples so far")
        
        print(f"Extracted features for {len(features_list)} samples")
        return tokenizer, features_list, labels_list
    
    def _create_features_for_sample(self, row, tokenizer):
        """Extract features for a single sample"""
        # Text features
        text = row['processed_text'] if pd.notna(row['processed_text']) else ''
        text_sequence = tokenizer.texts_to_sequences([text])[0]
        # Pad sequence to fixed length
        if len(text_sequence) > self.max_text_length:
            text_sequence = text_sequence[:self.max_text_length]
        else:
            text_sequence = text_sequence + [0] * (self.max_text_length - len(text_sequence))
        
        # Image features - load and preprocess image if available
        image_shape = tuple(self.config['model']['image']['input_shape'])
        # Explicitly create a float32 array to reduce memory usage
        image_features = np.zeros(image_shape, dtype=np.float32)
        
        if pd.notna(row['processed_image_path']) and os.path.exists(str(row['processed_image_path'])):
            # Check file permissions before processing
            img_path = str(row['processed_image_path'])
            if not os.access(img_path, os.R_OK):
                print(f"Warning: No read permission for image {img_path}")
            else:
                try:
                    # Check if it's a preprocessed .npy file
                    if img_path.endswith('.npy'):
                        try:
                            # Load directly as float32 to save memory
                            image_features = np.load(img_path).astype(np.float32)
                        except Exception as e:
                            print(f"Error loading .npy image {img_path}: {e}")
                    else:
                        # Process regular image file
                        from src.data.image_utils import preprocess_image
                        try:
                            # Process image with explicit float32 dtype - ensure target_size is correct
                            image_features = preprocess_image(
                                img_path,
                                target_size=image_shape[:2],
                            )
                        except np.core._exceptions._ArrayMemoryError as me:
                            print(f"Memory error processing image {img_path}. Using zeros instead: {me}")
                            # Keep the zero array created above
                        except Exception as e:
                            print(f"Error preprocessing image {img_path}: {e}")
                except Exception as e:
                    print(f"Error preprocessing image {img_path}: {e}")
        
        # Metadata features - extract key metadata and convert to numerical format
        metadata_features = np.zeros(10, dtype=np.float32)  # Fixed size for metadata features
        
        # Handle metadata which could be a string (from CSV) or dict
        metadata = row['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {}
        
        # Extract numeric features from metadata
        try:
            # Common features that might exist across datasets
            if 'score' in metadata:
                metadata_features[0] = np.float32(metadata['score']) if pd.notna(metadata['score']) else 0.0
            if 'num_comments' in metadata:
                metadata_features[1] = np.float32(metadata['num_comments']) if pd.notna(metadata['num_comments']) else 0.0
            if 'upvote_ratio' in metadata:
                metadata_features[2] = np.float32(metadata['upvote_ratio']) if pd.notna(metadata['upvote_ratio']) else 0.0
                
            # Indicator features
            metadata_features[3] = np.float32(1.0) if row['dataset_source'] == 'fakeddit' else np.float32(0.0)
            metadata_features[4] = np.float32(1.0) if row['dataset_source'] == 'fakenewsnet' else np.float32(0.0)
            
            # Text length feature
            if pd.notna(row['processed_text']):
                metadata_features[5] = np.float32(min(len(row['processed_text']) / 1000.0, 5.0))  # Normalize text length
                
            # Has image feature
            metadata_features[6] = np.float32(1.0) if pd.notna(row['processed_image_path']) else np.float32(0.0)
            
            # Timestamp feature if available
            if 'timestamp' in metadata or 'created_utc' in metadata:
                timestamp = metadata.get('timestamp', metadata.get('created_utc', 0))
                metadata_features[7] = np.float32(float(timestamp) / 1e10) if pd.notna(timestamp) else np.float32(0.0)
                
            # Keywords count if available
            if 'keywords' in metadata and isinstance(metadata['keywords'], list):
                metadata_features[8] = np.float32(min(len(metadata['keywords']) / 10.0, 1.0))
                
            # Authors count if available
            if 'authors' in metadata and isinstance(metadata['authors'], list):
                metadata_features[9] = np.float32(min(len(metadata['authors']) / 5.0, 1.0))
                
        except Exception as e:
            print(f"Error processing metadata: {e}")
            
        # Create features dictionary
        features = {
            'text': np.array(text_sequence, dtype=np.int32),
            'image': image_features,  # Already float32
            'metadata': metadata_features  # Already float32
        }
        
        # Label
        label = int(row['label']) if pd.notna(row['label']) else 0
        
        return features, label
    
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
