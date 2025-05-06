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
from PIL import Image, ImageDraw, ImageFont
import gc
import psutil
import ast

# Import utility functions
from src.utils.text_utils import TextProcessor
from src.utils.image_utils import ImageProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'preprocessing.log')
if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
        
        logger.info(f"Initializing DatasetProcessor with:")
        logger.info(f"  Raw data dir: {self.raw_dir}")
        logger.info(f"  Processed dir: {self.processed_dir}")
        logger.info(f"  Images dir: {self.images_dir}")
        logger.info(f"  Cache dir: {self.cache_dir}")
        
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
        self.allow_synthetic_images = config['data'].get('allow_synthetic_images', True)
        # For categorical vocabularies
        self.source_vocab = {}
        self.subreddit_vocab = {}
        self.author_vocab = {}
        self._vocab_ready = False
        
    def get_fakeddit_image_path(self, article_id, images_dir):
        """
        Get the image path for a Fakeddit article based on the article ID
        
        Args:
            article_id: ID of the article
            images_dir: Directory containing Fakeddit images
            
        Returns:
            list: List of possible image paths
        """
        image_paths = []
        # Only use the actual folder structure
        for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            path = os.path.join(images_dir, f"{article_id}.{ext}")
            if os.path.exists(path):
                image_paths.append(path)
                return image_paths
        # Log missing image
        missing_log = os.path.join('logs', 'missing_images.log')
        with open(missing_log, 'a') as f:
            f.write(f"Missing Fakeddit image: {article_id} in {images_dir}\n")
        return image_paths
        
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
                # Load in chunks to manage memory
                chunk_size = 5000
                chunks = []
                total_chunks = sum(1 for _ in pd.read_csv(output_path, chunksize=chunk_size))
                
                for i, chunk in enumerate(pd.read_csv(output_path, chunksize=chunk_size)):
                    print(f"Loading chunk {i+1}/{total_chunks} from processed Fakeddit data")
                    chunks.append(chunk)
                    if len(chunks) * chunk_size > sample_size if sample_size else False:
                        break
                    
                df = pd.concat(chunks, ignore_index=True)
                print(f"Fakeddit {subset} dataset loaded: {len(df)} samples")
                
                if sample_size and sample_size < len(df):
                    df = df.sample(sample_size, random_state=42)
                    print(f"Sampled {sample_size} examples from Fakeddit {subset}")
                return df
            except Exception as e:
                print(f"Error loading processed file {output_path}: {e}")
                print("Will reprocess the dataset...")
        
        # Map the subset to the corresponding file
        subset_to_file = {
            'train': 'multimodal_train.tsv',
            'val': 'multimodal_validate.tsv',
            'test': 'multimodal_test_public.tsv'
        }
        
        tsv_file = subset_to_file.get(subset, 'multimodal_train.tsv')
        tsv_path = os.path.join(self.raw_dir, 'fakeddit', tsv_file)
        
        if not os.path.exists(tsv_path):
            print(f"Fakeddit {subset} file not found at {tsv_path}, trying alternative paths...")
            alt_paths = [
                os.path.join("data/raw/fakeddit", tsv_file),
                os.path.join("data/fakeddit", tsv_file)
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    tsv_path = alt_path
                    print(f"Found Fakeddit {subset} file at {tsv_path}")
                    break
            else:
                raise FileNotFoundError(f"Fakeddit {subset} file not found in any location")
        
        # Load TSV data in chunks
        print(f"Loading Fakeddit {subset} dataset from {tsv_path}")
        chunk_size = 5000
        chunks = []
        total_chunks = sum(1 for _ in pd.read_csv(tsv_path, sep='\t', chunksize=chunk_size))
        
        for i, chunk in enumerate(pd.read_csv(tsv_path, sep='\t', chunksize=chunk_size)):
            print(f"Processing chunk {i+1}/{total_chunks}")
            
            # Create standardized data for this chunk
            standardized_chunk = pd.DataFrame()
            standardized_chunk['id'] = chunk['id']
            
            # Handle text content
            text_columns = ['title', 'clean_title', 'selftext', 'clean_selftext']
            standardized_chunk['text'] = chunk.apply(
                lambda row: ' '.join([
                    str(row[col]) for col in text_columns 
                    if col in chunk.columns and pd.notna(row[col]) and str(row[col]).strip() != ''
                ]),
            axis=1
        )
            
            # Set placeholder for empty text
            standardized_chunk.loc[standardized_chunk['text'] == '', 'text'] = "[No text content available]"
            
            # Process image paths for this chunk
            fakeddit_images_dir = os.path.join(self.images_dir, 'fakeddit/public_image_set')
            if not os.path.exists(fakeddit_images_dir):
                for alt_dir in ["data/images/fakeddit/public_image_set", "data/raw/fakeddit/public_image_set"]:
                    if os.path.exists(alt_dir):
                        fakeddit_images_dir = alt_dir
                        break
            
            # Process image paths
            image_paths = []
            has_images = []
            for _, row in chunk.iterrows():
                has_image = row.get('hasImage', False) if 'hasImage' in chunk.columns else False
                if has_image:
                    img_paths = self.get_fakeddit_image_path(row['id'], fakeddit_images_dir)
                    image_path = img_paths[0] if img_paths else None
                    image_paths.append(image_path)
                    has_images.append(image_path is not None)
                else:
                    image_paths.append(None)
                    has_images.append(False)
            
            standardized_chunk['image_path'] = image_paths
            standardized_chunk['has_image'] = has_images
            
            # Add other fields
            standardized_chunk['label'] = chunk['2_way_label']
            standardized_chunk['dataset_source'] = 'fakeddit'
        
        # Add metadata
            metadata_list = []
            for _, row in chunk.iterrows():
                metadata = {
                'source': row.get('domain', ''),
                'subreddit': row.get('subreddit', ''),
                'upvote_ratio': row.get('upvote_ratio', 0),
                'score': row.get('score', 0),
                'num_comments': row.get('num_comments', 0)
                }
                metadata_list.append(json.dumps(metadata))  # Convert to string to save memory
            
            standardized_chunk['metadata'] = metadata_list
            
            chunks.append(standardized_chunk)
            
            # Save intermediate results if memory usage is high
            process = psutil.Process()
            if process.memory_info().rss / 1024 / 1024 > 12000:  # If using more than 12GB
                print("Memory usage high, saving intermediate results...")
                temp_df = pd.concat(chunks, ignore_index=True)
                temp_path = os.path.join(self.processed_dir, f'temp_fakeddit_{subset}_{i}.csv')
                temp_df.to_csv(temp_path, index=False)
                chunks = []  # Clear chunks
                gc.collect()
            
            # Break if we have enough samples
            if sample_size and sum(len(chunk) for chunk in chunks) >= sample_size:
                break
        
        # Combine all chunks or load from temporary files
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            # Load and combine temporary files
            temp_files = glob(os.path.join(self.processed_dir, f'temp_fakeddit_{subset}_*.csv'))
            df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)
            
            # Clean up temporary files
            for f in temp_files:
                os.remove(f)
        
        print(f"Loaded {len(df)} records from Fakeddit {subset}")
        
        # Sample if needed
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"Sampled {sample_size} examples from Fakeddit {subset}")
        
        # Save processed data
        try:
            df.to_csv(output_path, index=False)
            print(f"Saved processed Fakeddit {subset} dataset to {output_path}")
        except Exception as e:
            print(f"Error saving processed file: {e}")
        
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
                        image_paths = self.get_fakenewsnet_image_paths(article_dir, source, label, images_dir)
                        
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

    def get_fakenewsnet_image_paths(self, article_id, source, label, images_dir):
        """
        Get the image paths for a FakeNewsNet article
        
        Args:
            article_id: ID of the article
            source: Source of the article (gossipcop or politifact)
            label: Label of the article (fake or real)
            images_dir: Base directory containing FakeNewsNet images
            
        Returns:
            list: List of image paths
        """
        image_paths = []
        
        # Construct the article directory path
        article_dir = os.path.join(images_dir, source, label, article_id)
        
        # Check if the article directory exists
        if not os.path.exists(article_dir):
            logger.warning(f"Article directory not found: {article_dir}")
            return image_paths
            
        # Look for images in the article directory
        extensions = ['jpg', 'jpeg', 'png', 'gif', 'webp']
        for ext in extensions:
            # Look for images with the article ID
            img_path = os.path.join(article_dir, f"{article_id}.{ext}")
            if os.path.exists(img_path):
                logger.info(f"Found FakeNewsNet image: {img_path}")
                image_paths.append(img_path)
                return image_paths
                
            # Look for any images in the directory
            for img_file in os.listdir(article_dir):
                if img_file.lower().endswith(f'.{ext}'):
                    img_path = os.path.join(article_dir, img_file)
                    if os.path.exists(img_path):
                        logger.info(f"Found FakeNewsNet image: {img_path}")
                        image_paths.append(img_path)
                        return image_paths
        
        # If no images found, try alternative locations
        alt_locations = [
            os.path.join(images_dir, source, label),
            os.path.join(images_dir, source),
            images_dir
        ]
        
        for location in alt_locations:
            if os.path.exists(location):
                for ext in extensions:
                    img_path = os.path.join(location, f"{article_id}.{ext}")
                    if os.path.exists(img_path):
                        logger.info(f"Found FakeNewsNet image in alt location: {img_path}")
                        image_paths.append(img_path)
                        return image_paths
        
        # Print debug info if no path was found
        if not image_paths:
            logger.warning(f"No image found for FakeNewsNet article {article_id} in {article_dir}")
            # Check if the directory exists
            if not os.path.exists(article_dir):
                logger.warning(f"  Warning: Article directory {article_dir} does not exist")
            else:
                # Get sample of files in the article directory
                files = os.listdir(article_dir)[:5] if os.path.exists(article_dir) and os.listdir(article_dir) else []
                logger.info(f"  Sample files in {article_dir}: {files}")
        
        return image_paths

    def preprocess_dataset(self, df=None):
        """Preprocess the dataset for training"""
        logger.info("Preprocessing dataset...")
        combined_path = os.path.join(self.processed_dir, 'combined_processed.csv')
        if df is None:
            # If no dataframe provided, try to load the combined dataset
            if os.path.exists(combined_path):
                logger.info(f"Loading combined dataset from {combined_path}")
                chunk_size = 5000  # Smaller chunk size
                chunks = []
                total_chunks = sum(1 for _ in pd.read_csv(combined_path, chunksize=chunk_size))
                for i, chunk in enumerate(pd.read_csv(combined_path, chunksize=chunk_size, low_memory=False)):
                    logger.info(f"Processing chunk {i+1}/{total_chunks}")
                    needed_columns = ['id', 'text', 'image_path', 'label', 'metadata', 'dataset_source', 'clean_text', 'image_paths', 'processed_image_path']
                    chunk = chunk[needed_columns]
                    for col in ['text', 'clean_text']:
                        if col in chunk.columns:
                            chunk[col] = chunk[col].astype('string')
                    if 'label' in chunk.columns:
                        chunk['label'] = chunk['label'].astype('int8')
                    chunks.append(chunk)
                    process = psutil.Process()
                    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
                    gc.collect()
                    if process.memory_info().rss / 1024 / 1024 > 12000:
                        logger.warning("Memory usage high, saving intermediate results...")
                        intermediate_df = pd.concat(chunks, ignore_index=True)
                        intermediate_path = os.path.join(self.processed_dir, f'intermediate_preprocessed_{i}.csv')
                        intermediate_df.to_csv(intermediate_path, index=False)
                        chunks = []
                        gc.collect()
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
            else:
                intermediate_files = glob(os.path.join(self.processed_dir, 'intermediate_preprocessed_*.csv'))
                if not intermediate_files:
                    logger.error(f"No combined dataset found at {combined_path} and no intermediate files found.")
                    return None
                df = pd.concat([pd.read_csv(f) for f in intermediate_files], ignore_index=True)
                for f in intermediate_files:
                    os.remove(f)
            logger.info(f"Loaded combined dataset with {len(df)} records")
        else:
            logger.info("Loading individual datasets...")
            fakeddit_df = self.load_fakeddit()
            if not fakeddit_df.empty:
                logger.info(f"Loaded Fakeddit dataset with {len(fakeddit_df)} records")
                df = fakeddit_df
                del fakeddit_df
                gc.collect()
            fakenewsnet_df = self.load_fakenewsnet()
            if not fakenewsnet_df.empty:
                logger.info(f"Loaded FakeNewsNet dataset with {len(fakenewsnet_df)} records")
                if df is not None:
                    df = pd.concat([df, fakenewsnet_df], ignore_index=True)
                else:
                    df = fakenewsnet_df
                del fakenewsnet_df
                gc.collect()
        if df is None:
            logger.error("No data could be loaded!")
            return None
        logger.info("Processing text data...")
        batch_size = 1000
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            mask = batch['clean_text'].isna() | (batch['clean_text'] == '')
            if mask.any():
                for idx in batch[mask].index:
                    if pd.notna(batch.loc[idx, 'text']) and batch.loc[idx, 'text'].strip():
                        try:
                            cleaned = preprocess_text(
                                batch.loc[idx, 'text'],
                max_length=self.max_text_length,
                                remove_stopwords=True,
                                use_lemmatization=True,
                                use_stemming=False,
                                expand_contractions=True,
                                convert_slang=True
                            )
                            df.loc[idx, 'clean_text'] = cleaned
                        except Exception as e:
                            logger.error(f"Error processing text for row {idx}: {e}")
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{len(df)} texts")
                gc.collect()
        logger.info("Processing image paths...")
        df['processed_image_path'] = df.apply(lambda row: self._process_image_path(row), axis=1)
        df['has_image'] = df['processed_image_path'].notna()
        preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
        chunk_size = 5000
        df.iloc[0:0].to_csv(preprocessed_path, index=False)
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk.to_csv(preprocessed_path, mode='a', header=False, index=False)
            if i % 50000 == 0:
                logger.info(f"Saved {i}/{len(df)} records")
                gc.collect()
        valid_texts = df['clean_text'].notna().sum() and df['clean_text'].str.strip().str.len().gt(0).sum()
        valid_images = df['has_image'].sum()
        logger.info(f"Preprocessed dataset size: {len(df)} records")
        logger.info(f"Records with valid text: {valid_texts} ({valid_texts/len(df)*100:.2f}%)")
        logger.info(f"Records with valid images: {valid_images} ({valid_images/len(df)*100:.2f}%)")
        logger.info("\nDataset source distribution:")
        logger.info(f"{df['dataset_source'].value_counts()}")
        return df
    
    def _process_image_path(self, row):
        """Process image paths and find valid images, with strict real image handling."""
        img_path = None
        dataset_source = row.get('dataset_source', '')
        article_id = row.get('id', '')
        # First check if we have a single image_path (from Fakeddit)
        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = row['image_path']
            if os.path.exists(img_path):
                return img_path
        # If direct path doesn't exist but it's Fakeddit, try to find it using the helper function
        if dataset_source == 'fakeddit' and 'id' in row:
            article_id = row['id']
            fakeddit_images_dir = self.config['data']['fakeddit']['images_dir']
            if os.path.exists(fakeddit_images_dir):
                fakeddit_paths = self.get_fakeddit_image_path(article_id, fakeddit_images_dir)
                if fakeddit_paths:
                    return fakeddit_paths[0]
        # If no valid image_path, try image_paths (from FakeNewsNet)
        if 'image_paths' in row and pd.notna(row['image_paths']):
            image_paths = row['image_paths']
            if isinstance(image_paths, str):
                try:
                    if image_paths.startswith('[') and image_paths.endswith(']'):
                        image_paths = json.loads(image_paths.replace("'", '"'))
                        for img_path in image_paths:
                            if isinstance(img_path, str) and os.path.exists(img_path):
                                return img_path
                    else:
                        if os.path.exists(image_paths):
                            return image_paths
                except Exception as e:
                    if os.path.exists(image_paths):
                        return image_paths
            elif isinstance(image_paths, list):
                for img_path in image_paths:
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        return img_path
        # Try to get image path based on article ID and dataset source for FakeNewsNet
        if dataset_source == 'fakenewsnet' and 'id' in row and pd.notna(row['id']):
            try:
                article_id = row['id']
                if '_' in article_id:
                    source, article_id = article_id.split('_', 1)
                    label = 'fake' if row.get('label', 0) == 1 else 'real'
                    fnn_images_dir = self.config['data']['fakenewsnet']['images_dir']
                    fnn_paths = []
                    article_img_dir = os.path.join(fnn_images_dir, source, label, article_id)
                    if os.path.exists(article_img_dir):
                        for filename in os.listdir(article_img_dir):
                            file_path = os.path.join(article_img_dir, filename)
                            if os.path.isfile(file_path) and file_path.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                                fnn_paths.append(file_path)
                    if fnn_paths:
                        return fnn_paths[0]
            except Exception as e:
                pass
        # If we get here, no valid image was found
        missing_log = os.path.join('logs', 'missing_images.log')
        with open(missing_log, 'a') as f:
            f.write(f"Missing image for article {article_id} in dataset {dataset_source}\n")
        return None
    
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
        """Apply balanced sampling to handle class imbalance (supports oversampling and downsampling)"""
        if not self.balanced_sampling:
            return df
        
        print("Applying balanced sampling...")
        
        # Get counts for each class
        label_counts = df['label'].value_counts()
        min_class_size = label_counts.min()
        max_class_size = label_counts.max()
        balanced_samples = []
        
        # Configurable strategy: 'oversample', 'downsample', or 'both'
        sampling_strategy = self.config['data'].get('sampling_strategy', 'oversample')
        print(f"Sampling strategy: {sampling_strategy}")
        
        for label, count in label_counts.items():
            class_df = df[df['label'] == label]
            if sampling_strategy == 'downsample':
                # Downsample majority class to min_class_size
                if count > min_class_size:
                    sampled = resample(
                        class_df,
                        replace=False,
                        n_samples=min_class_size,
                        random_state=42
                    )
                else:
                    sampled = class_df
            elif sampling_strategy == 'oversample':
                # Oversample minority class to max_class_size
                if count < max_class_size:
                    sampled = resample(
                        class_df,
                        replace=True,
                        n_samples=max_class_size,
                        random_state=42
                    )
                else:
                    sampled = class_df
            elif sampling_strategy == 'both':
                # Downsample majority, oversample minority to mean size
                target_size = int(label_counts.mean())
                if count > target_size:
                    sampled = resample(
                        class_df,
                        replace=False,
                        n_samples=target_size,
                        random_state=42
                    )
                elif count < target_size:
                    sampled = resample(
                        class_df,
                        replace=True,
                        n_samples=target_size,
                        random_state=42
                    )
                else:
                    sampled = class_df
            else:
                sampled = class_df
            balanced_samples.append(sampled)
        
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        print(f"Balanced dataset size: {len(balanced_df)} records")
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
        hash_columns = ['id', 'clean_text', 'processed_image_path', 'label']
        hash_df = df[hash_columns].copy()
        
        # Convert to string and hash
        df_str = hash_df.to_string()
        return hashlib.md5(df_str.encode()).hexdigest()
    
    def create_tf_dataset(self, df=None, shuffle=True):
        """Create TensorFlow train/val/test datasets from a DataFrame, and return word_index.
        Args:
            df: DataFrame containing the data (if None, loads preprocessed dataset)
            shuffle: Whether to shuffle the dataset (applies to train only)
        Returns:
            train_dataset, val_dataset, test_dataset, word_index
        """
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from sklearn.model_selection import train_test_split
        import numpy as np
        import os
        import json
        import random

        print("Preparing TensorFlow datasets (train/val/test) and tokenizer...")

        # 1. Load preprocessed dataset if df is None
        if df is None:
            preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
            if not os.path.exists(preprocessed_path):
                raise FileNotFoundError(f"Preprocessed dataset not found at {preprocessed_path}")
            print(f"Loading preprocessed dataset from {preprocessed_path}")
            df = pd.read_csv(preprocessed_path, low_memory=False)
        
        # 2. Drop rows with missing labels or text
        df = df[df['label'].notna() & df['clean_text'].notna() & df['clean_text'].str.strip().ne('')]
        df = df.reset_index(drop=True)
        print(f"Dataset size after dropping missing: {len(df)}")

        # Always apply balanced sampling if enabled in config
        if self.balanced_sampling:
            df = self.apply_balanced_sampling(df)

        # Rebuild categorical vocabs on the full DataFrame to ensure all categories are included
        self._build_categorical_vocabs(df)

        # 3. Split into train/val/test
        train_ratio = self.train_ratio
        val_ratio = self.val_split
        test_ratio = self.test_split
        random_seed = self.random_seed

        # First split train vs temp (val+test)
        stratify_col = df['label'] if 'label' in df.columns else None
        train_df, temp_df = train_test_split(
            df,
            test_size=val_ratio + test_ratio,
            random_state=random_seed,
            stratify=stratify_col
        )
        # Then split temp into val and test
        if val_ratio + test_ratio > 0:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_size,
                random_state=random_seed,
                stratify=temp_df['label'] if 'label' in temp_df.columns else None
            )
        else:
            val_df, test_df = pd.DataFrame(), pd.DataFrame()

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # 4. Fit tokenizer on train set
        print("Fitting tokenizer on train set...")
        tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_df['clean_text'].astype(str).tolist())
        word_index = tokenizer.word_index
        self.word_index = word_index
        print(f"Tokenizer fitted. Vocab size: {len(word_index)}")

        # Find the true maximum token index in all splits
        def get_max_token_index(df_split):
            max_idx = 0
            for text in df_split['clean_text'].astype(str):
                seq = tokenizer.texts_to_sequences([text])[0]
                if seq:
                    max_idx = max(max_idx, max(seq))
            return max_idx
        max_token_index = max(get_max_token_index(train_df), get_max_token_index(val_df), get_max_token_index(test_df))

        # 5. Helper to create tf.data.Dataset from a DataFrame
        def make_tf_dataset(df_split, shuffle_split, vocab_size=None):
            # Load tokenizer
            tokenizer_path = os.path.join(self.processed_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            max_len = self.config['data'].get('max_text_length', 128)
            if vocab_size is None:
                vocab_size = max_token_index + 1
            def generator():
                for _, row in df_split.iterrows():
                    text = row['clean_text'] if pd.notna(row['clean_text']) else ''
                    # Tokenize and pad
                    seq = tokenizer.texts_to_sequences([text])[0]
                    # Replace out-of-bounds indices with OOV index (1)
                    seq = [i if i < vocab_size else 1 for i in seq]
                    if len(seq) > max_len:
                        seq = seq[:max_len]
                    else:
                        seq = seq + [0] * (max_len - len(seq))
                    seq = np.array(seq, dtype=np.int32)
                    image_path = ''
                    if 'processed_image_path' in row and pd.notna(row['processed_image_path']) and isinstance(row['processed_image_path'], str) and os.path.exists(row['processed_image_path']):
                        image_path = row['processed_image_path']
                    elif 'image_path' in row and pd.notna(row['image_path']) and isinstance(row['image_path'], str) and os.path.exists(row['image_path']):
                        image_path = row['image_path']
                    elif 'image_paths' in row and pd.notna(row['image_paths']):
                        image_paths = row['image_paths']
                        if isinstance(image_paths, str):
                            try:
                                if image_paths.startswith('[') and image_paths.endswith(']'):
                                    paths = json.loads(image_paths.replace("'", '"'))
                                    if paths and len(paths) > 0 and isinstance(paths[0], str) and os.path.exists(paths[0]):
                                        image_path = paths[0]
                                else:
                                    if os.path.exists(image_paths):
                                        image_path = image_paths
                            except:
                                if os.path.exists(image_paths):
                                    image_path = image_paths
                        elif isinstance(image_paths, list) and len(image_paths) > 0 and isinstance(image_paths[0], str) and os.path.exists(image_paths[0]):
                            image_path = image_paths[0]
                    label = float(row['label']) if 'label' in row and pd.notna(row['label']) else 0.0
                    # Ensure label is always a float scalar
                    if isinstance(label, (list, np.ndarray)) and len(label) == 1:
                        label = float(label[0])
                    # Metadata: use 10-dim float32 vector if available, else zeros
                    if 'metadata' in row and isinstance(row['metadata'], str):
                        try:
                            metadata = json.loads(row['metadata'])
                        except:
                            metadata = {}
                    elif 'metadata' in row and isinstance(row['metadata'], dict):
                        metadata = row['metadata']
                    else:
                        metadata = {}
                    metadata_vec = np.zeros(10, dtype=np.float32)
                    try:
                        if 'score' in metadata:
                            metadata_vec[0] = np.float32(metadata['score']) if pd.notna(metadata['score']) else 0.0
                        if 'num_comments' in metadata:
                            metadata_vec[1] = np.float32(metadata['num_comments']) if pd.notna(metadata['num_comments']) else 0.0
                        if 'upvote_ratio' in metadata:
                            metadata_vec[2] = np.float32(metadata['upvote_ratio']) if pd.notna(metadata['upvote_ratio']) else 0.0
                        # Indicator features
                        metadata_vec[3] = np.float32(1.0) if row.get('dataset_source', '') == 'fakeddit' else np.float32(0.0)
                        metadata_vec[4] = np.float32(1.0) if row.get('dataset_source', '') == 'fakenewsnet' else np.float32(0.0)
                        # Text length
                        if pd.notna(row['clean_text']):
                            metadata_vec[5] = np.float32(min(len(row['clean_text']) / 1000.0, 5.0))
                        # Has image
                        metadata_vec[6] = np.float32(1.0) if pd.notna(row.get('processed_image_path', None)) else np.float32(0.0)
                        # Timestamp
                        if 'timestamp' in metadata or 'created_utc' in metadata:
                            timestamp = metadata.get('timestamp', metadata.get('created_utc', 0))
                            metadata_vec[7] = np.float32(float(timestamp) / 1e10) if pd.notna(timestamp) else np.float32(0.0)
                        # Keywords count
                        if 'keywords' in metadata and isinstance(metadata['keywords'], list):
                            metadata_vec[8] = np.float32(min(len(metadata['keywords']) / 10.0, 1.0))
                        # Authors count
                        if 'authors' in metadata and isinstance(metadata['authors'], list):
                            metadata_vec[9] = np.float32(min(len(metadata['authors']) / 5.0, 1.0))
                    except Exception as e:
                        pass
                    # Categorical indices
                    if not self._vocab_ready:
                        self._build_categorical_vocabs(df_split)
                    source_idx, subreddit_idx, author_idx = self._map_categorical_indices(row)
                    # Ensure indices are in range for their vocab size; fallback to <UNK> (0) if not
                    if source_idx >= len(self.source_vocab):
                        source_idx = self.source_vocab.get('<UNK>', 0)
                    if subreddit_idx >= len(self.subreddit_vocab):
                        subreddit_idx = self.subreddit_vocab.get('<UNK>', 0)
                    if author_idx >= len(self.author_vocab):
                        author_idx = self.author_vocab.get('<UNK>', 0)
                    yield (
                        {
                            'text': seq,
                            'image_path': image_path,
                            'metadata': metadata_vec,
                            'source_idx': source_idx,
                            'subreddit_idx': subreddit_idx,
                            'author_idx': author_idx
                        },
                        label
                    )
            output_signature = (
                    {
                    'text': tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
                    'image_path': tf.TensorSpec(shape=(), dtype=tf.string),
                    'metadata': tf.TensorSpec(shape=(10,), dtype=tf.float32),
                    'source_idx': tf.TensorSpec(shape=(), dtype=tf.int32),
                    'subreddit_idx': tf.TensorSpec(shape=(), dtype=tf.int32),
                    'author_idx': tf.TensorSpec(shape=(), dtype=tf.int32)
                    },
                tf.TensorSpec(shape=(), dtype=tf.float32)
                )
            ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            # Use configurable shuffle buffer size to limit in-memory shuffle
            shuffle_buffer_size = self.config['data'].get('shuffle_buffer_size', 1000)
            shuffle_buffer_size = min(shuffle_buffer_size, len(df_split))
            if shuffle_split:
                ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=random_seed)

            # Determine batch size and auto-adjust based on available RAM
            batch_size = self.config['training'].get('batch_size', 32)
            import psutil
            available_bytes = psutil.virtual_memory().available
            # Estimate bytes per batch (images only): height*width*channels*float_size
            bytes_per_image = 224 * 224 * 3 * 4
            batch_footprint = batch_size * bytes_per_image
            threshold = self.config['data'].get('batch_mem_threshold', 0.5)
            if batch_footprint > available_bytes * threshold:
                new_batch_size = max(1, batch_size // 2)
                logger.info(f"Auto-adjusting batch size from {batch_size} to {new_batch_size} due to low available RAM ({available_bytes/(1024**3):.2f} GB)")
                batch_size = new_batch_size

            ds = ds.batch(batch_size)
            # Map to load and preprocess images
            def load_and_preprocess_image(features, label):
                import tensorflow as tf
                def _load_image(path):
                    from src.data.image_utils import preprocess_image
                    img = preprocess_image(path.numpy().decode('utf-8'), target_size=(224, 224), normalize=True)
                    return img.astype(np.float32)
                image = tf.map_fn(
                    lambda p: tf.py_function(_load_image, [p], tf.float32),
                    features['image_path'],
                    fn_output_signature=tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32)
                )
                features['image'] = image
                features.pop('image_path', None)
                if 'metadata' not in features:
                    features['metadata'] = tf.zeros([10], dtype=tf.float32)
                else:
                    features['metadata'] = tf.cast(features['metadata'], tf.float32)
                label = tf.reshape(label, [-1])
                # tf.print('DEBUG: label shape', tf.shape(label), 'dtype', label.dtype)
                return features, label
            # Use config for num_parallel_calls and prefetch buffer size
            num_parallel_calls = self.config['data'].get('num_parallel_calls', 2)
            prefetch_buffer_size = self.config['data'].get('prefetch_buffer_size', 2)
            ds = ds.map(load_and_preprocess_image, num_parallel_calls=num_parallel_calls)
            # Remove .cache() to avoid excessive RAM usage
            ds = ds.prefetch(prefetch_buffer_size)
            # Document: .cache() removed for memory efficiency; prefetch buffer is small and configurable
            return ds
        # 6. Create datasets
        train_dataset = make_tf_dataset(train_df, shuffle_split=True, vocab_size=max_token_index+1)
        val_dataset = make_tf_dataset(val_df, shuffle_split=False, vocab_size=max_token_index+1)
        test_dataset = make_tf_dataset(test_df, shuffle_split=False, vocab_size=max_token_index+1)
        print("TensorFlow datasets ready.")
        return train_dataset, val_dataset, test_dataset, word_index, max_token_index
        
    def _extract_features(self, df):
        """Extract features from dataset in batches to reduce memory usage"""
        # Create a vocabulary from text data
        from tensorflow.keras.preprocessing.text import Tokenizer
        print("Creating tokenizer...")
        tokenizer = Tokenizer(num_words=self.max_vocab_size)
        
        # Process text data in chunks for tokenizer
        chunk_size = 10000
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            tokenizer.fit_on_texts(chunk['clean_text'].fillna(''))
        
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
        batch_size = 100  # Reduced batch size for better memory management
        
        print(f"Processing {len(df)} samples in batches of {batch_size}...")
        
        # Get system memory info for monitoring
        try:
            def get_memory_usage():
                process = psutil.Process()
                memory_info = process.memory_info()
                return f"Memory usage: {memory_info.rss / (1024 * 1024):.1f} MB"
        except ImportError:
            def get_memory_usage():
                return "Memory usage tracking not available"
        
        # Process data in batches
        for i in range(0, len(df), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            print(f"Memory usage before batch: {get_memory_usage()}")
            
            batch_df = df.iloc[i:i+batch_size]
            batch_features = []
            batch_labels = []
            
            # Process each row in the batch
            success_count = 0
            error_count = 0
            
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {i//batch_size + 1}"):
                try:
                    features, label = self._create_features_for_sample(row, tokenizer)
                    batch_features.append(features)
                    batch_labels.append(label)
                    success_count += 1
                except Exception as e:
                    print(f"Error processing row {row.get('id', 'unknown')}: {e}")
                    error_count += 1
                    continue
            
            # Extend main lists with batch results
            features_list.extend(batch_features)
            labels_list.extend(batch_labels)
            
            # Clear batch data to free memory
            batch_features = None
            batch_labels = None
            batch_df = None
            
            # Force garbage collection
            gc.collect()
            
            print(f"Batch {i//batch_size + 1} complete: {success_count} successful, {error_count} errors")
            print(f"Memory usage after batch: {get_memory_usage()}")
            
            # Save intermediate results every 5 batches
            if (i//batch_size + 1) % 5 == 0:
                print(f"Intermediate progress - processed {len(features_list)} samples so far")
                # Save intermediate results to disk
                intermediate_path = os.path.join(self.processed_dir, f'intermediate_features_{i}.pkl')
                try:
                    with open(intermediate_path, 'wb') as f:
                        pickle.dump({
                            'features': features_list,
                            'labels': labels_list,
                            'tokenizer': tokenizer
                        }, f)
                    print(f"Saved intermediate results to {intermediate_path}")
                except Exception as e:
                    print(f"Error saving intermediate results: {e}")
        
        print(f"Extracted features for {len(features_list)} samples")
        return tokenizer, features_list, labels_list
    
    def _create_features_for_sample(self, row, tokenizer):
        """Extract features for a single sample"""
        # Text features - use clean_text instead of processed_text
        text = row['clean_text'] if pd.notna(row['clean_text']) else ''
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
        
        has_valid_image = False
        image_loading_log = os.path.join('logs', 'image_loading_errors.log')
        def log_image_error(msg):
            print(msg)
            with open(image_loading_log, 'a') as f:
                f.write(msg + '\n')
        
        if pd.notna(row.get('processed_image_path', None)) and os.path.exists(str(row['processed_image_path'])):
            # Check file permissions before processing
            img_path = str(row['processed_image_path'])
            if not os.access(img_path, os.R_OK):
                log_image_error(f"Warning: No read permission for image {img_path}")
            else:
                try:
                    # Check if it's a preprocessed .npy file
                    if img_path.endswith('.npy'):
                        try:
                            # Load directly as float32 to save memory
                            image_features = np.load(img_path).astype(np.float32)
                            has_valid_image = True
                        except Exception as e:
                            log_image_error(f"Error loading .npy image {img_path}: {e}")
                    else:
                        # Process regular image file
                        from src.data.image_utils import preprocess_image
                        try:
                            # Process image with explicit float32 dtype - ensure target_size is correct
                            image_features = preprocess_image(
                                img_path,
                                target_size=image_shape[:2],
                            )
                            has_valid_image = True
                        except np.core._exceptions._ArrayMemoryError as me:
                            log_image_error(f"Memory error processing image {img_path}. Using zeros instead: {me}")
                            # Keep the zero array created above
                        except Exception as e:
                            log_image_error(f"Error preprocessing image {img_path}: {e}")
                except Exception as e:
                    log_image_error(f"Error preprocessing image {img_path}: {e}")
        
        # If we don't have a valid image, create a synthetic one
        if not has_valid_image:
            # Fill with random noise as placeholder (better than all zeros)
            # This is just for model training to work, real applications would need real images
            np.random.seed(hash(str(row['id'])) % 2**32)  # Create a deterministic seed based on the row id
            image_features = np.random.normal(0, 0.1, size=image_shape).astype(np.float32)
        
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
            
            # Text length feature - use clean_text instead of processed_text
            if pd.notna(row['clean_text']):
                metadata_features[5] = np.float32(min(len(row['clean_text']) / 1000.0, 5.0))  # Normalize text length
                
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
        label = float(row['label']) if pd.notna(row['label']) else 0.0
        
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

    def analyze_image_paths(self):
        """Analyze the dataset files to identify image issues"""
        # List of dataset files to check
        dataset_files = [
            os.path.join(self.processed_dir, 'fakeddit_processed.csv'),
            os.path.join(self.processed_dir, 'fakenewsnet_processed.csv'),
            os.path.join(self.processed_dir, 'combined_processed.csv'),
            os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
        ]
        
        # Check each file
        results = {}
        for file_path in dataset_files:
            if os.path.exists(file_path):
                print(f"Analyzing {os.path.basename(file_path)}...")
                results[file_path] = self._analyze_dataset_file(file_path)
            else:
                print(f"File not found: {file_path}")
        
        return results
    
    def _analyze_dataset_file(self, file_path):
        """Analyze a single dataset file"""
        try:
            # Load the file
            df = pd.read_csv(file_path, low_memory=False)
            
            # Get basic stats
            result = {
                'total_records': len(df),
                'columns': list(df.columns),
                'image_columns': [col for col in df.columns if 'image' in col.lower()],
                'image_stats': {}
            }
            
            # For each image column, check validity of paths
            for col in result['image_columns']:
                if col in df.columns:
                    # Count non-null entries
                    non_null = df[col].notna().sum()
                    
                    # Sample some paths to check if they exist
                    sample_size = min(100, non_null)
                    if sample_size > 0:
                        sample_paths = df[df[col].notna()][col].sample(sample_size).tolist()
                        existing_paths = sum(1 for path in sample_paths if os.path.exists(path))
                        
                        # Extrapolate to full dataset
                        estimated_existing = (existing_paths / sample_size) * non_null
                        
                        result['image_stats'][col] = {
                            'non_null': non_null,
                            'null_percentage': 100 - (non_null / len(df) * 100),
                            'sample_existing': existing_paths,
                            'sample_size': sample_size,
                            'estimated_existing': estimated_existing,
                            'estimated_existing_percentage': (estimated_existing / len(df) * 100)
                        }
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return {'error': str(e)}
            
    def find_actual_images(self):
        """Find all image files in the data directory"""
        # List of common image extensions
        extensions = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg']
        
        # Find all image files recursively
        all_images = []
        for ext in extensions:
            found = glob(f"data/**/*.{ext}", recursive=True)
            all_images.extend(found)
            print(f"Found {len(found)} files with extension .{ext}")
        
        return all_images
    
    def create_synthetic_images(self, num_images=100, output_dir=None):
        """Create synthetic images for missing ones"""
        if output_dir is None:
            self.synthetic_dir = os.path.join(self.images_dir, 'synthetic')
            output_dir = self.synthetic_dir
            
        print(f"Creating {num_images} synthetic images...")
        
        # Create synthetic images directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create synthetic images
        image_size = (224, 224)  # Size of the images
        image_paths = []
        
        for i in tqdm(range(num_images), desc="Creating synthetic images"):
            # Create a new image
            img = Image.new('RGB', image_size, color=(
                random.randint(200, 255),
                random.randint(200, 255),
                random.randint(200, 255)
            ))
            
            # Add some shapes
            draw = ImageDraw.Draw(img)
            
            # Draw random shapes
            for _ in range(5):
                shape_type = random.choice(['rectangle', 'ellipse'])
                x1 = random.randint(0, image_size[0] - 1)
                y1 = random.randint(0, image_size[1] - 1)
                x2 = random.randint(x1, image_size[0] - 1)
                y2 = random.randint(y1, image_size[1] - 1)
                
                # Random color for the shape
                color = (
                    random.randint(0, 200),
                    random.randint(0, 200),
                    random.randint(0, 200)
                )
                
                if shape_type == 'rectangle':
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                else:
                    draw.ellipse([x1, y1, x2, y2], fill=color)
            
            # Add some text
            try:
                # Try to get a font (will depend on system fonts)
                font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                # Use default font if arial not available
                font = ImageFont.load_default()
            
            # Add text
            text = f"Synthetic {i+1}"
            text_color = (0, 0, 0)  # Black
            
            # Calculate text position (centered)
            draw.text((20, 20), text, fill=text_color, font=font)
            
            # Save the image
            img_path = os.path.join(output_dir, f"synthetic_{i+1}.jpg")
            img.save(img_path)
            image_paths.append(img_path)
        
        print(f"Created {num_images} synthetic images in {output_dir}")
        return image_paths
    
    def update_dataset_with_synthetic_images(self, synthetic_image_paths=None):
        """Update the preprocessed dataset with synthetic images"""
        preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
        
        if not os.path.exists(preprocessed_path):
            print(f"Preprocessed dataset not found at {preprocessed_path}")
            return False
        
        # Create a backup
        backup_path = preprocessed_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy2(preprocessed_path, backup_path)
            print(f"Created backup of original dataset at {backup_path}")
        
        # Load dataset
        df = pd.read_csv(preprocessed_path, low_memory=False)
        print(f"Loaded dataset with {len(df)} rows")
        
        # Check for image path column
        if 'processed_image_path' not in df.columns:
            # Add the column if it doesn't exist
            df['processed_image_path'] = None
            print("Added 'processed_image_path' column")
        
        # Count valid image paths
        valid_before = df['processed_image_path'].notna().sum()
        valid_existing_before = sum(1 for path in df[df['processed_image_path'].notna()]['processed_image_path'] if os.path.exists(path))
        
        print(f"Before update: {valid_before} non-null paths, {valid_existing_before} existing files")
        
        # If synthetic images not provided, create them
        if synthetic_image_paths is None:
            synthetic_image_paths = self.create_synthetic_images(num_images=100)
        
        # Update missing or invalid image paths
        count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating image paths"):
            path = row['processed_image_path']
            if pd.isna(path) or not isinstance(path, str) or not os.path.exists(path):
                # Assign a synthetic image
                synthetic_img = synthetic_image_paths[idx % len(synthetic_image_paths)]
                df.at[idx, 'processed_image_path'] = synthetic_img
                count += 1
        
        print(f"Updated {count} rows with synthetic image paths")
        
        # Save the updated dataset
        df.to_csv(preprocessed_path, index=False)
        print(f"Saved updated dataset to {preprocessed_path}")
        
        # Verify the update
        valid_after = df['processed_image_path'].notna().sum()
        valid_existing_after = sum(1 for path in df[df['processed_image_path'].notna()]['processed_image_path'].sample(min(1000, valid_after)) if os.path.exists(path))
        
        print(f"After update: {valid_after} non-null paths, ~{valid_existing_after} existing files (sampled)")
        
        return True
    
    def fix_fakeddit_paths(self):
        """Fix image paths for Fakeddit dataset. Run before training if image path errors are detected."""
        fakeddit_path = os.path.join(self.processed_dir, 'fakeddit_processed.csv')
        if not os.path.exists(fakeddit_path):
            print(f"Fakeddit processed data not found at {fakeddit_path}")
            return False
        df = pd.read_csv(fakeddit_path)
        print(f"Loaded Fakeddit dataset with {len(df)} rows")
        possible_locations = [
            "data/images/fakeddit/public_image_set",
            "data/raw/fakeddit/public_image_set",
            "data/fakeddit/public_image_set",
            "data/images/fakeddit",
            "data/raw/fakeddit/images"
        ]
        actual_location = None
        for location in possible_locations:
            if os.path.exists(location) and len(os.listdir(location)) > 0:
                actual_location = location
                print(f"Found Fakeddit images at {location}")
                print(f"Sample files: {os.listdir(location)[:5]}")
                break
        if not actual_location:
            print("Could not find Fakeddit images directory")
            return False
        updated = 0
        total_with_image_flag = df['has_image'].sum()
        print(f"Records marked as having images: {total_with_image_flag}")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating Fakeddit paths"):
            if 'id' in row and pd.notna(row['id']):
                article_id = row['id']
                found_image = False
                for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    img_path = os.path.join(actual_location, f"{article_id}.{ext}")
                    if os.path.exists(img_path):
                        df.at[idx, 'image_path'] = img_path
                        df.at[idx, 'has_image'] = True
                        found_image = True
                        updated += 1
                        break
                if not found_image and row.get('has_image', False):
                    if row.get('hasImage', False) and 'hasImage' in df.columns:
                        df.at[idx, 'hasImage'] = False
                    df.at[idx, 'has_image'] = False
        print(f"Updated {updated} image paths in Fakeddit dataset")
        print(f"Records missing images but marked with image flag: {total_with_image_flag - updated}")
        df.to_csv(fakeddit_path, index=False)
        print(f"Saved updated Fakeddit dataset to {fakeddit_path}")
        return df

    def fix_fakenewsnet_paths(self):
        """Fix image paths for FakeNewsNet dataset. Run before training if image path errors are detected."""
        fakenewsnet_path = os.path.join(self.processed_dir, 'fakenewsnet_processed.csv')
        if not os.path.exists(fakenewsnet_path):
            print(f"FakeNewsNet processed data not found at {fakenewsnet_path}")
            return False
        df = pd.read_csv(fakenewsnet_path)
        print(f"Loaded FakeNewsNet dataset with {len(df)} rows")
        possible_locations = [
            "data/images/fakenewsnet",
            "data/raw/fakenewsnet/images",
            "data/fakenewsnet/images",
            "data/images"
        ]
        actual_location = None
        for location in possible_locations:
            if os.path.exists(location) and len(os.listdir(location)) > 0:
                sources = ['gossipcop', 'politifact']
                if any(source in os.listdir(location) for source in sources):
                    actual_location = location
                    print(f"Found FakeNewsNet images at {location}")
                    print(f"Sample directories: {os.listdir(location)[:5]}")
                    break
        if not actual_location:
            print("Could not find FakeNewsNet images directory")
            return False
        updated = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating FakeNewsNet paths"):
            if 'id' in row and pd.notna(row['id']) and 'dataset_source' in row and row['dataset_source'] == 'fakenewsnet':
                if '_' in row['id']:
                    source, article_id = row['id'].split('_', 1)
                    label = 'fake' if row['label'] == 1 else 'real'
                    article_dir = os.path.join(actual_location, source, label, article_id)
                    if os.path.exists(article_dir):
                        image_files = []
                        for ext in ['jpg', 'jpeg', 'png', 'gif']:
                            image_files.extend(glob(os.path.join(article_dir, f"*.{ext}")))
                        if image_files:
                            df.at[idx, 'image_paths'] = str(image_files)
                            df.at[idx, 'has_image'] = True
                            updated += 1
        print(f"Updated {updated} image paths in FakeNewsNet dataset")
        df.to_csv(fakenewsnet_path, index=False)
        print(f"Saved updated FakeNewsNet dataset to {fakenewsnet_path}")
        return df
    
    def fix_all_image_paths(self):
        """Analyze and fix all image paths across datasets. Run before training if image path errors are detected."""
        print("\n===== Analyzing and fixing all image paths =====")
        
        # Step 1: Analyze current image paths
        print("\nStep 1: Analyzing current image paths")
        self.analyze_image_paths()
        
        # Step 2: Find all images in data directory
        print("\nStep 2: Finding all images in data directory")
        all_images = self.find_actual_images()
        print(f"Found {len(all_images)} images in data directory")
        
        # Create images directory if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Step 3: Fix Fakeddit paths
        print("\nStep 3: Fixing Fakeddit paths")
        fakeddit_df = self.fix_fakeddit_paths()
        
        # Step 4: Fix FakeNewsNet paths
        print("\nStep 4: Fixing FakeNewsNet paths")
        fakenewsnet_df = self.fix_fakenewsnet_paths()
        
        # Step 5: Combine and analyze fixed datasets
        print("\nStep 5: Combining and analyzing fixed datasets")
        combined_df = None
        dfs_to_combine = []
        import pandas as pd
        if isinstance(fakeddit_df, pd.DataFrame) and not fakeddit_df.empty:
            dfs_to_combine.append(fakeddit_df)
        elif fakeddit_df is not None:
            print("Warning: Fakeddit fix function did not return a valid DataFrame.")
        if isinstance(fakenewsnet_df, pd.DataFrame) and not fakenewsnet_df.empty:
            dfs_to_combine.append(fakenewsnet_df)
        elif fakenewsnet_df is not None:
            print("Warning: FakeNewsNet fix function did not return a valid DataFrame.")
        if dfs_to_combine:
            combined_df = pd.concat(dfs_to_combine, ignore_index=True)
            combined_path = os.path.join(self.processed_dir, 'combined_processed.csv')
            combined_df.to_csv(combined_path, index=False)
            print(f"Combined dataset saved to {combined_path}")
            print(f"Combined size: {len(combined_df)} records")
            print(f"Records with valid images: {combined_df['has_image'].sum()} ({combined_df['has_image'].sum()/len(combined_df)*100:.2f}%)")
        
        # Step 6: Create synthetic images if needed
        print("\nStep 6: Creating synthetic images")
        self.synthetic_dir = os.path.join(self.images_dir, 'synthetic')
        if not os.path.exists(self.synthetic_dir) or len(os.listdir(self.synthetic_dir)) < 100:
            synthetic_images = self.create_synthetic_images(num_images=1000)
        else:
            print(f"Using existing synthetic images in {self.synthetic_dir}")
            synthetic_images = [os.path.join(self.synthetic_dir, f) for f in os.listdir(self.synthetic_dir) if f.endswith('.jpg')]
        
        # Step 7: Update dataset with synthetic images for missing ones
        print("\nStep 7: Updating dataset with synthetic images")
        preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
        
        # If we have a preprocessed file, update it
        if os.path.exists(preprocessed_path):
            self.update_dataset_with_synthetic_images(synthetic_image_paths=synthetic_images)
        # Otherwise, if we have a combined dataset, preprocess it and save
        elif combined_df is not None:
            print("Preprocessing combined dataset...")
            preprocessed_df = self.preprocess_dataset(combined_df)
            preprocessed_df.to_csv(preprocessed_path, index=False)
            print(f"Preprocessed dataset saved to {preprocessed_path}")
        
        print("\n===== Image path fixing completed =====")
        return True

    def _build_categorical_vocabs(self, df):
        # Source
        sources = df['metadata'].apply(lambda m: safe_json_loads(m).get('source', ''))
        sources = sources.apply(lambda v: '' if pd.isna(v) else str(v))
        self.source_vocab = {v: i+1 for i, v in enumerate(sorted(set(sources)))}
        self.source_vocab['<UNK>'] = 0
        # Subreddit
        if 'subreddit' in df.columns:
            subreddits = df['metadata'].apply(lambda m: safe_json_loads(m).get('subreddit', ''))
            subreddits = subreddits.apply(lambda v: '' if pd.isna(v) else str(v))
            self.subreddit_vocab = {v: i+1 for i, v in enumerate(sorted(set(subreddits)))}
            self.subreddit_vocab['<UNK>'] = 0
        # Author (if available)
        authors = set()
        for m in df['metadata']:
            try:
                meta = safe_json_loads(m)
                if 'authors' in meta and isinstance(meta['authors'], list):
                    for a in meta['authors']:
                        if pd.isna(a):
                            continue
                        authors.add(str(a))
            except:
                continue
        self.author_vocab = {v: i+1 for i, v in enumerate(sorted(authors))}
        self.author_vocab['<UNK>'] = 0
        self._vocab_ready = True
        # Log vocab sizes
        logger.info(f"Source vocab size: {len(self.source_vocab)}")
        logger.info(f"Subreddit vocab size: {len(self.subreddit_vocab)}")
        logger.info(f"Author vocab size: {len(self.author_vocab)}")
        # Save vocabs
        vocab_dir = os.path.join(self.processed_dir, 'vocabs')
        os.makedirs(vocab_dir, exist_ok=True)
        with open(os.path.join(vocab_dir, 'source_vocab.json'), 'w') as f:
            json.dump(self.source_vocab, f)
        with open(os.path.join(vocab_dir, 'subreddit_vocab.json'), 'w') as f:
            json.dump(self.subreddit_vocab, f)
        with open(os.path.join(vocab_dir, 'author_vocab.json'), 'w') as f:
            json.dump(self.author_vocab, f)

    def _map_categorical_indices(self, row):
        # Map source, subreddit, author to indices
        meta = safe_json_loads(row['metadata'])
        source = meta.get('source', '')
        subreddit = meta.get('subreddit', '')
        authors = meta.get('authors', []) if 'authors' in meta else []
        source_idx = self.source_vocab.get(source, self.source_vocab.get('<UNK>', 0))
        subreddit_idx = self.subreddit_vocab.get(subreddit, self.subreddit_vocab.get('<UNK>', 0))
        author_idx = self.author_vocab.get(authors[0], self.author_vocab.get('<UNK>', 0)) if authors else self.author_vocab.get('<UNK>', 0)
        return source_idx, subreddit_idx, author_idx

def safe_json_loads(s):
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}
