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
from src.data.image_utils import download_image, preprocess_image, augment_image
from src.data.text_utils import preprocess_text
import requests
from typing import List, Dict, Any
from .dataset_utils import get_image_paths, standardize_metadata, create_standardized_df, validate_dataset

class DatasetProcessor:
    def __init__(self, config: Dict[str, Any]):
        """Initialize dataset processor with configuration"""
        self.config = config
        self.raw_dir = config['data']['raw_dir']
        self.processed_dir = config['data']['processed_dir']
        self.images_dir = config['data']['images_dir']
        self.cache_dir = config['data']['cache_dir']
        
        # Check if the raw_dir is an absolute path (indicating it's on a different drive)
        self.raw_dir_is_absolute = os.path.isabs(self.raw_dir)
        
        # Create directories if they don't exist
        if self.raw_dir_is_absolute:
            os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up caching configuration
        self.use_cache = config['data'].get('cache_features', False)
        
        # Set up balanced sampling configuration
        self.balanced_sampling = config['data'].get('balanced_sampling', False)
        
    def process_datasets(self):
        """Process all datasets and save standardized data"""
        print("Processing datasets...")
        
        # Process Fakeddit dataset
        fakeddit_df = self.load_fakeddit()
        if not fakeddit_df.empty:
            self._save_processed_data(fakeddit_df, 'fakeddit')
        
        # Process FakeNewNet dataset
        fakenewnet_df = self.load_fakenewnet()
        if not fakenewnet_df.empty:
            self._save_processed_data(fakenewnet_df, 'fakenewnet')
        
        # Combine datasets if both are available
        if not fakeddit_df.empty and not fakenewnet_df.empty:
            combined_df = pd.concat([fakeddit_df, fakenewnet_df], ignore_index=True)
            self._save_processed_data(combined_df, 'combined')
            
        print("Dataset processing complete!")
    
    def _save_processed_data(self, df: pd.DataFrame, dataset_name: str):
        """Save processed dataset to CSV file"""
        output_path = os.path.join(self.processed_dir, f"{dataset_name}_processed.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved processed {dataset_name} dataset to {output_path}")
        
        # Save statistics
        stats_path = os.path.join(self.processed_dir, f"{dataset_name}_stats.txt")
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
                    authors.update(authors_list)
                f.write(f"Number of unique authors: {len(authors)}\n")
        
        print(f"Saved {dataset_name} statistics to {stats_path}")

    def load_fakeddit(self) -> pd.DataFrame:
        """Load and process Fakeddit dataset"""
        data = []
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
                
                # Print first few rows for debugging
                print(f"\nFirst few rows of {os.path.basename(file_path)}:")
                print(df.head())
                
                for _, row in df.iterrows():
                    try:
                        article_id = str(row['id'])
                        image_paths = get_image_paths(article_id, 'fakeddit', self.config['data']['images_dir'])
                        
                        # Create standardized data entry with error handling
                        entry = {
                            'id': article_id,
                            'text': str(row.get('title', '')) + ' ' + str(row.get('selftext', '')),
                            'clean_text': self._preprocess_text(str(row.get('title', '')) + ' ' + str(row.get('selftext', ''))),
                            'image_paths': image_paths,
                            'label': 1 if row.get('2_way_label') == 1 else 0,  # Convert to binary classification
                            'metadata': standardize_metadata({
                                'subreddit': row.get('subreddit', ''),
                                'score': row.get('score', 0),
                                'num_comments': row.get('num_comments', 0),
                                'upvote_ratio': row.get('upvote_ratio', 0.0),
                                'created_utc': row.get('created_utc', '')
                            }, 'fakeddit'),
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

    def load_fakenewnet(self) -> pd.DataFrame:
        """Load and process FakeNewNet dataset"""
        data = []
        base_dir = self.config['data']['fakenewnet']['base_dir']
        
        # Validate base directory exists
        if not os.path.exists(base_dir):
            print(f"Error: Base directory not found: {base_dir}")
            return pd.DataFrame()
            
        # Process each source (gossipcop and politifact)
        for source in self.config['data']['fakenewnet']['sources']:
            source_dir = os.path.join(base_dir, source)
            if not os.path.exists(source_dir):
                print(f"Warning: Source directory not found: {source_dir}")
                continue
                
            # Process each label (fake and real)
            for label in self.config['data']['fakenewnet']['labels']:
                label_dir = os.path.join(source_dir, label)
                if not os.path.exists(label_dir):
                    print(f"Warning: Label directory not found: {label_dir}")
                    continue
                    
                # Get all article directories
                article_dirs = [d for d in os.listdir(label_dir) 
                              if d.startswith(f"{source}-id") and 
                              os.path.isdir(os.path.join(label_dir, d))]
                
                print(f"Processing {source}/{label}: {len(article_dirs)} articles")
                
                # Process each article
                for article_dir in article_dirs:
                    try:
                        # Construct path to news content JSON
                        json_path = os.path.join(label_dir, article_dir, "news content.json")
                        if not os.path.exists(json_path):
                            print(f"Warning: JSON file not found: {json_path}")
                            continue
                            
                        # Load article data
                        with open(json_path, 'r', encoding='utf-8') as f:
                            article = json.load(f)
                            
                        # Extract article ID
                        article_id = article_dir.split('-id')[1]
                        
                        # Get image paths
                        image_paths = get_image_paths(article_id, 'fakenewnet', self.config['data']['images_dir'])
                        
                        # Create standardized data entry
                        data.append({
                            'id': f"{source}_{article_id}",
                            'text': article.get('text', ''),
                            'clean_text': self._preprocess_text(article.get('text', '')),
                            'image_paths': image_paths,
                            'label': 1 if label == 'fake' else 0,
                            'metadata': standardize_metadata(article, 'fakenewnet'),
                            'file_source': f"{source}/{label}/{article_dir}"
                        })
                        
                    except Exception as e:
                        print(f"Error processing article {article_dir}: {e}")
                        continue
                        
        if not data:
            print("No valid FakeNewNet data found")
            return pd.DataFrame()
            
        # Create standardized DataFrame
        df = create_standardized_df(data, 'fakenewnet')
        
        # Validate dataset and print statistics
        stats = validate_dataset(df, 'fakenewnet')
        print("\nFakeNewNet Dataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Label distribution: {stats['label_distribution']}")
        print(f"Text length - Mean: {stats['text_length_stats']['mean']:.2f}, "
              f"Min: {stats['text_length_stats']['min']}, "
              f"Max: {stats['text_length_stats']['max']}")
        print(f"Images available: {stats['image_stats']['total_with_images']} "
              f"({stats['image_stats']['percentage_with_images']:.2f}%)")
        
        # Print metadata statistics
        print("\nMetadata Statistics:")
        for field, field_stats in stats['metadata_stats'].items():
            if isinstance(field_stats, dict):
                if 'unique_values' in field_stats:
                    print(f"{field}: {field_stats['unique_values']} unique values, "
                          f"{field_stats['null_count']} null values")
                else:
                    print(f"{field}: Mean={field_stats['mean']:.2f}, "
                          f"Min={field_stats['min']}, Max={field_stats['max']}")
        
        return df

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text according to configuration"""
        # Implement text preprocessing based on config
        return text

    def _download_image(self, url, filename):
        """Download image from URL and save to disk"""
        try:
            # Create full path
            image_path = os.path.join(self.images_dir, filename)
            
            # Skip if image already exists
            if os.path.exists(image_path):
                return image_path
            
            # Download image
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save image
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return image_path
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def combine_datasets(self):
        """Combine standardized datasets"""
        fakeddit_df = self.load_fakeddit()
        fakenewnet_df = self.load_fakenewnet()
        
        # Check if datasets are empty
        if fakeddit_df.empty and fakenewnet_df.empty:
            raise ValueError("No valid data files found. Please check file paths and formats.")
            
        # Combine datasets
        dfs_to_combine = []
        if not fakeddit_df.empty:
            dfs_to_combine.append(fakeddit_df)
        if not fakenewnet_df.empty:
            dfs_to_combine.append(fakenewnet_df)
            
        combined_df = pd.concat(dfs_to_combine, ignore_index=True)
        
        # Save combined dataset
        combined_path = os.path.join(self.processed_dir, 'combined_dataset.csv')
        combined_df.to_csv(combined_path, index=False)
        
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
    
    def preprocess_dataset(self, df=None):
        """Preprocess the combined dataset for training"""
        if df is None:
            # Load from saved file if not provided
            combined_path = os.path.join(self.processed_dir, 'combined_dataset.csv')
            if os.path.exists(combined_path):
                df = pd.read_csv(combined_path)
            else:
                df = self.combine_datasets()
        
        # Apply balanced sampling if enabled
        df = self.apply_balanced_sampling(df)
        
        # Process text data
        print("Processing text data...")
        df['processed_text'] = df['clean_text'].apply(lambda x: preprocess_text(x, 
                                                     max_length=self.config['data']['max_text_length']))
        
        # Process image data
        print("Processing image data...")
        def process_image_url(row):
            image_id = row['id']
            image_url = row['image_path']
            image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
            
            if not pd.isna(image_url) and isinstance(image_url, str) and image_url.strip():
                if not os.path.exists(image_path):
                    try:
                        success = download_image(image_url, image_path)
                        if not success:
                            return None
                    except Exception as e:
                        print(f"Error downloading image for ID {image_id}: {e}")
                        return None
                return image_path
            return None
        
        df['image_path'] = df.apply(process_image_url, axis=1)
        
        # Save preprocessed dataset
        preprocessed_path = os.path.join(self.processed_dir, 'preprocessed_dataset.csv')
        df.to_csv(preprocessed_path, index=False)
        
        # Print statistics
        print(f"Preprocessed dataset size: {len(df)} records")
        print(f"Records with valid text: {df['processed_text'].notna().sum()} ({df['processed_text'].notna().sum()/len(df)*100:.2f}%)")
        print(f"Records with valid images: {df['image_path'].notna().sum()} ({df['image_path'].notna().sum()/len(df)*100:.2f}%)")
        
        return df
    
    def _get_cache_path(self, dataset_hash):
        """Generate a cache file path based on dataset hash"""
        return os.path.join(self.cache_dir, f"features_{dataset_hash}.pkl")
    
    def _compute_dataset_hash(self, df):
        """Compute a hash of the dataset for caching purposes"""
        # Use a subset of columns to compute the hash
        hash_columns = ['id', 'processed_text', 'image_path', 'label']
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
            tokenizer = Tokenizer(num_words=self.config['data']['vocab_size'])
            tokenizer.fit_on_texts(df['processed_text'].fillna(''))
            
            # Save the tokenizer
            with open(os.path.join(self.processed_dir, 'tokenizer.pickle'), 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Function to create features and labels
            def create_features(row):
                # Text features
                text = row['processed_text'] if pd.notna(row['processed_text']) else ''
                text_sequence = tokenizer.texts_to_sequences([text])[0]
                # Pad sequence to fixed length
                if len(text_sequence) > self.config['data']['max_text_length']:
                    text_sequence = text_sequence[:self.config['data']['max_text_length']]
                else:
                    text_sequence = text_sequence + [0] * (self.config['data']['max_text_length'] - len(text_sequence))
                
                # Image features - load and preprocess image if available
                image_features = np.zeros(self.config['model']['image']['input_shape'])
                if pd.notna(row['image_path']) and os.path.exists(str(row['image_path'])):
                    try:
                        image_features = preprocess_image(row['image_path'], 
                                                        target_size=self.config['model']['image']['input_shape'][:2])
                    except Exception as e:
                        print(f"Error preprocessing image {row['image_path']}: {e}")
                
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
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'features': features_list,
                        'labels': labels_list,
                        'tokenizer': tokenizer
                    }, f)
        
        # Split into train, validation, and test sets
        from sklearn.model_selection import train_test_split
        
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
                        'text': tf.TensorSpec(shape=(self.config['data']['max_text_length'],), dtype=tf.int32),
                        'image': tf.TensorSpec(shape=self.config['model']['image']['input_shape'], dtype=tf.float32),
                        'metadata': tf.TensorSpec(shape=(10,), dtype=tf.float32)
                    },
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            )
            
            if is_training:
                dataset = dataset.shuffle(buffer_size=min(len(features), 10000))
            
            dataset = dataset.batch(self.config['training']['batch_size'])
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

    def move_raw_data(self, source_dir, target_dir):
        """
        Move raw data from source directory to target directory
        
        Args:
            source_dir (str): Source directory path
            target_dir (str): Target directory path
        """
        if not os.path.exists(source_dir):
            print(f"Source directory does not exist: {source_dir}")
            return False
            
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # Get list of items in source directory
            items = os.listdir(source_dir)
            
            for item in items:
                source_item = os.path.join(source_dir, item)
                target_item = os.path.join(target_dir, item)
                
                # Skip if item already exists in target
                if os.path.exists(target_item):
                    print(f"Item already exists in target: {target_item}")
                    continue
                
                # Move directory or file
                if os.path.isdir(source_item):
                    print(f"Moving directory: {source_item} -> {target_item}")
                    shutil.copytree(source_item, target_item)
                    shutil.rmtree(source_item)
                else:
                    print(f"Moving file: {source_item} -> {target_item}")
                    shutil.copy2(source_item, target_item)
                    os.remove(source_item)
                    
            print(f"Successfully moved data from {source_dir} to {target_dir}")
            return True
            
        except Exception as e:
            print(f"Error moving data: {e}")
            return False 