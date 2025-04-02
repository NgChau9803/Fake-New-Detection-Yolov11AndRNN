import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import pickle
import hashlib
from tqdm import tqdm
from sklearn.utils import resample
from src.data.image_utils import download_image, preprocess_image, augment_image
from src.data.text_utils import preprocess_text
import requests

class DatasetProcessor:
    def __init__(self, config):
        self.config = config
        self.processed_dir = config['data']['processed_dir']
        self.images_dir = config['data']['images_dir']
        self.cache_dir = config['data']['cache_dir']
        
        # Create directories if they don't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up caching configuration
        self.use_cache = config['data'].get('cache_features', False)
        
        # Set up balanced sampling configuration
        self.balanced_sampling = config['data'].get('balanced_sampling', False)
        
    def load_fakeddit(self):
        """Load and standardize Fakeddit dataset files"""
        fakeddit_config = self.config['data']['fakeddit']
        fakeddit_files = fakeddit_config['files']
        file_type = fakeddit_config['file_type']
        
        all_data = []
        
        for file_path in fakeddit_files:
            try:
                if os.path.exists(file_path):
                    # Load TSV file
                    df = pd.read_csv(file_path, sep='\t')
                    
                    # Check for required columns
                    required_columns = [
                        'id',
                        'title',
                        'clean_title',
                        'image_url',
                        '2_way_label',
                        'author',
                        'subreddit',
                        'domain',
                        'score',
                        'upvote_ratio',
                        'num_comments'
                    ]
                    
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"Warning: Missing required columns in {file_path}: {missing_columns}")
                        continue
                    
                    # Map image paths using post IDs
                    df['image_path'] = df['id'].apply(lambda x: os.path.join(self.images_dir, f"{x}.jpg"))
                    
                    # Check if images exist
                    df['has_image'] = df['image_path'].apply(os.path.exists)
                    print(f"Found {df['has_image'].sum()} images out of {len(df)} posts")
                    
                    # Standardize column names and create metadata
                    standardized_df = pd.DataFrame({
                        'id': df['id'],
                        'text': df['title'],
                        'clean_text': df['clean_title'],
                        'image_path': df['image_path'],
                        'label': df['2_way_label'],
                        'metadata': df[['author', 'subreddit', 'domain', 'score', 'upvote_ratio', 'num_comments']].to_dict('records'),
                        'dataset_source': 'fakeddit',
                        'file_source': file_path,
                        'has_image': df['has_image']
                    })
                    
                    # Print dataset statistics
                    print(f"\nFakeddit Dataset Statistics from {file_path}:")
                    print(f"Total samples: {len(standardized_df)}")
                    print(f"Label distribution:\n{standardized_df['label'].value_counts()}")
                    print(f"Average text length: {standardized_df['text'].str.len().mean():.2f}")
                    print(f"Images available: {standardized_df['has_image'].sum()} ({standardized_df['has_image'].sum()/len(standardized_df)*100:.2f}%)")
                    
                    # Print metadata statistics
                    print("\nMetadata Statistics:")
                    print(f"Number of unique authors: {len(df['author'].unique())}")
                    print(f"Number of unique subreddits: {len(df['subreddit'].unique())}")
                    print(f"Number of unique domains: {len(df['domain'].unique())}")
                    print(f"Average score: {df['score'].mean():.2f}")
                    print(f"Average upvote ratio: {df['upvote_ratio'].mean():.2f}")
                    print(f"Average number of comments: {df['num_comments'].mean():.2f}")
                    
                    all_data.append(standardized_df)
                    print(f"\nLoaded {len(standardized_df)} records from {file_path}")
                else:
                    print(f"Warning: File not found: {file_path}")
            except Exception as e:
                print(f"Error loading Fakeddit dataset file {file_path}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Print combined dataset statistics
            print("\nCombined Fakeddit Dataset Statistics:")
            print(f"Total samples: {len(combined_df)}")
            print(f"Label distribution:\n{combined_df['label'].value_counts()}")
            print(f"Images available: {combined_df['has_image'].sum()} ({combined_df['has_image'].sum()/len(combined_df)*100:.2f}%)")
            
            return combined_df
        else:
            print("No valid Fakeddit data files found.")
            return pd.DataFrame()
            
    def load_fakenewnet(self):
        """Load and standardize FakeNewNet dataset files"""
        fakenewnet_config = self.config['data']['fakenewnet']
        base_dir = fakenewnet_config['base_dir']
        sources = fakenewnet_config['sources']
        labels = fakenewnet_config['labels']
        
        all_data = []
        
        for source in sources:
            for label in labels:
                source_dir = os.path.join(base_dir, source, label)
                
                if not os.path.exists(source_dir):
                    print(f"Warning: Directory not found: {source_dir}")
                    continue
                
                # Get all article directories
                article_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
                print(f"Processing {source}_{label}: {len(article_dirs)} articles")
                
                for article_dir in tqdm(article_dirs, desc=f"Processing {source}_{label}"):
                    try:
                        # Load news content JSON
                        json_path = os.path.join(source_dir, article_dir, "news content.json")
                        if not os.path.exists(json_path):
                            continue
                            
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Extract article ID from directory name
                        article_id = article_dir.split('-')[-1]
                        
                        # Create standardized data
                        standardized_data = {
                            'id': f"{source}_{article_id}",
                            'text': data.get('text', ''),
                            'clean_text': data.get('title', ''),
                            'image_paths': [],  # Will be populated after downloading
                            'label': 1 if label == 'fake' else 0,
                            'metadata': {
                                'source': source,
                                'publish_date': data.get('publish_date'),
                                'authors': data.get('authors', []),
                                'keywords': data.get('keywords', []),
                                'canonical_link': data.get('canonical_link', ''),
                                'summary': data.get('summary', ''),
                                'url': data.get('url', '')
                            },
                            'dataset_source': 'fakenewnet',
                            'file_source': f"{source}_{label}",
                            'has_image': False
                        }
                        
                        # Map image paths
                        image_paths = []
                        
                        # Add top image if available
                        if data.get('top_img'):
                            top_img_path = os.path.join(self.images_dir, f"{standardized_data['id']}_top.jpg")
                            if os.path.exists(top_img_path):
                                image_paths.append(top_img_path)
                        
                        # Add all images from the images list
                        for i, _ in enumerate(data.get('images', [])):
                            img_path = os.path.join(self.images_dir, f"{standardized_data['id']}_{i}.jpg")
                            if os.path.exists(img_path):
                                image_paths.append(img_path)
                        
                        standardized_data['image_paths'] = image_paths
                        standardized_data['has_image'] = len(image_paths) > 0
                        
                        all_data.append(pd.DataFrame([standardized_data]))
                        
                    except Exception as e:
                        print(f"Error processing {json_path}: {e}")
                
                if all_data:
                    print(f"\nFakeNewNet Dataset Statistics for {source}_{label}:")
                    current_df = pd.concat(all_data[-len(article_dirs):], ignore_index=True)
                    print(f"Total samples: {len(current_df)}")
                    print(f"Label distribution:\n{current_df['label'].value_counts()}")
                    print(f"Average text length: {current_df['text'].str.len().mean():.2f}")
                    print(f"Images available: {current_df['has_image'].sum()} ({current_df['has_image'].sum()/len(current_df)*100:.2f}%)")
                    
                    # Print metadata statistics
                    print("\nMetadata Statistics:")
                    print(f"Number of unique sources: {len(current_df['metadata'].apply(lambda x: x['source']).unique())}")
                    print(f"Number of unique authors: {len(set(author for authors in current_df['metadata'].apply(lambda x: x['authors']) for author in authors))}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Print combined dataset statistics
            print("\nCombined FakeNewNet Dataset Statistics:")
            print(f"Total samples: {len(combined_df)}")
            print(f"Label distribution:\n{combined_df['label'].value_counts()}")
            print(f"Images available: {combined_df['has_image'].sum()} ({combined_df['has_image'].sum()/len(combined_df)*100:.2f}%)")
            
            return combined_df
        else:
            print("No valid FakeNewNet data files found.")
            return pd.DataFrame()
    
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