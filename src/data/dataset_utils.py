import os
import pandas as pd
import json
from typing import List, Dict, Any

def get_image_paths(article_id: str, dataset_name: str, images_dir: str) -> List[str]:
    """Get image paths for an article based on dataset type"""
    image_paths = []
    if dataset_name == 'fakeddit':
        image_dir = os.path.join(images_dir, 'fakeddit', 'public_image_set')
        for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
            image_path = os.path.join(image_dir, f"{article_id}.{ext}")
            if os.path.exists(image_path):
                return [image_path]
        # Log missing image
        missing_log = os.path.join('logs', 'missing_images.log')
        with open(missing_log, 'a') as f:
            f.write(f"Missing Fakeddit image: {article_id} in {image_dir}\n")
        return []
    elif dataset_name == 'fakenewsnet':
        parts = article_id.split('_')
        if len(parts) < 2:
            return []
        source = parts[0]
        article_dir = parts[1]
        paths = []
        for label in ['real', 'fake']:
            article_img_dir = os.path.join(images_dir, source, label, article_dir)
            if os.path.exists(article_img_dir):
                for filename in os.listdir(article_img_dir):
                    file_path = os.path.join(article_img_dir, filename)
                    if os.path.isfile(file_path) and _is_image_file(filename):
                        paths.append(file_path)
        if not paths:
            missing_log = os.path.join('logs', 'missing_images.log')
            with open(missing_log, 'a') as f:
                f.write(f"Missing FakeNewsNet image: {article_id} in {images_dir}\n")
        return paths
    return []

def _is_image_file(filename: str) -> bool:
    """Check if a file is an image based on extension"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def standardize_metadata(metadata: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Standardize metadata across datasets"""
    if dataset_name == 'fakeddit':
        return {
            'subreddit': metadata.get('subreddit', ''),
            'author': metadata.get('author', ''),
            'timestamp': metadata.get('created_utc', ''),
            'score': metadata.get('score', 0),
            'num_comments': metadata.get('num_comments', 0),
            'upvote_ratio': metadata.get('upvote_ratio', 0.0),
            'domain': metadata.get('domain', '')
        }
    elif dataset_name == 'fakenewsnet':
        # Ensure authors is a list
        authors = metadata.get('authors', [])
        if not isinstance(authors, list):
            authors = [authors] if authors else []
            
        # Ensure keywords is a list
        keywords = metadata.get('keywords', [])
        if not isinstance(keywords, list):
            keywords = [keywords] if keywords else []
            
        return {
            'url': metadata.get('url', ''),
            'title': metadata.get('title', ''),
            'authors': authors,
            'keywords': keywords,
            'publish_date': metadata.get('publish_date', ''),
            'source': metadata.get('source', ''),
            'summary': metadata.get('summary', ''),
            'top_img': metadata.get('top_img', '')
        }
    return {}

def create_standardized_df(data: List[Dict[str, Any]], dataset_name: str) -> pd.DataFrame:
    """Create a standardized DataFrame from processed data"""
    if not data:
        return pd.DataFrame()
        
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add dataset source
    df['dataset_source'] = dataset_name
    
    # Check which images exist
    df['image_paths'] = df['image_paths'].apply(lambda paths: [p for p in paths if os.path.exists(p)])
    df['has_image'] = df['image_paths'].apply(len) > 0
    
    # Standardize columns
    required_columns = [
        'id',
        'text',
        'clean_text',
        'image_paths',
        'label',
        'metadata',
        'dataset_source',
        'file_source',
        'has_image'
    ]
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
            
    # Convert label to numeric if needed
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'fake': 1, 'real': 0})
    
    # Ensure metadata is a dictionary
    df['metadata'] = df['metadata'].apply(lambda x: x if isinstance(x, dict) else {})
    
    # Sort columns
    df = df[required_columns]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

def validate_dataset(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Validate dataset and return statistics"""
    stats = {
        'total_samples': len(df),
        'label_distribution': df['label'].value_counts().to_dict(),
        'text_length_stats': {
            'mean': df['text'].str.len().mean(),
            'min': df['text'].str.len().min(),
            'max': df['text'].str.len().max()
        },
        'image_stats': {
            'total_with_images': df['has_image'].sum(),
            'percentage_with_images': (df['has_image'].sum() / len(df)) * 100
        },
        'metadata_stats': {}
    }
    
    # Add metadata statistics
    if 'metadata' in df.columns:
        # Convert metadata to DataFrame for analysis
        try:
            metadata_df = pd.json_normalize(df['metadata'])
            for col in metadata_df.columns:
                if metadata_df[col].dtype == 'object':
                    stats['metadata_stats'][col] = {
                        'unique_values': metadata_df[col].nunique(),
                        'null_count': metadata_df[col].isnull().sum()
                    }
                else:
                    stats['metadata_stats'][col] = {
                        'mean': metadata_df[col].mean(),
                        'min': metadata_df[col].min(),
                        'max': metadata_df[col].max()
                    }
        except Exception as e:
            print(f"Error analyzing metadata: {e}")
    
    return stats 