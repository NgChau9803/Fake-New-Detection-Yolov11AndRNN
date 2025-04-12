import os
import pandas as pd
import json
from typing import List, Dict, Any

def get_image_paths(article_id: str, dataset_name: str, images_dir: str) -> List[str]:
    """Get image paths for an article based on dataset type"""
    if dataset_name == 'fakeddit':
        return [os.path.join(images_dir, f"{article_id}.jpg")]
    elif dataset_name == 'fakenewnet':
        # Try both top image and numbered images
        paths = []
        top_path = os.path.join(images_dir, f"{article_id}_top.jpg")
        if os.path.exists(top_path):
            paths.append(top_path)
        # Check for numbered images
        i = 0
        while True:
            img_path = os.path.join(images_dir, f"{article_id}_{i}.jpg")
            if os.path.exists(img_path):
                paths.append(img_path)
                i += 1
            else:
                break
        return paths
    return []

def standardize_metadata(metadata: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Standardize metadata across datasets"""
    if dataset_name == 'fakeddit':
        return {
            'subreddit': metadata.get('subreddit', ''),
            'timestamp': metadata.get('timestamp', ''),
            'score': metadata.get('score', 0),
            'num_comments': metadata.get('num_comments', 0),
            'upvote_ratio': metadata.get('upvote_ratio', 0.0)
        }
    elif dataset_name == 'fakenewnet':
        return {
            'url': metadata.get('url', ''),
            'title': metadata.get('title', ''),
            'authors': metadata.get('authors', []),
            'keywords': metadata.get('keywords', []),
            'publish_date': metadata.get('publish_date', ''),
            'source': metadata.get('source', ''),
            'summary': metadata.get('summary', '')
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
    
    return stats 