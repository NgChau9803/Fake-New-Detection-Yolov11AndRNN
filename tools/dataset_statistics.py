#!/usr/bin/env python3

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
from collections import Counter
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.text_utils import compute_text_features

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def analyze_dataset(file_path, file_type='csv', output_dir=None):
    """Analyze a single dataset file and generate statistics"""
    print(f"\nAnalyzing file: {file_path}")
    
    # Load the dataset
    try:
        if file_type.lower() == 'tsv':
            df = pd.read_csv(file_path, sep='\t')
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    print(f"Dataset shape: {df.shape}")
    
    # Basic statistics
    stats = {
        'file_path': file_path,
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
        'missing_values': {col: int(df[col].isna().sum()) for col in df.columns}
    }
    
    # Check for label columns
    label_columns = [col for col in df.columns if 'label' in col.lower()]
    if label_columns:
        stats['label_columns'] = {}
        for col in label_columns:
            value_counts = df[col].value_counts().to_dict()
            # Convert to strings for JSON serialization
            value_counts = {str(k): int(v) for k, v in value_counts.items()}
            stats['label_columns'][col] = value_counts
    
    # Check for text columns
    text_columns = [col for col in df.columns if any(x in col.lower() for x in ['title', 'text', 'content', 'comment'])]
    if text_columns:
        stats['text_columns'] = {}
        for col in text_columns:
            if df[col].dtype == 'object':  # Only analyze string columns
                # Sample a subset for text analysis
                sample_size = min(1000, len(df))
                sample_df = df.sample(sample_size, random_state=42)
                
                # Calculate text statistics
                text_stats = []
                for text in tqdm(sample_df[col].fillna(''), desc=f"Analyzing {col}"):
                    if isinstance(text, str):
                        text_stats.append(compute_text_features(text))
                
                # Convert to DataFrame for easy aggregation
                text_stats_df = pd.DataFrame(text_stats)
                
                # Calculate and store aggregated statistics
                stats['text_columns'][col] = {
                    'avg_char_count': float(text_stats_df['char_count'].mean()),
                    'avg_word_count': float(text_stats_df['word_count'].mean()),
                    'avg_sentence_count': float(text_stats_df['sentence_count'].mean()),
                    'url_percentage': float(text_stats_df['has_urls'].mean() * 100),
                    'email_percentage': float(text_stats_df['has_email'].mean() * 100),
                    'numbers_percentage': float(text_stats_df['has_numbers'].mean() * 100),
                }
    
    # Check for image URL columns
    image_columns = [col for col in df.columns if any(x in col.lower() for x in ['image', 'img', 'photo', 'picture'])]
    if image_columns:
        stats['image_columns'] = {}
        for col in image_columns:
            if df[col].dtype == 'object':  # Only analyze string columns
                non_null_count = df[col].notna().sum()
                stats['image_columns'][col] = {
                    'count': int(non_null_count),
                    'percentage': float(non_null_count / len(df) * 100)
                }
    
    # Save statistics to JSON file if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(file_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{file_name}_stats.json")
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"Statistics saved to: {output_path}")
        
        # Generate visualizations
        if label_columns:
            for col in label_columns:
                plt.figure(figsize=(10, 6))
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel('Label')
                plt.ylabel('Count')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{file_name}_{col}_distribution.png"))
                plt.close()
        
        if text_columns:
            for col in text_columns[:1]:  # Just plot first text column to avoid too many plots
                if df[col].dtype == 'object':
                    plt.figure(figsize=(10, 6))
                    sample_df = df.sample(min(1000, len(df)), random_state=42)
                    sample_df[col].fillna('').apply(len).plot(kind='hist', bins=50)
                    plt.title(f'Distribution of {col} Length')
                    plt.xlabel('Length (characters)')
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{file_name}_{col}_length_distribution.png"))
                    plt.close()
    
    return stats

def analyze_all_datasets(config, output_dir='data/statistics'):
    """Analyze all datasets specified in the configuration"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    all_stats = {
        'fakeddit': [],
        'fakenewsnet': []
    }
    
    # Analyze Fakeddit datasets
    fakeddit_files = config['data']['fakeddit']['files']
    fakeddit_type = config['data']['fakeddit']['file_type']
    
    for file_path in fakeddit_files:
        if os.path.exists(file_path):
            stats = analyze_dataset(file_path, fakeddit_type, output_dir)
            if stats:
                all_stats['fakeddit'].append(stats)
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Analyze FakeNewsNet datasets
    fakenewsnet_files = config['data']['fakenewsnet']['files']
    fakenewsnet_type = config['data']['fakenewsnet']['file_type']
    
    for file_path in fakenewsnet_files:
        if os.path.exists(file_path):
            stats = analyze_dataset(file_path, fakenewsnet_type, output_dir)
            if stats:
                all_stats['fakenewsnet'].append(stats)
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Save combined statistics
    with open(os.path.join(output_dir, 'all_datasets_stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    # Generate summary report
    total_rows = 0
    total_data_mb = 0
    label_distribution = Counter()
    
    # Count total rows and data size
    for dataset_type in all_stats:
        for dataset_stats in all_stats[dataset_type]:
            total_rows += dataset_stats['num_rows']
            total_data_mb += dataset_stats['memory_usage']
            
            # Aggregate label distribution if available
            if 'label_columns' in dataset_stats:
                for col, distribution in dataset_stats['label_columns'].items():
                    for label, count in distribution.items():
                        label_distribution[f"{dataset_type}_{col}_{label}"] += count
    
    # Create summary text file
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Dataset Statistics Summary Report\n")
        f.write("===============================\n\n")
        
        f.write(f"Total files analyzed: {len(all_stats['fakeddit']) + len(all_stats['fakenewsnet'])}\n")
        f.write(f"Total data rows: {total_rows:,}\n")
        f.write(f"Total data size: {total_data_mb:.2f} MB\n\n")
        
        # Fakeddit stats
        f.write(f"Fakeddit files: {len(all_stats['fakeddit'])}\n")
        f.write(f"FakeNewsNet files: {len(all_stats['fakenewsnet'])}\n\n")
        
        # Label distribution summary
        f.write("Label Distribution:\n")
        for label, count in label_distribution.most_common():
            f.write(f"  {label}: {count:,}\n")
    
    print(f"\nAnalysis complete. Reports saved to: {output_dir}")
    print(f"Summary report: {os.path.join(output_dir, 'summary_report.txt')}")

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset statistics')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--file', type=str, help='Analyze a specific file instead of all files in config')
    parser.add_argument('--type', type=str, choices=['csv', 'tsv'], default='csv', help='File type (for single file analysis)')
    parser.add_argument('--output', type=str, default='data/statistics', help='Output directory for statistics')
    
    args = parser.parse_args()
    
    # Check if analyzing a single file or all files from config
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        
        analyze_dataset(args.file, args.type, args.output)
    else:
        # Load configuration
        config = load_config(args.config)
        analyze_all_datasets(config, args.output)

if __name__ == "__main__":
    main() 