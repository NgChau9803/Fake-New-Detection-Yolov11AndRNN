import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import argparse
import yaml
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def check_processed_files(processed_dir):
    """Check processed data files"""
    print(f"\nChecking processed files in: {processed_dir}")
    
    # Check if directory exists
    if not os.path.exists(processed_dir):
        print(f"Error: Processed directory not found: {processed_dir}")
        return False
    
    # List all CSV files
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in processed directory")
        return False
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        file_path = os.path.join(processed_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        df = pd.read_csv(file_path)
        print(f"  - {file}: {len(df)} rows, {df.shape[1]} columns, {file_size:.2f} MB")
        
        # Print column names
        print(f"    Columns: {', '.join(df.columns.tolist())}")
        
        # Check for critical columns
        critical_columns = ['id', 'text', 'label']
        missing_columns = [col for col in critical_columns if col not in df.columns]
        if missing_columns:
            print(f"    Warning: Missing critical columns: {', '.join(missing_columns)}")
        
        # Check image paths if available
        if 'processed_image_path' in df.columns:
            valid_images = df['processed_image_path'].notna().sum()
            print(f"    Valid image paths: {valid_images} ({valid_images/len(df)*100:.2f}%)")
            
            # Verify a few image paths
            sample_paths = df[df['processed_image_path'].notna()]['processed_image_path'].sample(min(5, valid_images)).tolist()
            print(f"    Checking {len(sample_paths)} sample image paths:")
            for path in sample_paths:
                exists = os.path.exists(path)
                print(f"      - {path}: {'Exists' if exists else 'Not found'}")
    
    return True

def visualize_samples(processed_dir, images_dir, num_samples=3):
    """Visualize sample processed data"""
    print(f"\nVisualizing {num_samples} sample records")
    
    # Load preprocessed dataset
    preprocessed_path = os.path.join(processed_dir, 'preprocessed_dataset.csv')
    if not os.path.exists(preprocessed_path):
        print(f"Error: Preprocessed dataset not found: {preprocessed_path}")
        return False
    
    # Load dataset
    df = pd.read_csv(preprocessed_path)
    
    # Get samples with valid images
    samples = df[df['processed_image_path'].notna()].sample(min(num_samples, len(df)))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        print(f"\nSample {i+1}:")
        print(f"ID: {row['id']}")
        print(f"Label: {'Fake' if row['label'] == 1 else 'Real'}")
        print(f"Dataset Source: {row.get('dataset_source', 'Unknown')}")
        
        # Print text excerpt
        text = row.get('processed_text', row.get('text', 'No text available'))
        print(f"Text excerpt: {text[:200]}...")
        
        # Try to display image
        image_path = row.get('processed_image_path')
        if image_path and os.path.exists(image_path):
            print(f"Image path: {image_path}")
            print("Image exists: Yes")
            
            # Check sample preprocessed image if available
            sample_img_path = os.path.join(processed_dir, 'sample_images', f"{row['id']}_preprocessed.npy")
            if os.path.exists(sample_img_path):
                print(f"Preprocessed image: Available")
            else:
                print(f"Preprocessed image: Not available")
        else:
            print(f"Image path: {image_path}")
            print("Image exists: No")
    
    # Check preprocessed sample images
    sample_images_dir = os.path.join(processed_dir, 'sample_images')
    if os.path.exists(sample_images_dir):
        sample_files = os.listdir(sample_images_dir)
        print(f"\nFound {len(sample_files)} sample preprocessed images in {sample_images_dir}")
        
        # Show statistics for preprocessed and augmented images
        preprocessed_count = len([f for f in sample_files if "preprocessed" in f])
        augmented_count = len([f for f in sample_files if "augmented" in f])
        print(f"  - Preprocessed images: {preprocessed_count}")
        print(f"  - Augmented images: {augmented_count}")
    
    return True

def check_dataset_stats(processed_dir):
    """Check dataset statistics"""
    print("\nChecking dataset statistics")
    
    # Find statistics files
    stats_files = [f for f in os.listdir(processed_dir) if f.endswith('_stats.txt')]
    if not stats_files:
        print("No statistics files found")
        return False
    
    print(f"Found {len(stats_files)} statistics files:")
    for file in stats_files:
        print(f"\n=== {file} ===")
        with open(os.path.join(processed_dir, file), 'r') as f:
            print(f.read())
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Check Processed Dataset Files')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample data')
    parser.add_argument('--check-stats', action='store_true',
                        help='Check dataset statistics')
    parser.add_argument('--check-all', action='store_true',
                        help='Run all checks')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get paths from config
    processed_dir = os.path.join(os.getcwd(), config['data']['processed_dir'])
    images_dir = config['data']['images_dir']
    
    # Run checks
    check_processed_files(processed_dir)
    
    if args.visualize or args.check_all:
        visualize_samples(processed_dir, images_dir)
    
    if args.check_stats or args.check_all:
        check_dataset_stats(processed_dir)
    
    print("\nCheck completed!")

if __name__ == "__main__":
    main() 