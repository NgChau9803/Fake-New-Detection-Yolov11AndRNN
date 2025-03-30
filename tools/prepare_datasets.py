import os
import argparse
import yaml
import sys
import pandas as pd

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import DatasetProcessor

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_datasets(config):
    """Prepare and combine datasets"""
    print("Initializing dataset processor...")
    dataset_processor = DatasetProcessor(config)
    
    print("Loading and combining datasets...")
    combined_df = dataset_processor.combine_datasets()
    print(f"Combined dataset size: {len(combined_df)} rows")
    
    print("Preprocessing dataset...")
    preprocessed_df = dataset_processor.preprocess_dataset(combined_df)
    print(f"Preprocessed dataset size: {len(preprocessed_df)} rows")
    
    print(f"Datasets prepared and saved to: {config['data']['processed_dir']}")
    
    # Display dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(preprocessed_df)}")
    
    if 'label' in preprocessed_df.columns:
        label_counts = preprocessed_df['label'].value_counts()
        print("Label distribution:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} samples ({count/len(preprocessed_df)*100:.2f}%)")
    
    # Check for images
    if 'image_path' in preprocessed_df.columns:
        has_image = preprocessed_df['image_path'].notna().sum()
        print(f"Samples with images: {has_image} ({has_image/len(preprocessed_df)*100:.2f}%)")
    
    # Check for text
    if 'processed_text' in preprocessed_df.columns:
        has_text = preprocessed_df['processed_text'].notna().sum()
        print(f"Samples with text: {has_text} ({has_text/len(preprocessed_df)*100:.2f}%)")
    
    return preprocessed_df

def main():
    parser = argparse.ArgumentParser(description='Prepare and combine datasets')
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    prepare_datasets(config)

if __name__ == "__main__":
    main() 