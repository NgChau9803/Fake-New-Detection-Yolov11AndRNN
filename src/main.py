import os
import sys
import yaml
import argparse
from tqdm import tqdm
from src.data.dataset import DatasetProcessor

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def process_data(config, args):
    """Process datasets based on configuration"""
    # Initialize dataset processor
    processor = DatasetProcessor(config)
    
    # Process datasets
    if args.process_datasets:
        print("Processing raw datasets...")
        processor.process_datasets()
        
    # Combine datasets
    if args.combine:
        print("Combining datasets...")
        combined_df = processor.combine_datasets()
        
    # Preprocess dataset
    if args.preprocess:
        print("Preprocessing combined dataset...")
        preprocessed_df = processor.preprocess_dataset()
        
    # Create TensorFlow datasets
    if args.create_tf_datasets:
        print("Creating TensorFlow datasets...")
        train_ds, val_ds, test_ds, word_index = processor.create_tf_datasets()
        print(f"Created {len(word_index)} word vocabulary")
        
    # Create cross-dataset validation
    if args.cross_validation and config['evaluation'].get('cross_dataset_validation', False):
        print("Creating cross-dataset validation...")
        train_df, val_df = processor.create_cross_dataset_validation_set()
        if train_df is not None and val_df is not None:
            print("Cross-dataset validation sets created successfully")

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection Data Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--process_datasets', action='store_true',
                        help='Process raw datasets')
    parser.add_argument('--combine', action='store_true',
                        help='Combine processed datasets')
    parser.add_argument('--preprocess', action='store_true',
                        help='Preprocess combined dataset')
    parser.add_argument('--create_tf_datasets', action='store_true',
                        help='Create TensorFlow datasets')
    parser.add_argument('--cross_validation', action='store_true',
                        help='Create cross-dataset validation')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps of the pipeline')
    
    args = parser.parse_args()
    
    # If all flag is set, enable all steps
    if args.all:
        args.process_datasets = True
        args.combine = True
        args.preprocess = True
        args.create_tf_datasets = True
        args.cross_validation = True
    
    # If no steps specified, show help
    if not any([args.process_datasets, args.combine, args.preprocess,
                args.create_tf_datasets, args.cross_validation]):
        parser.print_help()
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Process data
    process_data(config, args)
    
    print("Data pipeline execution complete!")

if __name__ == "__main__":
    main() 