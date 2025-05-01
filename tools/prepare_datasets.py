import os
import argparse
import yaml
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import DatasetProcessor

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_datasets(config):
    """Prepare and combine datasets using DatasetProcessor"""
    print("Initializing dataset processor...")
    dataset_processor = DatasetProcessor(config)
    
    print("Processing datasets...")
    dataset_processor.process_datasets()
    
    print("Datasets prepared successfully!")

def main():
    parser = argparse.ArgumentParser(description='Prepare and combine datasets')
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    prepare_datasets(config)

if __name__ == "__main__":
    main() 