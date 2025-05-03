#!/usr/bin/env python3
"""
Image Path Fixing Script for Multimodal Fake News Detection System
This script helps fix image path issues for both Fakeddit and FakeNewsNet datasets.
"""

import yaml
import os
import sys
import argparse
from src.data.dataset import DatasetProcessor

def main():
    """Main entry point for fixing image paths"""
    parser = argparse.ArgumentParser(description='Fix image paths for datasets')
    parser.add_argument('--dataset', choices=['fakeddit', 'fakenewsnet', 'all'], 
                    default='all', help='Dataset to fix')
    parser.add_argument('--analyze', action='store_true',
                    help='Analyze image paths before fixing')
    parser.add_argument('--create-synthetic', action='store_true',
                    help='Create synthetic images for missing images')
    parser.add_argument('--num-synthetic', type=int, default=100,
                    help='Number of synthetic images to create')
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join('config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure paths
    config['data']['raw_dir'] = 'data/raw'
    config['data']['images_dir'] = 'data/images'
    
    # Create processor
    processor = DatasetProcessor(config)
    
    # Analyze image paths if requested
    if args.analyze:
        print("Analyzing image paths...")
        stats = processor.analyze_image_paths()
        print("\nActual images:")
        processor.find_actual_images()
    
    # Create synthetic images if requested
    if args.create_synthetic:
        print(f"\nCreating {args.num_synthetic} synthetic images...")
        synthetic_paths = processor.create_synthetic_images(num_images=args.num_synthetic)
        processor.update_dataset_with_synthetic_images(synthetic_paths)
    
    # Fix image paths based on dataset selection
    if args.dataset == 'fakeddit' or args.dataset == 'all':
        print("\nFixing Fakeddit image paths...")
        processor.fix_fakeddit_paths()
    
    if args.dataset == 'fakenewsnet' or args.dataset == 'all':
        print("\nFixing FakeNewsNet image paths...")
        processor.fix_fakenewsnet_paths()
    
    if args.dataset == 'all':
        print("\nFixing all image paths...")
        processor.fix_all_image_paths()
    
    print("\nImage path fixing complete!")

if __name__ == "__main__":
    main() 