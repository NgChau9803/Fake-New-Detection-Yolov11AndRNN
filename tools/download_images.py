import os
import pandas as pd
import argparse
import concurrent.futures
from tqdm import tqdm
import yaml
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.image_utils import download_image

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_images_from_dataset(dataset_path, output_dir, num_workers=4):
    """Download images from URLs in the dataset"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    if dataset_path.endswith('.tsv'):
        df = pd.read_csv(dataset_path, sep='\t')
    else:
        df = pd.read_csv(dataset_path)
    
    # Extract image URLs and IDs
    if 'image_url' in df.columns and 'id' in df.columns:
        image_data = list(zip(df['id'], df['image_url']))
    else:
        print("Error: Dataset must contain 'id' and 'image_url' columns")
        return
    
    # Filter out rows with missing URLs
    image_data = [(id_, url) for id_, url in image_data if isinstance(url, str) and url]
    
    print(f"Found {len(image_data)} images to download")
    
    # Download images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create download tasks
        def download_task(item):
            id_, url = item
            output_path = os.path.join(output_dir, f"{id_}.jpg")
            if os.path.exists(output_path):
                return True  # Skip if already downloaded
            return download_image(url, output_path)
        
        # Execute tasks with progress bar
        results = list(tqdm(
            executor.map(download_task, image_data),
            total=len(image_data),
            desc="Downloading images"
        ))
    
    # Report results
    success_count = sum(results)
    print(f"Downloaded {success_count} out of {len(image_data)} images")
    print(f"Images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Download images from dataset')
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset file (overrides config)')
    parser.add_argument('--output', type=str, help='Output directory for images (overrides config)')
    parser.add_argument('--workers', type=int, default=4, help='Number of download workers')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Use command-line arguments if provided, otherwise use config
    dataset_path = args.dataset or config['data']['fakeddit_path']
    output_dir = args.output or config['data']['images_dir']
    
    download_images_from_dataset(dataset_path, output_dir, args.workers)

if __name__ == "__main__":
    main() 