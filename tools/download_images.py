import os
import json
import argparse
import concurrent.futures
from tqdm import tqdm
import yaml
import sys
import requests

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_image(url, output_path):
    """Download image from URL and save to disk"""
    try:
        # Skip if image already exists
        if os.path.exists(output_path):
            return True
            
        # Download image
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save image
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return False

def download_fakenewnet_images(dataset_path, output_dir, num_workers=4):
    """Download images from FakeNewNet dataset"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON file
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract image URLs and IDs
    image_data = []
    
    # Add top image if available
    if data.get('top_img'):
        image_data.append((data['id'], data['top_img'], '_top'))
    
    # Add all images from the images list
    for i, url in enumerate(data.get('images', [])):
        image_data.append((data['id'], url, f'_{i}'))
    
    print(f"Found {len(image_data)} images to download")
    
    # Download images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create download tasks
        def download_task(item):
            id_, url, suffix = item
            output_path = os.path.join(output_dir, f"{id_}{suffix}.jpg")
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
    parser = argparse.ArgumentParser(description='Download images from FakeNewNet dataset')
    parser.add_argument('--config', type=str, default='../config/config.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset file (overrides config)')
    parser.add_argument('--output', type=str, help='Output directory for images (overrides config)')
    parser.add_argument('--workers', type=int, default=4, help='Number of download workers')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Use command-line arguments if provided, otherwise use config
    dataset_path = args.dataset or config['data']['fakenewnet']['files'][0]  # Use first file
    output_dir = args.output or config['data']['images_dir']
    
    # Download images
    download_fakenewnet_images(dataset_path, output_dir, args.workers)

if __name__ == "__main__":
    main() 