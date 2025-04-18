import os
import json
import argparse
import concurrent.futures
from tqdm import tqdm
import yaml
import sys
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.image_utils import download_image

def setup_logging(dataset_name):
    """Setup logging to file and console"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join('logs', 'downloads')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{dataset_name}_download_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_fakeddit_images(dataset_path, output_dir, dataset_type, num_workers=4):
    """Download images from Fakeddit dataset"""
    # Create Fakeddit-specific output directory with dataset type
    fakeddit_dir = os.path.join(output_dir, 'fakeddit', dataset_type)
    os.makedirs(fakeddit_dir, exist_ok=True)
    
    logging.info(f"Downloading Fakeddit {dataset_type} images to: {fakeddit_dir}")
    
    # Load TSV file
    df = pd.read_csv(dataset_path, sep='\t')
    df = df.replace(np.nan, '', regex=True)
    df.fillna('', inplace=True)
    
    # Filter rows with valid images
    valid_images = df[
        (df['hasImage'] == True) & 
        (df['image_url'] != '') & 
        (df['image_url'] != 'nan')
    ]
    
    # Check for existing files and filter out already downloaded images
    existing_files = set(os.listdir(fakeddit_dir))
    remaining_images = []
    
    for _, row in valid_images.iterrows():
        output_path = os.path.join(fakeddit_dir, f"{row['id']}.jpg")
        if not os.path.exists(output_path):
            remaining_images.append(row)
    
    logging.info(f"Found {len(valid_images)} total images")
    logging.info(f"Already downloaded: {len(valid_images) - len(remaining_images)}")
    logging.info(f"Remaining to download: {len(remaining_images)}")
    
    if not remaining_images:
        logging.info("All images already downloaded!")
        return
    
    # Download remaining images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create download tasks
        def download_task(row):
            image_url = row['image_url']
            output_path = os.path.join(fakeddit_dir, f"{row['id']}.jpg")
            try:
                return download_image(image_url, output_path)
            except Exception as e:
                logging.error(f"Error downloading image {image_url}: {str(e)}")
                return False
        
        # Execute tasks with progress bar
        results = list(tqdm(
            executor.map(download_task, remaining_images),
            total=len(remaining_images),
            desc=f"Downloading Fakeddit {dataset_type} images"
        ))
    
    # Report results
    success_count = sum(results)
    logging.info(f"Downloaded {success_count} out of {len(remaining_images)} remaining images")
    logging.info(f"Images saved to: {fakeddit_dir}")

def download_fakenewsnet_images(json_path, output_dir, num_workers=4):
    """Download images from a FakeNewsNet article JSON file"""
    try:
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
        
        # Get article ID and source/label from the directory path
        article_dir = os.path.dirname(json_path)
        article_id = os.path.basename(article_dir)
        
        # Extract source and label from the path
        path_parts = article_dir.split(os.sep)
        source = path_parts[-3]  # e.g., 'gossipcop' or 'politifact'
        label = path_parts[-2]   # e.g., 'fake' or 'real'
        
        # Create output directory structure: fakenewsnet/source/label/article_id
        article_output_dir = os.path.join(output_dir, 'fakenewsnet', source, label, article_id)
        os.makedirs(article_output_dir, exist_ok=True)
        
        logging.info(f"Processing article {article_id} from {source}/{label}")
        
        # Get image URLs from the article data
        image_urls = []
        if 'images' in article_data:
            image_urls.extend(article_data['images'])
        
        if not image_urls:
            logging.warning(f"No images found in article {article_id}")
            return
        
        # Check for existing images to resume download
        existing_images = set(os.listdir(article_output_dir))
        remaining_urls = [url for url in image_urls 
                         if os.path.basename(url) not in existing_images]
        
        if not remaining_urls:
            logging.info(f"All images already downloaded for article {article_id}")
            return
        
        # Download images in parallel
        def download_image(url):
            try:
                # Get filename from URL
                filename = os.path.basename(url)
                if not filename:
                    filename = f"image_{hash(url)}.jpg"
                
                output_path = os.path.join(article_output_dir, filename)
                
                # Skip if file already exists
                if os.path.exists(output_path):
                    return True
                
                # Download the image
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    logging.info(f"Successfully downloaded image: {filename}")
                    return True
                logging.warning(f"Failed to download image: {filename} (Status: {response.status_code})")
                return False
            except Exception as e:
                logging.error(f"Error downloading {url}: {str(e)}")
                return False
        
        # Use thread pool to download images in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(download_image, remaining_urls),
                total=len(remaining_urls),
                desc=f"Downloading images for {article_id} ({source}/{label})"
            ))
        
        # Count successful downloads
        successful = sum(results)
        logging.info(f"Downloaded {successful}/{len(remaining_urls)} images for article {article_id}")
        
    except Exception as e:
        logging.error(f"Error processing article {json_path}: {str(e)}")
        raise  # Re-raise the exception to handle it in the calling function

def load_fakenewsnet(base_dir, output_dir, num_workers=4):
    """Load and process FakeNewsNet dataset"""
    logging.info(f"Processing FakeNewsNet dataset from {base_dir}")
    
    # Create tracking file path
    tracking_dir = os.path.join(output_dir, 'fakenewsnet', '.tracking')
    os.makedirs(tracking_dir, exist_ok=True)
    tracking_file = os.path.join(tracking_dir, 'processed_articles.txt')
    
    # Load already processed articles
    processed_articles = set()
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            processed_articles = set(line.strip() for line in f)
        logging.info(f"Found {len(processed_articles)} previously processed articles")
    
    # Get all JSON files in the directory structure
    json_files = []
    for source in ['gossipcop', 'politifact']:
        for label in ['fake', 'real']:
            source_label_dir = os.path.join(base_dir, source, label)
            if not os.path.exists(source_label_dir):
                logging.warning(f"Directory not found: {source_label_dir}")
                continue
                
            logging.info(f"Processing {source}/{label}:")
            for article_dir in os.listdir(source_label_dir):
                article_path = os.path.join(source_label_dir, article_dir)
                if not os.path.isdir(article_path):
                    continue
                    
                json_path = os.path.join(article_path, 'news content.json')
                if os.path.exists(json_path):
                    # Skip if article was already processed
                    if article_dir in processed_articles:
                        logging.info(f"Skipping already processed article: {article_dir}")
                        continue
                    json_files.append(json_path)
            
            logging.info(f"Found {len(json_files)} new articles to process in {source}/{label}")
    
    if not json_files:
        logging.info("No new articles found to process")
        return
    
    # Process all found JSON files
    for json_path in json_files:
        try:
            logging.info(f"\nProcessing article: {json_path}")
            article_dir = os.path.basename(os.path.dirname(json_path))
            
            # Download images for this article
            download_fakenewsnet_images(json_path, output_dir, num_workers)
            
            # Mark article as processed
            with open(tracking_file, 'a') as f:
                f.write(f"{article_dir}\n")
            processed_articles.add(article_dir)
            
        except Exception as e:
            logging.error(f"Error processing article {json_path}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Download images from datasets')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--dataset', type=str, choices=['fakeddit', 'fakenewsnet'], required=True, help='Dataset to download images from')
    parser.add_argument('--type', type=str, choices=['train', 'validate', 'test'], help='Dataset type for Fakeddit')
    parser.add_argument('--output', type=str, help='Output directory for images (overrides config)')
    parser.add_argument('--workers', type=int, default=4, help='Number of download workers')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.dataset)
    logging.info(f"Starting download process for {args.dataset}")
    logging.info(f"Log file: {log_file}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set base output directory
    base_output_dir = args.output or os.path.join('E:/BDMProject/images')
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    logging.info(f"Using output directory: {base_output_dir}")
    
    try:
        if args.dataset == 'fakeddit':
            if not args.type:
                logging.error("--type argument is required for Fakeddit dataset")
                return
                
            # Get file path from config
            file_map = {
                'train': 'multimodal_train.tsv',
                'validate': 'multimodal_validate.tsv',
                'test': 'multimodal_test_public.tsv'
            }
            file_name = file_map[args.type]
            dataset_path = os.path.join(config['data']['raw_dir'], 'fakeddit', file_name)
            
            logging.info(f"Loading dataset from: {dataset_path}")
            
            # Download Fakeddit images with dataset type
            download_fakeddit_images(dataset_path, base_output_dir, args.type, args.workers)
            
        elif args.dataset == 'fakenewsnet':
            # Get base directory from config
            base_dir = config['data']['fakenewsnet']['base_dir']
            
            if not os.path.exists(base_dir):
                logging.error(f"Base directory not found: {base_dir}")
                return
                
            logging.info(f"Processing FakeNewsNet dataset from: {base_dir}")
            
            # Process each source (gossipcop and politifact)
            for source in config['data']['fakenewsnet']['sources']:
                source_dir = os.path.join(base_dir, source)
                if not os.path.exists(source_dir):
                    logging.warning(f"Source directory not found: {source_dir}")
                    continue
                    
                # Process each label (fake and real)
                for label in config['data']['fakenewsnet']['labels']:
                    label_dir = os.path.join(source_dir, label)
                    if not os.path.exists(label_dir):
                        logging.warning(f"Label directory not found: {label_dir}")
                        continue
                        
                    # Get all article directories
                    article_dirs = [d for d in os.listdir(label_dir) 
                                  if d.startswith(source) and 
                                  os.path.isdir(os.path.join(label_dir, d))]
                    
                    logging.info(f"\nProcessing {source}/{label}: {len(article_dirs)} articles")
                    
                    # Process each article
                    for article_dir in article_dirs:
                        try:
                            # Construct path to news content JSON
                            json_path = os.path.join(label_dir, article_dir, "news content.json")
                            if not os.path.exists(json_path):
                                logging.warning(f"JSON file not found: {json_path}")
                                continue
                                
                            # Download images for this article
                            download_fakenewsnet_images(json_path, base_output_dir, args.workers)
                        except Exception as e:
                            logging.error(f"Error processing article {article_dir}: {str(e)}")
    
    except KeyboardInterrupt:
        logging.warning("Download process interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        logging.info("Download process completed")

if __name__ == "__main__":
    main() 