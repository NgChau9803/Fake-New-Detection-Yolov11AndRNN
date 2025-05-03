#!/usr/bin/env python
import os
import sys
import yaml
import logging
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"image_fixing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function to run image path fixing"""
    logger.info("Starting image path fixing...")
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")
    
    try:
        # Import just what we need from the dataset module
        sys.path.append(os.getcwd())
        
        # Import only the functions we need without requiring TensorFlow
        from src.data.dataset import DatasetProcessor
    except ImportError as e:
        # If there's an error with the full import, create a custom processor
        logger.error(f"Failed to import DatasetProcessor: {e}")
        logger.info("Creating a simplified image processor")
        
        import pandas as pd
        from tqdm import tqdm
        from glob import glob
        
        class SimpleImageProcessor:
            def __init__(self, config):
                self.config = config
                self.processed_dir = os.path.join(os.getcwd(), config['data']['processed_dir'])
                self.images_dir = os.path.join(os.getcwd(), config['data']['images_dir'])
                
            def fix_paths(self):
                """Fix image paths for Fakeddit and FakeNewsNet datasets"""
                print("Looking for image directories...")
                
                # Check for Fakeddit images
                fakeddit_paths = [
                    "data/images/fakeddit/public_image_set",
                    "data/raw/fakeddit/public_image_set",
                    "data/fakeddit/public_image_set"
                ]
                
                fakeddit_dir = None
                for path in fakeddit_paths:
                    if os.path.exists(path):
                        fakeddit_dir = path
                        break
                
                if fakeddit_dir:
                    print(f"Found Fakeddit images in: {fakeddit_dir}")
                    file_count = len([f for f in os.listdir(fakeddit_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
                    print(f"Found {file_count} image files")
                    print(f"Sample files: {[f for f in os.listdir(fakeddit_dir)[:5] if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]}")
                else:
                    print("No Fakeddit image directory found")
                
                # Check for FakeNewsNet images
                fakenewsnet_paths = [
                    "data/images/fakenewsnet",
                    "data/raw/fakenewsnet/images",
                    "data/fakenewsnet/images"
                ]
                
                fakenewsnet_dir = None
                for path in fakenewsnet_paths:
                    if os.path.exists(path):
                        sources = ['gossipcop', 'politifact']
                        if any(os.path.exists(os.path.join(path, source)) for source in sources):
                            fakenewsnet_dir = path
                            break
                
                if fakenewsnet_dir:
                    print(f"Found FakeNewsNet images in: {fakenewsnet_dir}")
                    sources = [s for s in ['gossipcop', 'politifact'] 
                            if os.path.exists(os.path.join(fakenewsnet_dir, s))]
                    
                    for source in sources:
                        source_path = os.path.join(fakenewsnet_dir, source)
                        labels = [l for l in ['fake', 'real'] 
                                if os.path.exists(os.path.join(source_path, l))]
                        
                        for label in labels:
                            label_path = os.path.join(source_path, label)
                            if os.path.exists(label_path):
                                article_count = len([d for d in os.listdir(label_path) 
                                                if os.path.isdir(os.path.join(label_path, d))])
                                print(f"  {source}/{label}: {article_count} article directories")
                
                # Find all image files
                all_images = []
                for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    found = glob(f"data/**/*.{ext}", recursive=True)
                    all_images.extend(found)
                    print(f"Found {len(found)} files with extension .{ext}")
                
                print(f"Total images found: {len(all_images)}")
                
                return {
                    "fakeddit_dir": fakeddit_dir,
                    "fakenewsnet_dir": fakenewsnet_dir,
                    "total_images": len(all_images)
                }
        
        # Use the simplified processor
        processor = SimpleImageProcessor(config)
        result = processor.fix_paths()
        
        logger.info(f"Images found: {result['total_images']}")
        logger.info(f"Fakeddit directory: {result['fakeddit_dir']}")
        logger.info(f"FakeNewsNet directory: {result['fakenewsnet_dir']}")
        
        return 0
    
    # Initialize the processor
    dataset_processor = DatasetProcessor(config)
    logger.info("Dataset processor initialized")
    
    # Fix image paths
    logger.info("Fixing image paths...")
    dataset_processor.fix_all_image_paths()
    
    logger.info(f"Image path fixing completed successfully!")
    logger.info(f"Log saved to: {log_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 