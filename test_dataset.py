#!/usr/bin/env python
import os
import sys
import yaml
import logging
import traceback

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    # Import the DatasetProcessor
    logger.info("Importing DatasetProcessor...")
    from src.data.dataset import DatasetProcessor
    logger.info("Import successful")
    
    # Load configuration
    logger.info("Loading configuration...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded")
    
    # Initialize dataset processor
    logger.info("Initializing DatasetProcessor...")
    dataset_processor = DatasetProcessor(config)
    logger.info("DatasetProcessor initialized")
    
    # List paths in data/processed
    logger.info("Checking processed data directory...")
    processed_dir = os.path.join('data', 'processed')
    if os.path.exists(processed_dir):
        files = os.listdir(processed_dir)
        logger.info(f"Files in {processed_dir}: {files}")
    else:
        logger.warning(f"Directory {processed_dir} does not exist")
    
    # Try to load preprocessed dataset if it exists
    preprocessed_path = os.path.join(processed_dir, 'preprocessed_dataset.csv')
    if os.path.exists(preprocessed_path):
        logger.info(f"Found preprocessed dataset at {preprocessed_path}")
        
        # Check file size
        file_size = os.path.getsize(preprocessed_path) / (1024 * 1024)
        logger.info(f"File size: {file_size:.2f} MB")
        
        # Try to create TF datasets
        logger.info("Creating TensorFlow datasets...")
        try:
            train_dataset, val_dataset, test_dataset, word_index = dataset_processor.create_tf_datasets()
            logger.info("Successfully created TensorFlow datasets!")
            logger.info(f"Vocabulary size: {len(word_index)} tokens")
        except Exception as e:
            logger.error(f"Error creating TensorFlow datasets: {e}")
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"Preprocessed dataset not found at {preprocessed_path}")
    
except Exception as e:
    logger.error(f"Error: {e}")
    logger.error(traceback.format_exc())
    
logger.info("Test completed") 