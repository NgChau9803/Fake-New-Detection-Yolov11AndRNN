#!/usr/bin/env python
import os
import sys
import argparse
import yaml
import tensorflow as tf
import numpy as np
import logging
import pandas as pd
from datetime import datetime
import json

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import and configure TensorFlow
from src.tf_config import configure_tensorflow
configure_tensorflow()

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {log_file}")

# Import project modules
from src.data.dataset import DatasetProcessor
from src.models.fusion_model import MultiModalFusionModel
from src.training.trainer import ModelTrainer
from src.training.callbacks import MetricsVisualizer, ClassActivationLogger
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.explainer import ModelExplainer
from src.models.model_factory import create_model

def setup_environment():
    """Setup required environment and dependencies"""
    logger.info("Setting up environment...")
    
    # Setup NLTK if available
    try:
        from setup_nltk import setup_nltk
        setup_nltk()
        logger.info("NLTK setup complete")
    except ImportError:
        logger.warning("NLTK setup module not found, skipping NLTK initialization")
    
    # Setup GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.error(f"Error setting GPU memory growth: {e}")
    else:
        logger.info("No GPUs detected, using CPU")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    logger.info("Random seeds set for reproducibility")

def validate_paths(config):
    """Validate that all required paths exist and create if necessary"""
    logger.info("Validating data paths...")
    
    # Convert paths to absolute paths
    base_path = os.getcwd()
    required_dirs = [
        os.path.join(base_path, config['data']['raw_dir']),
        os.path.join(base_path, config['data']['processed_dir']),
        os.path.join(base_path, config['data']['images_dir']),
        os.path.join(base_path, config['data']['cache_dir'])
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Validate dataset files if specified
    if 'fakeddit' in config['data'] and 'files' in config['data']['fakeddit']:
        for file_path in config['data']['fakeddit']['files']:
            abs_file_path = os.path.join(base_path, file_path)
            if not os.path.exists(abs_file_path):
                logger.warning(f"Fakeddit file not found: {abs_file_path}")
    
    if 'fakenewsnet' in config['data'] and 'base_dir' in config['data']['fakenewsnet']:
        fakenewsnet_base = os.path.join(base_path, config['data']['fakenewsnet']['base_dir'])
        if not os.path.exists(fakenewsnet_base):
            logger.warning(f"FakeNewsNet base directory not found: {fakenewsnet_base}")
    
    logger.info("Path validation complete")

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    
    # Validate paths
    validate_paths(config)
    
    return config

def process_datasets(config):
    """Process and prepare datasets (now memory-optimized: no .cache, small prefetch, batch size warning)"""
    # Log memory-relevant config
    logger.info(f"Batch size: {config['training'].get('batch_size', 32)}")
    logger.info(f"num_parallel_calls: {config['data'].get('num_parallel_calls', 2)}")
    logger.info(f"prefetch_buffer_size: {config['data'].get('prefetch_buffer_size', 2)}")
    logger.info("Initializing dataset processor...")
    dataset_processor = DatasetProcessor(config)
    
    logger.info("Processing datasets...")
    try:
        dataset_processor.process_datasets()
    except Exception as e:
        logger.error(f"Error in dataset processing: {e}")
        logger.error("Attempting to continue with existing processed files...")
    
    logger.info("Combining datasets...")
    try:
        combined_df = dataset_processor.combine_datasets()
    except Exception as e:
        logger.error(f"Error combining datasets: {e}")
        logger.error("Attempting to continue with existing combined dataset...")
        combined_df = None
    
    logger.info("Preprocessing dataset...")
    try:
        if combined_df is not None:
            dataset_processor.preprocess_dataset(combined_df)
        else:
            # Try to preprocess without combined_df
            dataset_processor.preprocess_dataset()
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {e}")
        logger.error("Attempting to continue with existing preprocessed dataset...")
    
    logger.info("Creating TensorFlow datasets...")
    try:
        train_dataset, val_dataset, test_dataset, word_index, max_token_index = dataset_processor.create_tf_dataset()
        logger.info(f"Dataset processing complete: {len(word_index)} unique tokens in vocabulary")
        # Log dataset sizes robustly
        for name, ds in [('Train', train_dataset), ('Validation', val_dataset), ('Test', test_dataset)]:
            size = tf.data.experimental.cardinality(ds).numpy()
            if size < 0:
                logger.warning(f"{name} dataset cardinality is unknown (value={size}). This is expected for generator-based datasets. If you want the true size, check the DataFrame split size before creating the dataset.")
            else:
                logger.info(f"{name} dataset size: {size}")
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'word_index': word_index
        }
    except Exception as e:
        logger.error(f"Error creating TensorFlow datasets: {e}")
        logger.error("Aborting pipeline due to dataset creation failure.")
        sys.exit(1)

def build_model(config, vocab_size):
    """Build the multi-modal fusion model"""
    logger.info("Building multi-modal fusion model...")
    
    # Create model using factory
    model = create_model(config, vocab_size)
    
    # Display model summary with dummy data for introspection
    dummy_text = tf.zeros((1, config['data']['max_text_length']), dtype=tf.int32)
    dummy_image = tf.zeros((1, *config['model']['image']['input_shape']), dtype=tf.float32)
    dummy_metadata = tf.zeros((1, 10), dtype=tf.float32)
    dummy_source_idx = tf.zeros((1,), dtype=tf.int32)
    dummy_subreddit_idx = tf.zeros((1,), dtype=tf.int32)
    dummy_author_idx = tf.zeros((1,), dtype=tf.int32)
    
    # Forward pass to build model
    _ = model({
        'text': dummy_text,
        'image': dummy_image,
        'metadata': dummy_metadata,
        'source_idx': dummy_source_idx,
        'subreddit_idx': dummy_subreddit_idx,
        'author_idx': dummy_author_idx
    })
    
    # Print model summary
    model.summary(print_fn=logger.info)
    
    logger.info(f"Model built successfully with fusion method: {config['model']['fusion'].get('fusion_method', 'cross_attention')}")
    
    return model

def train_model(config, model, datasets, output_dir=None):
    """Train the multi-modal fusion model"""
    logger.info("Initializing model trainer...")
    trainer = ModelTrainer(config)
    
    # Create custom callbacks for training visualization
    metrics_visualizer = MetricsVisualizer()
    class_activation_logger = ClassActivationLogger(
        validation_data=datasets['val_dataset'],
        class_names=['Real', 'Fake'],
        num_samples=3
    )
    additional_callbacks = [metrics_visualizer, class_activation_logger]
    
    logger.info(f"Starting training for {config['training']['epochs']} epochs...")
    trained_model, history = trainer.train(
        model=model,
        train_dataset=datasets['train_dataset'],
        val_dataset=datasets['val_dataset'],
        additional_callbacks=additional_callbacks
    )
    
    # If output directory not specified by trainer, create one
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('models', f'fake_news_detector_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model configuration
        with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
            model_config = {
                'vocab_size': len(datasets['word_index']),
                'text_config': config['model']['text'],
                'image_config': config['model']['image'],
                'fusion_config': config['model']['fusion'],
                'timestamp': timestamp
            }
            json.dump(model_config, f, indent=4)
        
        # Save model
        model_save_path = os.path.join(output_dir, 'model')
        trained_model.save(model_save_path)
        logger.info(f"Model saved to: {model_save_path}")
    
    logger.info("Training completed.")
    return trained_model, history, output_dir

def evaluate_model(config, model, dataset, word_index=None, detailed=False):
    """Evaluate the trained model"""
    logger.info("Evaluating model on test dataset...")
    
    if detailed and word_index is not None:
        # Use comprehensive evaluator
        evaluator = ModelEvaluator(model, dataset, config, word_index)
        results = evaluator.evaluate()
        
        # Log detailed metrics
        logger.info("\nDetailed Evaluation Results:")
        for metric, value in results.items():
            if metric not in ['full_report', 'class_distribution', 'error_distribution']:
                logger.info(f"{metric.capitalize()}: {value:.4f}")
    else:
        # Basic evaluation
        test_results = model.evaluate(dataset)
        
        results = {}
        for i, metric_name in enumerate(model.metrics_names):
            results[metric_name] = float(test_results[i])
            logger.info(f"{metric_name}: {results[metric_name]:.4f}")
    
    return results

def explain_predictions(config, model, dataset, word_index):
    """Generate explanations for model predictions"""
    logger.info("Initializing model explainer...")
    
    # Initialize model explainer
    explainer = ModelExplainer(model, word_index, config)
    
    # Sample some examples from the test dataset
    logger.info("Generating explanations for sample predictions...")
    for examples, labels in dataset.take(5):
        explanations = explainer.explain_batch(examples, labels)
        explainer.save_explanations(explanations, config)
    
    logger.info("Explanation generation completed.")

def main():
    """Main pipeline function - unified from both main scripts"""
    # Setup command line parser with all options
    parser = argparse.ArgumentParser(description='Multimodal Fake News Detection Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    # Operation modes
    parser.add_argument('--setup', action='store_true', help='Setup environment and validate paths')
    parser.add_argument('--process_datasets', action='store_true', help='Process and prepare datasets')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--explain', action='store_true', help='Generate explanations for predictions')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    
    # Additional options
    parser.add_argument('--model_path', type=str, help='Path to saved model for evaluation or prediction')
    parser.add_argument('--detailed_evaluation', action='store_true', help='Perform detailed evaluation with additional metrics')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    # Always setup environment
    setup_environment()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print TensorFlow information
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Run the requested steps
    datasets = None
    model = None
    output_dir = None
    
    # Process datasets
    if args.all or args.process_datasets:
        datasets = process_datasets(config)
    
    # Train model
    if args.all or args.train:
        if datasets is None:
            logger.info("Loading pre-processed datasets...")
            dataset_processor = DatasetProcessor(config)
            try:
                train_dataset, val_dataset, test_dataset, word_index, max_token_index = dataset_processor.create_tf_dataset()
            except Exception as e:
                logger.error(f"Failed to create datasets: {e}")
                sys.exit(1)
            datasets = {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset,
                'word_index': word_index
            }
        vocab_size = max_token_index + 1
        model = build_model(config, vocab_size)
        model, history, output_dir = train_model(config, model, datasets)
    
    # Evaluate model
    if args.all or args.evaluate:
        if model is None or datasets is None:
            logger.info("Loading pre-trained model and datasets...")
            
            # Load datasets if not already loaded
            if datasets is None:
                dataset_processor = DatasetProcessor(config)
                train_dataset, val_dataset, test_dataset, word_index, max_token_index = dataset_processor.create_tf_dataset()
                datasets = {
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset,
                    'test_dataset': test_dataset,
                    'word_index': word_index
                }
            
            # Load model if not already loaded
            if model is None:
                model_path = args.model_path or config.get('model_path', 'models/latest/final_model.h5')
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
        
        # Perform evaluation
        test_metrics = evaluate_model(
            config, 
            model, 
            datasets['test_dataset'], 
            datasets['word_index'],
            detailed=args.detailed_evaluation
        )
    
    # Generate explanations
    if args.all or args.explain:
        if model is None or datasets is None:
            logger.info("Loading pre-trained model and datasets...")
            
            # Load datasets if not already loaded
            if datasets is None:
                dataset_processor = DatasetProcessor(config)
                train_dataset, val_dataset, test_dataset, word_index, max_token_index = dataset_processor.create_tf_dataset()
                datasets = {
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset,
                    'test_dataset': test_dataset,
                    'word_index': word_index
                }
            
            # Load model if not already loaded
            if model is None:
                model_path = args.model_path or config.get('model_path', 'models/latest/final_model.h5')
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
        
        explain_predictions(config, model, datasets['test_dataset'], datasets['word_index'])
    
    logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main() 