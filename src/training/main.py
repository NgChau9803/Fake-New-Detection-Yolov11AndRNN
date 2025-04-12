import os
import sys
import yaml
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.data.dataset import DatasetProcessor
from src.models.fusion_model import MultiModalFusionModel
from src.training.trainer import ModelTrainer
from src.training.callbacks import MetricsVisualizer, ClassActivationLogger
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.explainer import MultimodalExplainer
from src.models.model_factory import create_model

def setup_environment():
    """Setup required environment and dependencies"""
    # Setup NLTK
    from setup_nltk import setup_nltk
    setup_nltk()
    
    # Setup GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

def validate_paths(config):
    """Validate that all required paths exist"""
    required_dirs = [
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['images_dir'],
        config['data']['cache_dir']
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Validate dataset files
    for file_path in config['data']['fakeddit']['files']:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fakeddit file not found: {file_path}")
    
    fakenewnet_base = config['data']['fakenewnet']['base_dir']
    if not os.path.exists(fakenewnet_base):
        raise FileNotFoundError(f"FakeNewNet base directory not found: {fakenewnet_base}")

def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate paths
    validate_paths(config)
    return config

def train_model(config):
    """Train the multimodal fake news detection model"""
    print("\n=== Preparing Datasets ===")
    dataset_processor = DatasetProcessor(config)
    
    # Process datasets
    dataset_processor.process_datasets()
    
    # Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset, word_index = dataset_processor.create_tf_datasets()
    
    print(f"Vocabulary size: {len(word_index)}")
    print(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy()}")
    print(f"Test dataset size: {tf.data.experimental.cardinality(test_dataset).numpy()}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = create_model(config, len(word_index))
    
    # Display model summary
    for features, _ in train_dataset.take(1):
        _ = model(features)
        break
    model.summary()
    
    # Save model architecture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('models', f'fake_news_detector_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model configuration
    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        model_config = {
            'vocab_size': len(word_index),
            'text_config': config['model']['text'],
            'image_config': config['model']['image'],
            'fusion_config': config['model']['fusion'],
            'timestamp': timestamp
        }
        json.dump(model_config, f, indent=4)
    
    # Train model
    print("\n=== Training Model ===")
    trainer = ModelTrainer(config)
    
    # Add custom callbacks
    metrics_visualizer = MetricsVisualizer()
    class_activation_logger = ClassActivationLogger(
        validation_data=val_dataset,
        class_names=['Real', 'Fake'],
        num_samples=3
    )
    
    # Train the model with custom callbacks
    trained_model, history = trainer.train(
        model, 
        train_dataset, 
        val_dataset,
        additional_callbacks=[metrics_visualizer, class_activation_logger]
    )
    
    # Save model
    model_save_path = os.path.join(output_dir, 'model')
    trained_model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Evaluate model
    print("\n=== Evaluating Model ===")
    evaluator = ModelEvaluator(trained_model, test_dataset, config, word_index)
    results = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        if metric not in ['full_report', 'class_distribution', 'error_distribution']:
            print(f"{metric.capitalize()}: {value:.4f}")
    
    # Generate explanations
    print("\n=== Generating Explanations ===")
    explainer = MultimodalExplainer(trained_model, test_dataset, word_index, config)
    explainer.explain_predictions(num_samples=config['evaluation']['num_explanation_samples'])
    
    print("\nTraining and evaluation complete!")
    return trained_model, results

def evaluate_model(model_path, config):
    """Evaluate a pretrained model"""
    print("\n=== Loading Model ===")
    model = tf.keras.models.load_model(model_path)
    
    print("\n=== Preparing Test Dataset ===")
    dataset_processor = DatasetProcessor(config)
    
    # Create TensorFlow datasets
    _, _, test_dataset, word_index = dataset_processor.create_tf_datasets()
    
    print("\n=== Evaluating Model ===")
    evaluator = ModelEvaluator(model, test_dataset, config, word_index)
    results = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        if metric not in ['full_report', 'class_distribution', 'error_distribution']:
            print(f"{metric.capitalize()}: {value:.4f}")
    
    # Generate explanations
    print("\n=== Generating Explanations ===")
    explainer = MultimodalExplainer(model, test_dataset, word_index, config)
    explainer.explain_predictions(num_samples=config['evaluation']['num_explanation_samples'])
    
    print("\nEvaluation complete!")
    return results

def main():
    """Main function to run the entire pipeline"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fake News Detection with Multimodal Analysis")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="train", 
                        choices=["train", "evaluate", "predict", "process_data"],
                        help="Operation mode: train, evaluate, predict, or process_data")
    parser.add_argument("--model_path", type=str, help="Path to saved model for evaluation or prediction")
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode == "process_data":
        print("Processing datasets...")
        dataset_processor = DatasetProcessor(config)
        dataset_processor.process_datasets()
        print("Dataset processing completed.")
        
    elif args.mode == "train":
        train_model(config)
        
    elif args.mode == "evaluate":
        if not args.model_path:
            print("Error: Model path must be specified for evaluation mode")
            sys.exit(1)
        evaluate_model(args.model_path, config)
        
    elif args.mode == "predict":
        if not args.model_path:
            print("Error: Model path must be specified for prediction mode")
            sys.exit(1)
        # TODO: Implement prediction mode
        print("Prediction mode not yet implemented")

if __name__ == "__main__":
    main() 