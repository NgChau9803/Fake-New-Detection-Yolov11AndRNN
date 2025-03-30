import os
import yaml
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

from data.dataset import DatasetProcessor
from models.fusion_model import MultiModalFusionModel
from training.trainer import ModelTrainer
from evaluation.evaluator import ModelEvaluator
from evaluation.explainer import MultimodalExplainer

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config):
    """Train the multimodal fake news detection model"""
    print("Preparing dataset...")
    dataset_processor = DatasetProcessor(config)
    
    # Preprocess dataset
    combined_df = dataset_processor.combine_datasets()
    preprocessed_df = dataset_processor.preprocess_dataset(combined_df)
    
    # Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset, word_index = dataset_processor.create_tf_datasets(preprocessed_df)
    
    print(f"Vocabulary size: {len(word_index)}")
    print(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy()}")
    print(f"Test dataset size: {tf.data.experimental.cardinality(test_dataset).numpy()}")
    
    # Create model
    print("Creating model...")
    model = MultiModalFusionModel(len(word_index), config)
    
    # Display model summary
    # Create sample input to build the model
    for features, _ in train_dataset.take(1):
        _ = model(features)
        break
    
    model.summary()
    
    # Train model
    print("Training model...")
    trainer = ModelTrainer(config)
    trained_model, history = trainer.train(model, train_dataset, val_dataset)
    
    # Save model
    model_save_path = os.path.join('models', f'fake_news_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    trained_model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ModelEvaluator(trained_model, test_dataset, config)
    results = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        if metric != 'full_report':
            print(f"{metric.capitalize()}: {value:.4f}")
    
    # Generate explanations
    print("\nGenerating explanations...")
    explainer = MultimodalExplainer(trained_model, test_dataset, word_index, config)
    explainer.explain_predictions(num_samples=config['evaluation']['num_explanation_samples'])
    
    print("\nTraining and evaluation complete!")
    return trained_model, results

def evaluate_model(model_path, config):
    """Evaluate a pretrained model"""
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Preparing test dataset...")
    dataset_processor = DatasetProcessor(config)
    _, _, test_dataset, word_index = dataset_processor.create_tf_datasets()
    
    print("Evaluating model...")
    evaluator = ModelEvaluator(model, test_dataset, config)
    results = evaluator.evaluate()
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        if metric != 'full_report':
            print(f"{metric.capitalize()}: {value:.4f}")
    
    # Generate explanations
    print("\nGenerating explanations...")
    explainer = MultimodalExplainer(model, test_dataset, word_index, config)
    explainer.explain_predictions(num_samples=config['evaluation']['num_explanation_samples'])
    
    print("\nEvaluation complete!")
    return results

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection with Multimodal Analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train', help='Operation mode')
    parser.add_argument('--model', type=str, help='Path to pretrained model for evaluation mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set memory growth for GPUs
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
    
    if args.mode == 'train':
        train_model(config)
    elif args.mode == 'evaluate':
        if args.model is None:
            print("Error: Model path is required for evaluation mode")
            return
        evaluate_model(args.model, config)

if __name__ == "__main__":
    main()
