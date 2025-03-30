import os
import yaml
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import json

from src.data.dataset import DatasetProcessor
from src.models.fusion_model import MultiModalFusionModel
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.explainer import MultimodalExplainer

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(config):
    """Train the multimodal fake news detection model"""
    print("\n=== Preparing Datasets ===")
    dataset_processor = DatasetProcessor(config)
    
    # Preprocess dataset
    combined_df = dataset_processor.combine_datasets()
    preprocessed_df = dataset_processor.preprocess_dataset(combined_df)
    
    # Check if cross-dataset validation is enabled
    train_df, cross_val_df = None, None
    if config['evaluation'].get('cross_dataset_validation', False):
        train_df, cross_val_df = dataset_processor.create_cross_dataset_validation_set(preprocessed_df)
    
    # Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset, word_index = dataset_processor.create_tf_datasets(
        preprocessed_df if train_df is None else train_df
    )
    
    print(f"Vocabulary size: {len(word_index)}")
    print(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Validation dataset size: {tf.data.experimental.cardinality(val_dataset).numpy()}")
    print(f"Test dataset size: {tf.data.experimental.cardinality(test_dataset).numpy()}")
    
    # Create model
    print("\n=== Creating Model ===")
    model = MultiModalFusionModel(len(word_index), config)
    
    # Display model summary
    # Create sample input to build the model
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
    trained_model, history = trainer.train(model, train_dataset, val_dataset)
    
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
    
    # Perform cross-dataset validation if enabled
    if config['evaluation'].get('cross_dataset_validation', False) and cross_val_df is not None:
        print("\n=== Cross-Dataset Validation ===")
        cross_val_results = evaluator.evaluate_cross_dataset(train_df, cross_val_df)
    
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
    
    # Preprocess dataset for testing
    combined_df = dataset_processor.combine_datasets()
    preprocessed_df = dataset_processor.preprocess_dataset(combined_df)
    
    # Create TensorFlow datasets
    _, _, test_dataset, word_index = dataset_processor.create_tf_datasets(preprocessed_df)
    
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

def visualize_attention(model_path, config, sample_text):
    """Visualize attention weights for a sample text"""
    print("\n=== Loading Model for Attention Visualization ===")
    model = tf.keras.models.load_model(model_path)
    
    # Load tokenizer
    print("Loading tokenizer...")
    import pickle
    processed_dir = config['data']['processed_dir']
    tokenizer_path = os.path.join(processed_dir, 'tokenizer.pickle')
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return
    
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Tokenize and preprocess sample text
    from src.data.text_utils import preprocess_text
    
    preprocessed = preprocess_text(sample_text)
    token_ids = tokenizer.texts_to_sequences([preprocessed])[0]
    
    # Pad sequence
    max_length = config['data']['max_text_length']
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids = token_ids + [0] * (max_length - len(token_ids))
    
    # Create dummy inputs
    image_shape = config['model']['image']['input_shape']
    dummy_image = np.zeros((1, *image_shape))
    dummy_metadata = np.zeros((1, 10))
    
    inputs = {
        'text': tf.convert_to_tensor([token_ids], dtype=tf.int32),
        'image': tf.convert_to_tensor(dummy_image, dtype=tf.float32),
        'metadata': tf.convert_to_tensor(dummy_metadata, dtype=tf.float32)
    }
    
    # Get prediction
    prediction = model(inputs).numpy()[0][0]
    print(f"Prediction: {'Fake' if prediction >= 0.5 else 'Real'} ({prediction:.4f})")
    
    # Get token importance if the model has attention weights
    if hasattr(model.text_extractor, 'get_token_importance'):
        print("\nToken importance:")
        token_importance = model.text_extractor.get_token_importance(
            inputs['text'], tokenizer.word_index
        )
        
        for token, importance in token_importance:
            if token:
                print(f"{token}: {importance:.4f}")
    else:
        print("This model doesn't support attention visualization")

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection with Multimodal Analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'visualize'], default='train', help='Operation mode')
    parser.add_argument('--model', type=str, help='Path to pretrained model for evaluation or visualization mode')
    parser.add_argument('--text', type=str, help='Sample text for attention visualization')
    
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
    elif args.mode == 'visualize':
        if args.model is None:
            print("Error: Model path is required for visualization mode")
            return
        if args.text is None:
            print("Error: Sample text is required for visualization mode")
            return
        visualize_attention(args.model, config, args.text)

if __name__ == "__main__":
    main() 