#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import pickle
from PIL import Image
import textwrap

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.text_utils import preprocess_text

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_tokenizer(tokenizer_path):
    """Load the saved tokenizer"""
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print(f"Tokenizer loaded from: {tokenizer_path}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def process_text_input(text, tokenizer, max_length):
    """Process text input for model"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize
    sequences = tokenizer.texts_to_sequences([processed_text])
    token_ids = sequences[0]
    tokens = [tokenizer.index_word.get(id, '') for id in token_ids]
    
    # Pad sequence
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        tokens = tokens[:max_length]
    else:
        pad_length = max_length - len(token_ids)
        token_ids = token_ids + [0] * pad_length
        tokens = tokens + [''] * pad_length
    
    return token_ids, tokens

def process_image_input(image_path, target_size=(224, 224)):
    """Process image input for model"""
    try:
        from src.data.image_utils import preprocess_image
        image_array = preprocess_image(image_path, target_size)
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return np.zeros((*target_size, 3))

def visualize_attention(model, text, image_path=None, tokenizer=None, config=None, output_dir='visualization'):
    """Visualize attention weights for text and image inputs"""
    if model is None or tokenizer is None or config is None:
        print("Error: Model, tokenizer, and config are required")
        return
    
    max_length = config['data']['max_text_length']
    
    # Process text input
    token_ids, tokens = process_text_input(text, tokenizer, max_length)
    
    # Create input tensors
    text_tensor = tf.convert_to_tensor([token_ids], dtype=tf.int32)
    
    # Process image input if provided
    image_shape = config['model']['image']['input_shape']
    if image_path and os.path.exists(image_path):
        image_array = process_image_input(image_path, image_shape[:2])
        image_tensor = tf.convert_to_tensor([image_array], dtype=tf.float32)
    else:
        image_tensor = tf.zeros((1, *image_shape), dtype=tf.float32)
    
    # Create dummy metadata
    metadata_tensor = tf.zeros((1, 10), dtype=tf.float32)
    
    # Create input dictionary
    inputs = {
        'text': text_tensor,
        'image': image_tensor,
        'metadata': metadata_tensor
    }
    
    # Get model prediction and attention weights
    try:
        # Get model outputs
        model_outputs = model(inputs)
        prediction = model_outputs.numpy()[0][0]
        print(f"Prediction: {'Fake' if prediction >= 0.5 else 'Real'} ({prediction:.4f})")
        
        # Create visualization directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Visualize text attention if available
        if hasattr(model, 'text_extractor') and hasattr(model.text_extractor, 'attention_weights'):
            text_attention = model.text_extractor.attention_weights
            visualize_text_attention(tokens, text_attention, 
                                  os.path.join(output_dir, f"text_attention_{timestamp}.png"))
        
        # Visualize image attention if available
        if hasattr(model, 'image_extractor'):
            feature_maps = model.image_extractor.get_feature_maps(image_tensor)
            attention_map = model.image_extractor.get_attention_map(image_tensor)
            
            if image_path and os.path.exists(image_path):
                visualize_image_attention(image_path, attention_map, feature_maps,
                                       os.path.join(output_dir, f"image_attention_{timestamp}.png"))
        
        # Create multimodal visualization
        create_multimodal_visualization(text, tokens, prediction, 
                                      image_path if image_path and os.path.exists(image_path) else None,
                                      os.path.join(output_dir, f"multimodal_vis_{timestamp}.png"))
        
        print(f"Visualizations saved to: {output_dir}")
            
    except Exception as e:
        print(f"Error during visualization: {e}")

def visualize_text_attention(tokens, attention_weights, output_path):
    """Visualize text attention weights"""
    # Remove empty tokens (padding)
    valid_tokens = [t for t in tokens if t]
    
    if not valid_tokens:
        print("No valid tokens to visualize")
        return
    
    # Average attention weights across heads
    avg_weights = tf.reduce_mean(attention_weights, axis=-1)[0]  # Take first batch
    
    # Get attention for valid tokens only
    valid_attention = avg_weights[:len(valid_tokens), :len(valid_tokens)]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(valid_attention.numpy(), 
               xticklabels=valid_tokens, 
               yticklabels=valid_tokens,
               cmap='viridis')
    plt.title('Text Attention Weights')
    plt.ylabel('Query Tokens')
    plt.xlabel('Key Tokens')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Create token importance visualization
    token_importance = np.mean(valid_attention.numpy(), axis=0)
    
    plt.figure(figsize=(14, 6))
    plt.bar(valid_tokens, token_importance)
    plt.title('Token Importance Scores')
    plt.xlabel('Tokens')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_importance.png'), dpi=300)
    plt.close()

def visualize_image_attention(image_path, attention_map, feature_maps, output_path):
    """Visualize image attention and feature maps"""
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Attention map
    plt.subplot(132)
    plt.imshow(attention_map[0], cmap='jet')
    plt.title('Attention Map')
    plt.axis('off')
    
    # Overlay
    plt.subplot(133)
    plt.imshow(img)
    plt.imshow(attention_map[0], cmap='jet', alpha=0.5)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Create feature map visualization
    feature_path = output_path.replace('.png', '_features.png')
    num_features = min(8, feature_maps['backbone_output'].shape[-1])
    
    plt.figure(figsize=(15, 5))
    for i in range(num_features):
        plt.subplot(2, 4, i+1)
        plt.imshow(feature_maps['backbone_output'][0, :, :, i], cmap='viridis')
        plt.title(f'Feature {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(feature_path, dpi=300)
    plt.close()

def create_multimodal_visualization(original_text, tokens, prediction, image_path=None, output_path=None):
    """Create a comprehensive multimodal visualization"""
    plt.figure(figsize=(12, 10))
    
    # Layout setup
    if image_path:
        grid = plt.GridSpec(2, 1, height_ratios=[1, 1])
        ax_text = plt.subplot(grid[0])
        ax_image = plt.subplot(grid[1])
    else:
        ax_text = plt.gca()
    
    # Add original text with line wrapping
    wrapped_text = textwrap.fill(original_text, width=80)
    ax_text.text(0.05, 0.95, f"Original Text:\n{wrapped_text}", 
              transform=ax_text.transAxes, fontsize=12,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Add processed tokens
    valid_tokens = [t for t in tokens if t]
    tokens_text = ' '.join(valid_tokens)
    wrapped_tokens = textwrap.fill(tokens_text, width=80)
    ax_text.text(0.05, 0.6, f"Processed Tokens:\n{wrapped_tokens}", 
              transform=ax_text.transAxes, fontsize=12,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Add prediction
    ax_text.text(0.05, 0.3, f"Prediction: {'Fake' if prediction >= 0.5 else 'Real'} ({prediction:.4f})",
              transform=ax_text.transAxes, fontsize=14, fontweight='bold',
              verticalalignment='top', bbox=dict(boxstyle='round', 
                                                facecolor='red' if prediction >= 0.5 else 'green', 
                                                alpha=0.3))
    
    # Add confidence meter
    ax_text.barh(0.15, prediction, height=0.05, color='red' if prediction >= 0.5 else 'green')
    ax_text.barh(0.15, 1-prediction, height=0.05, left=prediction, color='lightgray')
    ax_text.set_xlim(0, 1)
    ax_text.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax_text.text(0.01, 0.15, "Real", fontsize=10, verticalalignment='center')
    ax_text.text(0.99, 0.15, "Fake", fontsize=10, verticalalignment='center', horizontalalignment='right')
    
    # Remove axis ticks for text subplot
    ax_text.set_xticks([])
    ax_text.set_yticks([])
    ax_text.set_title("Fake News Detection Analysis", fontsize=16)
    
    # Add image if provided
    if image_path:
        img = Image.open(image_path)
        ax_image.imshow(img)
        ax_image.set_title("Input Image")
        ax_image.axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize attention weights for text and images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--text', type=str, help='Input text to analyze')
    parser.add_argument('--image', type=str, help='Path to input image (optional)')
    parser.add_argument('--output', type=str, default='visualization', help='Output directory for visualizations')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Load tokenizer
    tokenizer_path = os.path.join(config['data']['processed_dir'], 'tokenizer.pickle')
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        return
    
    if args.interactive:
        print("\nFake News Detection Attention Visualization Tool")
        print("==============================================")
        
        while True:
            text = input("\nEnter text to analyze (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
                
            image_path = input("Enter image path (optional, press Enter to skip): ")
            if image_path and not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}")
                image_path = None
                
            visualize_attention(model, text, image_path, tokenizer, config, args.output)
    else:
        if args.text:
            visualize_attention(model, args.text, args.image, tokenizer, config, args.output)
        else:
            print("Error: Please provide input text with --text or use --interactive mode")

if __name__ == "__main__":
    main() 