#!/usr/bin/env python3

"""
Test script for the YOLOv11 backbone in the fake news detection model.
This script demonstrates how to initialize and use the YOLOv11 backbone for feature extraction.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.yolo_backbone import YOLOv11Backbone
from src.models.image_model import ImageFeatureExtractor
from src.data.image_utils import preprocess_image

def parse_args():
    parser = argparse.ArgumentParser(description='Test YOLOv11 backbone for fake news detection')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='yolo_outputs', help='Output directory for visualizations')
    parser.add_argument('--width_mult', type=float, default=0.75, help='Width multiplier for YOLO backbone')
    parser.add_argument('--depth_mult', type=float, default=0.67, help='Depth multiplier for YOLO backbone')
    parser.add_argument('--visualize', action='store_true', help='Visualize feature maps and attention')
    return parser.parse_args()

def visualize_features(feature_maps, output_dir):
    """Visualize feature maps from the YOLOv11 backbone"""
    os.makedirs(output_dir, exist_ok=True)
    
    for key, feature_map in feature_maps.items():
        # Skip non-tensor features
        if not isinstance(feature_map, np.ndarray) and not isinstance(feature_map, tf.Tensor):
            continue
            
        # Convert to numpy if needed
        if isinstance(feature_map, tf.Tensor):
            feature_map = feature_map.numpy()
        
        # Average the channels for visualization
        avg_feature = np.mean(feature_map[0], axis=-1)
        
        # Normalize for better visualization
        min_val = np.min(avg_feature)
        max_val = np.max(avg_feature)
        normalized = (avg_feature - min_val) / (max_val - min_val + 1e-7)
        
        # Save the visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(normalized, cmap='viridis')
        plt.title(f'Feature Map: {key}')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{key}_feature.png'), dpi=300)
        plt.close()
        
        # Visualize individual channels for the first few channels
        num_channels = min(16, feature_map.shape[-1])
        rows = int(np.ceil(np.sqrt(num_channels)))
        cols = int(np.ceil(num_channels / rows))
        
        plt.figure(figsize=(12, 12))
        for i in range(num_channels):
            plt.subplot(rows, cols, i+1)
            channel = feature_map[0, :, :, i]
            channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel) + 1e-7)
            plt.imshow(channel, cmap='viridis')
            plt.title(f'Channel {i+1}')
            plt.axis('off')
        
        plt.suptitle(f'{key} - Channels', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(output_dir, f'{key}_channels.png'), dpi=300)
        plt.close()

def visualize_attention(image_path, attention_map, output_dir):
    """Visualize attention map overlaid on the original image"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    original_img = Image.open(image_path)
    
    # Convert attention map to numpy if needed
    if isinstance(attention_map, tf.Tensor):
        attention_map = attention_map.numpy()
    
    # Create figure with original, attention map, and overlay
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(131)
    plt.imshow(original_img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Attention map
    plt.subplot(132)
    plt.imshow(attention_map, cmap='jet')
    plt.title('Attention Map')
    plt.axis('off')
    
    # Overlay
    plt.subplot(133)
    plt.imshow(original_img)
    plt.imshow(attention_map, cmap='jet', alpha=0.5)
    plt.title('Attention Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_visualization.png'), dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return
    
    # Load and preprocess image
    image_size = (224, 224)
    print(f"Loading image: {args.image}")
    image_array = preprocess_image(args.image, image_size)
    image_tensor = tf.convert_to_tensor([image_array], dtype=tf.float32)
    
    # Create YOLOv11 backbone
    print("Creating YOLOv11 backbone...")
    yolo_backbone = YOLOv11Backbone(
        input_shape=(224, 224, 3),
        width_mult=args.width_mult,
        depth_mult=args.depth_mult,
        use_fpn=True,
        pooling="avg"
    )
    
    # Create ImageFeatureExtractor with YOLOv11 backbone
    print("Creating ImageFeatureExtractor with YOLOv11 backbone...")
    image_extractor = ImageFeatureExtractor(
        input_shape=(224, 224, 3),
        backbone_type="yolov11",
        pretrained=False,
        output_dim=512,
        dropout_rate=0.2,
        use_attention=True,
        pooling="avg",
        backbone_params={
            "width_mult": args.width_mult,
            "depth_mult": args.depth_mult,
            "use_fpn": True
        }
    )
    
    # Extract features using both models
    print("Extracting features...")
    yolo_features = yolo_backbone(image_tensor)
    img_features = image_extractor(image_tensor)
    
    print(f"YOLOv11 backbone output shape: {yolo_features.shape}")
    print(f"ImageFeatureExtractor output shape: {img_features.shape}")
    
    # Visualize if requested
    if args.visualize:
        print("Generating visualizations...")
        
        # Get feature maps
        feature_maps = image_extractor.get_feature_maps(image_tensor)
        
        # Get attention map
        attention_map = image_extractor.get_attention_map(image_tensor)
        
        # Visualize features
        visualize_features(feature_maps, args.output)
        
        # Visualize attention
        visualize_attention(args.image, attention_map, args.output)
        
        print(f"Visualizations saved to: {args.output}")
    
    print("Done!")

if __name__ == "__main__":
    main() 