import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class MetricsVisualizer(tf.keras.callbacks.Callback):
    """Custom callback to visualize training metrics during training"""
    
    def __init__(self, log_dir='logs'):
        super(MetricsVisualizer, self).__init__()
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"), 'plots')
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_history = {}
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Store metrics
        for metric, value in logs.items():
            if metric not in self.metrics_history:
                self.metrics_history[metric] = []
            self.metrics_history[metric].append(value)
        
        # Create plots every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == self.params['epochs'] - 1:
            self._create_plots(epoch + 1)
    
    def _create_plots(self, epoch):
        # Create accuracy plot
        if 'accuracy' in self.metrics_history and 'val_accuracy' in self.metrics_history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics_history['accuracy'], label='Training Accuracy')
            plt.plot(self.metrics_history['val_accuracy'], label='Validation Accuracy')
            plt.title(f'Accuracy - Epoch {epoch}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f'accuracy_epoch_{epoch}.png'))
            plt.close()
        
        # Create loss plot
        if 'loss' in self.metrics_history and 'val_loss' in self.metrics_history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics_history['loss'], label='Training Loss')
            plt.plot(self.metrics_history['val_loss'], label='Validation Loss')
            plt.title(f'Loss - Epoch {epoch}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.log_dir, f'loss_epoch_{epoch}.png'))
            plt.close()

class ClassActivationLogger(tf.keras.callbacks.Callback):
    """Custom callback to log class activation maps during training"""
    
    def __init__(self, validation_data, class_names=['Real', 'Fake'], log_dir='logs', num_samples=3):
        super(ClassActivationLogger, self).__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.num_samples = min(num_samples, 5)  # Limit to 5 samples
        self.log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"), 'activations')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Get sample data robustly
        self.sample_data = []
        for batch in validation_data.take(1):
            # If batch is a tuple (features, labels)
            if isinstance(batch, tuple) and len(batch) == 2:
                features, labels = batch
            # If batch is a dict with 'label' key
            elif isinstance(batch, dict) and 'label' in batch:
                features = {k: v for k, v in batch.items() if k != 'label'}
                labels = batch['label']
            else:
                raise ValueError("Validation batch format not recognized for ClassActivationLogger.")
            for i in range(min(self.num_samples, len(labels))):
                label_val = labels[i].numpy() if hasattr(labels[i], 'numpy') else labels[i]
                # Robustly extract scalar
                if isinstance(label_val, (np.ndarray, list)) and len(label_val) == 1:
                    label_val = float(label_val[0])
                elif hasattr(label_val, 'item'):
                    label_val = float(label_val.item())
                else:
                    label_val = float(label_val)
                self.sample_data.append((
                    {k: v[i:i+1] for k, v in features.items()},
                    label_val
                ))
    
    def on_epoch_end(self, epoch, logs=None):
        # Log activations every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == self.params['epochs'] - 1:
            self._log_activations(epoch + 1)
    
    def _log_activations(self, epoch):
        for i, (features, true_label) in enumerate(self.sample_data):
            # Get model prediction and feature maps
            model_outputs = self.model(features, training=False)
            prediction = model_outputs.numpy()[0][0]
            predicted_label = 1 if prediction >= 0.5 else 0
            
            # Get feature maps from image extractor
            if hasattr(self.model, 'image_extractor'):
                feature_maps = self.model.image_extractor.get_feature_maps(features['image'])
                attention_map = self.model.image_extractor.get_attention_map(features['image'])
                
                # Create visualization
                plt.figure(figsize=(15, 5))
                
                # Original image
                plt.subplot(131)
                plt.imshow(features['image'][0])
                plt.title('Original Image')
                plt.axis('off')
                
                # Attention map
                plt.subplot(132)
                plt.imshow(attention_map[0], cmap='jet')
                plt.title('Attention Map')
                plt.axis('off')
                
                # Overlay
                plt.subplot(133)
                plt.imshow(features['image'][0])
                plt.imshow(attention_map[0], cmap='jet', alpha=0.5)
                plt.title('Overlay')
                plt.axis('off')
                
                plt.suptitle(f'Sample {i+1} - Epoch {epoch}\n'
                           f'True: {self.class_names[true_label]} | '
                           f'Pred: {self.class_names[predicted_label]} ({prediction:.4f})')
                plt.savefig(os.path.join(self.log_dir, f'sample_{i+1}_epoch_{epoch}_cam.png'))
                plt.close()
            
            # Create a summary text file
            summary_path = os.path.join(self.log_dir, f'sample_{i+1}_epoch_{epoch}.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Sample {i+1} - Epoch {epoch}\n")
                f.write(f"True label: {self.class_names[true_label]}\n")
                f.write(f"Predicted label: {self.class_names[predicted_label]} (confidence: {prediction:.4f})\n")
                f.write(f"Status: {'Correct' if predicted_label == true_label else 'Incorrect'}\n")
                
                # Add feature map statistics if available
                if hasattr(self.model, 'image_extractor'):
                    f.write("\nFeature Map Statistics:\n")
                    for name, feature_map in feature_maps.items():
                        f.write(f"{name}:\n")
                        f.write(f"  Shape: {feature_map.shape}\n")
                        f.write(f"  Mean: {np.mean(feature_map):.4f}\n")
                        f.write(f"  Std: {np.std(feature_map):.4f}\n")
                        f.write(f"  Max: {np.max(feature_map):.4f}\n")
                        f.write(f"  Min: {np.min(feature_map):.4f}\n") 