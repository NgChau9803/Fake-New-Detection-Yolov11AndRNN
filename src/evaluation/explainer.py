import os
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
import tensorflow as tf
import pandas as pd

class MultimodalExplainer:
    def __init__(self, model, test_dataset, tokenizer, config):
        self.model = model
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = os.path.join('evaluation', 'explanations')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize LIME explainers
        self.text_explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
        self.image_explainer = LimeImageExplainer()
        
    def explain_predictions(self, num_samples=5):
        """Generate explanations for a subset of test samples"""
        # Get a subset of test samples
        test_samples = []
        sample_count = 0
        
        for features, labels in self.test_dataset:
            for i in range(len(labels)):
                if sample_count >= num_samples:
                    break
                    
                sample = {
                    'text_features': features['text'][i].numpy(),
                    'image_features': features['image'][i].numpy(),
                    'metadata_features': features['metadata'][i].numpy(),
                    'label': labels[i].numpy()
                }
                test_samples.append(sample)
                sample_count += 1
                
            if sample_count >= num_samples:
                break
        
        # Generate explanations for each sample
        for i, sample in enumerate(test_samples):
            # Make prediction
            features = {
                'text': tf.expand_dims(tf.convert_to_tensor(sample['text_features']), 0),
                'image': tf.expand_dims(tf.convert_to_tensor(sample['image_features']), 0),
                'metadata': tf.expand_dims(tf.convert_to_tensor(sample['metadata_features']), 0)
            }
            
            prediction = self.model(features).numpy()[0][0]
            predicted_class = 'Fake' if prediction >= 0.5 else 'Real'
            true_class = 'Fake' if sample['label'] == 1 else 'Real'
            
            # Get text explanation
            text_explanation = self.explain_text(sample['text_features'], i)
            
            # Get image explanation
            image_explanation = self.explain_image(sample['image_features'], i)
            
            # Save combined explanation
            self.save_combined_explanation(text_explanation, image_explanation, 
                                         prediction, true_class, predicted_class, i)
    
    def explain_text(self, text_features, sample_idx):
        """Generate LIME explanation for text"""
        # Convert text features back to text
        # This is a simplification - in a real implementation you would need
        # to convert token IDs back to words using the tokenizer
        idx_to_word = {v: k for k, v in self.tokenizer.items()}
        text = ' '.join([idx_to_word.get(idx, '') for idx in text_features if idx > 0])
        
        # Create prediction function for LIME
        def predict_fn(texts):
            # Convert texts to token sequences
            sequences = []
            for t in texts:
                tokens = t.split()
                seq = [self.tokenizer.get(token, 0) for token in tokens]
                # Pad sequence
                if len(seq) > self.config['data']['max_text_length']:
                    seq = seq[:self.config['data']['max_text_length']]
                else:
                    seq = seq + [0] * (self.config['data']['max_text_length'] - len(seq))
                sequences.append(seq)
            
            # Create dummy image and metadata inputs
            batch_size = len(texts)
            dummy_image = np.zeros((batch_size, *self.config['model']['image']['input_shape']))
            dummy_metadata = np.zeros((batch_size, 10))
            
            # Get predictions
            features = {
                'text': tf.convert_to_tensor(sequences),
                'image': tf.convert_to_tensor(dummy_image),
                'metadata': tf.convert_to_tensor(dummy_metadata)
            }
            
            preds = self.model(features).numpy()
            # Return probabilities for both classes [P(real), P(fake)]
            return np.hstack([1-preds, preds])
        
        # Generate explanation
        try:
            exp = self.text_explainer.explain_instance(text, predict_fn, num_features=10)
            
            # Create visualization
            fig = plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title('Text Explanation')
            plt.tight_layout()
            
            # Save visualization
            plt.savefig(os.path.join(self.output_dir, f'text_explanation_{sample_idx}.png'), dpi=300)
            plt.close()
            
            return exp
        except Exception as e:
            print(f"Error generating text explanation: {e}")
            return None
    
    def explain_image(self, image_features, sample_idx):
        """Generate LIME explanation for image"""
        # Create prediction function for LIME
        def predict_fn(images):
            # Preprocess images
            processed_images = np.array(images) / 255.0  # Normalize if not already done
            
            # Create dummy text and metadata inputs
            batch_size = len(images)
            dummy_text = np.zeros((batch_size, self.config['data']['max_text_length']), dtype=np.int32)
            dummy_metadata = np.zeros((batch_size, 10))
            
            # Get predictions
            features = {
                'text': tf.convert_to_tensor(dummy_text),
                'image': tf.convert_to_tensor(processed_images),
                'metadata': tf.convert_to_tensor(dummy_metadata)
            }
            
            preds = self.model(features).numpy()
            # Return probabilities for both classes [P(real), P(fake)]
            return np.hstack([1-preds, preds])
        
        # Generate explanation
        try:
            # Convert to 0-255 range for LIME
            image = (image_features * 255).astype(np.uint8)
            
            exp = self.image_explainer.explain_instance(
                image, 
                predict_fn, 
                top_labels=2,
                hide_color=0, 
                num_samples=100
            )
            
            # Get the explanation for the predicted class (1 = Fake)
            temp, mask = exp.get_image_and_mask(1, positive_only=True, num_features=5, hide_rest=True)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            plt.imshow(temp)
            plt.title('Image Regions Supporting Fake News Classification')
            plt.axis('off')
            
            # Save visualization
            plt.savefig(os.path.join(self.output_dir, f'image_explanation_{sample_idx}.png'), dpi=300)
            plt.close()
            
            return exp
        except Exception as e:
            print(f"Error generating image explanation: {e}")
            return None
    
    def save_combined_explanation(self, text_exp, image_exp, prediction, true_class, predicted_class, sample_idx):
        """Save combined explanation with prediction results"""
        # Create a summary figure
        plt.figure(figsize=(12, 10))
        
        # Add title with prediction info
        plt.suptitle(f"Explanation for Sample #{sample_idx+1}\n" +
                    f"Prediction: {predicted_class} ({prediction:.2f}), True Class: {true_class}", 
                    fontsize=16)
        
        # No actual plotting - just save metadata
        plt.figtext(0.5, 0.5, 
                   "Combined explanation saved as separate text and image files.\n" +
                   f"Text explanation: text_explanation_{sample_idx}.png\n" +
                   f"Image explanation: image_explanation_{sample_idx}.png", 
                   wrap=True, horizontalalignment='center', fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.savefig(os.path.join(self.output_dir, f'combined_explanation_{sample_idx}.png'), dpi=300)
        plt.close() 