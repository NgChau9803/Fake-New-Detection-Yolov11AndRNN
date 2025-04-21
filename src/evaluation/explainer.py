import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
import pandas as pd
import logging
import shap
from datetime import datetime
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Class to provide explainability for model predictions"""
    
    def __init__(self, model, word_index, config=None):
        """
        Initialize the explainer
        
        Args:
            model: Trained model to explain
            word_index: Word index mapping from tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.word_index = word_index
        self.config = config or {}
        self.idx_to_word = {idx: word for word, idx in word_index.items()}
        
        # Create output directory for explanations
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join('explanations', timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize explainers
        self.text_explainer = LimeTextExplainer(class_names=['Real', 'Fake'])
        self.image_explainer = LimeImageExplainer()
        
        # For tracking processed examples
        self.processed_count = 0
    
    def explain_text(self, text_input, predictions, num_features=10):
        """
        Generate LIME explanation for text input
        
        Args:
            text_input: Tokenized text input
            predictions: Model predictions
            num_features: Number of features to include in explanation
            
        Returns:
            dict: Explanation with feature weights
        """
        # Convert token IDs back to text
        text = ' '.join([self.idx_to_word.get(idx, '[UNK]') for idx in text_input if idx > 0])
        
        # Define prediction function for LIME
        def predict_fn(texts):
            # Create a batch of tokenized inputs with same shape as original
            sequences = []
            for t in texts:
                # Simple tokenization - this would be replaced with proper tokenization
                tokens = t.split()
                # Convert to token IDs using word_index
                sequence = [self.word_index.get(word, 1) for word in tokens]  # 1 is typically OOV token
                # Pad/truncate to match original length
                if len(sequence) > len(text_input):
                    sequence = sequence[:len(text_input)]
                else:
                    sequence = sequence + [0] * (len(text_input) - len(sequence))
                
                sequences.append(sequence)
            
            # Create batch
            batch = tf.constant(sequences, dtype=tf.int32)
            
            # Make prediction using only text input
            # This assumes we can call the model with just the text component
            dummy_image = tf.zeros((len(texts), *self.model.input_shape[1][1:]), dtype=tf.float32)
            dummy_metadata = tf.zeros((len(texts), *self.model.input_shape[2][1:]), dtype=tf.float32)
            
            batch_preds = self.model.predict({
                'text': batch,
                'image': dummy_image,
                'metadata': dummy_metadata
            })
            
            # Return probabilities for both classes
            return np.hstack([1 - batch_preds, batch_preds])
        
        # Generate explanation
        try:
            explanation = self.text_explainer.explain_instance(
                text, 
                predict_fn, 
                num_features=num_features,
                num_samples=100
            )
            
            # Get feature weights
            features = []
            for word, weight in explanation.as_list():
                features.append({
                    'word': word,
                    'weight': float(weight),
                    'is_positive': weight > 0
                })
            
            # Get prediction probabilities
            probs = explanation.predict_proba
            
            return {
                'type': 'text',
                'text': text,
                'prediction': float(predictions),
                'probabilities': {
                    'real': float(1 - predictions),
                    'fake': float(predictions)
                },
                'features': features,
                'explanation_score': explanation.score
            }
            
        except Exception as e:
            logger.error(f"Error generating text explanation: {e}")
            return {
                'type': 'text',
                'text': text,
                'prediction': float(predictions),
                'probabilities': {
                    'real': float(1 - predictions),
                    'fake': float(predictions)
                },
                'error': str(e)
            }
    
    def explain_image(self, image_input, predictions, num_features=5):
        """
        Generate LIME explanation for image input
        
        Args:
            image_input: Image input array
            predictions: Model predictions
            num_features: Number of superpixel features to include
            
        Returns:
            dict: Explanation with superpixel weights
        """
        # Define prediction function for LIME
        def predict_fn(images):
            # Create a batch of images
            batch = np.array(images)
            
            # Make prediction using only image input
            # This assumes we can call the model with just the image component
            dummy_text = tf.zeros((len(images), *self.model.input_shape[0][1:]), dtype=tf.int32)
            dummy_metadata = tf.zeros((len(images), *self.model.input_shape[2][1:]), dtype=tf.float32)
            
            batch_preds = self.model.predict({
                'text': dummy_text,
                'image': batch,
                'metadata': dummy_metadata
            })
            
            # Return probabilities for both classes
            return np.hstack([1 - batch_preds, batch_preds])
        
        try:
            # Generate explanation
            explanation = self.image_explainer.explain_instance(
                image_input, 
                predict_fn, 
                top_labels=2,
                hide_color=0, 
                num_samples=50
            )
            
            # Get explanation for fake class (index 1)
            temp, mask = explanation.get_image_and_mask(
                1,  # Fake news class
                positive_only=False,
                num_features=num_features,
                hide_rest=False
            )
            
            # Create a visualization of the explanation
            # Save the visualization for later use
            img_path = os.path.join(self.output_dir, f"image_explanation_{self.processed_count}.png")
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image_input)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(temp)
            plt.title("Image Explanation")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(img_path)
            plt.close()
            
            return {
                'type': 'image',
                'prediction': float(predictions),
                'probabilities': {
                    'real': float(1 - predictions),
                    'fake': float(predictions)
                },
                'explanation_path': img_path,
                'superpixels': int(np.sum(mask)),
                'has_positive_regions': bool(np.any(mask > 0))
            }
            
        except Exception as e:
            logger.error(f"Error generating image explanation: {e}")
            return {
                'type': 'image',
                'prediction': float(predictions),
                'probabilities': {
                    'real': float(1 - predictions),
                    'fake': float(predictions)
                },
                'error': str(e)
            }
    
    def explain_batch(self, batch_inputs, batch_labels, sample_size=5):
        """
        Generate explanations for a batch of inputs
        
        Args:
            batch_inputs: Batch of inputs with text, image, and metadata
            batch_labels: Corresponding labels
            sample_size: Number of samples to explain
            
        Returns:
            list: List of explanations
        """
        # Get predictions from model
        predictions = self.model(batch_inputs, training=False).numpy()
        
        # Take a sample of the batch for explanation (explaining all can be time-consuming)
        if isinstance(batch_inputs, dict):
            sample_indices = np.random.choice(len(predictions), 
                                             min(sample_size, len(predictions)), 
                                             replace=False)
            
            text_inputs = batch_inputs['text'].numpy()[sample_indices]
            image_inputs = batch_inputs['image'].numpy()[sample_indices]
            sample_preds = predictions[sample_indices]
            sample_labels = batch_labels.numpy()[sample_indices]
        else:
            # Handle case where batch_inputs is not a dictionary
            logger.warning("batch_inputs is not a dictionary, cannot process")
            return []
        
        # Generate explanations for each sample
        explanations = []
        for i in range(len(sample_indices)):
            # Create explanation for this sample
            explanation = {
                'id': int(self.processed_count + i),
                'true_label': int(sample_labels[i]),
                'predicted_label': int(round(sample_preds[i][0])),
                'prediction_score': float(sample_preds[i][0])
            }
            
            # Explain text
            text_explanation = self.explain_text(text_inputs[i], sample_preds[i][0])
            explanation['text_explanation'] = text_explanation
            
            # Explain image if it's not a blank/dummy image
            if np.mean(image_inputs[i]) > 0.01:  # Simple check if image has content
                image_explanation = self.explain_image(image_inputs[i], sample_preds[i][0])
                explanation['image_explanation'] = image_explanation
            
            explanations.append(explanation)
        
        # Update processed count
        self.processed_count += len(sample_indices)
        
        return explanations
    
    def explain_features(self, batch_inputs, batch_labels, feature_names=None):
        """
        Generate global feature importance using SHAP
        
        Args:
            batch_inputs: Batch of inputs
            batch_labels: Corresponding labels
            feature_names: Names of features for display
            
        Returns:
            dict: Feature importance scores
        """
        try:
            # Create a background dataset for SHAP (subset of the data)
            if isinstance(batch_inputs, dict):
                # For text modality
                text_explainer = shap.DeepExplainer(
                    self.model,
                    {
                        'text': batch_inputs['text'][:50],
                        'image': batch_inputs['image'][:50],
                        'metadata': batch_inputs['metadata'][:50]
                    }
                )
                
                # Calculate SHAP values
                shap_values = text_explainer.shap_values(
                    {
                        'text': batch_inputs['text'][:100],
                        'image': batch_inputs['image'][:100],
                        'metadata': batch_inputs['metadata'][:100]
                    }
                )
                
                # Plot and save feature importance
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    features=batch_inputs['text'][:100].numpy(),
                    feature_names=feature_names or [f"Feature_{i}" for i in range(batch_inputs['text'].shape[1])],
                    show=False
                )
                plt.savefig(os.path.join(self.output_dir, "shap_feature_importance.png"))
                plt.close()
                
                return {
                    'type': 'shap',
                    'output_path': os.path.join(self.output_dir, "shap_feature_importance.png")
                }
                
            else:
                logger.warning("batch_inputs is not a dictionary, cannot process SHAP values")
                return {'error': 'Invalid input format for SHAP analysis'}
                
        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            return {'error': str(e)}
    
    def generate_confusion_matrix(self, test_dataset):
        """
        Generate and visualize confusion matrix
        
        Args:
            test_dataset: Test dataset for evaluation
            
        Returns:
            dict: Path to confusion matrix visualization
        """
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Collect predictions and true labels
        y_true = []
        y_pred = []
        
        for batch_inputs, batch_labels in test_dataset:
            batch_preds = self.model(batch_inputs, training=False).numpy()
            batch_preds_binary = (batch_preds > 0.5).astype(int)
            
            y_true.extend(batch_labels.numpy())
            y_pred.extend(batch_preds_binary)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
        disp.plot(cmap='Blues')
        plt.title('Confusion Matrix')
        
        # Save the visualization
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        
        return {'confusion_matrix_path': cm_path}
    
    def save_explanations(self, explanations, config=None):
        """
        Save explanations to file
        
        Args:
            explanations: List of explanation dictionaries
            config: Configuration dictionary
        """
        # Save explanations as JSON
        output_path = os.path.join(self.output_dir, f"explanations_{self.processed_count}.json")
        
        # Convert explanation to JSON-serializable format
        serializable_explanations = []
        for expl in explanations:
            serializable_expl = {}
            for key, value in expl.items():
                if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    serializable_expl[key] = value
                elif isinstance(value, np.ndarray):
                    serializable_expl[key] = value.tolist()
                else:
                    serializable_expl[key] = str(value)
            serializable_explanations.append(serializable_expl)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_explanations, f, indent=2)
        
        # Create visualizations for explanations
        for i, explanation in enumerate(explanations):
            try:
                if 'text_explanation' in explanation and 'features' in explanation['text_explanation']:
                    self._visualize_text_explanation(explanation, i)
            except Exception as e:
                logger.error(f"Error visualizing text explanation: {e}")
    
    def _visualize_text_explanation(self, explanation, index):
        """Create visualization for text explanation"""
        if 'text_explanation' not in explanation or 'features' not in explanation['text_explanation']:
            return
        
        features = explanation['text_explanation']['features']
        
        if not features:
            return
        
        # Extract words and weights
        words = [f['word'] for f in features]
        weights = [f['weight'] for f in features]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 6))
        colors = ['green' if w > 0 else 'red' for w in weights]
        plt.barh(range(len(words)), weights, color=colors)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Feature Importance')
        plt.ylabel('Word')
        plt.title(f"Text Explanation (ID: {explanation['id']}, Prediction: {'Fake' if explanation['predicted_label'] == 1 else 'Real'})")
        
        # Add truth label
        true_label = 'Fake' if explanation['true_label'] == 1 else 'Real'
        plt.text(0.01, 0.01, f"True Label: {true_label}", transform=plt.gca().transAxes)
        
        # Save the plot
        output_path = os.path.join(self.output_dir, f"text_explanation_{explanation['id']}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() 