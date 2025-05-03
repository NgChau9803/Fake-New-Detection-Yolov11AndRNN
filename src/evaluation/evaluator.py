import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import seaborn as sns
import json
import pandas as pd
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, test_dataset, config, tokenizer=None):
        self.model = model
        self.test_dataset = test_dataset
        self.config = config
        self.tokenizer = tokenizer
        self.metrics = config['evaluation']['metrics']
        self.output_dir = os.path.join('evaluation', 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def evaluate(self):
        """Evaluate the model on the test dataset"""
        # Get predictions
        y_true = []
        y_pred = []
        examples = []
        batch_count = 0
        
        print("Evaluating model on test dataset...")
        for features, labels in tqdm(self.test_dataset):
            batch_pred = self.model(features, training=False)
            batch_pred = tf.sigmoid(batch_pred).numpy()
            
            # Store true labels and predictions
            y_true.extend(labels.numpy())
            y_pred.extend(batch_pred.flatten())
            
            # Store some examples for detailed analysis
            if batch_count < 5:  # Store examples from the first 5 batches
                for i in range(len(labels)):
                    examples.append({
                        'features': {k: v[i].numpy() for k, v in features.items()},
                        'true_label': labels[i].numpy(),
                        'pred_score': batch_pred[i][0],
                        'pred_label': batch_pred[i][0] >= 0.5
                    })
            batch_count += 1
        
        # Convert to binary predictions (threshold = 0.5)
        y_pred_binary = np.array(y_pred) >= 0.5
        y_true = np.array(y_true)
        
        # Calculate metrics
        results = self.compute_performance_metrics(y_true, y_pred_binary, y_pred)
        
        # Generate visualizations
        self.generate_confusion_matrix(y_true, y_pred_binary)
        self.generate_roc_curve(y_true, y_pred)
        self.generate_precision_recall_curve(y_true, y_pred)
        
        # Error analysis
        self.analyze_errors(examples)
        
        # Save evaluation results
        self._save_evaluation_report(results)
        
        return results
        
    def evaluate_cross_dataset(self, train_df, val_df):
        """Evaluate model performance on a dataset different from training"""
        if not self.config['evaluation'].get('cross_dataset_validation', False):
            print("Cross-dataset validation not enabled in config")
            return None
            
        if train_df is None or val_df is None:
            print("No data available for cross-dataset validation")
            return None
            
        from src.data.dataset import DatasetProcessor
        
        print("Performing cross-dataset validation...")
        dataset_processor = DatasetProcessor(self.config)
        
        # Check dataset sources
        train_sources = train_df['dataset_source'].unique()
        val_sources = val_df['dataset_source'].unique()
        
        print(f"Training dataset sources: {train_sources}")
        print(f"Validation dataset sources: {val_sources}")
        
        # Create TensorFlow dataset for validation
        _, _, val_dataset, _ = dataset_processor.create_tf_dataset(val_df)
        
        # Evaluate on validation dataset
        y_true = []
        y_pred = []
        
        print("Evaluating model on cross-dataset validation data...")
        for features, labels in tqdm(val_dataset):
            batch_pred = self.model(features, training=False)
            batch_pred = tf.sigmoid(batch_pred).numpy()
            
            y_true.extend(labels.numpy())
            y_pred.extend(batch_pred.flatten())
        
        # Convert to binary predictions
        y_pred_binary = np.array(y_pred) >= 0.5
        y_true = np.array(y_true)
        
        # Calculate metrics
        results = self.compute_performance_metrics(y_true, y_pred_binary, y_pred)
        
        # Save results
        cross_val_dir = os.path.join(self.output_dir, 'cross_validation')
        os.makedirs(cross_val_dir, exist_ok=True)
        
        # Generate visualizations
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_true, y_pred_binary), annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Cross-Dataset Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(cross_val_dir, 'cross_dataset_confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Save to JSON
        cross_val_results = {
            'cross_dataset_sources': val_sources.tolist(),
            'training_sources': train_sources.tolist(),
            'metrics': results
        }
        
        with open(os.path.join(cross_val_dir, 'cross_dataset_results.json'), 'w') as f:
            json.dump(cross_val_results, f, indent=4)
            
        print("Cross-dataset validation results:")
        for metric, value in results.items():
            if metric != 'full_report':
                print(f"{metric.capitalize()}: {value:.4f}")
                
        return results
    
    def compute_performance_metrics(self, y_true, y_pred_binary, y_pred_proba):
        """Compute performance metrics"""
        # Use sklearn's classification_report for detailed metrics
        report = classification_report(y_true, y_pred_binary, output_dict=True)
        
        # Extract key metrics
        accuracy = report['accuracy']
        
        # Handle potential missing keys in the report
        if '1' in report:
            precision = report['1']['precision']
            recall = report['1']['recall']
            f1 = report['1']['f1-score']
        else:
            weighted_avg = report['weighted avg']
            precision = weighted_avg['precision']
            recall = weighted_avg['recall']
            f1 = weighted_avg['f1-score']
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Calculate class distribution
        class_distribution = {
            'real_count': int(np.sum(y_true == 0)),
            'fake_count': int(np.sum(y_true == 1)),
            'real_percent': float(np.sum(y_true == 0) / len(y_true) * 100),
            'fake_percent': float(np.sum(y_true == 1) / len(y_true) * 100)
        }
        
        # Calculate error distribution
        errors = y_pred_binary != y_true
        error_count = np.sum(errors)
        error_rate = error_count / len(y_true)
        
        false_positive_rate = np.sum((y_pred_binary == 1) & (y_true == 0)) / np.sum(y_true == 0)
        false_negative_rate = np.sum((y_pred_binary == 0) & (y_true == 1)) / np.sum(y_true == 1)
        
        error_distribution = {
            'error_count': int(error_count),
            'error_rate': float(error_rate),
            'false_positive_rate': float(false_positive_rate),
            'false_negative_rate': float(false_negative_rate)
        }
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'pr_auc': pr_auc,
            'class_distribution': class_distribution,
            'error_distribution': error_distribution,
            'full_report': report
        }
        
        return results
    
    def generate_confusion_matrix(self, y_true, y_pred):
        """Generate and save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], 
                   yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
    
    def generate_roc_curve(self, y_true, y_pred_proba):
        """Generate and save ROC curve visualization"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
    
    def generate_precision_recall_curve(self, y_true, y_pred_proba):
        """Generate and save precision-recall curve visualization"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pr_curve.png'), dpi=300)
        plt.close()
    
    def analyze_errors(self, examples):
        """Analyze prediction errors to gain insights"""
        error_examples = []
        correct_examples = []
        
        for ex in examples:
            if ex['true_label'] != (ex['pred_score'] >= 0.5):
                error_examples.append(ex)
            else:
                correct_examples.append(ex)
        
        # Prepare directories
        error_dir = os.path.join(self.output_dir, 'error_analysis')
        os.makedirs(error_dir, exist_ok=True)
        
        # Save error examples
        if error_examples:
            # Extract text features if tokenizer available
            if self.tokenizer:
                idx_to_word = {v: k for k, v in self.tokenizer.items()}
                
                for i, ex in enumerate(error_examples[:10]):  # Limit to 10 examples
                    text_tokens = [idx_to_word.get(idx, '') for idx in ex['features']['text'] if idx > 0]
                    text = ' '.join(text_tokens)
                    
                    with open(os.path.join(error_dir, f'error_example_{i}.txt'), 'w') as f:
                        f.write(f"True label: {'Fake' if ex['true_label'] == 1 else 'Real'}\n")
                        f.write(f"Predicted score: {ex['pred_score']:.4f}\n")
                        f.write(f"Predicted label: {'Fake' if ex['pred_score'] >= 0.5 else 'Real'}\n")
                        f.write(f"Text: {text}\n")
            
            # Save error analysis summary
            error_summary = {
                'total_errors': len(error_examples),
                'false_positives': sum(1 for ex in error_examples if ex['true_label'] == 0),
                'false_negatives': sum(1 for ex in error_examples if ex['true_label'] == 1),
                'example_count': min(len(error_examples), 10)
            }
            
            with open(os.path.join(error_dir, 'error_summary.json'), 'w') as f:
                json.dump(error_summary, f, indent=4)
    
    def _save_evaluation_report(self, results):
        """Save evaluation results to a JSON file"""
        # Convert numpy values to Python native types for JSON serialization
        for key, value in results.items():
            if isinstance(value, np.floating):
                results[key] = float(value)
            elif isinstance(value, np.integer):
                results[key] = int(value)
        
        # Convert nested dictionaries
        if 'full_report' in results:
            for category, metrics in results['full_report'].items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, np.floating):
                            results['full_report'][category][metric_name] = float(metric_value)
        
        # Save to JSON file
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # Also save a readable summary text file
        with open(os.path.join(self.output_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write("Fake News Detection Model Evaluation Summary\n")
            f.write("==========================================\n\n")
            
            # Print key metrics
            f.write("Performance Metrics:\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1 Score: {results['f1']:.4f}\n")
            f.write(f"ROC AUC: {results['auc']:.4f}\n")
            f.write(f"PR AUC: {results['pr_auc']:.4f}\n\n")
            
            # Print class distribution
            dist = results['class_distribution']
            f.write("Class Distribution:\n")
            f.write(f"Real samples: {dist['real_count']} ({dist['real_percent']:.1f}%)\n")
            f.write(f"Fake samples: {dist['fake_count']} ({dist['fake_percent']:.1f}%)\n\n")
            
            # Print error distribution
            err = results['error_distribution']
            f.write("Error Analysis:\n")
            f.write(f"Total errors: {err['error_count']} ({err['error_rate']*100:.1f}%)\n")
            f.write(f"False positive rate: {err['false_positive_rate']*100:.1f}%\n")
            f.write(f"False negative rate: {err['false_negative_rate']*100:.1f}%\n") 