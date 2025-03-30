import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.learning_rate = config['training']['learning_rate']
        self.epochs = config['training']['epochs']
        self.model_dir = os.path.join('models', datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.model_dir, exist_ok=True)
        
    def create_callbacks(self):
        """Create training callbacks"""
        # Model checkpoint to save best model
        checkpoint_path = os.path.join(self.model_dir, 'best_model.h5')
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        # TensorBoard for visualization
        log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        
        return [checkpoint_callback, early_stopping, tensorboard_callback]
    
    def train(self, model, train_dataset, val_dataset):
        """Train the model"""
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     tf.keras.metrics.Precision(), 
                     tf.keras.metrics.Recall(),
                     tf.keras.metrics.AUC()]
        )
        
        # Train the model
        callbacks = self.create_callbacks()
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks
        )
        
        # Save training history
        np.save(os.path.join(self.model_dir, 'training_history.npy'), history.history)
        
        return model, history 