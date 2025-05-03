import os
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.mixed_precision import global_policy, set_global_policy
from datetime import datetime
import json
import matplotlib.pyplot as plt
import keras

class GradientAccumulation(keras.Model):
    """Model wrapper for gradient accumulation"""
    def __init__(self, model, steps=2):
        super(GradientAccumulation, self).__init__()
        self.model = model
        self.steps = steps
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v), trainable=False) 
                                      for v in model.trainable_variables]
        self.accumulation_counter = tf.Variable(0, trainable=False)
        
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
    
    def train_step(self, data):
        self.accumulation_counter.assign_add(1)
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            scaled_loss = loss / self.steps  # Scale the loss
        
        # Calculate gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Accumulate gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
        
        # Apply gradients when we've accumulated enough
        if self.accumulation_counter % self.steps == 0:
            # Apply gradients and reset
            self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.model.trainable_variables))
            
            # Reset accumulated gradients
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign(tf.zeros_like(self.model.trainable_variables[i]))
        
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def get_config(self):
        return {
            "steps": self.steps,
            "model": self.model
        }


class ExponentialMovingAverage(keras.callbacks.Callback):
    """Apply Exponential Moving Average to model weights"""
    def __init__(self, model, decay=0.999):
        super(ExponentialMovingAverage, self).__init__()
        self.model = model
        self.decay = decay
        self.shadow_variables = [tf.Variable(w.numpy(), trainable=False) 
                                for w in model.weights]
        self.ema_initialized = False
        
    def on_batch_end(self, batch, logs=None):
        if not self.ema_initialized:
            # Initialize shadow variables on first batch
            for i, shadow_var in enumerate(self.shadow_variables):
                shadow_var.assign(self.model.weights[i])
            self.ema_initialized = True
        else:
            # Update shadow variables
            for i, shadow_var in enumerate(self.shadow_variables):
                shadow_var.assign(
                    self.decay * shadow_var + (1.0 - self.decay) * self.model.weights[i]
                )
    
    def on_epoch_end(self, epoch, logs=None):
        # Save original weights
        original_weights = [w.numpy() for w in self.model.weights]
        
        # Apply EMA weights for evaluation
        for i, shadow_var in enumerate(self.shadow_variables):
            self.model.weights[i].assign(shadow_var)
        
        # Store values using EMA weights
        logs['ema_val_loss'] = self.model.evaluate(
            self.validation_data, verbose=0
        )[0]
        
        # Restore original weights
        for i, original_w in enumerate(original_weights):
            self.model.weights[i].assign(original_w)
    
    def apply_ema_weights(self):
        """Apply EMA weights to the model (call after training)"""
        for i, shadow_var in enumerate(self.shadow_variables):
            self.model.weights[i].assign(shadow_var)


class CosineAnnealingScheduler(keras.callbacks.Callback):
    """Cosine annealing learning rate scheduler with warm restarts"""
    def __init__(self, initial_lr, min_lr, cycle_length, cycle_mult=2.0):
        super(CosineAnnealingScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.current_cycle = 0
        self.cycle_end = cycle_length
        self.batch_count = 0
        
    def on_train_begin(self, logs=None):
        # Set initial learning rate
        keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)
        
    def on_batch_begin(self, batch, logs=None):
        # Calculate and update learning rate
        self.batch_count += 1
        
        # Check if we're at the end of a cycle
        if self.batch_count >= self.cycle_end:
            # Start a new cycle
            self.current_cycle += 1
            self.cycle_length = int(self.cycle_length * self.cycle_mult)
            self.cycle_end += self.cycle_length
            self.batch_count = 0
        
        # Calculate progress within the current cycle (0 to 1)
        progress = self.batch_count / self.cycle_length
        
        # Calculate learning rate using cosine annealing formula
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        new_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        # Update learning rate
        keras.backend.set_value(self.model.optimizer.lr, new_lr)
        
    def on_batch_end(self, batch, logs=None):
        # Store current learning rate in logs
        logs = logs or {}
        logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)


class StochasticWeightAveraging(keras.callbacks.Callback):
    """Implement Stochastic Weight Averaging for better generalization"""
    def __init__(self, start_epoch, swa_freq=1):
        super(StochasticWeightAveraging, self).__init__()
        self.start_epoch = start_epoch  # Start SWA after this epoch
        self.swa_freq = swa_freq  # Frequency of weight updates
        self.swa_weights = None  # Averaged weights
        self.count = 0  # Counter for weight updates
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.swa_freq == 0:
            if self.swa_weights is None:
                # Initialize SWA weights
                self.swa_weights = [tf.identity(w) for w in self.model.weights]
            else:
                # Update SWA weights
                self.count += 1
                for i, w in enumerate(self.model.weights):
                    self.swa_weights[i].assign(
                        self.swa_weights[i] * (self.count / (self.count + 1)) + 
                        w * (1 / (self.count + 1))
                    )
    
    def apply_swa_weights(self):
        """Apply SWA weights to the model (call after training)"""
        if self.swa_weights is not None:
            for i, w in enumerate(self.swa_weights):
                self.model.weights[i].assign(w)


class LearningRateLogger(keras.callbacks.Callback):
    """Log learning rate at each batch update"""
    def __init__(self, log_dir):
        super(LearningRateLogger, self).__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.lr_history = []
        
    def on_batch_end(self, batch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        self.lr_history.append(lr)
        
    def on_train_end(self, logs=None):
        # Save learning rate history
        np.save(os.path.join(self.log_dir, 'lr_history.npy'), np.array(self.lr_history))
        
        # Plot learning rate curve
        plt.figure(figsize=(10, 5))
        plt.plot(self.lr_history)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')
        plt.savefig(os.path.join(self.log_dir, 'lr_schedule.png'))
        plt.close()


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
        # Training settings
        self.learning_rate = config['training']['learning_rate']
        self.min_learning_rate = config['training'].get('min_lr', self.learning_rate * 0.01)
        self.weight_decay = config['training'].get('weight_decay', 1e-4)
        self.epochs = config['training']['epochs']
        self.grad_accum_steps = config['training'].get('grad_accum_steps', 1)
        self.use_mixed_precision = config['training'].get('mixed_precision', False)
        self.use_swa = config['training'].get('swa', False)
        self.swa_start = config['training'].get('swa_start', int(self.epochs * 0.75))
        self.use_ema = config['training'].get('ema', False)
        self.ema_decay = config['training'].get('ema_decay', 0.999)
        self.grad_clip_value = config['training'].get('grad_clip', 0.0)
        self.optimizer_type = config['training'].get('optimizer', 'adamw')
        self.scheduler_type = config['training'].get('scheduler', 'cosine')
        
        # Set up directories
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_dir = os.path.join('models', timestamp)
        self.log_dir = os.path.join('logs', timestamp)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Set up mixed precision if requested
        if self.use_mixed_precision:
            enable_mixed_precision()
        
    def create_callbacks(self, additional_callbacks=None):
        """Create training callbacks"""
        callbacks = []
        
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
        callbacks.append(checkpoint_callback)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # TensorBoard for visualization
        tensorboard_callback = TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            profile_batch='500,520'  # Profile performance on batches 500-520
        )
        callbacks.append(tensorboard_callback)
        
        # Learning rate scheduler (if not using cosine annealing)
        if self.scheduler_type == 'reduce_on_plateau':
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config['training'].get('reduce_lr_factor', 0.5),
                patience=self.config['training'].get('reduce_lr_patience', 3),
                min_lr=self.min_learning_rate,
                verbose=1
            )
            callbacks.append(reduce_lr)
        elif self.scheduler_type == 'cosine':
            cosine_scheduler = CosineAnnealingScheduler(
                initial_lr=self.learning_rate,
                min_lr=self.min_learning_rate,
                cycle_length=self.config['training'].get('cycle_length', int(self.epochs / 3)),
                cycle_mult=self.config['training'].get('cycle_mult', 1.0)
            )
            callbacks.append(cosine_scheduler)
        
        # Learning rate logger
        lr_logger = LearningRateLogger(self.log_dir)
        callbacks.append(lr_logger)
        
        # Add SWA callback if enabled
        if self.use_swa:
            swa_callback = StochasticWeightAveraging(
                start_epoch=self.swa_start,
                swa_freq=self.config['training'].get('swa_freq', 1)
            )
            callbacks.append(swa_callback)
        
        # Add any additional callbacks
        if additional_callbacks:
            callbacks.extend(additional_callbacks)
        
        return callbacks
    
    def _create_optimizer(self):
        """Create optimizer based on configuration"""
        if self.optimizer_type.lower() == 'adamw':
            optimizer = tfa.optimizers.AdamW(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif self.optimizer_type.lower() == 'lamb':
            optimizer = tfa.optimizers.LAMB(
                learning_rate=self.learning_rate,
                weight_decay_rate=self.weight_decay
            )
        else:
            optimizer = keras.optimizers.Adam(
                learning_rate=self.learning_rate
            )
        
        # Apply gradient clipping if enabled
        if self.grad_clip_value > 0:
            optimizer = keras.mixed_precision.LossScaleOptimizer(
                optimizer
            ) if self.use_mixed_precision else optimizer
            
            if hasattr(optimizer, 'clipnorm'):
                optimizer.clipnorm = self.grad_clip_value
            else:
                print("Warning: Gradient clipping not supported for this optimizer")
        
        return optimizer
    
    def _f1_metric(self):
        # Robust F1 metric for binary classification
        def f1(y_true, y_pred):
            y_pred = tf.round(y_pred)
            tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
            fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
            fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
            precision = tp / (tp + fp + tf.keras.backend.epsilon())
            recall = tp / (tp + fn + tf.keras.backend.epsilon())
            f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
            return f1
        return f1

    def _create_metrics(self):
        """Create metrics for model evaluation"""
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            self._f1_metric()
        ]
        return metrics
    
    def train(self, model, train_dataset, val_dataset, additional_callbacks=None):
        """Train the model with advanced optimization techniques"""
        # Apply gradient accumulation if steps > 1
        if self.grad_accum_steps > 1:
            model = GradientAccumulation(model, steps=self.grad_accum_steps)
            print(f"Gradient Accumulation enabled with {self.grad_accum_steps} steps")
        
        # Create optimizer
        optimizer = self._create_optimizer()
        
        # Create metrics
        metrics = self._create_metrics()
        
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )
        
        # Create callbacks
        callbacks = self.create_callbacks(additional_callbacks)
        
        # Add EMA callback if requested
        ema_callback = None
        if self.use_ema:
            ema_callback = ExponentialMovingAverage(model, decay=self.ema_decay)
            ema_callback.validation_data = val_dataset
            callbacks.append(ema_callback)
        
        # Store SWA callback for later use
        swa_callback = None
        for callback in callbacks:
            if isinstance(callback, StochasticWeightAveraging):
                swa_callback = callback
                break
        
        # Train the model
        print(f"Starting training for {self.epochs} epochs")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks
        )
        
        # Apply EMA weights if enabled
        if ema_callback is not None:
            print("Applying EMA weights to model")
            ema_callback.apply_ema_weights()
            
            # Evaluate model with EMA weights
            print("Evaluating model with EMA weights")
            ema_metrics = model.evaluate(val_dataset)
            print(f"EMA validation metrics: {dict(zip(model.metrics_names, ema_metrics))}")
        
        # Apply SWA weights if enabled
        if swa_callback is not None:
            # Save current weights before applying SWA
            current_weights = [w.numpy() for w in model.weights]
            
            print("Applying SWA weights to model")
            swa_callback.apply_swa_weights()
            
            # Evaluate model with SWA weights
            print("Evaluating model with SWA weights")
            swa_metrics = model.evaluate(val_dataset)
            print(f"SWA validation metrics: {dict(zip(model.metrics_names, swa_metrics))}")
            
            # Compare SWA with current best weights
            if swa_metrics[model.metrics_names.index('val_loss')] > history.history['val_loss'][-1]:
                print("Reverting to non-SWA weights (better performance)")
                for i, w in enumerate(current_weights):
                    model.weights[i].assign(w)
            else:
                print("Keeping SWA weights (better performance)")
        
        # Save training history
        history_path = os.path.join(self.model_dir, 'training_history.npy')
        np.save(history_path, history.history)
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, 'final_model.h5')
        model.save(final_model_path)
        
        # Plot training curves
        self._plot_training_curves(history.history)
        
        return model, history
    
    def _plot_training_curves(self, history):
        """Plot training and validation curves"""
        # Accuracy plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_curves.png'))
        plt.close()

def enable_mixed_precision():
    """Enable mixed precision training for faster performance on supported GPUs"""
    if tf.config.list_physical_devices('GPU'):
        try:
            # For TF 2.10+
            set_global_policy('mixed_float16')
            print("Mixed precision training enabled")
        except Exception as e:
            print(f"Failed to set mixed precision policy: {e}")
            print("Using default precision policy")
    else:
        print("No GPU detected, using default precision policy") 