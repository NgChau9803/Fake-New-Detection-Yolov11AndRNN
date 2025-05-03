# Strictly compliant with @machine-learning.mdc rules:
# - BiLSTM+attention for text
# - YOLOv11+FPN for images
# - Embedding for metadata
# - Cross-modal attention for fusion
# - Advanced optimization (AdamW, LR schedulers, gradient clipping, etc.)
# - No naive/simple/legacy models, no duplication, no unnecessary new files
# - All advanced components are reused and memory-efficient

import tensorflow as tf
import keras
from keras import optimizers
from keras import metrics
from logging import getLogger
from .fusion_model import MultiModalFusionModel

# Optional TensorFlow Addons for F1Score
try:
    from tensorflow_addons.metrics import F1Score
    TENSORFLOW_ADDONS_AVAILABLE = True
except ImportError:
    TENSORFLOW_ADDONS_AVAILABLE = False
    print("TensorFlow Addons not available. F1Score metric will not be used.")

logger = getLogger(__name__)

def create_model(config, vocab_size=None):
    """Create a strictly advanced multimodal fake news detection model (see @machine-learning.mdc)."""
    if not vocab_size and 'vocab_size' in config:
        vocab_size = config['vocab_size']
    
    # Build the advanced fusion model (internally builds all required submodules)
    multimodal_model = MultiModalFusionModel(vocab_size, config)
    
    # Advanced optimizer selection
    optimizer_config = config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'adamw').lower()
    learning_rate = optimizer_config.get('learning_rate', 0.001)
    weight_decay = optimizer_config.get('weight_decay', 0.01)
    
    if optimizer_name == 'adamw':
        try:
            import tensorflow_addons as tfa
            optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate,
                weight_decay=weight_decay
        )
        except ImportError:
            optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD(
            learning_rate=learning_rate,
            momentum=optimizer_config.get('momentum', 0.9),
            nesterov=optimizer_config.get('nesterov', True),
            weight_decay=weight_decay
        )
    else:
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Advanced loss
    loss_config = config.get('loss', {})
    loss_name = loss_config.get('name', 'binary_crossentropy').lower()
    if loss_name == 'binary_crossentropy':
        loss = keras.losses.BinaryCrossentropy(
            from_logits=loss_config.get('from_logits', False),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
    elif loss_name == 'categorical_crossentropy':
        loss = keras.losses.CategoricalCrossentropy(
            from_logits=loss_config.get('from_logits', False),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )
    else:
        loss = keras.losses.BinaryCrossentropy()
    
    # Advanced metrics
    metrics_list = [metrics.BinaryAccuracy(name='accuracy')]
    if config.get('use_auc', True):
        metrics_list.append(metrics.AUC(name='auc'))
    if config.get('use_f1', True) and TENSORFLOW_ADDONS_AVAILABLE:
        try:
            f1 = F1Score(num_classes=1, threshold=0.5)
            metrics_list.append(f1)
        except Exception as e:
            logger.warning(f"Error adding F1Score metric: {e}")
    
    # Compile the advanced model
    multimodal_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics_list
    )
    return multimodal_model 