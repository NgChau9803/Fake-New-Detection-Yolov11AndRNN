import tensorflow as tf
import keras
from src.models.text_model import TextFeatureExtractor
from src.models.image_model import ImageFeatureExtractor
from src.models.fusion_model import MultiModalFusionModel

def create_model(config, vocab_size):
    """
    Create and configure the multimodal fake news detection model
    
    Args:
        config: Configuration dictionary
        vocab_size: Size of vocabulary for text processing
        
    Returns:
        Compiled model
    """
    # Get model configuration
    text_config = config['model']['text']
    image_config = config['model']['image']
    fusion_config = config['model']['fusion']
    training_config = config['training']
    
    # Create model
    fusion_model = MultiModalFusionModel(
        vocab_size=vocab_size,
        config=config
    )
    
    # Set up optimizer
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    learning_rate = training_config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    elif optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate
        )
    else:
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate
        )
    
    # Compile model with metrics
    metrics = [
        'accuracy',
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        keras.metrics.AUC()
    ]
    
    # Add F1 score if TensorFlow Addons is available
    try:
        import tensorflow_addons as tfa
        metrics.append(tfa.metrics.F1Score(num_classes=1, threshold=0.5))
    except ImportError:
        print("TensorFlow Addons not available, F1 score metric will not be used")
    
    fusion_model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )
    
    return fusion_model 