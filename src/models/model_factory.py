import tensorflow as tf
import keras
from keras import optimizers
from keras import metrics
from logging import getLogger

# Model imports
from .text_model import TextFeatureExtractor
from .image_model import ImageFeatureExtractor
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
    """Create a multimodal fake news detection model.
    
    Args:
        config: A dictionary containing model configuration parameters.
        vocab_size: Size of the vocabulary for text processing.
        
    Returns:
        A compiled Keras model.
    """
    # Handle default values for missing configuration
    if not vocab_size and 'vocab_size' in config:
        vocab_size = config['vocab_size']
    
    # Configure input shape
    image_input_shape = config.get('image_input_shape', (224, 224, 3))
    
    # Configure text feature extractor
    text_model_config = config.get('text_model', {})
    text_feature_extractor = TextFeatureExtractor(
        vocab_size=vocab_size,
        max_sequence_length=config.get('max_sequence_length', 512),
        embedding_dim=text_model_config.get('embedding_dim', 300),
        hidden_dim=text_model_config.get('hidden_dim', 128),
        dropout_rate=text_model_config.get('dropout_rate', 0.2),
        trainable_embedding=text_model_config.get('trainable_embedding', True),
        model_type=text_model_config.get('model_type', 'lstm'),
        bidirectional=text_model_config.get('bidirectional', True),
        embedding_matrix=text_model_config.get('embedding_matrix', None)
    )
    
    # Configure image feature extractor
    image_model_config = config.get('image_model', {})
    
    # Configure specific parameters for YOLOv11 if selected
    backbone_type = image_model_config.get('backbone_type', image_model_config.get('backbone', 'resnet50'))
    backbone_params = {}
    
    if backbone_type == 'yolov11':
        backbone_params = {
            'width_mult': image_model_config.get('width_mult', 0.75),
            'depth_mult': image_model_config.get('depth_mult', 0.67),
            'use_fpn': image_model_config.get('use_fpn', True)
        }
    
    image_feature_extractor = ImageFeatureExtractor(
        input_shape=image_input_shape,
        backbone_type=backbone_type,
        pretrained=image_model_config.get('pretrained', True),
        output_dim=image_model_config.get('output_dim', 512),
        dropout_rate=image_model_config.get('dropout_rate', 0.2),
        l2_regularization=image_model_config.get('l2_regularization', 1e-5),
        use_attention=image_model_config.get('use_attention', True),
        pooling=image_model_config.get('pooling', 'avg'),
        trainable=image_model_config.get('trainable', False),
        backbone_params=backbone_params
    )
    
    # Configure fusion model
    fusion_config = config.get('fusion_model', {})
    multimodal_model = MultiModalFusionModel(
        text_feature_extractor=text_feature_extractor,
        image_feature_extractor=image_feature_extractor,
        fusion_method=fusion_config.get('fusion_method', 'concat'),
        hidden_dim=fusion_config.get('hidden_dim', 256),
        num_classes=config.get('num_classes', 2)
    )
    
    # Configure optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer_name = optimizer_config.get('name', 'adam').lower()
    learning_rate = optimizer_config.get('learning_rate', 0.001)
    
    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=optimizer_config.get('beta_1', 0.9),
            beta_2=optimizer_config.get('beta_2', 0.999),
            epsilon=optimizer_config.get('epsilon', 1e-8),
            weight_decay=optimizer_config.get('weight_decay', 0)  # add weight decay/L2 regularization
        )
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD(
            learning_rate=learning_rate,
            momentum=optimizer_config.get('momentum', 0.9),
            nesterov=optimizer_config.get('nesterov', True),
            weight_decay=optimizer_config.get('weight_decay', 0)
        )
    elif optimizer_name == 'adamw':
        optimizer = optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            beta_1=optimizer_config.get('beta_1', 0.9),
            beta_2=optimizer_config.get('beta_2', 0.999),
            epsilon=optimizer_config.get('epsilon', 1e-8)
        )
    else:
        logger.warning(f"Unknown optimizer {optimizer_name}, using Adam instead")
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    # Configure loss
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
        logger.warning(f"Unknown loss {loss_name}, using binary_crossentropy instead")
        loss = keras.losses.BinaryCrossentropy()
    
    # Configure metrics
    metrics_list = [metrics.BinaryAccuracy(name='accuracy')]
    
    # Add AUC if requested
    if config.get('use_auc', True):
        metrics_list.append(metrics.AUC(name='auc'))
    
    # Add F1 score if TensorFlow Addons is available
    if config.get('use_f1', True) and TENSORFLOW_ADDONS_AVAILABLE:
        try:
            f1 = F1Score(num_classes=1, threshold=0.5)
            metrics_list.append(f1)
        except Exception as e:
            logger.warning(f"Error adding F1Score metric: {e}")
    
    # Compile the model
    multimodal_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics_list
    )
    
    return multimodal_model 