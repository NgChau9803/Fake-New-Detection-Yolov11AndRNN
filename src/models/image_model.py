import tensorflow as tf
import numpy as np

class ImageFeatureExtractor(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 3), backbone="yolov11-tiny"):
        super(ImageFeatureExtractor, self).__init__()
        
        self.input_shape = input_shape
        self.backbone = backbone
        
        # YOLOv11 backbone implementation
        def create_yolov11_backbone():
            # Initial convolution block
            x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False)(input_layer)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            
            # First downsampling block
            x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            
            # Residual blocks
            for _ in range(3):
                x = self._residual_block(x, 64)
            
            # Second downsampling block
            x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            
            # More residual blocks
            for _ in range(4):
                x = self._residual_block(x, 128)
            
            # Third downsampling block
            x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            
            # Final residual blocks
            for _ in range(6):
                x = self._residual_block(x, 256)
            
            return x
        
        # Create input layer
        input_layer = tf.keras.layers.Input(shape=input_shape)
        
        # Build backbone
        if backbone == "yolov11-tiny":
            backbone_output = create_yolov11_backbone()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature extraction layers
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone_output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)
        
        # Store intermediate layers for feature extraction
        self.intermediate_layers = {
            'backbone_output': backbone_output,
            'pooled_features': x
        }
    
    def _residual_block(self, x, filters):
        """Create a residual block with skip connection"""
        shortcut = x
        
        # First convolution
        x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        # Second convolution
        x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
        # Add skip connection
        x = tf.keras.layers.Add()([shortcut, x])
        
        return x
    
    def call(self, inputs, training=False):
        """Forward pass"""
        return self.model(inputs, training=training)
    
    def get_feature_maps(self, inputs):
        """Extract intermediate feature maps"""
        feature_maps = {}
        for layer_name, layer_output in self.intermediate_layers.items():
            feature_maps[layer_name] = layer_output
        return feature_maps
    
    def get_attention_map(self, inputs):
        """Generate attention map from feature maps"""
        feature_maps = self.get_feature_maps(inputs)
        backbone_output = feature_maps['backbone_output']
        
        # Global average pooling
        gap = tf.keras.layers.GlobalAveragePooling2D()(backbone_output)
        
        # Dense layer for attention weights
        attention_weights = tf.keras.layers.Dense(backbone_output.shape[-1], activation='softmax')(gap)
        
        # Reshape attention weights
        attention_weights = tf.reshape(attention_weights, [-1, 1, 1, backbone_output.shape[-1]])
        
        # Apply attention to feature maps
        attention_map = tf.reduce_sum(backbone_output * attention_weights, axis=-1)
        
        return attention_map 