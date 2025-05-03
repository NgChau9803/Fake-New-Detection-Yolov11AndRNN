import tensorflow as tf
# Directly use the keras namespace for clarity
import keras
from keras import layers
import numpy as np
from .yolo_backbone import YOLOv11Backbone

class SpatialAttention(keras.layers.Layer):
    """Spatial attention mechanism for focusing on important spatial locations"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )
        
    def call(self, inputs):
        # Generate max and average features along channel dimension
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate the pooled features
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Apply convolution to generate spatial attention map
        attention_map = self.conv(concat)
        
        # Apply attention map to input feature map
        return inputs * attention_map


class ChannelAttention(keras.layers.Layer):
    """Channel attention mechanism for focusing on important feature channels"""
    
    def __init__(self, ratio=8):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio
        
    def build(self, input_shape):
        channel = input_shape[-1]
        
        # Shared MLP for both max and avg pooled features
        self.shared_dense_1 = layers.Dense(
            channel // self.ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        
        self.shared_dense_2 = layers.Dense(
            channel,
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        
    def call(self, inputs):
        # Get spatial dimensions
        shape = tf.shape(inputs)
        batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
        
        # Global average pooling
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        avg_pool_flat = tf.reshape(avg_pool, [-1, channels])
        
        # Global max pooling
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_pool_flat = tf.reshape(max_pool, [-1, channels])
        
        # Shared MLP
        avg_fc1 = self.shared_dense_1(avg_pool_flat)
        avg_fc2 = self.shared_dense_2(avg_fc1)
        
        max_fc1 = self.shared_dense_1(max_pool_flat)
        max_fc2 = self.shared_dense_2(max_fc1)
        
        # Add the two features and apply sigmoid
        channel_attention = tf.nn.sigmoid(avg_fc2 + max_fc2)
        channel_attention = tf.reshape(channel_attention, [-1, 1, 1, channels])
        
        # Apply channel attention to input
        return inputs * channel_attention


class CBAM(keras.layers.Layer):
    """Convolutional Block Attention Module combining spatial and channel attention"""
    
    def __init__(self, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def call(self, inputs):
        # Apply channel attention first
        x = self.channel_attention(inputs)
        
        # Then apply spatial attention
        x = self.spatial_attention(x)
        
        return x


class CSPBlock(keras.layers.Layer):
    """Cross Stage Partial Block from YOLOv4/v5/v11"""
    
    def __init__(self, filters, num_blocks=1, use_attention=True):
        super(CSPBlock, self).__init__()
        self.filters = filters
        self.num_blocks = num_blocks
        self.use_attention = use_attention
        
    def build(self, input_shape):
        # Bottleneck the input to reduce channels
        self.conv1 = layers.Conv2D(
            self.filters, 
            kernel_size=1, 
            strides=1, 
            padding='same',
            use_bias=False
        )
        
        # Split into two branches
        # Branch 1: Skip connection
        self.conv2 = layers.Conv2D(
            self.filters // 2, 
            kernel_size=1, 
            strides=1, 
            padding='same',
            use_bias=False
        )
        
        # Branch 2: Residual connection
        self.conv3 = layers.Conv2D(
            self.filters // 2, 
            kernel_size=1, 
            strides=1, 
            padding='same',
            use_bias=False
        )
        
        # Residual blocks
        self.residual_blocks = []
        for _ in range(self.num_blocks):
            self.residual_blocks.append(ResidualBlock(self.filters // 2, use_attention=self.use_attention))
        
        # Merge branches
        self.conv4 = layers.Conv2D(
            self.filters, 
            kernel_size=1, 
            strides=1, 
            padding='same',
            use_bias=False
        )
        
        # Batch normalization and activation
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU(alpha=0.1)
        
    def call(self, inputs, training=False):
        # Split into two branches
        x = self.conv1(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        
        # Branch 1: Skip connection
        route1 = self.conv2(x)
        
        # Branch 2: Process through residual blocks
        route2 = self.conv3(x)
        for block in self.residual_blocks:
            route2 = block(route2, training=training)
        
        # Concatenate branches
        x = tf.concat([route2, route1], axis=-1)
        
        # Final convolution
        x = self.conv4(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        
        return x


class ResidualBlock(keras.layers.Layer):
    """Residual block with CBAM attention"""
    
    def __init__(self, filters, use_attention=True):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.use_attention = use_attention
        
    def build(self, input_shape):
        # First convolution
        self.conv1 = layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False
        )
        
        # Second convolution
        self.conv2 = layers.Conv2D(
            self.filters,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False
        )
        
        # Batch normalization and activation
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.act = layers.LeakyReLU(alpha=0.1)
        
        # Attention module
        if self.use_attention:
            self.attention = CBAM()
        
    def call(self, inputs, training=False):
        shortcut = inputs
        
        # First conv block
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Add skip connection
        x = x + shortcut
        
        return x


class FeaturePyramidNetwork(keras.layers.Layer):
    """Feature Pyramid Network for multi-scale feature extraction"""
    
    def __init__(self, filters=[256, 128, 64]):
        super(FeaturePyramidNetwork, self).__init__()
        self.filters = filters
        
    def build(self, input_shape):
        # Top-down pathway
        self.top_down_convs = []
        self.lateral_convs = []
        
        for i, filter_size in enumerate(self.filters):
            # Lateral (for skip connections) 1x1 conv
            self.lateral_convs.append(
                layers.Conv2D(
                    filter_size,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    use_bias=False
                )
            )
            
            # Top-down 3x3 conv
            self.top_down_convs.append(
                layers.Conv2D(
                    filter_size,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    use_bias=False
                )
            )
    
    def call(self, inputs, training=False):
        # inputs = list of feature maps from backbone
        # in decreasing resolution order (C3, C4, C5)
        
        # Initialize with the top feature map
        p5 = self.lateral_convs[0](inputs[-1])
        
        # Top-down pathway
        features = [p5]
        for i in range(len(inputs) - 2, -1, -1):
            # Upsample previous feature map
            upsampled = tf.image.resize(
                features[-1],
                tf.shape(inputs[i])[1:3],
                method='nearest'
            )
            
            # Lateral connection
            lateral = self.lateral_convs[len(inputs) - 1 - i](inputs[i])
            
            # Add and apply conv
            p = upsampled + lateral
            p = self.top_down_convs[len(inputs) - 1 - i](p)
            
            features.append(p)
        
        # Return feature maps in increasing resolution order
        return features[::-1]


class ImageFeatureExtractor(keras.Model):
    """Image feature extractor for fake news detection.
    
    This model extracts features from images using various backbone architectures.
    It supports ResNet50, EfficientNetB0, and YOLOv11 backbones with attention 
    mechanisms and feature pyramid networks.
    """
    
    def __init__(
        self,
        input_shape=(224, 224, 3),
        backbone_type="resnet50",
        pretrained=True,
        output_dim=512,
        dropout_rate=0.2,
        l2_regularization=1e-5,
        use_attention=True,
        pooling="avg",
        trainable=False,
        backbone_params=None
    ):
        """Initialize the image feature extractor.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            backbone_type: Backbone network ('resnet50', 'efficientnetb0', or 'yolov11')
            pretrained: Whether to use pretrained weights for backbone
            output_dim: Dimension of output feature vector
            dropout_rate: Dropout rate for regularization
            l2_regularization: L2 regularization factor
            use_attention: Whether to use attention mechanisms
            pooling: Pooling method ('avg', 'max', or None)
            trainable: Whether to train the backbone
            backbone_params: Additional configuration for the backbone
        """
        super().__init__()
        self.img_input_shape = input_shape  # Changed from input_shape to img_input_shape
        self.backbone_name = backbone_type
        self.pretrained = pretrained
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        self.use_attention = use_attention
        self.pooling = pooling
        self.trainable = trainable
        self.backbone_params = backbone_params or {}
        
        # Initialize regularizer
        self.regularizer = keras.regularizers.l2(l2_regularization)
        
        # Create backbone network
        self._create_backbone()
        
        # Create post-processing layers
        if self.pooling == "avg":
            self.global_pool = layers.GlobalAveragePooling2D()
        elif self.pooling == "max":
            self.global_pool = layers.GlobalMaxPooling2D()
        
        # Add attention if requested
        if use_attention:
            self.cbam = CBAM(ratio=8, kernel_size=7)
        
        # Final layers
        self.feature_dropout = layers.Dropout(dropout_rate)
        self.feature_dense = layers.Dense(
            output_dim,
            activation="relu",
            kernel_regularizer=self.regularizer,
            name="image_features"
        )
    
    def _create_backbone(self):
        """Create the backbone network for feature extraction."""
        if self.backbone_name == "resnet50":
            # Use ResNet50 backbone
            weights = "imagenet" if self.pretrained else None
            self.backbone = keras.applications.ResNet50(
                include_top=False,
                weights=weights,
                input_shape=self.img_input_shape  # Changed from input_shape to img_input_shape
            )
            
            # Set trainable status
            self.backbone.trainable = self.trainable
            
            # Add feature pyramid network if requested
            if self.backbone_params.get("use_fpn", False):
                self.fpn = FeaturePyramidNetwork([512, 1024, 2048])
                
        elif self.backbone_name == "efficientnetb0":
            # Use EfficientNetB0 backbone
            weights = "imagenet" if self.pretrained else None
            self.backbone = keras.applications.EfficientNetB0(
                include_top=False,
                weights=weights,
                input_shape=self.img_input_shape  # Changed from input_shape to img_input_shape
            )
            
            # Set trainable status
            self.backbone.trainable = self.trainable
            
            # Add feature pyramid network if requested
            if self.backbone_params.get("use_fpn", False):
                self.fpn = FeaturePyramidNetwork([32, 96, 320])
        
        elif self.backbone_name == "yolov11":
            # Use YOLOv11 backbone
            width_mult = self.backbone_params.get("width_mult", 0.75)
            depth_mult = self.backbone_params.get("depth_mult", 0.67)
            use_fpn = self.backbone_params.get("use_fpn", True)
            
            self.backbone = YOLOv11Backbone(
                input_shape=self.img_input_shape,  # Changed from input_shape to img_input_shape
                width_mult=width_mult,
                depth_mult=depth_mult,
                use_fpn=use_fpn,
                pooling=None,  # We'll handle pooling separately
                classification=False,
                num_classes=0,
                dropout_rate=0.0,
                name="yolov11_backbone"
            )
            
            # Set trainable status
            self.backbone.trainable = self.trainable
        
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
    
    def call(self, inputs, training=None):
        """Forward pass for the image feature extractor.
        
        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode
        
        Returns:
            Extracted image features
        """
        # Apply backbone
        if self.backbone_name == "yolov11":
            # YOLOv11 backbone directly outputs features
            features = self.backbone(inputs, training=training)
        else:
            # Apply standard backbones
            features = self.backbone(inputs, training=training)
            
            # Apply FPN if available
            if hasattr(self, "fpn") and self.backbone_params.get("use_fpn", False):
                # Extract intermediate features for FPN
                if self.backbone_name == "resnet50":
                    # Specific layers for ResNet
                    c3 = self.backbone.get_layer("conv3_block4_out").output
                    c4 = self.backbone.get_layer("conv4_block6_out").output
                    c5 = self.backbone.get_layer("conv5_block3_out").output
                    
                    # Create a new model to extract these features
                    feature_model = keras.Model(
                        inputs=self.backbone.input,
                        outputs=[c3, c4, c5]
                    )
                    feature_maps = feature_model(inputs, training=training)
                    
                    # Apply FPN
                    pyramid_features = self.fpn(feature_maps, training=training)
                    features = pyramid_features[-1]  # Use the most semantically rich feature
                
                elif self.backbone_name == "efficientnetb0":
                    # Specific layers for EfficientNet
                    c3 = self.backbone.get_layer("block4a_expand_activation").output
                    c4 = self.backbone.get_layer("block6a_expand_activation").output
                    c5 = self.backbone.get_layer("top_activation").output
                    
                    # Create a new model to extract these features
                    feature_model = keras.Model(
                        inputs=self.backbone.input,
                        outputs=[c3, c4, c5]
                    )
                    feature_maps = feature_model(inputs, training=training)
                    
                    # Apply FPN
                    pyramid_features = self.fpn(feature_maps, training=training)
                    features = pyramid_features[-1]  # Use the most semantically rich feature
        
        # Store feature maps for visualization if needed
        self.feature_maps = features
        
        # Apply global pooling if available
        if hasattr(self, "global_pool"):
            x = self.global_pool(features)
        else:
            x = features
        
        # Apply attention if requested
        if self.use_attention and hasattr(self, "cbam"):
            # Reshape for CBAM if needed
            if len(x.shape) < 4:
                x = tf.expand_dims(tf.expand_dims(x, 1), 1)
            x = self.cbam(x)
            # Flatten if needed
            if len(x.shape) > 2:
                x = tf.squeeze(x, [1, 2])
        
        # Final feature extraction
        x = self.feature_dropout(x, training=training)
        x = self.feature_dense(x)
        
        return x

    def get_feature_maps(self, inputs, training=False):
        """Return feature maps for visualization
        
        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Dictionary with feature maps at different levels
        """
        feature_maps = {}
        
        # Apply backbone
        if self.backbone_name == "yolov11":
            # Get the YOLO backbone features
            # Store multiple feature maps from different stages for YOLO
            # First get the backbone's input
            x = inputs
            
            # Run through stem and early stages to get feature maps
            stem_output = self.backbone.stem(x, training=training)
            feature_maps['stem'] = stem_output
            
            # Stage 1
            x = self.backbone.downsample1(stem_output, training=training)
            c2 = self.backbone.stage1(x, training=training)
            feature_maps['stage1'] = c2
            
            # Stage 2
            x = self.backbone.downsample2(c2, training=training)
            c3 = self.backbone.stage2(x, training=training)
            feature_maps['stage2'] = c3
            
            # Stage 3
            x = self.backbone.downsample3(c3, training=training)
            c4 = self.backbone.stage3(x, training=training)
            feature_maps['stage3'] = c4
            
            # Stage 4
            x = self.backbone.downsample4(c4, training=training)
            c5 = self.backbone.stage4(x, training=training)
            feature_maps['stage4'] = c5
            
            # FPN if available
            if hasattr(self.backbone, 'fpn') and self.backbone.use_fpn:
                fpn_features = self.backbone.fpn([c3, c4, c5], training=training)
                for i, feat in enumerate(fpn_features):
                    feature_maps[f'fpn_p{i+3}'] = feat
            
            # Final backbone output
            backbone_output = self.backbone(inputs, training=training)
            feature_maps['backbone_output'] = backbone_output
            
        else:
            # For other backbone types
            backbone_output = self.backbone(inputs, training=training)
            feature_maps['backbone_output'] = backbone_output
            
            # Extract intermediate layers for ResNet50
            if self.backbone_name == "resnet50":
                try:
                    # Extract common feature maps from ResNet
                    c3 = self.backbone.get_layer("conv3_block4_out").output
                    c4 = self.backbone.get_layer("conv4_block6_out").output
                    c5 = self.backbone.get_layer("conv5_block3_out").output
                    
                    # Create temporary models to extract these features
                    c3_model = keras.Model(inputs=self.backbone.input, outputs=c3)
                    c4_model = keras.Model(inputs=self.backbone.input, outputs=c4)
                    c5_model = keras.Model(inputs=self.backbone.input, outputs=c5)
                    
                    feature_maps['c3'] = c3_model(inputs, training=training)
                    feature_maps['c4'] = c4_model(inputs, training=training)
                    feature_maps['c5'] = c5_model(inputs, training=training)
                except:
                    # If layer names don't match, just use the final output
                    pass
                
            # Extract intermediate layers for EfficientNet
            elif self.backbone_name == "efficientnetb0":
                try:
                    # Extract common feature maps from EfficientNet
                    c3 = self.backbone.get_layer("block4a_expand_activation").output
                    c4 = self.backbone.get_layer("block6a_expand_activation").output
                    c5 = self.backbone.get_layer("top_activation").output
                    
                    # Create temporary models to extract these features
                    c3_model = keras.Model(inputs=self.backbone.input, outputs=c3)
                    c4_model = keras.Model(inputs=self.backbone.input, outputs=c4)
                    c5_model = keras.Model(inputs=self.backbone.input, outputs=c5)
                    
                    feature_maps['c3'] = c3_model(inputs, training=training)
                    feature_maps['c4'] = c4_model(inputs, training=training)
                    feature_maps['c5'] = c5_model(inputs, training=training)
                except:
                    # If layer names don't match, just use the final output
                    pass
        
        return feature_maps
    
    def get_attention_map(self, inputs, training=False):
        """Generate attention map for visualization
        
        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode
            
        Returns:
            Attention map highlighting important regions in the image
        """
        # Get feature maps
        feature_maps = self.get_feature_maps(inputs, training=training)
        
        # Get backbone output
        backbone_output = feature_maps['backbone_output']
        
        if self.backbone_name == "yolov11":
            # For YOLO, use the highest level feature map (after FPN if available)
            if 'fpn_p5' in feature_maps:
                # If FPN is used, use the P5 feature map
                feature_map = feature_maps['fpn_p5']
            else:
                # Otherwise use the stage4 output
                feature_map = feature_maps['stage4']
                
            # Calculate channel-wise importance
            # Get channel-wise mean and max to highlight important channels
            channel_mean = tf.reduce_mean(feature_map, axis=-1, keepdims=True)
            channel_max = tf.reduce_max(feature_map, axis=-1, keepdims=True)
            importance_map = channel_mean + channel_max
            
            # Normalize to [0, 1]
            min_val = tf.reduce_min(importance_map)
            max_val = tf.reduce_max(importance_map)
            attention_map = (importance_map - min_val) / (max_val - min_val + 1e-7)
            
            # Resize to match input resolution
            input_shape = tf.shape(inputs)
            attention_map = tf.image.resize(
                attention_map, 
                (input_shape[1], input_shape[2]), 
                method='bilinear'
            )
            
            # Squeeze to remove channel dimension
            attention_map = tf.squeeze(attention_map, axis=-1)
        else:
            # For other backbones
            # Create a class activation map using global average pooling weights
            # We use a simple gradient-free approach here
            
            # Get the last convolutional feature map
            if 'c5' in feature_maps:
                feature_map = feature_maps['c5']
            else:
                feature_map = backbone_output
            
            # Calculate spatial importance
            # Average across channels to get spatial attention
            attention_map = tf.reduce_mean(tf.abs(feature_map), axis=-1)
            
            # Normalize to [0, 1]
            min_val = tf.reduce_min(attention_map)
            max_val = tf.reduce_max(attention_map)
            attention_map = (attention_map - min_val) / (max_val - min_val + 1e-7)
            
            # Resize to match input resolution
            input_shape = tf.shape(inputs)
            attention_map = tf.image.resize(
                tf.expand_dims(attention_map, -1), 
                (input_shape[1], input_shape[2]), 
                method='bilinear'
            )
            
            # Squeeze to remove channel dimension
            attention_map = tf.squeeze(attention_map, axis=-1)
        
        return attention_map 