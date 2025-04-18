import tensorflow as tf
# Directly use the keras namespace for clarity
import keras
from keras import layers
import numpy as np

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
        batch_size, height, width, channels = tf.shape(inputs)
        
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
    """Image feature extractor with various backbone options and FPN"""
    
    def __init__(self, input_shape=(224, 224, 3), backbone="yolov11", trainable=True, pooling='avg'):
        super(ImageFeatureExtractor, self).__init__(name='image_feature_extractor')
        self.input_shape = input_shape
        self.backbone_type = backbone
        self.trainable = trainable
        self.pooling = pooling
        
        # Build model
        self._build_model()
        
    def _build_model(self):
        # Create input layer
        self.input_layer = layers.Input(shape=self.input_shape)
        
        # Build backbone based on type
        if self.backbone_type.lower() in ['yolov11', 'yolo']:
            backbone_outputs = self._create_yolov11_backbone(self.input_layer)
        elif 'resnet' in self.backbone_type.lower():
            backbone_outputs = self._create_resnet_backbone(self.input_layer, self.backbone_type)
        elif 'efficient' in self.backbone_type.lower():
            backbone_outputs = self._create_efficientnet_backbone(self.input_layer, self.backbone_type)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_type}")
            
        # Feature pyramid network
        self.fpn = FeaturePyramidNetwork()
        fpn_features = self.fpn(backbone_outputs)
        
        # Apply pooling to get fixed-size feature vector
        if self.pooling == 'avg':
            self.global_pool = layers.GlobalAveragePooling2D()
        elif self.pooling == 'max':
            self.global_pool = layers.GlobalMaxPooling2D()
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
            
        pooled_features = []
        for feature in fpn_features:
            pooled_features.append(self.global_pool(feature))
            
        # Concatenate all features
        if len(pooled_features) > 1:
            self.features = layers.Concatenate()(pooled_features)
        else:
            self.features = pooled_features[0]
            
        # Add final layers
        self.dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(512, activation='relu')
        
        # Store feature maps for visualization if needed
        self.feature_maps = backbone_outputs
        
    def _build_backbone(self, input_layer):
        """Build selected backbone network"""
        if self.backbone_type.lower() in ['yolov11', 'yolo']:
            return self._create_yolov11_backbone(input_layer)
        elif 'resnet' in self.backbone_type.lower():
            return self._create_resnet_backbone(input_layer, self.backbone_type)
        elif 'efficient' in self.backbone_type.lower():
            return self._create_efficientnet_backbone(input_layer, self.backbone_type)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_type}")
            
    def _create_yolov11_backbone(self, input_layer):
        """Create a simplified YOLOv11-like backbone"""
        # Downsample to 112x112
        x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        
        # Downsample to 56x56
        x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        
        # First CSP block
        x = CSPBlock(64, num_blocks=1)(x)
        
        # Downsample to 28x28
        x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        
        # Second CSP block
        x = CSPBlock(128, num_blocks=2)(x)
        feature1 = x  # 28x28 feature map
        
        # Downsample to 14x14
        x = layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        
        # Third CSP block
        x = CSPBlock(256, num_blocks=3)(x)
        feature2 = x  # 14x14 feature map
        
        # Downsample to 7x7
        x = layers.Conv2D(512, 3, strides=2, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.1)(x)
        
        # Fourth CSP block
        x = CSPBlock(512, num_blocks=1)(x)
        feature3 = x  # 7x7 feature map
        
        return [feature1, feature2, feature3]
        
    def _create_resnet_backbone(self, input_layer, version='resnet50'):
        """Create a ResNet backbone"""
        # Get base model without top layers
        if version == 'resnet50':
            base_model = keras.applications.ResNet50(
                include_top=False, 
                weights='imagenet', 
                input_tensor=input_layer
            )
        elif version == 'resnet101':
            base_model = keras.applications.ResNet101(
                include_top=False, 
                weights='imagenet', 
                input_tensor=input_layer
            )
        else:
            base_model = keras.applications.ResNet152(
                include_top=False, 
                weights='imagenet', 
                input_tensor=input_layer
            )
            
        # Freeze base model if not trainable
        base_model.trainable = self.trainable
        
        # Get outputs from different stages for FPN
        feature1 = base_model.get_layer('conv3_block4_out').output
        feature2 = base_model.get_layer('conv4_block6_out').output
        feature3 = base_model.get_layer('conv5_block3_out').output
        
        return [feature1, feature2, feature3]
        
    def _create_efficientnet_backbone(self, input_layer, version='efficientnetb0'):
        """Create an EfficientNet backbone"""
        # Get base model without top layers
        if version == 'efficientnetb0':
            base_model = keras.applications.EfficientNetB0(
                include_top=False, 
                weights='imagenet', 
                input_tensor=input_layer
            )
        elif version == 'efficientnetb3':
            base_model = keras.applications.EfficientNetB3(
                include_top=False, 
                weights='imagenet', 
                input_tensor=input_layer
            )
        else:
            base_model = keras.applications.EfficientNetB7(
                include_top=False, 
                weights='imagenet', 
                input_tensor=input_layer
            )
            
        # Freeze base model if not trainable
        base_model.trainable = self.trainable
        
        # Get feature maps from different stages of the network
        # Note: Layer names will differ based on EfficientNet version
        feature_names = ['block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation']
        features = [base_model.get_layer(name).output for name in feature_names]
        
        return features
        
    def call(self, inputs, training=False):
        x = self._build_backbone(inputs)
        fpn_features = self.fpn(x)
        
        pooled_features = []
        for feature in fpn_features:
            pooled_features.append(self.global_pool(feature))
            
        if len(pooled_features) > 1:
            features = layers.Concatenate()(pooled_features)
        else:
            features = pooled_features[0]
            
        features = self.dropout(features, training=training)
        features = self.output_layer(features)
        
        return features
    
    def get_feature_maps(self):
        """Return feature maps for visualization"""
        return self.feature_maps
    
    def get_attention_map(self, feature_map):
        """Generate attention map for visualization"""
        # Simply use the average across all channels
        attention = tf.reduce_mean(feature_map, axis=-1)
        attention = tf.expand_dims(attention, -1)
        attention = tf.image.resize(attention, self.input_shape[:2])
        
        # Normalize to [0, 1]
        attention_min = tf.reduce_min(attention)
        attention_max = tf.reduce_max(attention)
        attention = (attention - attention_min) / (attention_max - attention_min + 1e-9)
        
        return attention 