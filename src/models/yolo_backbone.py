"""
YOLOv11 Backbone implementation for fake news detection.
This module provides a modern implementation of the YOLOv11 architecture,
optimized for feature extraction from images in the context of multimodal
fake news detection.
"""

import tensorflow as tf
import keras
from keras import layers
import keras.backend as K
import math


class ConvolutionBlock(layers.Layer):
    """Convolution block with batch normalization and activation.
    
    This is a basic building block used throughout the YOLOv11 architecture.
    It consists of a convolution layer followed by optional batch normalization
    and activation function.
    """
    
    def __init__(
        self,
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        use_bn=True,
        activation="silu",
        name=None
    ):
        """Initialize the convolution block.
        
        Args:
            filters: Number of output filters
            kernel_size: Size of the convolution kernel
            strides: Stride of the convolution
            padding: Padding mode ('valid' or 'same')
            use_bias: Whether to use bias in the convolution
            use_bn: Whether to use batch normalization
            activation: Activation function to use
            name: Optional name for the layer
        """
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.activation = activation
        
        # Define layers
        self.conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=keras.initializers.HeNormal()
        )
        
        if use_bn:
            self.bn = layers.BatchNormalization()
        
        if activation:
            if activation == "silu":
                self.act = lambda x: x * tf.nn.sigmoid(x)  # SiLU/Swish
            elif activation == "relu":
                self.act = layers.ReLU()
            elif activation == "leaky_relu":
                self.act = layers.LeakyReLU(alpha=0.1)
            else:
                self.act = layers.Activation(activation)
    
    def call(self, inputs, training=None):
        """Forward pass for the convolution block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
        
        Returns:
            Output tensor after convolution, batch norm, and activation
        """
        x = self.conv(inputs)
        
        if self.use_bn:
            x = self.bn(x, training=training)
        
        if self.activation:
            x = self.act(x)
        
        return x


class CSPBottleneck(layers.Layer):
    """Cross Stage Partial bottleneck with residual connection.
    
    This implements a bottleneck block similar to ResNet but with CSP design
    for more efficient feature extraction.
    """
    
    def __init__(
        self,
        filters,
        expansion=0.5,
        shortcut=True,
        activation="silu",
        name=None
    ):
        """Initialize the CSP bottleneck.
        
        Args:
            filters: Number of output filters
            expansion: Channel expansion factor
            shortcut: Whether to use residual connection
            activation: Activation function to use
            name: Optional name for the layer
        """
        super().__init__(name=name)
        self.filters = filters
        self.expansion = expansion
        self.shortcut = shortcut
        
        hidden_filters = int(filters * expansion)
        
        # Define layers
        self.conv1 = ConvolutionBlock(
            filters=hidden_filters,
            kernel_size=1,
            activation=activation
        )
        
        self.conv2 = ConvolutionBlock(
            filters=filters,
            kernel_size=3,
            activation=activation
        )
    
    def call(self, inputs, training=None):
        """Forward pass for the CSP bottleneck.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
        
        Returns:
            Output tensor after bottleneck processing
        """
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        
        if self.shortcut and inputs.shape[-1] == self.filters:
            return x + inputs
        return x


class ChannelAttention(layers.Layer):
    """Channel attention module for adaptive feature refinement.
    
    This implements a squeeze-and-excitation operation that adaptively 
    weights channels based on their importance.
    """
    
    def __init__(
        self,
        in_channels,
        reduction_ratio=16,
        activation="silu",
        name=None
    ):
        """Initialize the channel attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck
            activation: Activation function to use
            name: Optional name for the layer
        """
        super().__init__(name=name)
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # Ensure reduction doesn't make the channels too small
        reduced_channels = max(1, in_channels // reduction_ratio)
        
        # Define layers
        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.global_max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        
        # Shared MLP
        self.fc1 = layers.Conv2D(
            filters=reduced_channels,
            kernel_size=1,
            use_bias=True,
            kernel_initializer=keras.initializers.HeNormal()
        )
        
        if activation == "silu":
            self.act = lambda x: x * tf.nn.sigmoid(x)  # SiLU/Swish
        elif activation == "relu":
            self.act = layers.ReLU()
        else:
            self.act = layers.Activation(activation)
        
        self.fc2 = layers.Conv2D(
            filters=in_channels,
            kernel_size=1,
            use_bias=True,
            kernel_initializer=keras.initializers.HeNormal()
        )
    
    def call(self, inputs, training=None):
        """Forward pass for the channel attention module.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
        
        Returns:
            Channel-weighted feature map
        """
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)
        
        avg_feat = self.fc1(avg_pool)
        avg_feat = self.act(avg_feat)
        avg_feat = self.fc2(avg_feat)
        
        max_feat = self.fc1(max_pool)
        max_feat = self.act(max_feat)
        max_feat = self.fc2(max_feat)
        
        attention = tf.nn.sigmoid(avg_feat + max_feat)
        
        return inputs * attention


class CSPStage(layers.Layer):
    """Cross Stage Partial (CSP) stage with multiple bottleneck layers.
    
    This implements a CSP stage that consists of multiple bottleneck blocks,
    which is a key component of YOLOv11 architecture for efficient feature extraction.
    """
    
    def __init__(
        self,
        filters,
        num_blocks,
        expansion=0.5,
        use_attention=True,
        activation="silu",
        name=None
    ):
        """Initialize the CSP stage.
        
        Args:
            filters: Number of output filters
            num_blocks: Number of bottleneck blocks in the stage
            expansion: Channel expansion factor
            use_attention: Whether to use channel attention
            activation: Activation function to use
            name: Optional name for the layer
        """
        super().__init__(name=name)
        self.num_blocks = num_blocks
        
        # Define layers
        self.conv1 = ConvolutionBlock(
            filters=filters,
            kernel_size=1,
            activation=activation
        )
        
        self.conv2 = ConvolutionBlock(
            filters=filters,
            kernel_size=1,
            activation=activation
        )
        
        self.bottlenecks = [
            CSPBottleneck(
                filters=filters,
                expansion=expansion,
                shortcut=True,
                activation=activation,
                name=f"bottleneck_{i}"
            ) for i in range(num_blocks)
        ]
        
        if use_attention:
            self.attention = ChannelAttention(filters * 2)
        else:
            self.attention = None
        
        self.conv3 = ConvolutionBlock(
            filters=filters,
            kernel_size=1,
            activation=activation
        )
    
    def call(self, inputs, training=None):
        """Forward pass for the CSP stage.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
        
        Returns:
            Output tensor after CSP stage processing
        """
        x1 = self.conv1(inputs, training=training)
        x2 = self.conv2(inputs, training=training)
        
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1, training=training)
        
        x = tf.concat([x1, x2], axis=-1)
        
        if self.attention is not None:
            x = self.attention(x, training=training)
        
        x = self.conv3(x, training=training)
        
        return x


class FeaturePyramidNetwork(layers.Layer):
    """Feature Pyramid Network for multi-scale feature integration.
    
    This implements a top-down feature pyramid that enhances feature 
    representation at different scales, which is crucial for detecting 
    objects of varying sizes.
    """
    
    def __init__(
        self,
        filters=[256, 512, 1024],
        activation="silu",
        name=None
    ):
        """Initialize the Feature Pyramid Network.
        
        Args:
            filters: List of filter counts for each pyramid level
            activation: Activation function to use
            name: Optional name for the layer
        """
        super().__init__(name=name)
        self.filters = filters
        
        # Define layers
        self.lateral_convs = []
        self.output_convs = []
        
        for i, f in enumerate(filters):
            self.lateral_convs.append(
                ConvolutionBlock(
                    filters=f,
                    kernel_size=1,
                    activation=activation,
                    name=f"lateral_conv_{i}"
                )
            )
            
            self.output_convs.append(
                ConvolutionBlock(
                    filters=f,
                    kernel_size=3,
                    activation=activation,
                    name=f"output_conv_{i}"
                )
            )
    
    def call(self, inputs, training=None):
        """Forward pass for the Feature Pyramid Network.
        
        Args:
            inputs: List of input feature maps [C3, C4, C5] from bottom to top
            training: Whether in training mode
        
        Returns:
            List of enhanced pyramid features [P3, P4, P5]
        """
        assert len(inputs) == len(self.filters), "Number of inputs must match filters"
        
        # Process top level
        laterals = [self.lateral_convs[-1](inputs[-1], training=training)]
        
        # Build top-down path
        for i in range(len(self.filters) - 2, -1, -1):
            # Upsample and add
            upsampled = tf.image.resize(
                laterals[0], 
                tf.shape(inputs[i])[1:3],
                method="nearest"
            )
            lateral = self.lateral_convs[i](inputs[i], training=training)
            laterals.insert(0, lateral + upsampled)
        
        # Apply output convolutions
        outputs = [
            self.output_convs[i](laterals[i], training=training)
            for i in range(len(self.filters))
        ]
        
        return outputs


class YOLOv11Backbone(keras.Model):
    """YOLOv11 backbone architecture for feature extraction.
    
    This implements an advanced YOLOv11 backbone with efficient feature extraction
    through Cross Stage Partial (CSP) blocks, attention mechanisms, and feature pyramid
    integration for state-of-the-art performance.
    """
    
    def __init__(
        self,
        input_shape=(224, 224, 3),
        width_mult=0.75,
        depth_mult=0.67,
        use_fpn=True,
        pooling="avg",
        classification=False,
        num_classes=0,
        dropout_rate=0.0,
        name="yolov11_backbone"
    ):
        """Initialize the YOLOv11 backbone.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            width_mult: Width multiplier for controlling model capacity
            depth_mult: Depth multiplier for controlling model capacity
            use_fpn: Whether to use Feature Pyramid Network
            pooling: Pooling method ('avg', 'max', or None)
            classification: Whether to add classification head
            num_classes: Number of classes for classification head
            dropout_rate: Dropout rate for classification head
            name: Name for the model
        """
        super().__init__(name=name)
        self.input_shape = input_shape
        self.width_mult = width_mult
        self.depth_mult = depth_mult
        self.use_fpn = use_fpn
        self.pooling = pooling
        self.classification = classification
        self.num_classes = num_classes
        
        # Base channels and depths
        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [3, 4, 6, 3]
        
        # Apply width and depth multipliers
        self.channels = [make_divisible(c * width_mult, 8) for c in base_channels]
        self.depths = [max(1, round(d * depth_mult)) for d in base_depths]
        
        # Define stem
        self.stem = tf.keras.Sequential([
            ConvolutionBlock(
                filters=self.channels[0],
                kernel_size=3,
                strides=2,
                activation="silu"
            ),
            ConvolutionBlock(
                filters=self.channels[0],
                kernel_size=3,
                strides=1,
                activation="silu"
            ),
            ConvolutionBlock(
                filters=self.channels[0],
                kernel_size=3,
                strides=1,
                activation="silu"
            )
        ], name="stem")
        
        # Define stages
        self.downsample1 = ConvolutionBlock(
            filters=self.channels[1],
            kernel_size=3,
            strides=2,
            activation="silu"
        )
        
        self.stage1 = CSPStage(
            filters=self.channels[1],
            num_blocks=self.depths[0],
            use_attention=True,
            activation="silu",
            name="stage1"
        )
        
        self.downsample2 = ConvolutionBlock(
            filters=self.channels[2],
            kernel_size=3,
            strides=2,
            activation="silu"
        )
        
        self.stage2 = CSPStage(
            filters=self.channels[2],
            num_blocks=self.depths[1],
            use_attention=True,
            activation="silu",
            name="stage2"
        )
        
        self.downsample3 = ConvolutionBlock(
            filters=self.channels[3],
            kernel_size=3,
            strides=2,
            activation="silu"
        )
        
        self.stage3 = CSPStage(
            filters=self.channels[3],
            num_blocks=self.depths[2],
            use_attention=True,
            activation="silu",
            name="stage3"
        )
        
        self.downsample4 = ConvolutionBlock(
            filters=self.channels[4],
            kernel_size=3,
            strides=2,
            activation="silu"
        )
        
        self.stage4 = CSPStage(
            filters=self.channels[4],
            num_blocks=self.depths[3],
            use_attention=True,
            activation="silu",
            name="stage4"
        )
        
        # Feature Pyramid Network
        if use_fpn:
            self.fpn = FeaturePyramidNetwork(
                filters=self.channels[2:5],
                activation="silu"
            )
        
        # Pooling layer
        if pooling == "avg":
            self.pool = layers.GlobalAveragePooling2D()
        elif pooling == "max":
            self.pool = layers.GlobalMaxPooling2D()
        
        # Classification head (if needed)
        if classification and num_classes > 0:
            self.dropout = layers.Dropout(dropout_rate)
            self.classifier = layers.Dense(
                num_classes,
                activation="softmax" if num_classes > 1 else "sigmoid",
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01)
            )
    
    def call(self, inputs, training=None):
        """Forward pass for the YOLOv11 backbone.
        
        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels)
            training: Whether in training mode
        
        Returns:
            Extracted features or class predictions
        """
        # Stem
        x = self.stem(inputs, training=training)
        
        # Stage 1
        x = self.downsample1(x, training=training)
        x = self.stage1(x, training=training)
        
        # Stage 2
        x = self.downsample2(x, training=training)
        c3 = self.stage2(x, training=training)
        
        # Stage 3
        x = self.downsample3(c3, training=training)
        c4 = self.stage3(x, training=training)
        
        # Stage 4
        x = self.downsample4(c4, training=training)
        c5 = self.stage4(x, training=training)
        
        # Apply FPN if enabled
        if self.use_fpn:
            features = self.fpn([c3, c4, c5], training=training)
            x = features[-1]  # Use the highest level feature
        else:
            x = c5
        
        # Apply pooling if specified
        if hasattr(self, "pool"):
            x = self.pool(x)
            
            # Apply classification head if needed
            if self.classification and self.num_classes > 0:
                x = self.dropout(x, training=training)
                x = self.classifier(x)
        
        return x


def make_divisible(x, divisor):
    """Make channels divisible by divisor.
    
    This function ensures that all layers have a channel number that is divisible
    by the given divisor. It is used to ensure that the resulting dimensions
    are compatible with hardware optimizations.
    
    Args:
        x: The value to make divisible
        divisor: The divisor to make x divisible by
    
    Returns:
        The rounded value
    """
    return max(int(x + divisor / 2) // divisor * divisor, divisor) 