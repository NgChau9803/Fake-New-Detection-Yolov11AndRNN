import tensorflow as tf
from src.models.text_model import TextFeatureExtractor
from src.models.image_model import ImageFeatureExtractor

class MultiModalFusionModel(tf.keras.Model):
    def __init__(self, vocab_size, config):
        super(MultiModalFusionModel, self).__init__()
        
        # Store configuration
        self.config = config
        self.fusion_method = config['model']['fusion'].get('fusion_method', 'concat')
        self.visualize_attention = config['model']['text'].get('visualize_attention', False)
        
        # Text feature extractor
        self.text_extractor = TextFeatureExtractor(
            vocab_size=vocab_size,
            embedding_dim=config['model']['text']['embedding_dim'],
            rnn_units=config['model']['text']['rnn_units'],
            dropout=config['model']['text']['dropout']
        )
        
        # Image feature extractor
        self.image_extractor = ImageFeatureExtractor(
            input_shape=config['model']['image']['input_shape'],
            backbone=config['model']['image']['backbone']
        )
        
        # Metadata processing layers
        self.metadata_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.metadata_dense2 = tf.keras.layers.Dense(32, activation='relu')
        
        # Feature dimension normalization
        self.text_feature_norm = tf.keras.layers.Dense(256, activation='relu')
        self.image_feature_norm = tf.keras.layers.Dense(256, activation='relu')
        self.metadata_feature_norm = tf.keras.layers.Dense(256, activation='relu')
        
        # Cross-modal attention for attention fusion
        if self.fusion_method == 'attention':
            self.text_image_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=64
            )
            self.image_text_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=64
            )
            self.attention_dense = tf.keras.layers.Dense(256, activation='relu')
        
        # Bilinear fusion for bilinear method
        if self.fusion_method == 'bilinear':
            self.bilinear_layer = BilinearFusion(256)
        
        # Fusion layers
        fusion_dims = config['model']['fusion']['hidden_dims']
        fusion_dropout = config['model']['fusion']['dropout']
        
        self.fusion_layers = []
        for dim in fusion_dims:
            self.fusion_layers.append(tf.keras.layers.Dense(dim, activation='relu'))
            self.fusion_layers.append(tf.keras.layers.BatchNormalization())
            self.fusion_layers.append(tf.keras.layers.Dropout(fusion_dropout))
        
        # Output layer for binary classification (fake or real)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        
        # Store attention weights for visualization if needed
        self.attention_weights = None
        
    def call(self, inputs, training=False):
        # Process text input
        text_features = self.text_extractor(inputs['text'], training=training)
        
        # Process image input
        image_features = self.image_extractor(inputs['image'], training=training)
        
        # Process metadata
        metadata = inputs['metadata']
        metadata_features = self.metadata_dense1(metadata)
        metadata_features = self.metadata_dense2(metadata_features)
        
        # Normalize feature dimensions
        text_features = self.text_feature_norm(text_features)
        image_features = self.image_feature_norm(image_features)
        metadata_features = self.metadata_feature_norm(metadata_features)
        
        # Apply fusion based on selected method
        if self.fusion_method == 'concat':
            # Simple concatenation
            combined_features = tf.concat([text_features, image_features, metadata_features], axis=1)
        
        elif self.fusion_method == 'attention':
            # Cross-modal attention fusion
            # Reshape for attention mechanism
            batch_size = tf.shape(text_features)[0]
            text_for_attn = tf.reshape(text_features, [batch_size, 1, -1])
            image_for_attn = tf.reshape(image_features, [batch_size, 1, -1])
            
            # Apply cross-attention
            text_image_attn = self.text_image_attention(
                query=text_for_attn,
                key=image_for_attn,
                value=image_for_attn
            )
            image_text_attn = self.image_text_attention(
                query=image_for_attn,
                key=text_for_attn,
                value=text_for_attn
            )
            
            # Reshape back and combine
            text_image_attn = tf.reshape(text_image_attn, [batch_size, -1])
            image_text_attn = tf.reshape(image_text_attn, [batch_size, -1])
            
            # Process attended features
            attended_features = tf.concat([text_image_attn, image_text_attn], axis=1)
            attended_features = self.attention_dense(attended_features)
            
            # Combine all features
            combined_features = tf.concat([
                text_features, image_features, metadata_features, attended_features
            ], axis=1)
            
        elif self.fusion_method == 'bilinear':
            # Bilinear fusion combines modalities using multiplicative interactions
            # Combine text and image first
            text_image_bilinear = self.bilinear_layer([text_features, image_features])
            
            # Then concatenate with metadata
            combined_features = tf.concat([text_image_bilinear, metadata_features], axis=1)
            
        else:
            # Default to simple concatenation
            combined_features = tf.concat([text_features, image_features, metadata_features], axis=1)
        
        # Apply fusion layers
        x = combined_features
        for layer in self.fusion_layers:
            x = layer(x, training=training)
        
        # Final classification
        output = self.output_layer(x)
        
        return output
    
    def get_attention_weights(self):
        """Return attention weights for visualization if available"""
        if hasattr(self.text_extractor, 'attention') and self.visualize_attention:
            return self.text_extractor.get_attention_weights()
        return None


class BilinearFusion(tf.keras.layers.Layer):
    """Bilinear fusion layer for combining two modalities with multiplicative interactions"""
    
    def __init__(self, output_dim):
        super(BilinearFusion, self).__init__()
        self.output_dim = output_dim
        
    def build(self, input_shape):
        # Create the bilinear weights tensor
        dim1 = input_shape[0][-1]
        dim2 = input_shape[1][-1]
        
        self.bilinear_weights = self.add_weight(
            shape=(dim1, dim2, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='bilinear_weights'
        )
        
        self.built = True
        
    def call(self, inputs):
        # Implement bilinear fusion: x1^T W x2
        x1 = inputs[0]  # [batch_size, dim1]
        x2 = inputs[1]  # [batch_size, dim2]
        
        # Expand dimensions for broadcasting
        x1_expanded = tf.expand_dims(x1, axis=2)  # [batch_size, dim1, 1]
        x2_expanded = tf.expand_dims(x2, axis=1)  # [batch_size, 1, dim2]
        
        # Compute the bilinear product
        # First compute x1^T * W: [batch_size, dim1, 1] * [dim1, dim2, output_dim] -> [batch_size, dim2, output_dim]
        x1_w = tf.tensordot(x1_expanded, self.bilinear_weights, axes=[[1], [0]])
        
        # Then compute (x1^T * W) * x2: [batch_size, dim2, output_dim] * [batch_size, 1, dim2] -> [batch_size, output_dim]
        bilinear_out = tf.matmul(x2_expanded, x1_w)
        bilinear_out = tf.squeeze(bilinear_out, axis=1)
        
        return bilinear_out 