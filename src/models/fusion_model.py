import tensorflow as tf
# Try to import tensorflow_addons, but handle gracefully if not available
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    print("TensorFlow Addons not available. Using standard layers instead.")
    TFA_AVAILABLE = False
from src.models.text_model import TextFeatureExtractor
from src.models.image_model import ImageFeatureExtractor
import numpy as np
import os
import json

class MultiModalFusionModel(tf.keras.Model):
    def __init__(self, vocab_size, config):
        super(MultiModalFusionModel, self).__init__()
        
        # Store configuration
        self.config = config
        self.fusion_method = config['model']['fusion'].get('fusion_method', 'cross_attention')
        self.visualize_attention = config['model']['text'].get('visualize_attention', False)
        self.use_spectral_norm = config['model']['fusion'].get('use_spectral_norm', True) and TFA_AVAILABLE
        self.use_stochastic_depth = config['model']['fusion'].get('use_stochastic_depth', False)
        self.stochastic_depth_rate = config['model']['fusion'].get('stochastic_depth_rate', 0.1)
        self.use_gradient_reversal = config['model']['fusion'].get('use_gradient_reversal', False)
        
        # Text feature extractor
        self.text_extractor = TextFeatureExtractor(
            vocab_size=vocab_size,
            embedding_dim=config['model']['text']['embedding_dim'],
            rnn_units=config['model']['text']['lstm_units'],
            dropout=config['model']['text']['dropout_rate'],
            attention_heads=config['model']['text'].get('attention_heads', 4)
        )
        
        # Configure specific parameters for YOLOv11 if selected
        backbone_type = config['model']['image'].get('backbone_type', config['model']['image'].get('backbone', 'yolov11'))
        backbone_params = {}
        if backbone_type == 'yolov11' or 'yolo' in backbone_type.lower():
            print("Using YOLOv11 backbone for image feature extraction")
            backbone_type = 'yolov11'  # Normalize the name
            backbone_params = {
                'width_mult': config['model']['image'].get('width_mult', 0.75),
                'depth_mult': config['model']['image'].get('depth_mult', 0.67),
                'use_fpn': config['model']['image'].get('use_fpn', True)
            }
        elif backbone_type != 'resnet50' and backbone_type != 'efficientnetb0':
            print(f"Unsupported backbone type '{backbone_type}', falling back to YOLOv11")
            backbone_type = 'yolov11'
            backbone_params = {
                'width_mult': config['model']['image'].get('width_mult', 0.75),
                'depth_mult': config['model']['image'].get('depth_mult', 0.67),
                'use_fpn': config['model']['image'].get('use_fpn', True)
            }
        
        # Image feature extractor
        self.image_extractor = ImageFeatureExtractor(
            input_shape=config['model']['image']['input_shape'],
            backbone_type=backbone_type,
            pretrained=config['model']['image'].get('pretrained', True),
            output_dim=config['model']['image'].get('output_dim', 512),
            dropout_rate=config['model']['image'].get('dropout_rate', 0.2),
            l2_regularization=config['model']['image'].get('l2_regularization', 1e-5),
            use_attention=config['model']['image'].get('use_attention', True),
            pooling=config['model']['image'].get('pooling', 'avg'),
            trainable=config['model']['image'].get('trainable', False),
            backbone_params=backbone_params
        )
        
        # Metadata processing layers with spectral normalization if enabled
        if self.use_spectral_norm:
            self.metadata_dense1 = tfa.layers.SpectralNormalization(
                tf.keras.layers.Dense(128, activation='relu')
            )
            self.metadata_dense2 = tfa.layers.SpectralNormalization(
                tf.keras.layers.Dense(64, activation='relu')
            )
            self.metadata_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        else:
            self.metadata_dense1 = tf.keras.layers.Dense(128, activation='relu')
            self.metadata_dense2 = tf.keras.layers.Dense(64, activation='relu')
            self.metadata_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Feature dimension normalization and projection
        feature_dim = config['model']['fusion'].get('feature_dim', 256)
        self.text_feature_norm = self._create_projection_layer(feature_dim)
        self.image_feature_norm = self._create_projection_layer(feature_dim)
        self.metadata_feature_norm = self._create_projection_layer(feature_dim)
        
        # Cross-Modal Transformer for fusion
        if self.fusion_method in ['cross_attention', 'transformer']:
            self.cross_attention_layers = []
            self.feed_forward_layers = []
            self.layer_norms = []
            
            # Number of transformer layers
            num_layers = config['model']['fusion'].get('transformer_layers', 2)
            
            for i in range(num_layers):
                # Multi-head cross-attention
                self.cross_attention_layers.append(
                    tf.keras.layers.MultiHeadAttention(
                        num_heads=config['model']['fusion'].get('attention_heads', 8),
                        key_dim=feature_dim // config['model']['fusion'].get('attention_heads', 8),
                        dropout=config['model']['fusion'].get('attention_dropout', 0.1),
                        name=f"cross_attention_{i}"
                    )
                )
                
                # Feed-forward network
                self.feed_forward_layers.append(
                    self._create_feed_forward_network(
                        feature_dim, 
                        config['model']['fusion'].get('ff_dim', feature_dim * 4)
                    )
                )
                
                # Layer normalization
                self.layer_norms.append([
                    tf.keras.layers.LayerNormalization(epsilon=1e-6),
                    tf.keras.layers.LayerNormalization(epsilon=1e-6)
                ])
            
            # Final cross-modal projection
            self.cross_modal_projection = self._create_projection_layer(feature_dim * 2)
        
        # Gated Multimodal Unit for fusion
        elif self.fusion_method == 'gmu':
            self.gated_multimodal_unit = GatedMultimodalUnit(feature_dim)
        
        # Low-rank Bilinear Pooling for fusion
        elif self.fusion_method == 'mutan':
            self.mutan_fusion = MutanFusion(
                output_dim=feature_dim,
                rank=config['model']['fusion'].get('mutan_rank', 15)
            )
            
        # FiLM conditioning
        elif self.fusion_method == 'film':
            self.film_generator = FiLMGenerator(feature_dim)
        
        # Fusion layers
        fusion_dims = config['model']['fusion'].get('hidden_units', [256, 128])
        fusion_dropout = config['model']['fusion'].get('dropout_rate', 0.3)
        
        self.fusion_layers = []
        for i, dim in enumerate(fusion_dims):
            if self.use_spectral_norm:
                dense = tfa.layers.SpectralNormalization(
                    tf.keras.layers.Dense(dim, activation='relu')
                )
            else:
                dense = tf.keras.layers.Dense(dim, activation='relu')
                
            self.fusion_layers.append(dense)
            self.fusion_layers.append(tf.keras.layers.BatchNormalization())
            self.fusion_layers.append(tf.keras.layers.Dropout(fusion_dropout))
            
            # Add stochastic depth if enabled
            if self.use_stochastic_depth and i > 0:
                self.fusion_layers.append(
                    StochasticDepth(self.stochastic_depth_rate)
                )
        
        # L1/L2 regularization with the regularizer's strength
        l1_reg = float(config['model']['fusion'].get('l1_reg', 1e-5))
        l2_reg = float(config['model']['fusion'].get('l2_reg', 1e-4))
        regularizer = tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
        
        # Output layer for binary classification with regularization
        self.output_layer = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',
            kernel_regularizer=regularizer,
            name="output"
        )
        
        # Store attention weights for visualization if needed
        self.attention_weights = None
        
        # --- Categorical metadata embedding sizes ---
        # Try to load vocab sizes from disk if available
        processed_dir = config['data']['processed_dir'] if 'processed_dir' in config['data'] else 'data/processed'
        vocab_dir = os.path.join(processed_dir, 'vocabs')
        def load_vocab_size(name):
            try:
                with open(os.path.join(vocab_dir, f'{name}_vocab.json'), 'r') as f:
                    vocab = json.load(f)
                size = len(vocab)
                if size == 0:
                    print(f"Warning: {name}_vocab.json is empty. Setting vocab size to 1.")
                    return 1
                return size
            except Exception:
                print(f"Warning: Could not load {name}_vocab.json. Using fallback size 1.")
                return 1  # fallback
        self.source_vocab_size = max(1, load_vocab_size('source'))
        self.subreddit_vocab_size = max(1, load_vocab_size('subreddit'))
        self.author_vocab_size = max(1, load_vocab_size('author'))
        self.categorical_embedding_dim = config['model']['fusion'].get('categorical_embedding_dim', 32)
        # Embedding layers for categorical metadata
        self.source_embedding = tf.keras.layers.Embedding(self.source_vocab_size, self.categorical_embedding_dim, mask_zero=True, name='source_embedding')
        self.subreddit_embedding = tf.keras.layers.Embedding(self.subreddit_vocab_size, self.categorical_embedding_dim, mask_zero=True, name='subreddit_embedding')
        self.author_embedding = tf.keras.layers.Embedding(self.author_vocab_size, self.categorical_embedding_dim, mask_zero=True, name='author_embedding')
        
    def _create_projection_layer(self, dim):
        """Helper function to create projection layers with optional spectral normalization"""
        if self.use_spectral_norm:
            return tf.keras.Sequential([
                tfa.layers.SpectralNormalization(tf.keras.layers.Dense(dim)),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Activation('relu')
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Activation('relu')
            ])
    
    def _create_feed_forward_network(self, dim, ff_dim):
        """Helper function to create feed-forward network with optional spectral normalization"""
        if self.use_spectral_norm:
            return tf.keras.Sequential([
                tfa.layers.SpectralNormalization(tf.keras.layers.Dense(ff_dim, activation='relu')),
                tf.keras.layers.Dropout(self.config['model']['fusion'].get('dropout_rate', 0.1)),
                tfa.layers.SpectralNormalization(tf.keras.layers.Dense(dim))
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Dense(ff_dim, activation='relu'),
                tf.keras.layers.Dropout(self.config['model']['fusion'].get('dropout_rate', 0.1)),
                tf.keras.layers.Dense(dim)
            ])
        
    def call(self, inputs, training=False):
        # Process text input
        text_features = self.text_extractor(inputs['text'], training=training)
        
        # Process image input
        image_features = self.image_extractor(inputs['image'], training=training)
        
        # Categorical metadata embeddings
        source_emb = self.source_embedding(inputs['source_idx'])
        subreddit_emb = self.subreddit_embedding(inputs['subreddit_idx'])
        author_emb = self.author_embedding(inputs['author_idx'])
        # Concatenate all categorical embeddings
        cat_emb = tf.concat([source_emb, subreddit_emb, author_emb], axis=-1)
        
        # Process metadata
        metadata = inputs['metadata']
        metadata_features = self.metadata_dense1(metadata)
        metadata_features = self.metadata_dense2(metadata_features)
        metadata_features = self.metadata_norm(metadata_features)
        # Concatenate dense metadata and categorical embeddings
        metadata_all = tf.concat([metadata_features, cat_emb], axis=-1)
        
        # Project features to the same dimension
        text_features = self.text_feature_norm(text_features)
        image_features = self.image_feature_norm(image_features)
        metadata_features = self.metadata_feature_norm(metadata_all)
        
        # Apply fusion based on selected method
        if self.fusion_method == 'concat':
            # Simple concatenation
            combined_features = tf.concat([text_features, image_features, metadata_features], axis=1)
        
        elif self.fusion_method in ['cross_attention', 'transformer']:
            # Transformer-based cross-modal fusion
            # Reshape features for attention mechanism
            batch_size = tf.shape(text_features)[0]
            
            # Create token sequences for each modality
            # Add sequence dimension to make it [batch, seq_len=1, feat_dim]
            text_tokens = tf.expand_dims(text_features, 1)
            image_tokens = tf.expand_dims(image_features, 1)
            metadata_tokens = tf.expand_dims(metadata_features, 1)
            
            # Concatenate tokens from different modalities to form a sequence
            # Result: [batch, seq_len=3, feat_dim]
            multimodal_tokens = tf.concat([text_tokens, image_tokens, metadata_tokens], axis=1)
            
            # Apply transformer layers
            x = multimodal_tokens
            for i in range(len(self.cross_attention_layers)):
                # Multi-head cross-attention
                attn_output = self.cross_attention_layers[i](
                    query=x, 
                    key=x, 
                    value=x,
                    training=training
                )
                
                # Add & Norm (residual connection)
                x = self.layer_norms[i][0](x + attn_output)
                
                # Feed Forward network
                ffn_output = self.feed_forward_layers[i](x)
                
                # Add & Norm (residual connection)
                x = self.layer_norms[i][1](x + ffn_output)
            
            # Extract modality-specific features after cross-attention
            text_attended = x[:, 0, :]    # First token corresponds to text
            image_attended = x[:, 1, :]   # Second token corresponds to image
            metadata_attended = x[:, 2, :] # Third token corresponds to metadata
            
            # Combine attended features
            combined_features = tf.concat([
                text_attended, image_attended, metadata_attended
            ], axis=1)
            
            # Project to final fusion representation
            combined_features = self.cross_modal_projection(combined_features)
            
        elif self.fusion_method == 'gmu':
            # Gated multimodal fusion
            text_image_gated = self.gated_multimodal_unit(
                [text_features, image_features]
            )
            combined_features = tf.concat([text_image_gated, metadata_features], axis=1)
            
        elif self.fusion_method == 'mutan':
            # Multimodal Tucker fusion (MUTAN)
            text_image_mutan = self.mutan_fusion([text_features, image_features])
            combined_features = tf.concat([text_image_mutan, metadata_features], axis=1)
            
        elif self.fusion_method == 'film':
            # Feature-wise Linear Modulation (FiLM)
            # Use text to modulate image features
            film_params = self.film_generator(text_features)
            modulated_image = film_params['gamma'] * image_features + film_params['beta']
            combined_features = tf.concat([text_features, modulated_image, metadata_features], axis=1)
            
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


class GatedMultimodalUnit(tf.keras.layers.Layer):
    """Gated Multimodal Unit for adaptive fusion of two modalities"""
    
    def __init__(self, output_dim):
        super(GatedMultimodalUnit, self).__init__()
        self.output_dim = output_dim
        
    def build(self, input_shape):
        dim1 = input_shape[0][-1]
        dim2 = input_shape[1][-1]
        
        # Transform each modality to common dimension
        self.transform_x1 = tf.keras.layers.Dense(self.output_dim, name='transform_x1')
        self.transform_x2 = tf.keras.layers.Dense(self.output_dim, name='transform_x2')
        
        # Create the gating mechanism
        self.gate_dense = tf.keras.layers.Dense(self.output_dim, activation='sigmoid', name='gate')
        
        self.built = True
        
    def call(self, inputs):
        x1 = inputs[0]  # First modality (e.g., text)
        x2 = inputs[1]  # Second modality (e.g., image)
        
        # Transform to common dimension
        h1 = self.transform_x1(x1)
        h2 = self.transform_x2(x2)
        
        # Compute the gate
        concat = tf.concat([x1, x2], axis=-1)
        z = self.gate_dense(concat)
        
        # Fuse modalities using the gate
        output = z * h1 + (1 - z) * h2
        
        return output


class MutanFusion(tf.keras.layers.Layer):
    """Multimodal Tucker Fusion (MUTAN) for efficient bilinear fusion"""
    
    def __init__(self, output_dim, rank=15):
        super(MutanFusion, self).__init__()
        self.output_dim = output_dim
        self.rank = rank
        
    def build(self, input_shape):
        dim1 = input_shape[0][-1]
        dim2 = input_shape[1][-1]
        
        # Create low-rank factors
        self.W1 = self.add_weight(
            shape=(dim1, self.rank),
            initializer='glorot_uniform',
            trainable=True,
            name='W1'
        )
        
        self.W2 = self.add_weight(
            shape=(dim2, self.rank),
            initializer='glorot_uniform',
            trainable=True,
            name='W2'
        )
        
        self.W3 = self.add_weight(
            shape=(self.rank, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='W3'
        )
        
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
        self.built = True
        
    def call(self, inputs):
        x1 = inputs[0]  # [batch_size, dim1]
        x2 = inputs[1]  # [batch_size, dim2]
        
        # Project inputs to low-rank space
        x1_proj = tf.matmul(x1, self.W1)  # [batch_size, rank]
        x2_proj = tf.matmul(x2, self.W2)  # [batch_size, rank]
        
        # Element-wise multiplication in low-rank space
        fusion = x1_proj * x2_proj  # [batch_size, rank]
        
        # Project back to output dimension
        output = tf.matmul(fusion, self.W3) + self.bias  # [batch_size, output_dim]
        
        return output


class FiLMGenerator(tf.keras.layers.Layer):
    """Feature-wise Linear Modulation (FiLM) generator"""
    
    def __init__(self, feature_dim):
        super(FiLMGenerator, self).__init__()
        self.feature_dim = feature_dim
        
    def build(self, input_shape):
        dim = input_shape[-1]
        
        # Dense layer to generate FiLM parameters
        self.film_generator = tf.keras.layers.Dense(self.feature_dim * 2, name='film_generator')
        
        self.built = True
        
    def call(self, x):
        # Generate gamma and beta
        film_params = self.film_generator(x)
        
        # Split into gamma and beta
        gamma, beta = tf.split(film_params, 2, axis=-1)
        
        # Apply scaling to gamma to make it centered around 1
        gamma = tf.nn.sigmoid(gamma) * 2.0  # Scale to [0, 2]
        
        return {'gamma': gamma, 'beta': beta}


class StochasticDepth(tf.keras.layers.Layer):
    """Stochastic Depth layer for residual networks"""
    
    def __init__(self, drop_rate=0.2):
        super(StochasticDepth, self).__init__()
        self.drop_rate = drop_rate
        
    def call(self, inputs, training=None):
        if not training or self.drop_rate == 0:
            return inputs
            
        # Random variable for dropping the entire residual path
        keep_prob = 1.0 - self.drop_rate
        random_tensor = keep_prob + tf.random.uniform([], 0, 1)
        binary_tensor = tf.floor(random_tensor)
        
        # Either keep the entire path or drop it completely
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output 