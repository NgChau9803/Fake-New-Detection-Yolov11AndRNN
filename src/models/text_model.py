import tensorflow as tf
import numpy as np

class TextFeatureExtractor(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=300, rnn_units=128, dropout=0.2, attention_heads=4):
        super(TextFeatureExtractor, self).__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.dropout_rate = dropout
        self.attention_heads = attention_heads
        
        # Word embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size + 1,  # Add 1 for padding token (0)
            output_dim=embedding_dim,
            mask_zero=True,
            name="word_embedding"
        )
        
        # Positional encoding for better sequence understanding
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Bidirectional LSTM for better context capture
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(rnn_units, return_sequences=True),
            name="bidirectional_lstm"
        )
        
        # Attention mechanism to focus on important words
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=attention_heads, 
            key_dim=rnn_units,
            name="multi_head_attention"
        )
        
        # Layer normalization for better training stability
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network after attention
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(rnn_units * 4, activation='relu'),
            tf.keras.layers.Dense(rnn_units * 2)
        ], name="feed_forward_network")
        
        # Global average pooling to get fixed-size representation
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pooling")
        
        # Final output layers
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(embedding_dim, activation='relu', name="final_dense")
        
        # Store attention weights for visualization
        self.attention_weights = None
        
    def call(self, inputs, training=False, return_attention=False):
        # Create mask for padding tokens (0)
        mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        
        # Get embeddings
        x = self.embedding(inputs)
        
        # Apply mask to zero out padding tokens
        x = x * mask
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply Bidirectional LSTM
        lstm_output = self.bi_lstm(x)
        
        # Apply attention with residual connection and layer normalization
        # Store attention weights for later visualization
        attention_output, attention_weights = self.attention(
            query=lstm_output, 
            key=lstm_output, 
            value=lstm_output,
            return_attention_scores=True
        )
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights
        
        # Add residual connection and normalize
        x = self.layer_norm1(lstm_output + attention_output)
        
        # Apply feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        
        # Pool to get fixed-size representation
        x = self.global_pool(x)
        
        # Apply dropout and final dense layer
        x = self.dropout(x, training=training)
        x = self.dense(x)
        
        if return_attention:
            return x, attention_weights
        
        return x
    
    def get_attention_weights(self):
        """Return attention weights for visualization"""
        return self.attention_weights
    
    def get_token_importance(self, inputs, tokenizer):
        """
        Get token importance based on attention weights
        
        Args:
            inputs: Input token IDs
            tokenizer: Tokenizer with word index mapping
            
        Returns:
            List of (token, importance_score) pairs
        """
        # Get predictions with attention weights
        _, attention_weights = self.call(inputs, return_attention=True)
        
        # Convert token IDs to words
        idx_to_word = {v: k for k, v in tokenizer.items()}
        
        # Average attention weights across heads and positions
        # Shape: [batch_size, seq_len, seq_len, num_heads]
        avg_weights = tf.reduce_mean(attention_weights, axis=-1)  # Average across heads
        
        # Get importance per token
        token_importance = []
        for i, token_id in enumerate(inputs[0]):  # Use first example in batch
            if token_id == 0:  # Skip padding
                continue
            
            # Get word for token ID
            word = idx_to_word.get(token_id.numpy(), '<UNK>')
            
            # Get average attention for this token (based on how much other tokens attend to it)
            importance = tf.reduce_mean(avg_weights[0, :, i]).numpy()
            
            token_importance.append((word, importance))
        
        return token_importance


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer to provide sequence position information
    """
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
    
    def build(self, input_shape):
        _, max_len, _ = input_shape
        
        # Create positional encoding matrix
        pos_enc = self.positional_encoding(max_len, self.d_model)
        self.pos_encoding = tf.constant(pos_enc, dtype=tf.float32)
        
        self.built = True
    
    def call(self, inputs):
        # Add positional encoding to input embeddings
        return inputs + self.pos_encoding[:tf.shape(inputs)[1], :]
    
    def positional_encoding(self, position, d_model):
        # Implementation of positional encoding from "Attention Is All You Need" paper
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates 