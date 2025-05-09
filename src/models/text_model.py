import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from transformers import TFDistilBertModel, DistilBertTokenizerFast

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        
        # Store attention weights for visualization if needed
        self.attention_weights = None
        
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        
        # Scale dot-product attention
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        
        # Apply softmax to get attention weights
        weights = tf.nn.softmax(scaled_score, axis=-1)
        
        # Store for visualization
        self.attention_weights = weights
        
        # Apply attention weights to values
        output = tf.matmul(weights, value)
        
        return output, weights
        
    def separate_heads(self, x, batch_size):
        # Reshape to [batch_size, seq_length, num_heads, projection_dim]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        
        # Transpose to [batch_size, num_heads, seq_length, projection_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Linear projections
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)      # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        
        # Separate heads
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)      # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        
        # Compute attention for each head
        attention, weights = self.attention(query, key, value)
        
        # Reshape to [batch_size, seq_length, num_heads, projection_dim]
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        
        # Reshape to [batch_size, seq_length, embed_dim]
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        
        # Final linear projection
        output = self.combine_heads(concat_attention)
        
        return output
    
    def get_attention_weights(self):
        return self.attention_weights


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with multi-head self-attention and feed-forward layers"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, inputs, training=False):
        # Apply attention with residual connection and normalization
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Apply feed-forward network with residual connection and normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_attention_weights(self):
        return self.att.get_attention_weights()


class TransformerTextFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, model_name='distilbert-base-uncased', trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.transformer = TFDistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.transformer.trainable = trainable

    def call(self, input_texts):
        # input_texts: batch of strings (not token ids)
        inputs = self.tokenizer(
            input_texts, padding=True, truncation=True, return_tensors='tf'
        )
        outputs = self.transformer(inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return cls_embedding


class TextFeatureExtractor(tf.keras.Model):
    """Text feature extractor using BiLSTM and attention mechanisms"""
    
    def __init__(self, vocab_size, embedding_dim=300, rnn_units=128, 
                 dropout=0.3, attention_heads=4, use_transformer=True, use_hf_transformer=False, transformer_trainable=False, concat_transformer_bilstm=False):
        super(TextFeatureExtractor, self).__init__()
        self.use_hf_transformer = use_hf_transformer
        self.concat_transformer_bilstm = concat_transformer_bilstm
        if use_hf_transformer:
            self.transformer = TransformerTextFeatureExtractor(trainable=transformer_trainable)
            self.output_dense = layers.Dense(rnn_units * 2)
            self.dropout = layers.Dropout(dropout)
        if not use_hf_transformer or concat_transformer_bilstm:
            # ... existing BiLSTM+attention code ...
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.rnn_units = rnn_units
            self.attention_heads = attention_heads
            self.use_transformer = use_transformer
            self.embedding = layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                mask_zero=True
            )
            self.bilstm = layers.Bidirectional(
                layers.LSTM(
                    units=rnn_units,
                    return_sequences=True,
                    return_state=True,
                    dropout=dropout,
                    recurrent_dropout=0,
                    recurrent_activation='sigmoid'
                )
            )
            if use_transformer:
                self.transformer_block = TransformerBlock(
                    embed_dim=rnn_units * 2,
                    num_heads=attention_heads,
                    ff_dim=rnn_units * 4,
                    rate=dropout
                )
            self.attention = MultiHeadSelfAttention(
                embed_dim=rnn_units * 2,
                num_heads=attention_heads
            )
            self.global_max_pool = layers.GlobalMaxPooling1D()
            self.global_avg_pool = layers.GlobalAveragePooling1D()
            self.bilstm_output_dense = layers.Dense(rnn_units * 2)
            self.bilstm_dropout = layers.Dropout(dropout)
        
    def call(self, inputs, training=False):
        features = []
        if self.use_hf_transformer:
            # inputs: batch of strings
            transformer_features = self.transformer(inputs)
            features.append(transformer_features)
        if not self.use_hf_transformer or self.concat_transformer_bilstm:
            # inputs: batch of token ids
            mask = tf.cast(tf.math.not_equal(inputs, 0), tf.float32)
            mask = tf.expand_dims(mask, -1)
            x = self.embedding(inputs)
            lstm_outputs, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)
            lstm_outputs = lstm_outputs * mask
            if self.use_transformer:
                transformed = self.transformer_block(lstm_outputs, training=training)
                transformed = transformed * mask
            else:
                transformed = lstm_outputs
            attended = self.attention(transformed)
            attended = attended * mask
            max_pool = self.global_max_pool(attended)
            avg_pool = self.global_avg_pool(attended)
            concat_h = tf.concat([forward_h, backward_h], axis=-1)
            concat_pooled = tf.concat([max_pool, avg_pool], axis=-1)
            bilstm_features = self.bilstm_output_dense(tf.concat([concat_pooled, concat_h], axis=-1))
            bilstm_features = self.bilstm_dropout(bilstm_features, training=training)
            features.append(bilstm_features)
        if len(features) > 1:
            features = tf.concat(features, axis=-1)
        else:
            features = features[0]
        return features
    
    def get_attention_weights(self):
        """Return attention weights for visualization"""
        if self.use_transformer:
            return self.transformer_block.get_attention_weights()
        return self.attention.get_attention_weights()


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