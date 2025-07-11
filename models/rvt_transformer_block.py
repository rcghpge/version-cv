
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling2D, Add

# ---------------------------
# Position-Aware Attention Scaling (PAAS) Utility
# ---------------------------
class PAAS(tf.keras.layers.Layer):
    def __init__(self, num_heads):
        super(PAAS, self).__init__()
        self.num_heads = num_heads

    def build(self, input_shape):
        self.w_p = self.add_weight(
            shape=(1, 1, self.num_heads, input_shape[1]),
            initializer="ones",
            trainable=True,
            name="Wp"
        )

    def call(self, attn_scores):
        # Apply learnable position scaling to attention scores
        return attn_scores * self.w_p

# ---------------------------
# RVT-Inspired Transformer Block
# ---------------------------
def rvt_transformer_block(x, heads=2, key_dim=64, dropout_rate=0.1, ffn_units=128):
    # Self-attention with PAAS
    attn_layer = MultiHeadAttention(num_heads=heads, key_dim=key_dim)
    attn_output = attn_layer(x, x)
    
    # Apply PAAS scaling
    paas = PAAS(num_heads=heads)
    scaled_attn = paas(attn_output)
    
    attn_output = Dropout(dropout_rate)(scaled_attn)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Convolutional Feed-Forward Network (ConvFFN)
    ffn = DepthwiseConv2D(kernel_size=3, padding="same")(x)
    ffn = Dense(ffn_units, activation='relu')(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(x.shape[-1])(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)

    return x
