
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, LSTM, DepthwiseConv2D, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Position-Aware Attention Scaling (PAAS) Layer
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
        return attn_scores * self.w_p

# ---------------------------
# RVT Transformer Block
# ---------------------------
def rvt_transformer_block(x, heads=4, key_dim=64, dropout_rate=0.1, ffn_units=256):
    attn_layer = MultiHeadAttention(num_heads=heads, key_dim=key_dim)
    attn_output = attn_layer(x, x)

    paas = PAAS(num_heads=heads)
    scaled_attn = paas(attn_output)

    attn_output = Dropout(dropout_rate)(scaled_attn)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = DepthwiseConv2D(kernel_size=3, padding="same")(x)
    ffn = Dense(ffn_units, activation='relu')(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(x.shape[-1])(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)

    return x

# ---------------------------
# Build Model: RVT Transformer Block Build
# ---------------------------
def build_rvt_transformer_model(input_shape=(224, 224, 3), print_summary=True):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model(inputs)
    b, h, w, c = x.shape
    x = tf.reshape(x, (-1, h * w, c))

    x = rvt_transformer_block(x, heads=4, key_dim=64, dropout_rate=0.1, ffn_units=256)
    x = LSTM(128)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)

    if print_summary:
        model.summary()

    return model
