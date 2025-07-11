
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, LSTM, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Original Transformer Block
# ---------------------------
def transformer_block(x, heads=2, key_dim=64, dropout_rate=0.1, ffn_units=128):
    attn_output = MultiHeadAttention(num_heads=heads, key_dim=key_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = Dense(ffn_units, activation='relu')(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(x.shape[-1])(ffn)
    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

# ---------------------------
# Build Model: Original Transformer Block Build
# ---------------------------
def build_original_transformer_model(input_shape=(224, 224, 3), print_summary=True):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model(inputs)
    b, h, w, c = x.shape
    x = tf.reshape(x, (-1, h * w, c))

    x = transformer_block(x, heads=2, key_dim=64, dropout_rate=0.1, ffn_units=128)
    x = LSTM(128)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)

    if print_summary:
        model.summary()

    return model
