
import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, LSTM, Input
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Global Configs
# ---------------------------
SEED = 2025
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# ---------------------------
# Modular Model Transformer Block
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
# Build Refined Spatial Model (Stops before pooling/flattening)
# ---------------------------
def build_refined_model(input_shape=(224, 224, 3), print_summary=True):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model(inputs)
    # Stop here: no pooling or flattening, preserves spatial feature maps

    # Example: optional projection or attention blocks can go here
    x = transformer_block(x, heads=2, key_dim=64, dropout_rate=0.1, ffn_units=128)
    x = LSTM(128)(tf.reshape(x, (-1, x.shape[1]*x.shape[2], x.shape[3])))

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)

    if print_summary:
        model.summary()

    return model

# Example usage
# model = build_refined_model()
# model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
