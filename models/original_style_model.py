
import tensorflow as tf
import numpy as np
import random
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
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
# Build Original Style Model (Stops before pooling/flattening)
# ---------------------------
def build_original_model(input_shape=(224, 224, 3), print_summary=True):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model(inputs)
    # Stop here: no pooling or flattening
    # Users can extract x as feature maps directly

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(3, activation='softmax')(x)  # Adjust to your number of classes

    model = Model(inputs=inputs, outputs=output)

    if print_summary:
        model.summary()

    return model

# Example usage
# model = build_original_model()
# model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
