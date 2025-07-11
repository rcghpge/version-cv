from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input

# Pre-trained CNN base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze CNN for transfer learning initially

# Flatten feature maps
x = base_model.output
x = Flatten()(x)

# MLP head
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

# Combine
model = Model(inputs=base_model.input, outputs=output)

# Image Augmentation Pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1./255
)

train_generator = datagen.flow_from_directory(
    'path_to_images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Example of fine-tuning
base_model.trainable = True

# Compile with a lower learning rate
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
