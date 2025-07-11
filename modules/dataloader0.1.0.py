from tensorflow import data as tf_data

def create_dataset(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    train_dataset = tf_data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, train_gen.num_classes), dtype=tf.float32)
        )
    ).prefetch(tf_data.AUTOTUNE)

    val_dataset = tf_data.Dataset.from_generator(
        lambda: val_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, val_gen.num_classes), dtype=tf.float32)
        )
    ).prefetch(tf_data.AUTOTUNE)

    return train_dataset, val_dataset


# Example usage:
#train_ds, val_ds = create_dataset("path/to/data")
#model.fit(train_ds, validation_data=val_ds, epochs=5)

