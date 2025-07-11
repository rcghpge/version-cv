
import tensorflow as tf
import numpy as np
from build3_hybrid_stacked import build_hybrid_stacked_model

# Example dummy data generator
def generate_dummy_data(num_samples=10, img_shape=(224, 224, 3), tab_shape=(10, 128), num_classes=3):
    X_img = np.random.rand(num_samples, *img_shape).astype(np.float32)
    X_tab = np.random.rand(num_samples, *tab_shape).astype(np.float32)
    y = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes=num_classes)
    return X_img, X_tab, y

def main():
    print("Choose dataset option:")
    print("1: Math Dataset (tabular/NLP symbolic only)")
    print("2: MathWriting Dataset (images only)")
    print("3: Hybrid (Math Dataset + MathWriting)")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        print("Running model on Math Dataset (tabular/NLP)...")
        # For tabular/NLP-only example
        X_img = np.zeros((10, 224, 224, 3), dtype=np.float32)  # Dummy zeros since images are not used
        X_tab, _, y = generate_dummy_data(num_samples=10)[1:]  # Only use tabular + y

    elif choice == "2":
        print("Running model on MathWriting Dataset (images only)...")
        X_img, _, y = generate_dummy_data(num_samples=10)
        X_tab = np.zeros((10, 10, 128), dtype=np.float32)  # Dummy zeros since tabular is not used

    else:
        print("Running model on Hybrid Dataset...")
        X_img, X_tab, y = generate_dummy_data(num_samples=10)

    # Build model
    model = build_hybrid_stacked_model()

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train (dummy run)
    model.fit([X_img, X_tab], y, epochs=2, batch_size=2)

    # Evaluate (dummy run)
    loss, acc = model.evaluate([X_img, X_tab], y, verbose=0)
    print(f"Evaluation — Loss: {loss:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
