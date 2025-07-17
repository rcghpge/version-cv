import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
from datasets import load_dataset

# -------------------------------
# Settings
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -------------------------------
# Load dataset
# -------------------------------
dataset = load_dataset("deepcopy/MathWriting-human")
train_dataset = dataset["train"]
total_samples = len(train_dataset)
print(f"Loaded {total_samples} samples from train split.")

# -------------------------------
# GPT-2 tokenizer and model
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = TFAutoModel.from_pretrained("gpt2")

# Freeze GPT-2 weights 
gpt2_model.trainable = False

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# Define embedding function
# -------------------------------
def get_embeddings(text_list):
    tokens = tokenizer(
        text_list,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=64
    )

    # Check token IDs
    max_id = tf.reduce_max(tokens.input_ids)
    vocab_size = tokenizer.vocab_size
    if max_id >= vocab_size:
        raise ValueError(f"Token ID {max_id.numpy()} out of range — vocab size is {vocab_size}.")

    outputs = gpt2_model(**tokens)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

# -------------------------------
# Preprocess function
# -------------------------------
def preprocess(example):
    img_array = example["image"]["array"] if isinstance(example["image"], dict) and "array" in example["image"] else example["image"]
    img = tf.image.resize(tf.image.convert_image_dtype(img_array, tf.float32), IMG_SIZE)

    text = example.get("latex", "")
    text_emb = get_embeddings([text])[0]  # Get single embedding

    return img, text_emb

# -------------------------------
# Create dataset
# -------------------------------
images, text_embeds, labels = [], [], []

for ex in train_dataset:
    img, text_emb = preprocess(ex)
    images.append(img.numpy())
    text_embeds.append(text_emb.numpy())
    labels.append(1)  # Dummy binary label

images = np.stack(images)
text_embeds = np.stack(text_embeds)
labels = np.array(labels)

# -------------------------------
# TensorFlow dataset
# -------------------------------
dataset_tf = tf.data.Dataset.from_tensor_slices(((images, text_embeds), labels))
dataset_tf = dataset_tf.shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Export function
# -------------------------------
def get_dataset():
    return dataset_tf