#!/usr/bin/env python
# coding: utf-8

# # model 2

# In[1]:


# Run development environment checks for gpu compute
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("GPUs detected:", tf.config.list_physical_devices('GPU'))


# In[ ]:


# Run these dependency checks if pixi gets buggy
#!pip install transformers


# In[ ]:


#!pip install datasets


# In[ ]:


#!pip install tf-keras


# In[ ]:


#!pip install --upgrade pillow


# In[5]:


#!rm -rf ~/.cache/huggingface


# In[6]:


#!rm -rf ~/.keras


# In[7]:


#!rm -rf ~/.cache/huggingface/datasets


# In[11]:


import os
import random
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, GPT2Tokenizer
import datasets
from datasets import load_dataset

# GPT-2 Classifier
model = TFAutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model._name = "model_2"

# Load dataset
"""
The backend for HF datasets format datasets a certain way. For dataset configs set these configs as needed
"""
#dataset_dict = load_dataset("deepcopy/MathWriting-human", trust_remote_code=True)
#dataset_dict = dataset_dict.cast_column("image", datasets.Image(decode=False))
dataset_dict = load_dataset("deepcopy/MathWriting-human")

# Set this config for percentage of the data
USE_PERCENTAGE = 0.10
train_samples = int(USE_PERCENTAGE * len(dataset_dict["train"]))
val_samples = int(USE_PERCENTAGE * len(dataset_dict["val"]))

train_small = dataset_dict["train"].select(range(train_samples))
val_small = dataset_dict["val"].select(range(val_samples))

# Add binary_label
latex_pool = train_small["latex"]

def add_binary_label(example):
    if random.random() > 0.5:
        example["binary_label"] = 1
        example["latex_used"] = example["latex"]
    else:
        wrong_latex = random.choice(latex_pool)
        while wrong_latex == example["latex"]:
            wrong_latex = random.choice(latex_pool)
        example["binary_label"] = 0
        example["latex_used"] = wrong_latex
    return example

train_small = train_small.map(add_binary_label)
val_small = val_small.map(add_binary_label)

def encode_example(example):
    encoded = tokenizer(
        example["latex_used"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )
    return {
        "input_ids": encoded["input_ids"][0],
        "attention_mask": encoded["attention_mask"][0],
        "label": tf.cast(example["binary_label"], tf.float32)
    }

def hf_to_tf_dataset(hf_dataset, batch_size=8):
    def gen():
        for example in hf_dataset:
            encoded = encode_example(example)
            yield ({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }, encoded["label"])

    output_signature = (
        {
            "input_ids": tf.TensorSpec(shape=(128,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(shape=(128,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    return tf.data.Dataset.from_generator(gen, output_signature=output_signature)\
             .shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

tf_train = hf_to_tf_dataset(train_small)
tf_val = hf_to_tf_dataset(val_small)

# Compile GPT-2 model for classification
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metric_fn = tf.keras.metrics.BinaryAccuracy()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=loss_fn,
    metrics=[metric_fn],
    #run_eagerly=True # TODO: research
)

# Train
history = model.fit(
    tf_train,
    validation_data=tf_val,
    epochs=5,
    verbose=1
)


# In[12]:


# Model summary
model.summary()


# In[16]:


import matplotlib.pyplot as plt

plt.figure(figsize=(16, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy: GPT2')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss: GPT2')

plt.show()


# In[17]:


y_true = []
y_pred = []

for batch in tf_val:
    inputs, labels = batch
    logits = model.predict(inputs, verbose=0).logits
    predictions = tf.sigmoid(logits).numpy().flatten()
    predicted_labels = (predictions > 0.5).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# In[18]:


from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print("F1 Score:", round(f1, 4))


# In[20]:


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Print full report
print(classification_report(y_true, y_pred, digits=4))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[24]:


# Save builds
model.save_weights("model2_weights.h5")
model.save("models/model2_savedmodel", save_format="tf")
model.save_pretrained("models/model2.h5")
model.save_pretrained("models/model2.keras")
tokenizer.save_pretrained("models/model2.h5")
tokenizer.save_pretrained("models/model2.keras")

# Loading legacy model weights
# Example usage:
#model.load_weights("model2_weights.h5")


# GPT2 took a bit to run. The results show how rigorous math is as a field. Further research and work is needed.

# In[ ]:


# end of Model 2 build and test runs

