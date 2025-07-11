#!/usr/bin/env python
# coding: utf-8

"""
dataloader.py

A robust, unified data loading module for:
- Computer vision (image datasets)
- NLP (text datasets)
- Tabular data (CSV, Parquet)
- LLM engineering pipelines (tokenization, embeddings, vectorization)

Author: Your Name
"""

import os
import pandas as pd
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import AutoTokenizer, TFAutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

# === Vision: Image data ===
def create_image_generators(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2, augment=False):
    """
    Creates image data generators for training and validation.

    Args:
        data_dir (str): Directory with subfolders per class.
        img_size (tuple): Image resize shape.
        batch_size (int): Batch size.
        val_split (float): Validation split.
        augment (bool): Enable data augmentation.

    Returns:
        train_gen, val_gen (ImageDataGenerator flow objects)
    """
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=val_split,
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42,
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=True,
        seed=42,
    )
    return train_gen, val_gen

# === NLP: Text data ===
def load_text_data(file_path, tokenizer_name="t5-small", max_len=128, vectorizer=None):
    """
    Loads text data and returns raw texts, tokenized tensors, and optional vector embeddings.

    Args:
        file_path (str): Path to TXT or CSV file.
        tokenizer_name (str): HF tokenizer model name.
        max_len (int): Max token length.
        vectorizer: Optional sklearn-style vectorizer (ESA/SA).

    Returns:
        texts (list), tokens (dict), vectors (sparse matrix or None)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load text
    if file_path.endswith(".txt"):
        with open(file_path, encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        texts = df["text"].tolist() if "text" in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError("Unsupported text file format. Use .txt or .csv")

    # Tokenize
    tokens = tokenizer(
        texts,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )

    # Vector embeddings
    if vectorizer is not None:
        vectors = vectorizer.transform(texts)
    else:
        vectors = None

    return texts, tokens, vectors

# === Tabular data ===
def load_tabular_data(file_path, vectorizer=None):
    """
    Loads tabular data from CSV or Parquet and optionally vectorizes text columns.

    Args:
        file_path (str): Path to CSV or Parquet file.
        vectorizer: Optional sklearn-style vectorizer.

    Returns:
        df (DataFrame), vectors (optional)
    """
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported tabular file format. Use .csv or .parquet")

    # Example: vectorize text column
    if vectorizer is not None and "text" in df.columns:
        vectors = vectorizer.transform(df["text"].tolist())
    else:
        vectors = None

    return df, vectors

# === InkML data (markup) ===
from lxml import etree

def parse_inkml_file(path):
    tree = etree.parse(path)
    root = tree.getroot()
    traces = []
    for trace in root.findall('{http://www.w3.org/2003/InkML}trace'):
        trace_data = trace.text.strip().split(',')
        traces.append([float(t.strip()) for t in trace_data if t.strip()])
    return traces

def load_inkml_data(folder_path):
    """
    Loads .inkml files from a folder and parses traces.

    Args:
        folder_path (str): Path to folder containing inkml files.

    Returns:
        List of parsed traces.
    """
    paths = glob.glob(os.path.join(folder_path, "*.inkml"))
    parsed = [parse_inkml_file(p) for p in paths]
    return parsed

# === Master universal loader ===
def load_data(
    data_path,
    data_type,
    batch_size=32,
    img_size=(224, 224),
    tokenizer_name="t5-small",
    max_text_len=128,
    vectorizer=None,
    augment=False
):
    """
    Master universal data loader that delegates to specialized loaders.

    Args:
        data_path (str): File or folder path.
        data_type (str): One of ['image', 'text', 'tabular', 'inkml'].
        batch_size (int): Batch size for images.
        img_size (tuple): Target image size.
        tokenizer_name (str): HF tokenizer model name.
        max_text_len (int): Max text length.
        vectorizer: Optional vectorizer (ESA/SA).
        augment (bool): Image augmentation.

    Returns:
        Data generator, or tuple of processed data.
    """
    if data_type == "image":
        return create_image_generators(data_path, img_size, batch_size, augment=augment)
    elif data_type == "text":
        return load_text_data(data_path, tokenizer_name, max_text_len, vectorizer)
    elif data_type == "tabular":
        return load_tabular_data(data_path, vectorizer)
    elif data_type == "inkml":
        return load_inkml_data(data_path)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

# === Optional: ESA/SA vectorizer examples ===
# sa_vec = joblib.load("../builds/sa_vectorizer.joblib")
# esa_vec = joblib.load("../builds/esa_vectorizer.joblib")

# Example usage
# texts, tokens, vectors = load_text_data("data/descriptions.txt", vectorizer=sa_vec)

# df, tab_vectors = load_tabular_data("data/data.parquet", vectorizer=esa_vec)
