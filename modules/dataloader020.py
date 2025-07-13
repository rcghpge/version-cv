import os
import random
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from lxml import etree
from sklearn.model_selection import train_test_split

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from PIL import Image

# ---------------------------
# Image Data Generator Loader
# ---------------------------
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_image_generators(
    data_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    augment: bool = False,
    class_mode: str = "binary"
):
    """
    Create Keras image data generators from directory structure.
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
        class_mode=class_mode,
        subset='training',
        shuffle=True,
        seed=42,
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='validation',
        shuffle=False,
        seed=42,
    )
    return train_gen, val_gen


# ---------------------------
# Parquet Loader
# ---------------------------
def load_parquet(
    parquet_path: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[list[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    verbose: bool = True
):
    """
    Load parquet files and split into train/val/test.
    """
    parquet_path = os.path.expanduser(parquet_path)

    if "*" in parquet_path or "?" in parquet_path or "[" in parquet_path:
        file_list = glob(parquet_path, recursive=True)
        if not file_list:
            raise FileNotFoundError(f"No parquet files matched the pattern: {parquet_path}")
        if verbose:
            print(f"Found {len(file_list)} parquet files. Loading individually...")

        df_list = [pd.read_parquet(f) for f in file_list]
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_parquet(parquet_path)

    if verbose:
        print(f"Final merged shape: {df.shape}")

    if not target_column:
        return df, None, None, None, None

    X = df[feature_columns] if feature_columns else df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=random_state
    )
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state
    )

    if verbose:
        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    return X_train, X_val, y_train, y_val, X_test


# ---------------------------
# HuggingFace Datasets Image + LaTeX Pipeline
# ---------------------------
def load_datasets_pipeline(
    dataset_name: str = "deepcopy/MathWriting-Human",
    tokenizer_name: str = "gpt2",
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    num_proc: int = 12,
    max_length: int = 64,
    verbose: bool = True
):
    """
    Load HF dataset, resize images, create binary labels, and tokenize LaTeX.
    """
    ds = load_dataset(dataset_name)
    if verbose:
        print(ds)

    def resize_image(example, size=image_size):
        example["image"] = example["image"].resize(size)
        return example

    ds["train"] = ds["train"].map(resize_image, num_proc=num_proc)
    ds["val"] = ds["val"].map(resize_image, num_proc=num_proc)
    ds["test"] = ds["test"].map(resize_image, num_proc=num_proc)

    latex_pool = ds["train"]["latex"]

    def add_binary_label(example, latex_list):
        if random.random() > 0.5:
            example["label"] = 1
            example["latex_used"] = example["latex"]
        else:
            wrong_latex = random.choice(latex_list)
            while wrong_latex == example["latex"]:
                wrong_latex = random.choice(latex_list)
            example["label"] = 0
            example["latex_used"] = wrong_latex
        return example

    ds["train"] = ds["train"].map(lambda x: add_binary_label(x, latex_pool), num_proc=num_proc)
    ds["val"] = ds["val"].map(lambda x: add_binary_label(x, latex_pool), num_proc=num_proc)
    ds["test"] = ds["test"].map(lambda x: add_binary_label(x, latex_pool), num_proc=num_proc)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_latex(example):
        tokens = tokenizer(
            example["latex_used"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        example["latex_ids"] = tokens["input_ids"]
        return example

    ds["train"] = ds["train"].map(tokenize_latex, num_proc=num_proc)
    ds["val"] = ds["val"].map(tokenize_latex, num_proc=num_proc)
    ds["test"] = ds["test"].map(tokenize_latex, num_proc=num_proc)

    tf_train = ds["train"].to_tf_dataset(columns=["image", "latex_ids"], label_cols=["label"], shuffle=shuffle, batch_size=batch_size)
    tf_val = ds["val"].to_tf_dataset(columns=["image", "latex_ids"], label_cols=["label"], shuffle=False, batch_size=batch_size)
    tf_test = ds["test"].to_tf_dataset(columns=["image", "latex_ids"], label_cols=["label"], shuffle=False, batch_size=batch_size)

    return tf_train, tf_val, tf_test


# ---------------------------
# Text Q&A Loader
# ---------------------------
def load_txt_qa_pipeline(txt_dir: str, tokenizer_name: str = "gpt2", max_length: int = 256, verbose: bool = True):
    """
    Load Q&A text files, split question/answer, and tokenize.
    Assumes files contain text separated by '\nAnswer:\n'
    """
    txt_files = list(Path(txt_dir).glob("*.txt"))
    examples = []

    for file in txt_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
        if "\nAnswer:\n" in content:
            question, answer = content.split("\nAnswer:\n", 1)
            examples.append({"question": question.strip(), "answer": answer.strip()})
        else:
            print(f"Skipping file without expected delimiter: {file}")

    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_qa(example):
        question_tokens = tokenizer(example["question"], truncation=True, padding="max_length", max_length=max_length)
        answer_tokens = tokenizer(example["answer"], truncation=True, padding="max_length", max_length=max_length)
        example["question_ids"] = question_tokens["input_ids"]
        example["answer_ids"] = answer_tokens["input_ids"]
        return example

    dataset = dataset.map(tokenize_qa)
    return dataset


# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    # Example for HF dataset pipeline
    # tf_train, tf_val, tf_test = load_datasets_pipeline()

    # Example for txt Q&A pipeline
    # qa_dataset = load_txt_qa_pipeline("./qa_txts")

    # Example for parquet
    # X_train, X_val, y_train, y_val, X_test = load_parquet("./data/math_dataset.parquet", target_column="solution")
    pass
