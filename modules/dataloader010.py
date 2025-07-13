import os
import spacy
import sympy
import random
from glob import glob
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------
# Image Data Generator Loader
# ---------------------------
def create_image_generators(
    data_dir: Optional[str] = None,
    dataframe: Optional[pd.DataFrame] = None,
    x_col: str = "filename",
    y_col: str = "class",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    augment: bool = False,
    class_mode: str = "categorical"
):
    """
    Generates images either from a directory structure or from a dataframe (e.g., parquet or CSV).
    If a dataframe is provided, it will be used. Otherwise, it defaults to directory mode.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=10 if augment else 0,
        zoom_range=0.1 if augment else 0,
        horizontal_flip=augment,
    )

    if dataframe is not None:
        train_gen = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=data_dir,
            x_col=x_col,
            y_col=y_col,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset="training",
            shuffle=True,
            seed=42,
        )
        val_gen = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=data_dir,
            x_col=x_col,
            y_col=y_col,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset="validation",
            shuffle=False,
            seed=42,
        )
    else:
        if data_dir is None:
            raise ValueError("Either `dataframe` or `data_dir` must be provided.")

        train_gen = datagen.flow_from_directory(
            data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset="training",
            shuffle=True,
            seed=42,
        )
        val_gen = datagen.flow_from_directory(
            data_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset="validation",
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

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state)
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_val_size, random_state=random_state)

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
        tokens = tokenizer(example["latex_used"], truncation=True, padding="max_length", max_length=max_length)
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
# Text Pipeline (Vectorizer or Tokenized)
# ---------------------------
# Load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If not installed, ask user to run: python -m spacy download en_core_web_sm
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

def load_text_pipeline(
    base_path: str,
    vectorizer: TfidfVectorizer = None,
    use_sympy: bool = False,
    max_features: int = 10000,
    stop_words: str = "english",
    verbose: bool = True
):
    paths = glob.glob(os.path.join(base_path, "*.txt"))
    docs = []
    labels = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            docs.append(f.read())
        filename = os.path.basename(path)
        label = filename.split("__")[0]
        labels.append(label)

    if verbose:
        print(f"Loaded {len(docs)} docs from {base_path}")

    processed_docs = []

    for doc in docs:
        # Use SpaCy tokenizer
        if nlp:
            spacy_doc = nlp(doc)
            tokens = [token.text for token in spacy_doc if not token.is_space]

            # Optionally, further filter with stop words
            if stop_words == "english":
                tokens = [t for t in tokens if not t.lower() in spacy.lang.en.stop_words.STOP_WORDS]

        else:
            tokens = doc.split()

        # Optionally parse symbolic expressions
        if use_sympy:
            try:
                expr = sympy.sympify(doc)
                sym_tokens = [str(s) for s in expr.free_symbols]
                tokens.extend(sym_tokens)
            except sympy.SympifyError:
                # Skip if cannot parse
                pass

        processed_docs.append(" ".join(tokens))

    le = LabelEncoder()
    y = le.fit_transform(labels)

    if vectorizer is None:
        X = [doc.split() for doc in processed_docs]
        vec = None
        if verbose:
            print("Tokenized documents using SpaCy and optional SymPy instead of vectorizing.")
            print("Classes:", list(le.classes_))
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None)
        vectorizer.fit(processed_docs)
        X = vectorizer.transform(processed_docs)
        vec = vectorizer
        if verbose:
            print("Vectorized documents with custom preprocessing.")
            print("X shape:", X.shape)
            print("Classes:", list(le.classes_))

    return X, y, le, vec


# ---------------------------
# Text Pipeline (Hugging Face Dataset)
# ---------------------------
def load_text_pipeline_hf(
    base_path: str,
    use_sympy: bool = False,
    stop_words: str = "english",
    verbose: bool = True
):
    import importlib
    glob = importlib.import_module("glob")
    os_path = importlib.import_module("os").path

    paths = glob.glob(os_path.join(base_path, "*.txt"))
    docs = []
    labels = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()
        filename = os_path.basename(path)
        label = filename.split("__")[0]
        labels.append(label)

        # === Preprocess text ===
        if nlp:
            spacy_doc = nlp(raw_text)
            tokens = [token.text for token in spacy_doc if not token.is_space]

            if stop_words == "english":
                tokens = [t for t in tokens if not t.lower() in spacy.lang.en.stop_words.STOP_WORDS]
        else:
            tokens = raw_text.split()

        if use_sympy:
            try:
                expr = sympy.sympify(raw_text)
                sym_tokens = [str(s) for s in expr.free_symbols]
                tokens.extend(sym_tokens)
            except sympy.SympifyError:
                pass

        cleaned_text = " ".join(tokens)
        docs.append(cleaned_text)

    if verbose:
        print(f"Loaded {len(docs)} docs from {base_path}")
        print("Label examples:", set(labels))

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)

    data_dict = {
        "text": docs,
        "label": encoded_labels
    }

    hf_dataset = Dataset.from_dict(data_dict)

    if verbose:
        print(hf_dataset)

    return hf_dataset, le

