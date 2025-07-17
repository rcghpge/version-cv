import os
import spacy
import sympy
import random
from glob import glob
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from PIL import Image
import pytesseract
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
            raise FileNotFoundError(f"No parquet files matched: {parquet_path}")
        if verbose:
            print(f"Found {len(file_list)} parquet files.")
        df_list = [pd.read_parquet(f) for f in file_list]
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_parquet(parquet_path)

    if verbose:
        print(f"Final shape: {df.shape}")

    if not target_column:
        return df, None, None, None, None

    X = df[feature_columns] if feature_columns else df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state)
    rel_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=rel_val_size, random_state=random_state)

    if verbose:
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, y_train, y_val, X_test

# ---------------------------
# OCR Block
# ---------------------------
def ocr_extract_text(image_path: str, vectorizer: Optional[TfidfVectorizer] = None, verbose: bool = True):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    if verbose:
        print(f"OCR Text Preview:\n{text[:300]}...")
    if vectorizer:
        text_vector = vectorizer.transform([text])
        return text, text_vector
    else:
        return text, None

# ---------------------------
# InkML Parser and Loader
# ---------------------------
def parse_inkml_file(file_path, use_time=True, normalize=True):
    import lxml.etree as ET
    tree = ET.parse(file_path)
    root = tree.getroot()

    label_elem = root.find(".//{http://www.w3.org/2003/InkML}annotation[@type='normalizedLabel']")
    norm_label = label_elem.text if label_elem is not None else None

    split_elem = root.find(".//{http://www.w3.org/2003/InkML}annotation[@type='splitTagOriginal']")
    split_tag = split_elem.text.lower() if split_elem is not None else "train"

    traces = []
    for trace in root.findall(".//{http://www.w3.org/2003/InkML}trace"):
        coords_strs = trace.text.strip().split(",")
        trace_coords = []
        for coord_str in coords_strs:
            points = coord_str.strip().split(" ")
            points = [p for p in points if p != ""]
            if use_time and len(points) >= 3:
                x, y, t = float(points[0]), float(points[1]), float(points[2])
                trace_coords.append([x, y, t])
            elif not use_time and len(points) >= 2:
                x, y = float(points[0]), float(points[1])
                trace_coords.append([x, y])
        if trace_coords:
            traces.append(np.array(trace_coords))

    if normalize and traces:
        all_points = np.vstack([t[:, :2] for t in traces if t.shape[0] > 0])
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        for i, t in enumerate(traces):
            t[:, :2] = (t[:, :2] - min_vals) / (max_vals - min_vals + 1e-8)
            traces[i] = t

    return norm_label, traces, split_tag

def pad_trace_sequences(trace_list, max_len=300):
    flat_coords = []
    for trace in trace_list:
        flat_trace = np.concatenate(trace, axis=0) if len(trace) > 0 else np.zeros((1, 2))
        flat_coords.append(flat_trace)
    padded = pad_sequences(flat_coords, maxlen=max_len, dtype="float32", padding="post", truncating="post")
    return padded

def load_inkml_pipeline(
    inkml_dir: str,
    tokenizer_name: str = "gpt2",
    max_length: int = 64,
    use_time: bool = True,
    normalize: bool = True,
    pad_traces: bool = True,
    max_trace_len: int = 300,
    batch_size: int = 32,
    verbose: bool = True
):
    from collections import defaultdict
    inkml_files = glob(os.path.join(inkml_dir, "*.inkml"))
    splits = defaultdict(list)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    for file_path in inkml_files:
        norm_label, traces, split_tag = parse_inkml_file(file_path, use_time=use_time, normalize=normalize)
        if norm_label is None or not traces:
            continue

        tokens = tokenizer(norm_label, truncation=True, padding="max_length", max_length=max_length)
        encoded_latex = tokens["input_ids"]

        sample = {
            "traces": [t.tolist() for t in traces],
            "latex_ids": encoded_latex,
            "normalized_label": norm_label,
            "file_path": file_path
        }
        splits[split_tag].append(sample)

    if verbose:
        for split_name in splits:
            print(f"{split_name}: {len(splits[split_name])} samples")

    dataset_dict = {split_name: Dataset.from_list(samples) for split_name, samples in splits.items()}
    dataset_dict = DatasetDict(dataset_dict)

    if pad_traces:
        def pad_and_prepare(example):
            padded_traces = pad_trace_sequences([example["traces"]], max_len=max_trace_len)[0]
            example["padded_traces"] = padded_traces.tolist()
            return example

        for split_name in dataset_dict:
            dataset_dict[split_name] = dataset_dict[split_name].map(pad_and_prepare)

    if verbose:
        print("Final dataset_dict keys:", list(dataset_dict.keys()))

    # Convert to TensorFlow datasets
    tf_datasets = {}
    for split_name in dataset_dict:
        tf_dataset = dataset_dict[split_name].to_tf_dataset(
            columns=["padded_traces", "latex_ids"],
            shuffle=(split_name == "train"),
            batch_size=batch_size,
        )
        tf_datasets[split_name] = tf_dataset

    # Check explicit keys only — no fallback
    if "train" not in tf_datasets:
        raise ValueError("❌ No 'train' split found in dataset! Please ensure your InkML files have correct splitTagOriginal annotations.")

    return dataset_dict, tf_datasets

# ---------------------------
# Hugging Face Dataset (Image + LaTeX)
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
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
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
    docs, labels = [], []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            docs.append(f.read())
        label = os.path.basename(path).split("__")[0]
        labels.append(label)

    if verbose:
        print(f"Loaded {len(docs)} docs from {base_path}")

    processed_docs = []

    for doc in docs:
        if nlp:
            spacy_doc = nlp(doc)
            tokens = [token.text for token in spacy_doc if not token.is_space]
            if stop_words == "english":
                tokens = [t for t in tokens if not t.lower() in spacy.lang.en.stop_words.STOP_WORDS]
        else:
            tokens = doc.split()

        if use_sympy:
            try:
                expr = sympy.sympify(doc)
                sym_tokens = [str(s) for s in expr.free_symbols]
                tokens.extend(sym_tokens)
            except sympy.SympifyError:
                pass

        processed_docs.append(" ".join(tokens))

    le = LabelEncoder()
    y = le.fit_transform(labels)

    if vectorizer is None:
        X = [doc.split() for doc in processed_docs]
        vec = None
        if verbose:
            print("Tokenized using SpaCy/SymPy (no vectorizer). Classes:", list(le.classes_))
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None)
        vectorizer.fit(processed_docs)
        X = vectorizer.transform(processed_docs)
        vec = vectorizer
        if verbose:
            print("Vectorized. X shape:", X.shape, "Classes:", list(le.classes_))

    return X, y, le, vec

