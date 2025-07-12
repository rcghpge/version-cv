from glob import glob
import pandas as pd
from lxml import etree
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pylatex import Document, Math
from typing import Optional, Tuple

# --------------------------------
# Image data loader
# --------------------------------
def create_image_generators(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2, augment=False):
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
        shuffle=False,
        seed=42,
    )
    return train_gen, val_gen

# --------------------------------
# Parquet loader with target split
# --------------------------------
def load_parquet(
    parquet_path: str,
    target_column: Optional[str] = None,
    feature_columns: Optional[list] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    verbose: bool = True
):
    # Expand ~ before using glob
    parquet_path = os.path.expanduser(parquet_path)

    if "*" in parquet_path or "?" in parquet_path or "[" in parquet_path:
        file_list = glob(parquet_path, recursive=True)
        if not file_list:
            raise FileNotFoundError(f"No parquet files matched the pattern: {parquet_path}")
        if verbose:
            print(f"Found {len(file_list)} parquet files. Loading individually...")

        df_list = []
        for f in file_list:
            print(f"Loading {f} ...")
            df = pd.read_parquet(f)
            df_list.append(df)

        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.read_parquet(parquet_path)

    if verbose:
        print(f"Final merged shape: {df.shape}")

    if not target_column:
        return df, None, None, None, None

    if feature_columns:
        X = df[feature_columns]
    else:
        X = df.drop(columns=[target_column])

    y = df[target_column]

    from sklearn.model_selection import train_test_split
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
    
# --------------------------------
# Inkml loader with namespace parsing
# --------------------------------
def parse_inkml(file_path):
    tree = etree.parse(str(file_path))
    ns = {'ns': 'http://www.w3.org/2003/InkML'}

    label = None
    normalized = None

    label_elements = tree.xpath('//ns:annotation[@type="label"]', namespaces=ns)
    if label_elements and label_elements[0].text:
        label = label_elements[0].text.strip()

    normalized_elements = tree.xpath('//ns:annotation[@type="normalizedLabel"]', namespaces=ns)
    if normalized_elements and normalized_elements[0].text:
        normalized = normalized_elements[0].text.strip()

    return {"label": label, "normalizedLabel": normalized}

# --------------------------------
# Txt (LaTeX code) loader
# --------------------------------
def load_latex_txt(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

# --------------------------------
# PyLaTeX helper for generating LaTeX documents
# --------------------------------
def create_latex_doc(latex_code, output_path="output.tex"):
    doc = Document()
    doc.append(Math(data=[latex_code]))
    doc.generate_pdf(output_path.replace(".tex", ""), clean_tex=False)

# --------------------------------
# Data Pipeline
# --------------------------------
def load_data(image_dir=None, parquet_path=None, target_column=None, feature_columns=None, 
              test_size=0.2, val_size=0.1, inkml_dir=None, txt_dir=None, 
              img_size=(224, 224), batch_size=32, val_split=0.2, augment=False, verbose=True):
    train_gen, val_gen, parquet_splits, inkml_latex, txt_latex = None, None, None, [], []

    if image_dir:
        train_gen, val_gen = create_image_generators(image_dir, img_size, batch_size, val_split, augment)

    if parquet_path:
        parquet_splits = load_parquet(parquet_path, target_column, feature_columns, test_size, val_size, verbose=verbose)

    if parquet_path and not target_column:
        raise ValueError("If 'parquet_path' is provided, you should also specify 'target_column'.")

    if inkml_dir:
        inkml_files = Path(inkml_dir).rglob("*.inkml")
        inkml_latex = [parse_inkml(f) for f in inkml_files]

    if txt_dir:
        txt_files = Path(txt_dir).glob("*.txt")
        txt_latex = [load_latex_txt(f) for f in txt_files]

    return {
        "train_gen": train_gen,
        "val_gen": val_gen,
        "parquet_splits": parquet_splits,
        "inkml_latex": inkml_latex,
        "txt_latex": txt_latex
    }


# Example usage
# data = load_data(
#     image_dir="./images",
#     parquet_path="./data/math_dataset.parquet",
#     target_column="solution",
#     inkml_dir="./inkml_files",
#     txt_dir="./latex_txts",
#     augment=True
# )
# print(data)