import pandas as pd
from lxml import etree
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pylatex import Document, Math

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
# Parquet loader
# --------------------------------
def load_parquet(parquet_path):
    df = pd.read_parquet(parquet_path)
    return df

# --------------------------------
# INKML loader using lxml
# --------------------------------
def parse_inkml(file_path):
    tree = etree.parse(str(file_path))
    latex_elements = tree.xpath('//annotation[@type="truth"]')
    if latex_elements:
        return latex_elements[0].text
    return None

# --------------------------------
# TXT (LaTeX code) loader
# --------------------------------
def load_latex_txt(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

# --------------------------------
# PyLaTeX helper for generating LaTeX documents (optional)
# --------------------------------
def create_latex_doc(latex_code, output_path="output.tex"):
    doc = Document()
    doc.append(Math(data=[latex_code]))
    doc.generate_pdf(output_path.replace(".tex", ""), clean_tex=False)

# --------------------------------
# Orchestrator
# --------------------------------
def load_all_data(image_dir=None, parquet_path=None, inkml_dir=None, txt_dir=None, img_size=(224, 224), batch_size=32, val_split=0.2, augment=False):
    train_gen, val_gen, df, inkml_latex, txt_latex = None, None, None, [], []

    if image_dir:
        train_gen, val_gen = create_image_generators(image_dir, img_size, batch_size, val_split, augment)

    if parquet_path:
        df = load_parquet(parquet_path)

    if inkml_dir:
        inkml_files = Path(inkml_dir).glob("*.inkml")
        inkml_latex = [parse_inkml(f) for f in inkml_files]

    if txt_dir:
        txt_files = Path(txt_dir).glob("*.txt")
        txt_latex = [load_latex_txt(f) for f in txt_files]

    return {
        "train_gen": train_gen,
        "val_gen": val_gen,
        "parquet_df": df,
        "inkml_latex": inkml_latex,
        "txt_latex": txt_latex
    }

# Example usage (uncomment and adjust paths to test)
# data = load_all_data(
#     image_dir="./images",
#     parquet_path="./data/data.parquet",
#     inkml_dir="./inkml_files",
#     txt_dir="./latex_txts",
#     augment=True
# )
# print(data)

