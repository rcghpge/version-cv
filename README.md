# version-cv

* **One Sentence Summary** 

  This is a GitHub repository for ML/DL in symbolic math classification using image-based, text-based, and classical learning models including EfficientNetB0, GPT2, Word2Vec, and XGBoost.

---

## Overview

* **Definition of the tasks / challenge**

  The project explores symbolic reasoning and multimodal classification using visual and textual inputs. A binary classification task was constructed using the Hugging Face version of the MathWriting dataset by shuffling LaTeX ID and image pairings to generate synthetic binary labels (0 or 1).

* **Approach**

  Four models were developed and compared:
  - `EfficientNetB0` for visual symbol classification
  - `GPT2` fine-tuned on LaTeX token sequences
  - `Word2Vec` for symbolic embeddings
  - `XGBoost` applied to TF-IDF vectors

Prototypes were also designed but not fully built out one of which is a `GPT2 prototype`

* **Summary of the performance achieved** 
  - EfficientNetB0: ~51% accuracy 
  - GPT2: ~47% accuracy 
  - Word2Vec: ~48% macro F1 score 
  - XGBoost: ~49% macro F1 score 
  - GPT2 prototype: ~76.1% accuracy, ROC AUC score .832 (83.2%)

---

## Summary of Workdone

### Data

* **Data:**
  * **Type:**
    - Image: 224×224 LaTeX-rendered symbols (image column - `.parquet` dataset)
    - Text: Tokenized LaTeX expressions
    - Output: Multi-class and binary labels
  * **Size:**
    - ~230,000 total samples (multi-class)
    - Binary samples from MathWriting 
    - 1.25 GB `.parquet` dataset
    - original dataset is in `.inkml` format
  * **Split:**
    - ~10% of the training data was utilized (~18,400 training samples, ~2,300 validation samples)
    - GPT2 prototype utilized the full 1.25GB dataset
    - Multi-class: 90% train / 10% val 
    - Binary: 90% train / 10% val
      
#### Preprocessing / Clean up

* Image resizing and normalization 
* GPT2 tokenization of LaTeX strings 
* Shuffling LaTeX ID–image pairs to generate 0/1 binary labels 
* Oversampling used for class balancing

#### Data Visualization

* Token length distributions, symbol histograms 
* t-SNE clustering of Word2Vec vectors 
* Confusion matrices and ROC curves for binary task
* EDA

### Problem Formulation

* **Input / Output:** 
  - Input: math symbol image or LaTeX sequence 
  - Output: categorical label or binary label

* **Models:** 
  - `EfficientNetB0` (image classifier) 
  - `GPT2` (LaTeX sequence transformer) 
  - `Word2Vec` + `XGBoost` (vectorized + classical)
  - `GPT2 prototype` (LaTeX sequence tranformer optimized)

* **Loss, Optimizer, Hyperparameters:** 
  - Loss: Categorical/Binary Crossentropy 
  - Optimizers: Adam, AdamW 
  - Batch Size: 32, Epochs: 30, Early stopping applied

---

### Training

* **Software & Hardware:** 
  - Ubuntu 24.04 LTS WSL2
  - Python 3.12.11 
  - TensorFlow 2.15, 2.18, and 2.19, HuggingFace Transformers, Gensim, XGBoost (newer TF versions for cloud dev) 
  - Trained on: Dell Precision Workstation 5510, Lambda Cloud compute - NVIDIA RTX A6000, NVIDIA Quadro RTX 6000, NVIDIA GH200 Grace Hopper Superchip, NVIDIA A100 Tensor Core GPU, and NVIDIA A10 Tensor Core GPU + CPU clusters 

* **Training time:**
  
  - Models took a couple of weeks ~1-3 to build and train/validate
  - Full training and tuning not implemented

* **Training curves:** 

  - Models converged within 10–20 epochs 
  - Validation monitoring and checkpointing used 
  - Binary classifier reached high AUC after 5–8 epochs

* **Challenges:** 

  - GPT2 token memory and batch size constraints 
  - Symbol label imbalance and binary class noise 
  - Feature drift during Word2Vec–XGBoost transfer
  - Scoping the work only to what is understood as data science today
  - Time constraints and restrictions on what to build and how to build
  - Complete Python built models not developed due to time constraints
  - Scope of requirements for academic level projects
  - `.parquet` dataset not fully utilized including original `.inkml` dataset 
  - Did not fully test builds

### Performance Comparison

* **Metrics:** Accuracy, Macro F1, ROC AUC (binary)

| Model                  | Accuracy | F1 Score (macro) |
|-----------------------|----------|------------------|
| EfficientNetB0        | 51%      | NaN              |
| GPT2                  | 47%      | 0.34             |
| Word2Vec              | 48%      | 0.32             |
| XGBoost               | 49%      | 0.48             |
| GPT2 Prototype        | 76.1%    | 0.76             |

---

## 📊 Visual Overview

Click images to enlarge

---

### EfficientNetB0 Results

<table align="center" style="margin:auto">
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/efficientneb0.png" alt="EfficientNetB0 Output" width="400" height="500"/></td>
  </tr>
</table>

---

### GPT-2 Results
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/gpt2.png" alt="GPT-2 Output" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/gpt2confusionmatrix.png" alt="GPT-2 Confusion Matrix" width="400"/></td>
  </tr>
</table>

---

### Word2Vec + LSTM Results
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/word2vec.png" alt="Word2Vec Output" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/word2vecconfusionmatrix.png" alt="Word2Vec Confusion Matrix" width="400"/></td>
  </tr>
</table>

---

### XGBoost Results
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/xgboost.png" alt="XGBoost Output" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/xgboostconfusionmatrix.png" alt="XGBoost Confusion Matrix" width="400"/></td>
  </tr>
</table>

---

### GPT-2 Prototype Results
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/prototypesroccurve.png" alt="Prototype Drafts" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/prototypes.png" alt="Prototype ROC Curve" width="400"/></td>
  </tr>
</table>

---

### Model Prototyping + Inference

<table align="center" style="margin:auto">
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/samplemodeloutputs.png" alt="Sample Model Outputs" width="300"/></td>
  </tr>
</table>
---

### Prototype Samples
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/deepmind_latex_expression.png" alt="LaTeX Expression" width="325" height="350"/></td>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/formula_with_ocr_boxes.png" alt="Formula with OCR Boxes" width="325" height="350"/></td>
  </tr>
  <tr>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/formula_with_boxes.png" alt="Formula with Bounding Boxes" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/rcghpge/version-cv/main/assets/combined_synthetic_ink.png" alt="Synthetic Ink Sample" width="400"/></td>
  </tr>
</table>

---

### Conclusions

* EfficientNetB0 model inference on symbolic image classification for math-based images challenges the model   
* GPT2 struggles to generate results for token sequence inputs 
* MathWriting binary label task showed promising generalization from shuffled samples 
* Word2Vec + XGBoost were generally lightweight and interpretable - XGBoost generalized well when comparing confusion matrix results
* GPT2 prototype outperformed all model builds (though was trained on full `.parquet` dataset)

---

### Future Work

* Integrate CROHME-style handwritten data 
* Explore layout-aware ViT + LSTM hybrids 
* Build MathOCR pipeline with MathPix-style parsing 
* Expand symbolic corpora beyond MathWriting
* Explainable AI (XAI)
* Secure ML/DL systems
* Responsible data science

---

## How to reproduce results

```bash
git clone https://github.com/rcghpge/version-cv.git
cd version-cv
pixi shell
pixi install 
pixi info
```

## Launch Jupyter Lab

```bash
jupyter lab
```

Notebook index:

 * `basemodel.ipynb`: basel model + EfficientNetB0 backbone model
 * `model2.ipynb`: GPT2 model
 * `model3.ipynb`: Word2Vec training + classical classifier
 * `model4.ipynb`: XGBoost integration
 * `mathwriting.ipynb`: prototyping builds
 * `comparemodels.ipynb`: side-by-side model analysis
 * `prototypes.ipynb`: GPT2 prototype (performed the best)

---

## Overview of files in repository

```bash
version-cv/
├── CITATION.cff              # Citation metadata
├── LICENSE                   # Open-source license
├── README.md                 # Main project documentation
├── assets/                   # Diagrams, plots, confusion matrices, LaTeX images 
├── cloud/                    # LambdaCloud configs and scripts
├── data/                     # MathWriting, DeepMind MATH, CSAI, and related corpora
├── docs/                     # Academic PDFs and source papers
├── install_pixi.sh           # Environment setup script
├── models/                   # Saved model files (.h5, .keras, .json)
├── notebooks/                # All core modeling and evaluation notebooks
├── pixi.lock                 # Pixi environment lock
├── pixi.toml                 # Pixi environment definition
└── sandbox/                  # Experimental code (dev branch)
```

---

## Software Setup

```bash
# Preferred
pixi shell
pixi install

# Virtual evironment setup
pixi info
```

Dependencies:

 * `tensorflow`==2.15, 2.18, 2.19 (newer versions for cloud dev)
 * `transformers`
 * `gensim`
 * `xgboost`
 * `scikit-learn`, `matplotlib`, `numpy`, `pandas`
 * `pixi`
 * `python` >=3.12.11

---

## Data

* Public data:
  * [MathWriting Dataset](https://huggingface.co/datasets/deepcopy/MathWriting-human)
  * DeepMind MATH (local generation)
* Binary labels generated by shuffling MathWriting LaTeX IDs

Run preprocessing via:

```bash
jupyter lab notebooks/<notebooks>.ipynb
```

---

## Training

All model training is notebook-based. Start with:

 * `basemodel.ipynb` for image classification
 * `model2.ipynb` for text modeling
 * `model3.ipynb` and `model4.ipynb` for classical approaches
 * `protopes.ipynb` for GPT-based designs from prototyping

## Performance Evaluation

Use:
```bash
jupyter lab notebooks/comparemodels.ipynb
```
To reproduce visualizations, metrics, and ROC plots.

---

## Citations

Further citations of works cited are referenced at `data/` and `docs/` 

* Tan & Le (2019). EfficientNet
* Vaswani et al. (2017). Attention Is All You Need
* Mikolov et al. (2013). Word2Vec
* Chen & Guestrin (2016). XGBoost
* Gervais et al. (2025). MathWriting Dataset
* Hendrycks et al. (2021). MATH Dataset
* Saxton et al. (2019). MATH Dataset
* Cocker, R. (2025). Version-Tab (Version 0.1.0) [Computer software]. https://github.com/rcghpge/version-tab

---
