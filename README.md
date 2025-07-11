# version-cv

[![Dependabot Updates](https://github.com/rcghpge/version-cv/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/dependabot/dependabot-updates)
[![CodeQL Advanced](https://github.com/rcghpge/version-cv/actions/workflows/codeql.yml/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/codeql.yml)
[![Bandit](https://github.com/rcghpge/version-cv/actions/workflows/bandit.yml/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/bandit.yml)
[![pages-build-deployment](https://github.com/rcghpge/version-cv/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/pages/pages-build-deployment)

A research-driven computer vision repository focused on mathematical image recognition, symbolic diagram parsing, handwriting analysis, and deep learning for mathematical reasoning.

This project extends `version-tab` into vision-based tasks, exploring computer vision transformers (ViTs), T5-based vision-text pipelines, robust image augmentation, and symbolic reasoning. It integrates with `version-sdk`, a custom software development kit for data workflows.

**Research Paper Reference:** [Analysing Mathematical Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)

**Key Datasets:**
- [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset) 
- [MathWriting](https://huggingface.co/datasets/mathwriting) 

Initial benchmarks were considered but not implemented due to time constrained builds. They are located under `data/` and `docs/` directories.

---

## 📁 Project Structure

```
version-tab-cv/
├── data/            
├── docs/            
├── models/           
├── modules/        
├── notebooks/       
├── pixi.toml
├── pixi.lock
└── README.md
```

---

## ⚡ Setup

```bash
# Install Pixi if not already installed
curl -sSf https://pixi.sh/install.sh | bash

# Initialize Pixi (creates pixi.toml and pixi.lock)
pixi init

# Install dependencies
pixi install

# Enter Pixi environment
pixi shell
```

---

## 🚀Quick Start

### Option 1: In Pixi shell

```bash
python models/basemodel.py
jupyter lab
```

### Option 2: One-liner without shell

```bash
pixi run python models/basemodel.py
pixi run jupyter lab
```

---

## 📊Running & Viewing Results

1️⃣ Place your image data in `data/` folders (e.g., `data/handwriting`, `data/formulas`).

2️⃣ Run:

```bash
python models/basemodel.py
```

- Performs training or inference on image datasets
- Outputs accuracy, mAP, symbolic parsing metrics
- Saves predictions to `predictions/`

3️⃣ Check model summaries and logs in `models/summary/`.

---

## 🧪 Notebooks

Launch Jupyter Lab:

```bash
jupyter lab
```

Open `cv_basemodel.ipynb` or `vision_transformer.ipynb` inside `notebooks/` for experiments and visual analysis.

---

## 📚Research & References

- [DeepMind Math Dataset](https://github.com/google-deepmind/mathematics_dataset)
- [MathWriting Dataset](https://huggingface.co/datasets/mathwriting)

---

## 🛡️ Security Note

This repository is intended for academic research, experimentation, and open source collaboration.  
It is **not intended for production use** without further rigorous evaluation and security review.

Security-focused contributions, reproducibility enhancements, and audit feedback are highly encouraged via PRs or issues.

---


