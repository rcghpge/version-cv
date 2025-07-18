# version-cv

[![Dependabot Updates](https://github.com/rcghpge/version-cv/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/dependabot/dependabot-updates)
[![CodeQL Advanced](https://github.com/rcghpge/version-cv/actions/workflows/codeql.yml/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/codeql.yml)
[![Bandit](https://github.com/rcghpge/version-cv/actions/workflows/bandit.yml/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/bandit.yml)


A research-driven deep learning repository for mathematical problem solving, image recognition, and symbolic diagram parsing for mathematical reasoning. It builds on research and development from `version-tab`, a project focused on mathematical problem solving, symbolic reasoning, math-based vectorization, and large language model (LLM) development in tabular formats. `version-cv` builds from this for vision-based tasks.

The project is developed from a software development kit deisgned for data science workflows and other domains.

**Research Publications/References:** 
- [Gervais et al., MathWriting: A Dataset for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2404.10690)
- [Saxton et al., Analyzing Mathematical Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)

---

## 📊 Key Datasets

* [MathWriting](https://huggingface.co/datasets/mathwriting)
* [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)

Initial benchmarks were considered but not implemented due to time constraint research and builds. They are provided in [`data/`](https://github.com/rcghpge/version-cv/tree/main/data) and [`docs/`](https://github.com/rcghpge/version-cv/tree/main/docs) directories.

---

## 📁 Project Structure

```
version-cv/
├── cloud
├── data
├── docs
├── models
├── notebooks
├── sandbox
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── install_pixi.sh
├── pixi.lock
└── pixi.toml
```

---

## ⚡ Setup

This project is built with [Pixi](https://pixi.sh/latest/) to manage environments and Python dependencies.

```bash
# Install Pixi if not already installed
curl -sSf https://pixi.sh/install.sh | bash

# Or run the install script
./install_pixi.sh

# Initialize Pixi (creates pixi.toml and pixi.lock)
pixi init

# Install dependencies
pixi install

# Enter Pixi environment
pixi shell

# Pixi Environment information
pixi info
```

---

## 🚀 Quick Start

### Option 1: Using Pixi shell

```bash
python models/basemodel.py
jupyter lab
```

### Option 2: One-liner

```bash
pixi run python models/basemodel.py
pixi run jupyter lab
```

---

## 📊 Running & Viewing Results

1. Place your images and data in the `data/` directory (e.g., `data/handwriting`, `data/formulas`).

2. Run training/inference:

```bash
python models/basemodel.py
```

3. View logs and summaries:

* `models/summary/`

---

## 🧪 Notebooks

Jupyter Lab:

```bash
jupyter lab
```

Run Jupyter notebooks `basemodel.ipynb` or `mathwriting.ipynb` from the `notebooks/` folder for exploratory workflows.

---

## 📃 Research & References

* Gervais et al., *MathWriting: A Dataset for Handwritten Mathematical Expression Recognition*
  [arXiv:2404.10690](https://arxiv.org/abs/2404.10690)
* Saxton et al., *Analyzing Mathematical Reasoning Abilities of Neural Models*
  [arXiv:1904.01557](https://arxiv.org/abs/1904.01557)
* OpenAI, *Improving Mathematical Reasoning with Process Supervision* (2022)
  [Blog Link](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/)
* Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*
  [arXiv:2103.03874](https://arxiv.org/abs/2103.03874)

Additional implementation notes are in [`docs/`](https://github.com/rcghpge/version-cv/tree/main/docs) and data usage info is in [`data/`](https://github.com/rcghpge/version-cv/tree/main/data).

---

## 🛡️ Security Note
`version-cv` is built with integrated security and Python dependency management tools Bandit and pip-audit.
Security and reproducibility improvements are important and welcome via PR's

### Bandit

[Bandit](https://bandit.readthedocs.io/en/latest/) is a static analysis tool that is utilized to identify common security issues in Python code.

To run manually:

```bash
bandit -r models/ notebooks/
```

### pip-audit

[pip-audit](https://pypi.org/project/pip-audit/)  is a tool for scanning Python dependencies and environments for packages for vulnerabilities.

To run:

```bash
pixi run pip-audit
```

Or:

```bash
pip-audit
```

---

