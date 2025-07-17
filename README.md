# version-cv

[![Dependabot Updates](https://github.com/rcghpge/version-cv/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/dependabot/dependabot-updates)
[![CodeQL Advanced](https://github.com/rcghpge/version-cv/actions/workflows/codeql.yml/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/codeql.yml)
[![Bandit](https://github.com/rcghpge/version-cv/actions/workflows/bandit.yml/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/bandit.yml)
[![pages-build-deployment](https://github.com/rcghpge/version-cv/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/rcghpge/version-cv/actions/workflows/pages/pages-build-deployment)


**version-cv** is a research-focused computer vision repository designed for mathematical image recognition, symbolic diagram parsing, and deep learning for mathematical reasoning. It builds on the foundational work from `version-tab`, a project focused on mathematical language modeling and symbolic abstraction in tabular formats. `version-cv` extends these capabilities into vision-based tasks using state-of-the-art deep learning methods.

This project explores:

* TODO: Vision project summary
* Symbolic reasoning through visual inputs
* Advanced image augmentation

It integrates seamlessly with `version-sdk`, a custom software development kit for managing data and model workflows.

**Research Paper Reference:** [Gervais et al., MathWriting: A Dataset for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2404.10690)
**Research Paper Reference:** [Saxton et al., Analyzing Mathematical Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)

---

## 📊 Key Datasets

* [MathWriting](https://huggingface.co/datasets/mathwriting)
* [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)

> *Note:* Initial benchmarks are available in `data/` and `docs/`, but not yet implemented in code due to time constraints.

---

## 📁 Project Structure

```
version-cv/
├── data/            # Dataset storage
├── docs/            # Design notes, benchmarking plans
├── models/          # Model definitions and training logic
├── modules/         # Helper modules for parsing, transforms
├── notebooks/       # Jupyter notebooks for exploration
├── pixi.toml        # Pixi environment configuration
├── pixi.lock        # Lockfile for reproducibility
└── README.md
```

---

## ⚡ Setup

This project uses [Pixi](https://prefix.dev/pixi) to manage environments and dependencies.

```bash
# Install Pixi if not already installed
curl -sSf https://pixi.sh/install.sh | bash

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

* Outputs metrics: accuracy, mAP, symbolic parsing quality
* Saves predictions to `predictions/`

3. View logs and summaries:

* `models/summary/`

---

## 🧪 Notebooks

Launch:

```bash
jupyter lab
```

Run Jupyter notebooks `cv_basemodel.ipynb` or `vision_transformer.ipynb` from the `notebooks/` folder.

These provide experiments and visualizations for ongoing development.

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

## 🛡️ Security & Auditing

This repository emphasizes security and reproducibility, especially for research workflows. While not intended for production deployment without further hardening, the following tools and workflows are integrated to improve code quality and security posture:

### Bandit

[Bandit](https://bandit.readthedocs.io/en/latest/) is used to identify common security issues in Python code.

To run manually:

```bash
bandit -r models/ modules/ notebooks/
```

### pip-audit

[pip-audit](https://pypi.org/project/pip-audit/) checks for known vulnerabilities in installed Python dependencies.

To run:

```bash
pixi run pip-audit
```

Or:

```bash
pip-audit
```

### CodeQL & Dependabot

* **CodeQL** scans for potential vulnerabilities in GitHub Actions workflows and source code.
* **Dependabot** automatically opens pull requests to update dependencies and alerts about security patches.

> ⚠️ This repository is intended for **academic and industry research**, prototyping, and open collaboration.
> It is research-based though not tailored for production-grade. It is academic-focused.
> Security reviews and reproducibility audits are welcomed through GitHub Issues and Pull Requests.

---

