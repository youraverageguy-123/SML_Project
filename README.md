# SML Project — Sparse Features vs Contextual Embeddings

## Overview

This project compares classical machine learning models on multiple NLP feature representations for multi-class text classification.

The pipeline includes:

* Bag-of-Words
* TF-IDF
* GloVe embeddings
* DistilBERT embeddings
* PCA / Truncated SVD
* Multiple classical ML classifiers
* Visualization and automated LaTeX report generation

The project uses the AG News dataset from Hugging Face and generates plots, metrics, and reports automatically.

---

# Features

## Sparse Features

* Unigram Bag-of-Words
* Unigram + Bigram TF-IDF
* Stop-word removal
* Vocabulary pruning

## Dense Features

* Mean-pooled GloVe embeddings
* Precomputed DistilBERT embeddings

## Models

* Logistic Regression
* Linear SVM
* RBF SVM
* KNN
* KMeans

## Dimensionality Reduction

* Truncated SVD on TF-IDF
* PCA on embeddings

## Outputs

* EDA plots
* PCA visualizations
* t-SNE visualizations
* Confusion matrix
* Learning curves
* LaTeX results table
* Auto-generated PDF reports

---

# Project Structure

```text
SML_Project/
│
├── data/                     # Cached embeddings and data files
├── outputs/                  # Generated plots and tables
├── src/                      # Source modules
├── main.py                   # Main pipeline
├── requirements.txt          # Python dependencies
├── report_final.tex
├── submission_report.tex
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/youraverageguy-123/SML_Project.git
cd SML_Project
```

Alternatively:

* Download ZIP from GitHub
* Extract the folder
* Open terminal inside the extracted folder

---

# Python Environment Setup

It is recommended to use a virtual environment.

---

## Windows

### Create virtual environment

```powershell
python -m venv venv
```

### Activate virtual environment

```powershell
venv\Scripts\activate
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Linux

### Create virtual environment

```bash
python3 -m venv venv
```

### Activate virtual environment

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## macOS

### Create virtual environment

```bash
python3 -m venv venv
```

### Activate virtual environment

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

# Optional: LaTeX Installation

LaTeX is required only for automatic PDF report compilation.

If LaTeX is not installed:

* The ML pipeline will still run
* Plots and outputs will still generate
* Only PDF compilation will fail

---

## Fedora

```bash
sudo dnf install texlive
```

## Ubuntu / Debian

```bash
sudo apt install texlive-full
```

## Arch Linux

```bash
sudo pacman -S texlive-most
```

## Windows

Install either:

* MiKTeX
* TeX Live

## macOS

Install:

* MacTeX

---

# Running the Project

After installing dependencies and optional LaTeX support, the project can be run:

* directly from terminal
* from VS Code
* from PyCharm
* or from other Python IDEs

Note:

* Virtual environment setup and dependency installation should be done through terminal.
* `requirements.txt` must be installed before running the project.
* LaTeX installation is required only for automatic PDF report generation.

---

## Windows

### Terminal

```powershell
python main.py
```

### VS Code / IDEs

Open the project folder and run:

```python
main.py
```

---

## Linux / macOS

### Terminal

```bash
python3 main.py
```

### VS Code / IDEs

Open the project folder and run:

```python
main.py
```

---

# Generated Outputs

## outputs/

The following files are generated inside:

```text
outputs/
```

This includes:

* PNG plots
* Evaluation visualizations
* Learning curves
* Confusion matrices
* LaTeX tables
* Statistical summaries

---

## Root Directory Outputs

The following LaTeX source files are already included in the repository:

```text
report_final.tex
submission_report.tex
```

During execution, these files are updated and compiled into PDFs:

```text
report_final.pdf
submission_report.pdf
```

Additional LaTeX auxiliary files may also be generated:

```text
*.aux
*.log
*.out
```

These are produced automatically during PDF compilation if LaTeX is installed.

---

# Dataset

Dataset used:

* AG News Dataset (Hugging Face)

Loaded automatically using:

```python
load_dataset("ag_news")
```

Internet connection is required for first-time dataset download.

---

# Notes

* DistilBERT embeddings are precomputed and cached in `.npy` format.
* GloVe embeddings are downloaded automatically if not cached.
* Runtime may vary depending on system performance.
* Typical runtime is approximately 1–3 minutes on a standard CPU machine.

---

# Technologies Used

* Python
* NumPy
* pandas
* scikit-learn
* Hugging Face datasets
* transformers
* gensim
* matplotlib
* seaborn
* LaTeX

---
