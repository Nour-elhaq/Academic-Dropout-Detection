# Academic-Dropout-Detection — Student Dropout Prediction & Academic Success

## 🚀 Project Overview

This repository provides a **deployment-ready machine learning pipeline** to predict student dropout risk using **binary classification**:

* **Dropout = 1**
* **Non-Dropout (Graduate + Enrolled) = 0**

It is designed for an **early-warning use case after the 1st semester**, helping institutions identify at-risk students and intervene early.

### Key Features

* **Leakage Prevention**: Drops *2nd semester* features to avoid using future information.
* **Production Pipelines**: Uses `sklearn.pipeline.Pipeline` to bundle preprocessing + model consistently.
* **Imbalance Handling (No SMOTE)**:

  * Random Forest uses `class_weight="balanced"`
  * XGBoost uses `scale_pos_weight`
* **Threshold Tuning**: Selects an optimal threshold (e.g., to maximize **F1** or meet a target **recall**).

---

## 📊 Key Results (Test Set)

Optimized for dropout detection (minority class):

| Metric                  |   XGBoost   | Random Forest |
| :---------------------- | :---------: | :-----------: |
| **PR-AUC**              | **0.88** 🏆 |      0.86     |
| **Recall (Dropout)**    |   **0.81**  |      0.74     |
| **Precision (Dropout)** |   **0.78**  |      0.77     |
| **Accuracy**            |     0.86    |      0.85     |

> **Insight**: Using information available **after the 1st semester**, the XGBoost model identifies about **4 out of 5** students who will drop out.

---

## 🛠 Installation

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

Minimal requirements:

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib`
* `seaborn`
* `joblib`

---

## 🏃 Usage

### 1) Train models

This will:

* Load the dataset
* Apply leakage prevention (drop 2nd semester features if enabled)
* Tune Random Forest (CV search)
* Train XGBoost
* Select a decision threshold
* Save trained pipelines + metadata to `models/`

**Windows (PowerShell)**

```powershell
python src/train.py --data data/data.csv --drop_2nd_sem --threshold_objective f1
```

**macOS/Linux**

```bash
python src/train.py --data data/data.csv --drop_2nd_sem --threshold_objective f1
```

Useful options:

* `--data path/to/data.csv`
* `--drop_2nd_sem` (recommended)
* `--threshold_objective f1` or a recall objective (if supported by your script)

### 2) Evaluate saved models

Evaluates Random Forest and XGBoost using:

* default threshold (0.50)
* chosen threshold from training (saved in metadata)

```powershell
python src/evaluate.py
```

Outputs:

* classification reports
* PR-AUC / ROC-AUC
* plots saved to `plots/`

---

## 📂 Project Structure

```
├── data/
│   └── data.csv
├── models/
│   ├── rf_pipeline.joblib
│   ├── xgb_pipeline.joblib
│   └── metadata.json
├── plots/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
└── README.md
```

---

## 📈 Visualizations

After running evaluation, check `plots/` for:

* Confusion matrices (at selected thresholds)
* ROC curves
* Precision–Recall curves

---
[![DOI](https://zenodo.org/badge/1123243750.svg)](https://doi.org/10.5281/zenodo.18061762)

