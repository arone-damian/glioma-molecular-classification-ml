# Automated Molecular Classification of Brain Gliomas (LGG vs. GBM) using Machine Learning

[](https://www.python.org/)
[](https://scikit-learn.org/)
[](https://xgboost.readthedocs.io/)
[Dataset](https://www.google.com/search?q=https://archive.ics.uci.edu/dataset/759/glioma%2Bgrading%2Bclinical%2Band%2Bmutation%2Bfeatures)
[License: MIT](https://www.google.com/search?q=LICENSE)

## Overview

Gliomas are the most prevalent primary malignant tumors of the central nervous system (CNS). Accurate grading between **Lower-Grade Gliomas (LGG)** and **Glioblastoma Multiforme (GBM, WHO Grade 4)** is critical for prognosis and therapeutic decision-making. Modern neuro-oncology guidelines—including the **2021 WHO Classification of Tumors of the Central Nervous System** and **EANO guidelines**—emphasize integrating somatic mutation profiles (`IDH1`, `IDH2`, `TP53`, `ATRX`, `PTEN`, `EGFR`, `CIC`, among others) alongside clinical and histopathological features.

This repository presents an end-to-end **supervised Machine Learning and biomarker interpretability pipeline** that classifies patients into **LGG (`Grade = 0`)** or **GBM (`Grade = 1`)** using clinical demographics and 20 somatic gene mutations from The Cancer Genome Atlas (TCGA) cohort curated in the [UCI Machine Learning Repository (ID 759)](https://www.google.com/search?q=https://archive.ics.uci.edu/dataset/759/glioma%2Bgrading%2Bclinical%2Band%2Bmutation%2Bfeatures).

## Key Findings

* **Strong Discriminative Performance:** **XGBoost** achieved the highest global discriminative capacity with a **Test ROC-AUC of `0.924`** and a **10-Fold Stratified Cross-Validation ROC-AUC of `0.920`**. **SVM (RBF)** achieved the highest test set accuracy (**`86.3%`**) and GBM recall (**`88.6%`**), while **Logistic Regression** achieved the highest 10-Fold CV F1-score (**`0.851`**) and accuracy (**`86.8%`**).
* **Cross-Method Biomarker Consensus:** Across four distinct interpretability approaches (Random Forest Gini importance, XGBoost gain importance, standardized Logistic Regression coefficients, and Permutation Importance on test ROC-AUC), **`IDH1` mutation status** emerged as the single most influential predictor (`XGBoost Gain = 0.411`, `Mean ROC-AUC Decrease = 0.315`, `Logistic Odds Ratio = 0.139`), followed by **`Age_at_diagnosis`**, **`IDH2`**, **`CIC`**, **`PTEN`**, **`ATRX`**, and **`TP53`**.
* **Feature Group Ablation Study:** Somatic mutations alone (20 binary features) achieved a Test ROC-AUC of **`0.906`**, outperforming clinical demographics alone (5 features, ROC-AUC = **`0.829`**). Combining **clinical + mutational features (25 variables)** yielded the best overall performance (**ROC-AUC = `0.924`**), confirming that age at diagnosis provides complementary prognostic signal to the genomic profile.
* **Unsupervised Structural Validation:** 2D manifold projections (**PCA** and **t-SNE**) and unsupervised **K-Means clustering ($k = 2$)** recovered a meaningful intrinsic separation (`Adjusted Rand Index = 0.502`, `Normalized Mutual Information = 0.425`) aligned with true LGG/GBM labels prior to supervised training.

## Dataset Description

The dataset is fetched directly via the official `ucimlrepo` Python package ([UCI Dataset ID 759](https://www.google.com/search?q=https://archive.ics.uci.edu/dataset/759/glioma%2Bgrading%2Bclinical%2Band%2Bmutation%2Bfeatures)):

* **Cohort Size:** $N = 839$ patients ($487$ LGG [`58.0%`] and $352$ GBM [`42.0%`]).
* **Missing Values:** $0$ missing values across all 24 columns.
* **Target Variable:** `Grade` (`0` = LGG, `1` = GBM).
* **Predictor Variables (23 raw features $\rightarrow$ 25 features after One-Hot Encoding):**
  * **Clinical Features:** `Age_at_diagnosis` (continuous, years), `Gender` (binary), and `Race` (categorical: White, Black or African American, Asian, American Indian or Alaska Native $\rightarrow$ one-hot encoded with `drop_first=True`).
  * **Somatic Mutation Features (20 binary indicators: `0` = Not Mutated, `1` = Mutated):** `IDH1`, `TP53`, `ATRX`, `PTEN`, `EGFR`, `CIC`, `MUC16`, `PIK3CA`, `NF1`, `PIK3R1`, `FUBP1`, `RB1`, `NOTCH1`, `BCOR`, `CSMD3`, `SMARCA4`, `GRIN2A`, `IDH2`, `FAT4`, and `PDGFRA`.

## Methodology & Pipeline Architecture

1. **Exploratory Data Analysis & Preprocessing:** Verification of class balance and data integrity, followed by One-Hot Encoding of categorical demographic features (`Race`) and `StandardScaler` normalization embedded inside `scikit-learn` pipelines for scale-sensitive estimators (Logistic Regression and SVM).
2. **Stratified Data Partitioning:** 80% training ($n = 671$) and 20% hold-out test set ($n = 168$) preserving the exact LGG/GBM class ratio (`random_state = 42`), complemented by **10-Fold Stratified Cross-Validation** across the full cohort.
3. **Supervised Model Benchmarking:**
   * **Logistic Regression:** `Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000, random_state=42))])`
   * **SVM (RBF Kernel):** `Pipeline([('scaler', StandardScaler()), ('model', SVC(kernel='rbf', probability=True, random_state=42))])`
   * **Random Forest:** `RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)`
   * **XGBoost:** `XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)`
4. **Biomarker Interpretability & Feature Ablation:** Evaluation of tree-based feature importance, logistic regression standardized coefficients, 30-repeat permutation importance on the hold-out test set, and feature-group ablation experiments.

## Quantitative Results

### 1. Hold-Out Test Set Performance ($n = 168$)

| Model | Accuracy | Precision | Recall (GBM) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.857 | **0.811** | 0.857 | 0.833 | 0.918 |
| **SVM (RBF Kernel)** | **0.863** | 0.805 | **0.886** | **0.844** | 0.910 |
| **Random Forest** | 0.792 | 0.761 | 0.729 | 0.745 | 0.902 |
| **XGBoost** | 0.851 | 0.800 | 0.857 | 0.828 | **0.924** |

> **XGBoost Test Confusion Matrix ($n = 168$):** True Negatives (LGG) = `83`, False Positives = `15`, False Negatives = `10`, True Positives (GBM) = `60`.

### 2. 10-Fold Stratified Cross-Validation ($N = 839$)

| Model | Accuracy CV | Precision CV | Recall CV | F1-Score CV | ROC-AUC CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.868** | **0.812** | 0.898 | **0.851** | 0.918 |
| **SVM (RBF Kernel)** | 0.858 | 0.794 | **0.901** | 0.843 | 0.907 |
| **Random Forest** | 0.840 | 0.799 | 0.833 | 0.814 | 0.905 |
| **XGBoost** | 0.857 | 0.807 | 0.872 | 0.837 | **0.920** |

### 3. Feature Group Ablation Study (XGBoost on Test Set)

| Feature Subset | Number of Features | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clinical Features Only** | 5 | 0.732 | 0.727 | 0.571 | 0.640 | 0.829 |
| **Mutation Features Only** | 20 | **0.851** | 0.785 | **0.886** | **0.832** | 0.906 |
| **Clinical + Mutations (Full)** | 25 | **0.851** | **0.800** | 0.857 | 0.828 | **0.924** |

### 4. Top Biomarkers Across Interpretability Methods

| Rank | Random Forest (Gini) | XGBoost (Gain) | Logistic Regression ($\|\beta\|$ / OR) | Permutation Importance ($\Delta$ ROC-AUC) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `Age_at_diagnosis` (0.317) | `IDH1` (0.411) | `IDH1` ($\beta = -1.976$, OR = 0.139) | `IDH1` ($-0.315 \pm 0.032$) |
| **2** | `IDH1` (0.298) | `IDH2` (0.084) | `NOTCH1` ($\beta = -1.076$, OR = 0.341) | `Age_at_diagnosis` ($-0.029 \pm 0.012$) |
| **3** | `ATRX` (0.052) | `CIC` (0.074) | `TP53` ($\beta = +0.665$, OR = 1.944) | `PTEN` ($-0.006 \pm 0.005$) |
| **4** | `CIC` (0.047) | `PTEN` (0.037) | `IDH2` ($\beta = -0.604$, OR = 0.547) | `IDH2` ($-0.004 \pm 0.004$) |
| **5** | `PTEN` (0.043) | `NF1` (0.029) | `Age_at_diagnosis` ($\beta = +0.601$, OR = 1.823) | `GRIN2A` ($-0.004 \pm 0.003$) |

## Repository Structure

```text
glioma-molecular-classification-ml/
├── notebooks/
│   └── glioma_molecular_classification.ipynb   # Full ML, interpretability & clustering notebook
├── docs/
│   └── IEEE_Paper_Glioma_Classification_ES.pdf # 5-page IEEE academic report (Spanish)
├── figures/                                    # Generated ROC curves, feature importances & t-SNE plots
├── requirements.txt                            # Python dependencies for reproducibility
├── .gitignore                                  # Standard Python/Jupyter gitignore
└── README.md                                   # Project documentation
```

## Getting Started & Reproducibility

### 1. Clone the repository
```bash
git clone https://github.com/arone-damian/glioma-molecular-classification-ml.git
cd glioma-molecular-classification-ml
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn xgboost ucimlrepo jupyter
```

### 3. Run the notebook
Launch Jupyter Lab or Jupyter Notebook to execute `notebooks/glioma_molecular_classification.ipynb` (the dataset is automatically downloaded from the UCI repository via `ucimlrepo`):
```bash
jupyter notebook notebooks/glioma_molecular_classification.ipynb
```

## Author

**Frank Damian Arone Barrientos**  
Bioengineering Undergraduate — Universidad de Ingeniería y Tecnología (UTEC), Lima, Peru  
GitHub: [@arone-damian](https://www.google.com/search?q=https://github.com/arone-damian)
