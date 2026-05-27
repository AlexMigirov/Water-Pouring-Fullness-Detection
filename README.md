# 🎵 Water Level Detection from Audio — ML Project

Predicting when a cup is full based on audio recordings of the pouring process, using MFCC features and classic Machine Learning classifiers.

> **Pipeline status: ✅ COMPLETE — all 4 scripts have been run and all outputs generated.**

---

## 📁 Project Structure

```
Machine Learning Project/
│
├── הקלטות/                          # Raw audio recordings (60 × .mp4)
│   ├── 1.mp4  …  15.mp4             #   Class 1 – Thick_Glass       (~14.5s avg)
│   ├── 16.mp4 … 30.mp4              #   Class 2 – Tall_Thin_Glass    (~19.0s avg)
│   ├── 31.mp4 … 45.mp4              #   Class 3 – Ceramic_Cup        (~24.5s avg)
│   └── 46.mp4 … 60.mp4              #   Class 4 – Plastic_Cup        (~11.5s avg)
│
├── master_mfcc_dataset.xlsx          # Pre-extracted MFCC features (10,416 rows × 32 cols)
├── Spectrogram_Code.txt              # Original spectrogram + MFCC extraction reference
├── Project Instructions.pdf          # Full project specification
│
├── 01_explore_and_preprocess.py  ✅  # EDA + preprocessing → 6 figures + 2 CSVs
├── 02_spectrogram_analysis.py    ✅  # Mel-spectrogram + frequency-tracker → 4 figures
├── 03_model_training.py          ✅  # Group K-Fold CV (K=5), 3 models, 2 tasks → 7 figures + 4 CSVs
├── 04_evaluation_and_report.py   ✅  # Confusion matrices, dashboard, final table → 7 figures + 2 CSVs
│
├── outputs/
│   ├── figures/                      # 23 publication-quality PNG figures (300 DPI)
│   └── tables/                       # 8 CSV tables (UTF-8, Word-ready)
│
└── README.md                         # This file
```

---

## 🗂️ Dataset Overview

| Column | Description |
|---|---|
| `Recording_ID` | e.g. `הקלטה_01` — links each window to its source recording |
| `Group` | Group_1_to_15 / Group_16_to_30 / Group_31_to_45 / Group_46_to_60 |
| `Cup_Type` | Thick_Glass / Tall_Thin_Glass / Ceramic_Cup / Plastic_Cup |
| `Window_Num` | Sliding-window index (300 ms window, 100 ms hop = 10 ms step) |
| `Start_Time_s` / `End_Time_s` | Window boundaries in seconds |
| `Is_Full` | **Target A** — 1 if window falls in last 0.5 s of recording, else 0 |
| `MFCC_1_Raw` … `MFCC_13_Raw` | Raw MFCC coefficients at the mid-frame of the window |
| `MFCC_1_Mean` … `MFCC_4_Mean` | Per-window mean of the first 4 MFCCs |
| `MFCC_1_Std`  … `MFCC_4_Std`  | Per-window std of the first 4 MFCCs |
| `MFCC_1_Delta` … `MFCC_4_Delta` | Per-window mean delta (velocity) of first 4 MFCCs |

- **10,416 rows**, **25 numeric MFCC features**, **0 missing values**
- **Class imbalance (Task A)**: 97.04 % Not Full / 2.96 % Full
- **Group K-Fold**: 60 recordings → 5 folds of 12 recordings each

---

## 🧩 Cup Classes

| Class | Recording Range | Windows | Avg Duration |
|---|---|---|---|
| Thick_Glass | 1 – 15 | 2,239 | ~14.5 s |
| Tall_Thin_Glass | 16 – 30 | 2,815 | ~19.0 s |
| Ceramic_Cup | 31 – 45 | 3,657 | ~24.5 s |
| Plastic_Cup | 46 – 60 | 1,705 | ~11.5 s |

---

## ⚙️ Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl librosa av
```

| Package | Version tested | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading and manipulation |
| `numpy` | ≥ 1.24 | Numerical computation |
| `matplotlib` / `seaborn` | ≥ 3.7 | Plotting and figure generation |
| `scikit-learn` | ≥ 1.4 | ML models, Group K-Fold, metrics |
| `openpyxl` | ≥ 3.1 | Reading `.xlsx` files |
| `librosa` | ≥ 0.10 | Mel-spectrogram computation |
| `av` (PyAV) | ≥ 12.0 | Decoding `.mp4` audio streams |

> **Windows note:** Scripts set `sys.stdout.reconfigure(encoding='utf-8')` automatically to handle Hebrew filenames in the console.

---

## 🚀 How to Run

Run the scripts **in order** from the project root directory:

### Step 1 — Data Exploration & Preprocessing
```bash
python 01_explore_and_preprocess.py
```
| Output | Description |
|---|---|
| `outputs/figures/class_distribution.png` | Windows per cup class |
| `outputs/figures/is_full_distribution.png` | Is_Full balance (overall + per class) |
| `outputs/figures/windows_per_recording.png` | Recording length variability |
| `outputs/figures/mfcc_raw_distributions.png` | MFCC violin plots by cup type |
| `outputs/figures/mfcc_feature_boxplots.png` | Mean/Std/Delta boxplots by cup type |
| `outputs/figures/correlation_heatmap.png` | 25×25 feature correlation matrix |
| `outputs/tables/dataset_summary.csv` | Descriptive statistics (mean, std, quartiles) |
| `outputs/tables/preprocessed_dataset.csv` | Clean feature matrix used by steps 3 & 4 |

### Step 2 — Spectrogram & Frequency-Tracker Analysis
```bash
python 02_spectrogram_analysis.py
```
| Output | Description |
|---|---|
| `outputs/figures/Tracker_Thick_Glass.png` | Average mel-spectrogram + frequency tracker |
| `outputs/figures/Tracker_Tall_Thin_Glass.png` | (same for Class 2) |
| `outputs/figures/Tracker_Ceramic_Cup.png` | (same for Class 3) |
| `outputs/figures/Tracker_Plastic_Cup.png` | (same for Class 4) |

### Step 3 — Model Training & Cross-Validation
```bash
python 03_model_training.py
```
| Output | Description |
|---|---|
| `outputs/figures/roc_curves_fullness.png` | ROC curves (Task A, mean ± std over 5 folds) |
| `outputs/figures/pr_curves_fullness.png` | Precision-Recall curves (Task A) |
| `outputs/figures/roc_curves_cup_type.png` | One-vs-Rest ROC per cup class (Task B) |
| `outputs/figures/feature_importance.png` | Top-15 features for all 3 models (Task A) |
| `outputs/figures/metrics_comparison_fullness.png` | Bar chart comparison (Task A) |
| `outputs/figures/metrics_comparison_cup_type.png` | Bar chart comparison (Task B) |
| `outputs/figures/f1_heatmap_cup_type.png` | Per-class F1 heatmap (Task B) |
| `outputs/tables/fold_metrics_fullness.csv` | Raw per-fold scores (Task A) |
| `outputs/tables/cv_results_fullness.csv` | Mean ± std per model (Task A) |
| `outputs/tables/fold_metrics_cup_type.csv` | Raw per-fold scores (Task B) |
| `outputs/tables/cv_results_cup_type.csv` | Mean ± std per model (Task B) |

### Step 4 — Final Evaluation & Report Figures
```bash
python 04_evaluation_and_report.py
```
| Output | Description |
|---|---|
| `outputs/figures/confusion_matrix_fullness.png` | Confusion matrices — all 3 models (Task A) |
| `outputs/figures/confusion_matrix_cup_type.png` | Confusion matrices — all 3 models (Task B) |
| `outputs/figures/fold_variability_fullness.png` | Per-fold metric stability (Task A) |
| `outputs/figures/fold_variability_cup_type.png` | Per-fold metric stability (Task B) |
| `outputs/figures/per_recording_accuracy.png` | Per-recording accuracy heatmap (best model) |
| `outputs/figures/performance_dashboard.png` | 6-panel combined dashboard |
| `outputs/tables/final_summary_table.csv` | **Master Word-ready summary table** |
| `outputs/tables/per_recording_accuracy.csv` | Per-recording accuracy (Gradient Boosting) |

---

## 📊 Key Results

### Task A — Fullness Detection (`Is_Full`, binary)

| Model | Accuracy | F1-Score | AUC-ROC | Recall |
|---|---|---|---|---|
| **Gradient Boosting** 🥇 | 0.956 ± 0.010 | **0.417 ± 0.053** | **0.913 ± 0.030** | 0.528 |
| Random Forest | 0.946 ± 0.009 | 0.350 ± 0.049 | 0.910 ± 0.036 | 0.496 |
| Logistic Regression | 0.838 ± 0.030 | 0.234 ± 0.026 | 0.900 ± 0.035 | 0.828 |

### Task B — Cup Type Classification (4-class)

| Model | Accuracy | F1-Macro | AUC-ROC |
|---|---|---|---|
| **Gradient Boosting** 🥇 | **0.881 ± 0.024** | **0.872 ± 0.025** | **0.980 ± 0.006** |
| Random Forest | 0.854 ± 0.033 | 0.845 ± 0.033 | 0.971 ± 0.009 |
| Logistic Regression | 0.803 ± 0.026 | 0.796 ± 0.031 | 0.946 ± 0.012 |

> All results use **GroupKFold (K=5)** grouped by recording — no data leakage between windows of the same recording.

---

## 📋 Methodology

| Aspect | Choice | Rationale |
|---|---|---|
| Feature extraction | 25 MFCC features (13 raw + 4×mean + 4×std + 4×delta) | Captures spectral shape & temporal dynamics |
| Cross-validation | `GroupKFold(n_splits=5)` by recording number | Prevents leakage between time-windows of the same recording |
| Class imbalance | `class_weight='balanced'` (LR, RF) + `compute_sample_weight` (GB) | Avoids majority-class bias in 97/3 split |
| Primary metric | F1-Score + AUC-ROC | Appropriate for severely imbalanced binary classification |
| Confusion matrices | Out-of-Fold (OOF) accumulated predictions | Uses every sample once without train-set contamination |
| Figure format | 300 DPI PNG, white background | Ready for direct insertion into Word report |

---

## 👥 Project Team

Signal Processing & Machine Learning — Academic Project, 2025–2026
