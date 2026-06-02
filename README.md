Here is the seamlessly combined and formatted README.md file. It merges your existing project structure, results, and team information with the new, comprehensive summary of the architecture, DSP pipeline, and custom algorithms we built.
You can copy this entire block and replace your current README.md on GitHub.
Markdown
# 🎵 Acoustic Liquid Level Detection & Cup Classification — ML Project

Predicting when a cup is full based on audio recordings of the pouring process, using custom Digital Signal Processing (DSP), MFCC features, and classic Machine Learning classifiers.

> **Pipeline status: ✅ COMPLETE — all scripts have been run and all outputs generated.**

---

## 📌 Project Overview
The goal of this project is to analyze the sound of water pouring into a cup (e.g., from a water bar) and build a dataset capable of training a Machine Learning model to:
1. **Classify the type of cup** being filled (Glass, Ceramic, Plastic, etc.).
2. **Detect the "Full" state** (the last 0.5 seconds of the pour) regardless of the cup's material.

To achieve this, we engineered a custom **Dynamic Pathfinding Algorithm** that traces the specific resonant "pitch hook" of the rising liquid, bypassing ambient noise and splashing artifacts, and creating a highly structured dataset for training.

---

## 🛠️ The Data Pipeline & Architecture

### 1. Audio Preprocessing & Noise Reduction
* **PyAV Extraction**: Audio is extracted directly from `.mp4` containers and converted to mono arrays.
* **Mel-Spectrogram Generation**: Processed using `librosa` with an FFT window of 2048.
* **Spectral Median Filtering**: A critical step applied to subtract static background noise (e.g., the constant 600Hz hum of a water machine's motor) while preserving the dynamic, rising frequencies of the pour.

### 2. The "Pathfinder" Algorithm (Dynamic Pitch Tracking)
Standard polynomial curve-fitting failed due to extreme high-frequency splash noises and low-frequency rumbles. To solve this, we engineered a custom step-by-step pathfinding tracker:
* **Initial Anchoring**: The algorithm searches for the cup's base resonance frequency strictly within a 400Hz - 800Hz pocket.
* **Bounded Climbing**: The pathfinder analyzes 0.1-second steps, enforcing a strict physics-based rule: *The pitch can only rise (max 60Hz per step) or stay flat, and cannot exceed an 1800Hz ceiling.*
* **Noise Immunity**: If a loud, out-of-bounds splash occurs, the algorithm ignores it, mathematically maintaining the true trajectory of the rising water level.

### 3. Feature Extraction (MFCC + Pathfinder Data)
To prepare the data for Machine Learning classification, we extract highly granular acoustic features:
* **Windowing Strategy**: Audio is analyzed in **300ms windows** with a **100ms hop size** (66.6% overlap), ensuring high-resolution temporal tracking without inflating the dataset size.
* **Integrated Pathfinder Features**: The dataset includes the exact frequency tracked by our custom algorithm (`Pathfinder_Hz`) and the slope of its climb (`Hz_Slope`). 

---

## 📁 Project Structure

```text
Machine Learning Project/
│
├── הקלטות/                          # Raw audio recordings (60 × .mp4)
│   ├── 1.mp4  …  15.mp4             #   Class 1 – Thick_Glass        (~14.5s avg)
│   ├── 16.mp4 … 30.mp4              #   Class 2 – Tall_Thin_Glass    (~19.0s avg)
│   ├── 31.mp4 … 45.mp4              #   Class 3 – Ceramic_Cup        (~24.5s avg)
│   └── 46.mp4 … 60.mp4              #   Class 4 – Plastic_Cup        (~11.5s avg)
│
├── master_mfcc_dataset.xlsx          # Pre-extracted features (10,416 rows × 32 cols)
├── Spectrogram_Code.txt              # Original spectrogram + MFCC extraction reference
├── Project Instructions.pdf          # Full project specification
│
├── Codes/
│   ├── 01_explore_and_preprocess.py  ✅  # EDA + preprocessing → 6 figures + 2 CSVs
│   ├── 02_spectrogram_analysis.py    ✅  # Mel-spectrogram + frequency-tracker → 4 figures
│   ├── 03_model_training.py          ✅  # Group K-Fold CV (K=5), 3 models → 4 figures + 2 CSVs
│   └── 04_evaluation_and_report.py   ✅  # Confusion matrices, dashboard, final table → 3 figures + 1 CSV
│
├── figures/                          # 17 publication-quality PNG figures (300 DPI)
├── tables/                           # 4 CSV tables (UTF-8, Word-ready)
│
└── README.md                         # This file
🗂️ Dataset Overview
Column	Description
Recording_ID	e.g. הקלטה_01 — links each window to its source recording
Group	Group_1_to_15 / Group_16_to_30 / Group_31_to_45 / Group_46_to_60
Cup_Type	Thick_Glass / Tall_Thin_Glass / Ceramic_Cup / Plastic_Cup
Window_Num	Sliding-window index (300 ms window, 100 ms hop = 10 ms step)
Start_Time_s / End_Time_s	Window boundaries in seconds
Pathfinder_Hz	Custom Feature — The precise frequency tracked by the algorithm
Hz_Slope	Custom Feature — The climb velocity of the tracked frequency
Is_Full	Target A — 1 if window falls in last 0.5 s of recording, else 0
MFCC_1_Raw … MFCC_13_Raw	Raw MFCC coefficients at the mid-frame of the window
MFCC_1_Mean … MFCC_4_Mean	Per-window mean of the first 4 MFCCs
MFCC_1_Std … MFCC_4_Std	Per-window std of the first 4 MFCCs
MFCC_1_Delta … MFCC_4_Delta	Per-window mean delta (velocity) of first 4 MFCCs
•	10,416 rows, 27 numeric features, 0 missing values
•	Class imbalance (Task A): 97.04 % Not Full / 2.96 % Full
•	Group K-Fold: 60 recordings → 5 folds of 12 recordings each
🧩 Cup Classes
Class	Recording Range	Windows	Avg Duration
Thick_Glass	1 – 15	2,239	~14.5 s
Tall_Thin_Glass	16 – 30	2,815	~19.0 s
Ceramic_Cup	31 – 45	3,657	~24.5 s
Plastic_Cup	46 – 60	1,705	~11.5 s
⚙️ Dependencies
Bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl librosa av
Package	Purpose
pandas / numpy	Data loading and numerical computation
matplotlib / seaborn	Plotting and figure generation
scikit-learn	ML models, Group K-Fold, metrics
openpyxl	Reading .xlsx files
librosa	Mel-spectrogram computation & DSP
av (PyAV)	Decoding .mp4 audio streams
Windows note: Scripts set sys.stdout.reconfigure(encoding='utf-8') automatically to handle Hebrew filenames in the console.
🚀 How to Run
Run the scripts in order from the project root directory:
Step 1 — Data Exploration & Preprocessing
Bash
python Codes/01_explore_and_preprocess.py
Step 2 — Spectrogram & Frequency-Tracker Analysis
Bash
python Codes/02_spectrogram_analysis.py
Generates High-resolution .png charts plotting the Averaged Mel-Spectrogram for each cup class, overlaid with the Pathfinder's bright green tracking line and a red bounded box indicating the Is_Full region.
Step 3 — Model Training & Cross-Validation
Bash
python Codes/03_model_training.py
Step 4 — Final Evaluation & Report Figures
Bash
python Codes/04_evaluation_and_report.py
📊 Key Results
Fullness Detection (Is_Full, binary)
Model	Accuracy	F1-Score	AUC-ROC	Recall
Gradient Boosting 🥇	0.956 ± 0.010	0.417 ± 0.053	0.913 ± 0.030	0.528
Random Forest	0.946 ± 0.009	0.350 ± 0.049	0.910 ± 0.036	0.496
Logistic Regression	0.838 ± 0.030	0.234 ± 0.026	0.900 ± 0.035	0.828
All results use GroupKFold (K=5) grouped by recording — preventing data leakage between time-windows of the same recording. Class imbalance was handled using class_weight='balanced' and sample weighting techniques.
👥 Project Team
Alex Migirov, Dor Ohana, Alex Dryunkin, Yarin Navon
SCE, Shamoon College of Engineering

