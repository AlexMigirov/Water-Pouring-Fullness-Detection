import sys
# Ensure UTF-8 output on Windows (avoids cp1255 UnicodeEncodeError)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
01_explore_and_preprocess.py
============================
Step 1 of the Water-Level Detection ML Pipeline.

Purpose:
  - Load the pre-extracted MFCC dataset from master_mfcc_dataset.xlsx
  - Perform Exploratory Data Analysis (EDA)
  - Preprocess the data for downstream modelling
  - Save publication-quality figures and a cleaned CSV for later steps

Outputs (in outputs/figures/ and outputs/tables/):
  - class_distribution.png
  - is_full_distribution.png
  - mfcc_raw_distributions.png
  - mfcc_feature_boxplots.png
  - correlation_heatmap.png
  - windows_per_recording.png
  - dataset_summary.csv
  - preprocessed_dataset.csv   ← cleaned feature matrix used by step 03
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend (safe for saving)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
XLSX_PATH = "master_mfcc_dataset.xlsx"
XLSX_SHEET = 1                           # Sheet index 1 = "מאגר נתונים מאוחד"
OUT_DIR_FIGS   = Path("outputs/figures")
OUT_DIR_TABLES = Path("outputs/tables")

# Consistent colour palette for the 4 cup classes
CUP_PALETTE = {
    "Thick_Glass":      "#4E79A7",
    "Tall_Thin_Glass":  "#F28E2B",
    "Ceramic_Cup":      "#59A14F",
    "Plastic_Cup":      "#E15759",
}

# Publication-quality plot defaults
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F9F9F9",
    "axes.grid":         True,
    "grid.alpha":        0.4,
})

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure to the outputs/figures directory."""
    path = OUT_DIR_FIGS / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved figure -> {path}")


def save_table(df: pd.DataFrame, name: str) -> None:
    """Save a DataFrame as CSV to the outputs/tables directory."""
    path = OUT_DIR_TABLES / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Saved table  -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 0. Create output directories
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR_FIGS.mkdir(parents=True, exist_ok=True)
OUT_DIR_TABLES.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 1 – Load & Inspect Dataset")
print("="*65)

df_raw = pd.read_excel(XLSX_PATH, sheet_name=XLSX_SHEET, header=0)

# Rename columns to clean English names (columns may contain Hebrew in the xlsx)
ORIG_COLS = list(df_raw.columns)
NEW_COLS = [
    "Recording_ID", "Group", "Cup_Type",
    "Window_Num", "Start_Time_s", "End_Time_s", "Is_Full",
    "MFCC_1_Raw",  "MFCC_2_Raw",  "MFCC_3_Raw",  "MFCC_4_Raw",
    "MFCC_5_Raw",  "MFCC_6_Raw",  "MFCC_7_Raw",  "MFCC_8_Raw",
    "MFCC_9_Raw",  "MFCC_10_Raw", "MFCC_11_Raw", "MFCC_12_Raw", "MFCC_13_Raw",
    "MFCC_1_Mean", "MFCC_2_Mean", "MFCC_3_Mean", "MFCC_4_Mean",
    "MFCC_1_Std",  "MFCC_2_Std",  "MFCC_3_Std",  "MFCC_4_Std",
    "MFCC_1_Delta","MFCC_2_Delta","MFCC_3_Delta","MFCC_4_Delta",
]
df_raw.columns = NEW_COLS

print(f"\nDataset loaded:  {df_raw.shape[0]:,} rows  ×  {df_raw.shape[1]} columns")
print(f"From: {XLSX_PATH}")
print("\nColumn names:\n ", "\n  ".join(NEW_COLS))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract numeric recording number from Recording_ID
#    e.g.  "הקלטה_01"  →  1
# ─────────────────────────────────────────────────────────────────────────────
def extract_rec_num(rec_id: str) -> int:
    """Pull the integer from any string like 'הקלטה_01' or 'recording_01'."""
    m = re.search(r"(\d+)", str(rec_id))
    return int(m.group(1)) if m else -1

df_raw["Recording_Num"] = df_raw["Recording_ID"].apply(extract_rec_num)

# Map recording number → cup type (ground truth from instructions)
def recording_to_cup(n: int) -> str:
    if  1 <= n <= 15:  return "Thick_Glass"
    if 16 <= n <= 30:  return "Tall_Thin_Glass"
    if 31 <= n <= 45:  return "Ceramic_Cup"
    if 46 <= n <= 60:  return "Plastic_Cup"
    return "Unknown"

# Cross-check with the xlsx cup type
df_raw["Cup_Type_Check"] = df_raw["Recording_Num"].apply(recording_to_cup)
mismatches = (df_raw["Cup_Type"] != df_raw["Cup_Type_Check"]).sum()
print(f"\nCup-type cross-check mismatches: {mismatches}  (expected 0)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Basic dataset statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 2 – Descriptive Statistics")
print("="*65)

FEATURE_COLS = [c for c in NEW_COLS if c.startswith("MFCC")]
META_COLS    = ["Recording_ID", "Recording_Num", "Group", "Cup_Type",
                "Window_Num", "Start_Time_s", "End_Time_s", "Is_Full"]

print(f"\nFeature columns ({len(FEATURE_COLS)}):  {FEATURE_COLS}")
print(f"\nMissing values per column:\n{df_raw[FEATURE_COLS].isnull().sum()}")
print(f"\nIs_Full distribution:\n{df_raw['Is_Full'].value_counts()}")
print(f"\nCup_Type distribution:\n{df_raw['Cup_Type'].value_counts()}")

# Numeric summary saved as table
desc = df_raw[FEATURE_COLS + ["Is_Full"]].describe().round(4)
save_table(desc.reset_index().rename(columns={"index":"Statistic"}),
           "dataset_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. EDA – Class (cup type) distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 3 – EDA Visualisations")
print("="*65)

# ── 4a. Windows per cup class ──────────────────────────────────────────────
cup_counts = df_raw["Cup_Type"].value_counts().reindex(list(CUP_PALETTE.keys()))
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(cup_counts.index, cup_counts.values,
              color=[CUP_PALETTE[c] for c in cup_counts.index],
              edgecolor="white", linewidth=1.5, zorder=3)
for bar, val in zip(bars, cup_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Number of Windows per Cup Class", fontweight="bold")
ax.set_xlabel("Cup Type")
ax.set_ylabel("Number of Windows (rows)")
ax.set_ylim(0, cup_counts.max() * 1.12)
ax.set_xticklabels(cup_counts.index, rotation=10)
fig.tight_layout()
save_fig(fig, "class_distribution.png")

# ── 4b. Is_Full distribution (overall + per cup) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Overall
overall = df_raw["Is_Full"].value_counts()
axes[0].pie(
    overall.values,
    labels=["Not Full (0)", "Full (1)"],
    autopct="%1.2f%%",
    startangle=90,
    colors=["#4E79A7", "#E15759"],
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11},
)
axes[0].set_title("Overall Is_Full Distribution", fontweight="bold")

# Per cup type
per_cup = df_raw.groupby("Cup_Type")["Is_Full"].value_counts(normalize=True).unstack(fill_value=0) * 100
per_cup.plot(kind="bar", ax=axes[1],
             color=["#4E79A7", "#E15759"],
             edgecolor="white", linewidth=1.2)
axes[1].set_title("Is_Full Proportion per Cup Type (%)", fontweight="bold")
axes[1].set_xlabel("Cup Type")
axes[1].set_ylabel("Percentage (%)")
axes[1].set_xticklabels(per_cup.index, rotation=12)
axes[1].legend(["Not Full (0)", "Full (1)"])
fig.tight_layout()
save_fig(fig, "is_full_distribution.png")

# ── 4c. Windows per recording (to visualise variability) ──────────────────
wpd = df_raw.groupby(["Recording_Num", "Cup_Type"])["Window_Num"].max().reset_index()
wpd.columns = ["Recording_Num", "Cup_Type", "Num_Windows"]
wpd.sort_values("Recording_Num", inplace=True)

fig, ax = plt.subplots(figsize=(14, 5))
colors_rec = [CUP_PALETTE[ct] for ct in wpd["Cup_Type"]]
ax.bar(wpd["Recording_Num"], wpd["Num_Windows"],
       color=colors_rec, edgecolor="white", linewidth=0.8, zorder=3)
ax.set_title("Total Windows per Recording", fontweight="bold")
ax.set_xlabel("Recording Number")
ax.set_ylabel("Number of Windows")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Custom legend for cup classes
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in CUP_PALETTE.items()]
ax.legend(handles=legend_elements, title="Cup Type", loc="upper right")
fig.tight_layout()
save_fig(fig, "windows_per_recording.png")

# ── 4d. MFCC Raw distributions (violin plot) ──────────────────────────────
raw_mfcc_cols = [f"MFCC_{i}_Raw" for i in range(1, 14)]
df_melt = df_raw[["Cup_Type"] + raw_mfcc_cols].melt(
    id_vars="Cup_Type", var_name="Feature", value_name="Value"
)

fig, ax = plt.subplots(figsize=(16, 6))
sns.violinplot(
    data=df_melt, x="Feature", y="Value", hue="Cup_Type",
    palette=CUP_PALETTE, cut=0, linewidth=0.7, ax=ax,
    inner="quartile", density_norm="width",
)
ax.set_title("Distribution of MFCC Raw Coefficients by Cup Type", fontweight="bold")
ax.set_xlabel("MFCC Feature")
ax.set_ylabel("Coefficient Value")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Cup Type", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
save_fig(fig, "mfcc_raw_distributions.png")

# ── 4e. MFCC Mean/Std/Delta box-plots ─────────────────────────────────────
summary_features = (
    [f"MFCC_{i}_Mean" for i in range(1, 5)] +
    [f"MFCC_{i}_Std"  for i in range(1, 5)] +
    [f"MFCC_{i}_Delta" for i in range(1, 5)]
)
df_melt2 = df_raw[["Cup_Type"] + summary_features].melt(
    id_vars="Cup_Type", var_name="Feature", value_name="Value"
)
fig, ax = plt.subplots(figsize=(16, 6))
sns.boxplot(
    data=df_melt2, x="Feature", y="Value", hue="Cup_Type",
    palette=CUP_PALETTE, linewidth=0.8, fliersize=2, ax=ax,
)
ax.set_title("MFCC Summary Features (Mean / Std / Delta) by Cup Type", fontweight="bold")
ax.set_xlabel("Feature")
ax.set_ylabel("Value")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.legend(title="Cup Type", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
save_fig(fig, "mfcc_feature_boxplots.png")

# ── 4f. Correlation heat-map (feature vs feature) ─────────────────────────
corr = df_raw[FEATURE_COLS].corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.4, linecolor="white",
    square=True, ax=ax,
)
ax.set_title("Feature Correlation Matrix (MFCC Features)", fontweight="bold")
fig.tight_layout()
save_fig(fig, "correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 4 – Preprocessing")
print("="*65)

# ── 5a. Drop rows with any missing feature values ─────────────────────────
n_before = len(df_raw)
df_clean = df_raw.dropna(subset=FEATURE_COLS).copy()
n_after  = len(df_clean)
print(f"Dropped {n_before - n_after} rows with NaN features.  Remaining: {n_after:,}")

# ── 5b. Create a clean Group-ID for GroupKFold ────────────────────────────
#    Each recording = one group (60 groups, 5 folds → 12 recordings per fold)
df_clean["Group_KFold"] = df_clean["Recording_Num"].astype(int)
print(f"\nUnique GroupKFold groups (recordings): {df_clean['Group_KFold'].nunique()}")

# ── 5c. Feature matrix & target vector ────────────────────────────────────
X          = df_clean[FEATURE_COLS].values.astype(np.float64)
y_fullness = df_clean["Is_Full"].values.astype(int)   # binary target
groups     = df_clean["Group_KFold"].values             # GroupKFold splitter

print(f"\nFeature matrix X : {X.shape}")
print(f"Target y_fullness: {y_fullness.shape}  (Is_Full – binary)")
print(f"Groups           : {groups.shape}  ({np.unique(groups).size} unique recordings)")

# ── 5d. Save preprocessed dataset as CSV ─────────────────────────────────
df_out = df_clean[
    ["Recording_ID", "Recording_Num", "Group_KFold",
     "Cup_Type", "Window_Num",
     "Start_Time_s", "End_Time_s", "Is_Full"]
    + FEATURE_COLS
].copy()
save_table(df_out, "preprocessed_dataset.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary printout
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)
print(f"  Total windows (rows)   : {len(df_clean):,}")
print(f"  Feature columns        : {len(FEATURE_COLS)}")
print(f"  Unique recordings      : {df_clean['Recording_Num'].nunique()}")
print(f"  Is_Full = 1  (positive): {(y_fullness == 1).sum():,}  ({(y_fullness==1).mean()*100:.2f}%)")
print(f"  Is_Full = 0  (negative): {(y_fullness == 0).sum():,}  ({(y_fullness==0).mean()*100:.2f}%)")
print(f"\n  Cup type window counts:")
for cup in ["Thick_Glass","Tall_Thin_Glass","Ceramic_Cup","Plastic_Cup"]:
    n = (df_clean["Cup_Type"] == cup).sum()
    print(f"    {cup:<20}: {n:>5,} windows")

print("\n" + "="*65)
print("  Step 1 COMPLETE - outputs saved to outputs/")
print("  Next: run  python 02_spectrogram_analysis.py")
print("="*65 + "\n")
