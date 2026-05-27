import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
03_model_training.py
====================
Step 3 of the Water-Level Detection ML Pipeline.

Classification task:
  Binary: Is_Full  (is the cup full right now?)

Models:
  1. Gradient Boosting  (GradientBoostingClassifier)
  2. Logistic Regression
  3. Random Forest

Cross-validation:
  GroupKFold(n_splits=5), grouped by Recording_Num
  => 12 recordings per fold, zero leakage between windows of the same recording

Class imbalance (97/3 split):
  - Logistic Regression / Random Forest: class_weight='balanced'
  - Gradient Boosting: sample_weight computed per fold via compute_sample_weight

Outputs (outputs/figures/ and outputs/tables/):
    roc_curves_fullness.png
    pr_curves_fullness.png
    feature_importance.png
    metrics_comparison_fullness.png
    cv_results_fullness.csv
    fold_metrics_fullness.csv
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, auc,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_PATH      = PROJECT_ROOT / "tables/preprocessed_dataset.csv"
OUT_FIGS       = PROJECT_ROOT / "figures"
OUT_TABLES     = PROJECT_ROOT / "tables"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 42

CUP_PALETTE = {
    "Thick_Glass":     "#4E79A7",
    "Tall_Thin_Glass": "#F28E2B",
    "Ceramic_Cup":     "#59A14F",
    "Plastic_Cup":     "#E15759",
}
MODEL_COLORS = {
    "Gradient Boosting":   "#E63946",
    "Logistic Regression": "#457B9D",
    "Random Forest":       "#2A9D8F",
}

plt.rcParams.update({
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.facecolor": "white",
    "axes.facecolor":   "#F9F9F9",
    "axes.grid":        True,
    "grid.alpha":       0.35,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load preprocessed data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 3 - Model Training with Group K-Fold CV (K=5)")
print("="*65)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

FEATURE_COLS = [c for c in df.columns if c.startswith("MFCC")]
print(f"\nLoaded: {len(df):,} rows, {len(FEATURE_COLS)} features")

X      = df[FEATURE_COLS].values.astype(np.float64)
y_full = df["Is_Full"].values.astype(int)
groups = df["Group_KFold"].values.astype(int)

print(f"Groups (recordings): {np.unique(groups).size}")
print(f"Is_Full positives  : {y_full.sum():,} / {len(y_full):,}  ({y_full.mean()*100:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model definitions
# ─────────────────────────────────────────────────────────────────────────────
def get_models_binary():
    """Return dict of models tuned for binary classification (Is_Full)."""
    return {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE,
        ),
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Task A — Binary: Is_Full
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*65)
print("  TASK A: Fullness Detection (binary Is_Full)  [GroupKFold K=5]")
print("-"*65)

gkf = GroupKFold(n_splits=N_SPLITS)
models_a = get_models_binary()

# Storage for fold-level metrics and curve data
fold_records_a  = []
roc_data_a      = {m: {"fprs":[], "tprs":[], "aucs":[]} for m in models_a}
pr_data_a       = {m: {"precs":[], "recs":[], "aps":[]}  for m in models_a}
feat_imp_a      = {m: np.zeros(len(FEATURE_COLS)) for m in models_a if m != "Logistic Regression"}
feat_imp_a["Logistic Regression"] = np.zeros(len(FEATURE_COLS))

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y_full, groups), start=1):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y_full[train_idx], y_full[test_idx]

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Sample weights for Gradient Boosting (handles imbalance)
    sw_tr = compute_sample_weight("balanced", y_tr)

    train_recs = np.unique(groups[train_idx])
    test_recs  = np.unique(groups[test_idx])
    print(f"\n  Fold {fold_idx}  |  train recs: {sorted(train_recs)}  |  test recs: {sorted(test_recs)}")
    print(f"           train size: {len(y_tr):,}  (pos={y_tr.sum()})  |  test size: {len(y_te):,}  (pos={y_te.sum()})")

    for model_name, clf in models_a.items():
        # Fit
        if model_name == "Gradient Boosting":
            clf.fit(X_tr_s, y_tr, sample_weight=sw_tr)
        else:
            clf.fit(X_tr_s, y_tr)

        y_pred   = clf.predict(X_te_s)
        y_prob   = clf.predict_proba(X_te_s)[:, 1]

        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        f1   = f1_score(y_te, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_te, y_prob) if y_te.sum() > 0 else np.nan
        ap   = average_precision_score(y_te, y_prob) if y_te.sum() > 0 else np.nan

        fold_records_a.append({
            "Fold": fold_idx, "Model": model_name,
            "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc_roc, 4) if not np.isnan(auc_roc) else np.nan,
            "Avg-Precision": round(ap, 4) if not np.isnan(ap) else np.nan,
        })

        # ROC curve data
        if y_te.sum() > 0:
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            roc_data_a[model_name]["fprs"].append(fpr)
            roc_data_a[model_name]["tprs"].append(tpr)
            roc_data_a[model_name]["aucs"].append(auc_roc)

            prec_arr, rec_arr, _ = precision_recall_curve(y_te, y_prob)
            pr_data_a[model_name]["precs"].append(prec_arr)
            pr_data_a[model_name]["recs"].append(rec_arr)
            pr_data_a[model_name]["aps"].append(ap)

        # Feature importance accumulation
        if model_name in ["Gradient Boosting", "Random Forest"]:
            feat_imp_a[model_name] += clf.feature_importances_
        else:  # Logistic Regression
            feat_imp_a[model_name] += np.abs(clf.coef_[0])

        print(f"    {model_name:<22}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={auc_roc:.3f}")

# Average feature importances over folds
for m in feat_imp_a:
    feat_imp_a[m] /= N_SPLITS

# Fold-level CSV
df_folds_a = pd.DataFrame(fold_records_a)
df_folds_a.to_csv(OUT_TABLES / "fold_metrics_fullness.csv", index=False, encoding="utf-8-sig")
print(f"\n  [OK] Saved -> tables/fold_metrics_fullness.csv")

# Summary (mean ± std across folds)
summary_a = df_folds_a.groupby("Model")[["Accuracy","Precision","Recall","F1-Score","AUC-ROC","Avg-Precision"]].agg(
    ["mean","std"]).round(4)
summary_a.columns = [f"{col}_{stat}" for col, stat in summary_a.columns]
summary_a = summary_a.reset_index()
summary_a.to_csv(OUT_TABLES / "cv_results_fullness.csv", index=False, encoding="utf-8-sig")
print(f"  [OK] Saved -> tables/cv_results_fullness.csv")

print("\n  ---- Summary (Mean across 5 folds) ----")
for _, row in summary_a.iterrows():
    print(f"  {row['Model']:<22}  "
          f"Acc={row['Accuracy_mean']:.3f}+/-{row['Accuracy_std']:.3f}  "
          f"F1={row['F1-Score_mean']:.3f}+/-{row['F1-Score_std']:.3f}  "
          f"AUC={row['AUC-ROC_mean']:.3f}+/-{row['AUC-ROC_std']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FIGURES — ROC + PR curves
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "-"*65)
print("  Generating figures...")
print("-"*65)

# ── 5a. ROC curves — Task A ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.50)")

mean_fpr = np.linspace(0, 1, 200)
for model_name, color in MODEL_COLORS.items():
    data = roc_data_a[model_name]
    if not data["fprs"]:
        continue
    interp_tprs = []
    for fpr, tpr in zip(data["fprs"], data["tprs"]):
        interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, color=color, alpha=0.18, lw=0.9)
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr  = np.std(interp_tprs,  axis=0)
    mean_auc = np.mean(data["aucs"])
    std_auc  = np.std(data["aucs"])
    ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5,
            label=f"{model_name}  (AUC = {mean_auc:.3f} +/- {std_auc:.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color=color, alpha=0.12)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Fullness Detection (Is_Full)\nGroupKFold (K=5) Mean +/- Std", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
fig.tight_layout()
fig.savefig(OUT_FIGS / "roc_curves_fullness.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> figures/roc_curves_fullness.png")

# ── 5b. PR curves — Task A ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
baseline = y_full.mean()
ax.axhline(y=baseline, color="k", linestyle="--", lw=1.2,
           label=f"Random baseline (AP = {baseline:.3f})")

mean_rec_grid = np.linspace(0, 1, 200)
for model_name, color in MODEL_COLORS.items():
    data = pr_data_a[model_name]
    if not data["precs"]:
        continue
    interp_precs = []
    for prec, rec in zip(data["precs"], data["recs"]):
        interp_precs.append(np.interp(mean_rec_grid, rec[::-1], prec[::-1]))
        ax.plot(rec, prec, color=color, alpha=0.18, lw=0.9)
    mean_prec = np.mean(interp_precs, axis=0)
    std_prec  = np.std(interp_precs,  axis=0)
    mean_ap   = np.mean(data["aps"])
    std_ap    = np.std(data["aps"])
    ax.plot(mean_rec_grid, mean_prec, color=color, lw=2.5,
            label=f"{model_name}  (AP = {mean_ap:.3f} +/- {std_ap:.3f})")
    ax.fill_between(mean_rec_grid, mean_prec - std_prec, mean_prec + std_prec,
                    color=color, alpha=0.12)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Fullness Detection (Is_Full)\nGroupKFold (K=5) Mean +/- Std", fontweight="bold")
ax.legend(loc="upper right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
fig.tight_layout()
fig.savefig(OUT_FIGS / "pr_curves_fullness.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> figures/pr_curves_fullness.png")

fig.suptitle("Top 15 Feature Importances — Fullness Detection (Is_Full)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_FIGS / "feature_importance.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> figures/feature_importance.png")

# ── 5e. CV metrics bar chart — Task A ────────────────────────────────────
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
x = np.arange(len(metrics_to_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
for i, (model_name, color) in enumerate(MODEL_COLORS.items()):
    row = summary_a[summary_a["Model"] == model_name].iloc[0]
    means = [row[f"{m}_mean"] for m in metrics_to_plot]
    stds  = [row[f"{m}_std"]  for m in metrics_to_plot]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, means, width, label=model_name,
                  color=color, yerr=stds, capsize=4,
                  edgecolor="white", linewidth=1.0, alpha=0.9)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylabel("Score")
ax.set_ylim([0, 1.12])
ax.set_title("Model Comparison — Fullness Detection (Is_Full)\nGroupKFold (K=5) Mean +/- Std",
             fontweight="bold")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(OUT_FIGS / "metrics_comparison_fullness.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> figures/metrics_comparison_fullness.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  FINAL SUMMARY")
print("="*65)

print("\n  Fullness Detection (Is_Full):")
for _, row in summary_a.sort_values("F1-Score_mean", ascending=False).iterrows():
    print(f"    {row['Model']:<22}  F1={row['F1-Score_mean']:.3f}  "
          f"AUC={row['AUC-ROC_mean']:.3f}  "
          f"Recall={row['Recall_mean']:.3f}")

print("\n  Outputs saved:")
print("    tables/  ->  fold_metrics_fullness.csv, cv_results_fullness.csv")
print("    figures/ ->  roc_curves_fullness.png, pr_curves_fullness.png")
print("                 feature_importance.png, metrics_comparison_fullness.png")
print("\n  Step 3 COMPLETE - Next: run  python Codes/04_evaluation_and_report.py")
print("="*65 + "\n")
