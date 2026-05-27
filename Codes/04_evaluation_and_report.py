import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
04_evaluation_and_report.py
===========================
Step 4 of the Water-Level Detection ML Pipeline.

Generates all final report-ready figures and tables:
  - Confusion matrices (all 3 models, fullness detection)
  - Per-fold metric variability plot
  - Consolidated Word-ready summary table (CSV)
  - Combined performance dashboard

Strategy: accumulate OOF (Out-Of-Fold) predictions across all 5 GroupKFold
splits to produce one global confusion matrix per model — this uses every
sample exactly once and avoids train-set leakage.
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
from itertools import product

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH  = PROJECT_ROOT / "tables/preprocessed_dataset.csv"
OUT_FIGS   = PROJECT_ROOT / "figures"
OUT_TABLES = PROJECT_ROOT / "tables"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

N_SPLITS     = 5
RANDOM_STATE = 42

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
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F8F8",
    "axes.grid":        True,
    "grid.alpha":       0.35,
})

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  STEP 4 - Final Evaluation & Report Figures")
print("="*65)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
FEATURE_COLS = [c for c in df.columns if c.startswith("MFCC")]

X      = df[FEATURE_COLS].values.astype(np.float64)
y_full = df["Is_Full"].values.astype(int)
groups = df["Group_KFold"].values.astype(int)

print(f"Loaded {len(df):,} rows, {len(FEATURE_COLS)} features, {np.unique(groups).size} groups")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model factory  (identical hyperparameters to step 3)
# ─────────────────────────────────────────────────────────────────────────────
def make_models_binary():
    return {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. OOF prediction engine
# ─────────────────────────────────────────────────────────────────────────────
def collect_oof(X, y, groups, model_factory):
    """
    Run GroupKFold and return OOF predictions (labels + probabilities)
    for every model in model_factory().
    Returns dict: model_name -> {"y_true", "y_pred", "y_prob", "rec_ids"}
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    models = model_factory()
    results = {m: {"y_true":[], "y_pred":[], "y_prob":[], "rec_ids":[]} for m in models}

    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        g_te = groups[te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        sw_tr  = compute_sample_weight("balanced", y_tr)

        for mname, clf in models.items():
            fresh = type(clf)(**clf.get_params())   # fresh clone per fold
            if mname == "Gradient Boosting":
                fresh.fit(X_tr_s, y_tr, sample_weight=sw_tr)
            else:
                fresh.fit(X_tr_s, y_tr)

            y_pred = fresh.predict(X_te_s)
            y_prob = fresh.predict_proba(X_te_s)

            results[mname]["y_true"].extend(y_te.tolist())
            results[mname]["y_pred"].extend(y_pred.tolist())
            results[mname]["y_prob"].extend(y_prob.tolist())
            results[mname]["rec_ids"].extend(g_te.tolist())

        print(f"  Fold {fold_idx} done")

    # Convert to arrays
    for m in results:
        for k in results[m]:
            results[m][k] = np.array(results[m][k])
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run OOF predictions
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Collecting OOF predictions: Fullness Detection (Is_Full) ---")
oof = collect_oof(X, y_full, groups, make_models_binary)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Confusion matrices (all 3 models)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Generating figures ---")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices — Fullness Detection (Is_Full)\n"
             "Out-of-Fold Predictions (GroupKFold K=5)",
             fontsize=13, fontweight="bold", y=1.03)

for ax, (mname, color) in zip(axes, MODEL_COLORS.items()):
    yt = oof[mname]["y_true"]
    yp = oof[mname]["y_pred"]
    cm = confusion_matrix(yt, yp)
    cm_norm = confusion_matrix(yt, yp, normalize="true")

    # Annotate with count AND percentage
    annot = np.empty_like(cm, dtype=object)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        annot[i, j] = f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Not Full", "Full"],
                yticklabels=["Not Full", "Full"],
                vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 11}, ax=ax,
                cbar_kws={"shrink": 0.8})

    acc  = accuracy_score(yt, yp)
    f1   = f1_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    prec = precision_score(yt, yp, zero_division=0)

    ax.set_title(f"{mname}\nAcc={acc:.3f}  F1={f1:.3f}\nPrec={prec:.3f}  Rec={rec:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

fig.tight_layout()
fig.savefig(OUT_FIGS / "confusion_matrix_fullness.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> figures/confusion_matrix_fullness.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-fold metric variability
# ─────────────────────────────────────────────────────────────────────────────
df_fold = pd.read_csv(OUT_TABLES / "fold_metrics_fullness.csv")
metrics_show = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

fig, axes = plt.subplots(1, len(metrics_show), figsize=(16, 5), sharey=False)
fig.suptitle("Per-Fold Metric Variability — Fullness Detection (Is_Full)",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, metrics_show):
    for mname, color in MODEL_COLORS.items():
        vals  = df_fold[df_fold["Model"] == mname][metric].values
        folds = np.arange(1, len(vals)+1)
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=6, label=mname)

    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_xticks(range(1, N_SPLITS+1))
    ax.set_ylim([0, 1.05])

axes[0].set_ylabel("Score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.12), frameon=True)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fold_variability_fullness.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> figures/fold_variability_fullness.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Word-ready summary table
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for mname in MODEL_COLORS:
    yt = oof[mname]["y_true"]
    yp = oof[mname]["y_pred"]
    yb = oof[mname]["y_prob"][:, 1]
    rows.append({
        "Model":      mname,
        "Accuracy":   round(accuracy_score(yt, yp), 4),
        "Precision":  round(precision_score(yt, yp, zero_division=0), 4),
        "Recall":     round(recall_score(yt, yp, zero_division=0), 4),
        "F1-Score":   round(f1_score(yt, yp, zero_division=0), 4),
        "AUC-ROC":    round(roc_auc_score(yt, yb), 4),
        "Avg-Prec":   round(average_precision_score(yt, yb), 4),
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv(OUT_TABLES / "final_summary_table.csv", index=False, encoding="utf-8-sig")
print("  [OK] Saved -> tables/final_summary_table.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Performance dashboard (single-task, 3 metrics × 3 models)
# ─────────────────────────────────────────────────────────────────────────────
df_cv = pd.read_csv(OUT_TABLES / "cv_results_fullness.csv")

model_names = list(MODEL_COLORS.keys())
colors      = list(MODEL_COLORS.values())
x           = np.arange(len(model_names))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Model Performance Dashboard — Fullness Detection (Is_Full)\n"
             "GroupKFold (K=5)  Mean ± Std",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, ["F1-Score", "AUC-ROC", "Recall"]):
    means = [df_cv.loc[df_cv["Model"]==m, f"{metric}_mean"].values[0] for m in model_names]
    stds  = [df_cv.loc[df_cv["Model"]==m, f"{metric}_std"].values[0]  for m in model_names]
    bars  = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                   edgecolor="white", linewidth=1.2, alpha=0.88)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=9)
    ax.set_ylim([0, 1.14])
    ax.set_title(metric, fontweight="bold")
    ax.set_ylabel("Score")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=m) for m, c in MODEL_COLORS.items()]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.08), fontsize=10, frameon=True)
fig.tight_layout()
fig.savefig(OUT_FIGS / "performance_dashboard.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> figures/performance_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Classification report printout (best model = Gradient Boosting)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CLASSIFICATION REPORT — Gradient Boosting (best F1)")
print("="*65)
print(classification_report(
    oof["Gradient Boosting"]["y_true"],
    oof["Gradient Boosting"]["y_pred"],
    target_names=["Not Full (0)", "Full (1)"],
    zero_division=0
))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Final file listing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ALL OUTPUTS")
print("="*65)
print("\n  figures/")
for f in sorted(OUT_FIGS.glob("*.png")):
    print(f"    {f.name}")
print("\n  tables/")
for f in sorted(OUT_TABLES.glob("*.csv")):
    print(f"    {f.name}")

print("\n" + "="*65)
print("  Step 4 COMPLETE - Project pipeline finished!")
print("  All figures are 300 DPI and ready for the Word report.")
print("="*65 + "\n")
