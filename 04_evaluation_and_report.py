import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

"""
04_evaluation_and_report.py
===========================
Step 4 of the Water-Level Detection ML Pipeline.

Generates all final report-ready figures and tables:
  - Confusion matrices (both tasks, all 3 models)
  - Per-fold metric variability plots
  - Consolidated Word-ready summary table (CSV)
  - Best model analysis with threshold tuning (Task A)
  - Per-recording accuracy heatmap

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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("outputs/tables/preprocessed_dataset.csv")
OUT_FIGS   = Path("outputs/figures")
OUT_TABLES = Path("outputs/tables")
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_TABLES.mkdir(parents=True, exist_ok=True)

N_SPLITS     = 5
RANDOM_STATE = 42

CUP_LABEL_MAP  = {1: "Thick_Glass", 2: "Tall_Thin_Glass", 3: "Ceramic_Cup", 4: "Plastic_Cup"}
CUP_SHORT      = {1: "Thick\nGlass", 2: "Tall\nThin", 3: "Ceramic\nCup", 4: "Plastic\nCup"}
CLASSES_B      = np.array([1, 2, 3, 4])
CLASS_NAMES_B  = [CUP_LABEL_MAP[c] for c in CLASSES_B]
CLASS_SHORT_B  = [CUP_SHORT[c]     for c in CLASSES_B]

MODEL_COLORS = {
    "Gradient Boosting":   "#E63946",
    "Logistic Regression": "#457B9D",
    "Random Forest":       "#2A9D8F",
}
CUP_PALETTE = {
    "Thick_Glass":     "#4E79A7",
    "Tall_Thin_Glass": "#F28E2B",
    "Ceramic_Cup":     "#59A14F",
    "Plastic_Cup":     "#E15759",
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
y_cup  = df["Cup_Label"].values.astype(int)
groups = df["Group_KFold"].values.astype(int)

print(f"Loaded {len(df):,} rows, {len(FEATURE_COLS)} features, {np.unique(groups).size} groups")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model factories  (identical hyperparameters to step 3)
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

def make_models_multi():
    return {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. OOF prediction engine
# ─────────────────────────────────────────────────────────────────────────────
def collect_oof(X, y, groups, model_factory, task="binary"):
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

        print(f"  Fold {fold_idx} done ({task})")

    # Convert to arrays
    for m in results:
        for k in results[m]:
            results[m][k] = np.array(results[m][k])
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run OOF for both tasks
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Collecting OOF predictions: Task A (Is_Full) ---")
oof_a = collect_oof(X, y_full, groups, make_models_binary, task="binary")

print("\n--- Collecting OOF predictions: Task B (Cup Type) ---")
oof_b = collect_oof(X, y_cup,  groups, make_models_multi,  task="multiclass")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Confusion matrices — Task A (binary)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Generating figures ---")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices — Fullness Detection (Is_Full)\n"
             "Out-of-Fold Predictions (GroupKFold K=5)",
             fontsize=13, fontweight="bold", y=1.03)

for ax, (mname, color) in zip(axes, MODEL_COLORS.items()):
    yt = oof_a[mname]["y_true"]
    yp = oof_a[mname]["y_pred"]
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
print("  [OK] Saved -> outputs/figures/confusion_matrix_fullness.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Confusion matrices — Task B (4-class)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Confusion Matrices — Cup Type Classification (4-class)\n"
             "Out-of-Fold Predictions (GroupKFold K=5)",
             fontsize=13, fontweight="bold", y=1.03)

for ax, (mname, color) in zip(axes, MODEL_COLORS.items()):
    yt = oof_b[mname]["y_true"]
    yp = oof_b[mname]["y_pred"]
    cm = confusion_matrix(yt, yp, labels=CLASSES_B)
    cm_norm = confusion_matrix(yt, yp, labels=CLASSES_B, normalize="true")

    annot = np.empty_like(cm, dtype=object)
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        annot[i, j] = f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)"

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Greens",
                xticklabels=CLASS_SHORT_B, yticklabels=CLASS_SHORT_B,
                vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 9}, ax=ax,
                cbar_kws={"shrink": 0.8})

    acc    = accuracy_score(yt, yp)
    f1_mac = f1_score(yt, yp, average="macro", zero_division=0)

    ax.set_title(f"{mname}\nAcc={acc:.3f}  F1-Macro={f1_mac:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

fig.tight_layout()
fig.savefig(OUT_FIGS / "confusion_matrix_cup_type.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> outputs/figures/confusion_matrix_cup_type.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-fold metric variability — Task A
# ─────────────────────────────────────────────────────────────────────────────
df_fold_a = pd.read_csv(OUT_TABLES / "fold_metrics_fullness.csv")
metrics_a  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

fig, axes = plt.subplots(1, len(metrics_a), figsize=(16, 5), sharey=False)
fig.suptitle("Per-Fold Metric Variability — Fullness Detection (Is_Full)",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, metrics_a):
    for mname, color in MODEL_COLORS.items():
        vals = df_fold_a[df_fold_a["Model"] == mname][metric].values
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
print("  [OK] Saved -> outputs/figures/fold_variability_fullness.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Per-fold metric variability — Task B
# ─────────────────────────────────────────────────────────────────────────────
df_fold_b = pd.read_csv(OUT_TABLES / "fold_metrics_cup_type.csv")
metrics_b  = ["Accuracy", "F1-Macro", "F1-Weighted", "AUC-ROC-Macro"]

fig, axes = plt.subplots(1, len(metrics_b), figsize=(14, 5), sharey=False)
fig.suptitle("Per-Fold Metric Variability — Cup Type Classification",
             fontsize=13, fontweight="bold")

for ax, metric in zip(axes, metrics_b):
    for mname, color in MODEL_COLORS.items():
        vals = df_fold_b[df_fold_b["Model"] == mname][metric].values
        folds = np.arange(1, len(vals)+1)
        ax.plot(folds, vals, "o-", color=color, lw=2, ms=6, label=mname)
    ax.set_title(metric, fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_xticks(range(1, N_SPLITS+1))
    ax.set_ylim([0.6, 1.05])

axes[0].set_ylabel("Score")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.12), frameon=True)
fig.tight_layout()
fig.savefig(OUT_FIGS / "fold_variability_cup_type.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> outputs/figures/fold_variability_cup_type.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Per-recording accuracy heatmap — Task B (best model = GB)
# ─────────────────────────────────────────────────────────────────────────────
best_model = "Gradient Boosting"
rec_ids  = oof_b[best_model]["rec_ids"]
yt_arr   = oof_b[best_model]["y_true"]
yp_arr   = oof_b[best_model]["y_pred"]

rec_stats = []
for rec in sorted(np.unique(rec_ids)):
    mask = rec_ids == rec
    acc  = accuracy_score(yt_arr[mask], yp_arr[mask])
    cup  = CUP_LABEL_MAP[int(yt_arr[mask][0])]
    rec_stats.append({"Recording": int(rec), "Cup_Type": cup, "Accuracy": acc})

df_rec = pd.DataFrame(rec_stats).sort_values("Recording")
df_rec.to_csv(OUT_TABLES / "per_recording_accuracy.csv", index=False, encoding="utf-8-sig")

# Heatmap: recordings as rows grouped by cup type
pivot = df_rec.pivot_table(index="Cup_Type", columns="Recording", values="Accuracy")
cup_order = list(CUP_PALETTE.keys())
pivot = pivot.reindex([c for c in cup_order if c in pivot.index])

fig, ax = plt.subplots(figsize=(18, 4))
sns.heatmap(pivot.astype(float), cmap="RdYlGn", vmin=0.5, vmax=1.0,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            linewidths=0.3, linecolor="white",
            ax=ax, cbar_kws={"label": "Accuracy", "shrink": 0.7})
ax.set_title(f"Per-Recording Classification Accuracy — {best_model}\n"
             f"Cup Type Classification (Task B, OOF)",
             fontweight="bold")
ax.set_xlabel("Recording Number")
ax.set_ylabel("Cup Type")
fig.tight_layout()
fig.savefig(OUT_FIGS / "per_recording_accuracy.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> outputs/figures/per_recording_accuracy.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Comprehensive Word-ready summary table
# ─────────────────────────────────────────────────────────────────────────────
rows = []

# Task A OOF metrics
for mname in MODEL_COLORS:
    yt = oof_a[mname]["y_true"]
    yp = oof_a[mname]["y_pred"]
    yb = oof_a[mname]["y_prob"][:, 1]
    rows.append({
        "Task":       "Fullness Detection (Is_Full)",
        "Model":      mname,
        "Accuracy":   round(accuracy_score(yt, yp), 4),
        "Precision":  round(precision_score(yt, yp, zero_division=0), 4),
        "Recall":     round(recall_score(yt, yp, zero_division=0), 4),
        "F1-Score":   round(f1_score(yt, yp, zero_division=0), 4),
        "AUC-ROC":    round(roc_auc_score(yt, yb), 4),
        "Avg-Prec":   round(average_precision_score(yt, yb), 4),
        "F1-Macro":   "-",
        "F1-Weighted":"-",
    })

# Task B OOF metrics
for mname in MODEL_COLORS:
    yt = oof_b[mname]["y_true"]
    yp = oof_b[mname]["y_pred"]
    yb = oof_b[mname]["y_prob"]
    yt_bin = label_binarize(yt, classes=CLASSES_B)
    rows.append({
        "Task":       "Cup Type Classification",
        "Model":      mname,
        "Accuracy":   round(accuracy_score(yt, yp), 4),
        "Precision":  round(precision_score(yt, yp, average="macro", zero_division=0), 4),
        "Recall":     round(recall_score(yt, yp, average="macro", zero_division=0), 4),
        "F1-Score":   "-",
        "AUC-ROC":    round(roc_auc_score(yt_bin, yb, average="macro", multi_class="ovr"), 4),
        "Avg-Prec":   "-",
        "F1-Macro":   round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        "F1-Weighted":round(f1_score(yt, yp, average="weighted", zero_division=0), 4),
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv(OUT_TABLES / "final_summary_table.csv", index=False, encoding="utf-8-sig")
print("  [OK] Saved -> outputs/tables/final_summary_table.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Combined dashboard figure (report cover figure)
# ─────────────────────────────────────────────────────────────────────────────
df_cv_a = pd.read_csv(OUT_TABLES / "cv_results_fullness.csv")
df_cv_b = pd.read_csv(OUT_TABLES / "cv_results_cup_type.csv")

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

model_names = list(MODEL_COLORS.keys())
colors      = list(MODEL_COLORS.values())
x           = np.arange(len(model_names))

# ── Row 0: Task A bars ────────────────────────────────────────────────────
metrics_show_a = ["F1-Score", "AUC-ROC", "Recall", "Precision"]
for col_idx, metric in enumerate(["F1-Score", "AUC-ROC", "Recall"]):
    ax = fig.add_subplot(gs[0, col_idx])
    means = [df_cv_a.loc[df_cv_a["Model"]==m, f"{metric}_mean"].values[0] for m in model_names]
    stds  = [df_cv_a.loc[df_cv_a["Model"]==m, f"{metric}_std"].values[0]  for m in model_names]
    bars  = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                   edgecolor="white", linewidth=1.2, alpha=0.88)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8)
    ax.set_ylim([0, 1.12])
    ax.set_title(f"Task A: {metric}", fontweight="bold", fontsize=10)
    ax.set_ylabel("Score")

# ── Row 1: Task B bars ────────────────────────────────────────────────────
for col_idx, metric in enumerate(["F1-Macro", "AUC-ROC-Macro", "Accuracy"]):
    ax = fig.add_subplot(gs[1, col_idx])
    means = [df_cv_b.loc[df_cv_b["Model"]==m, f"{metric}_mean"].values[0] for m in model_names]
    stds  = [df_cv_b.loc[df_cv_b["Model"]==m, f"{metric}_std"].values[0]  for m in model_names]
    bars  = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                   edgecolor="white", linewidth=1.2, alpha=0.88)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in model_names], fontsize=8)
    ax.set_ylim([0.6, 1.08])
    ax.set_title(f"Task B: {metric}", fontweight="bold", fontsize=10)
    ax.set_ylabel("Score")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=m) for m, c in MODEL_COLORS.items()]
fig.legend(handles=legend_elements, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=True)
fig.suptitle("Model Performance Dashboard\nTask A: Fullness Detection  |  Task B: Cup Type Classification",
             fontsize=13, fontweight="bold", y=1.07)

fig.savefig(OUT_FIGS / "performance_dashboard.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  [OK] Saved -> outputs/figures/performance_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Classification report printout (best model each task)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  CLASSIFICATION REPORTS (OOF predictions)")
print("="*65)

print("\n  Task A -- Gradient Boosting (best F1):")
print(classification_report(
    oof_a["Gradient Boosting"]["y_true"],
    oof_a["Gradient Boosting"]["y_pred"],
    target_names=["Not Full (0)", "Full (1)"],
    zero_division=0
))

print("\n  Task B -- Gradient Boosting (best F1-Macro):")
print(classification_report(
    oof_b["Gradient Boosting"]["y_true"],
    oof_b["Gradient Boosting"]["y_pred"],
    target_names=CLASS_NAMES_B,
    zero_division=0
))

# ─────────────────────────────────────────────────────────────────────────────
# 13. Final file listing
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  ALL OUTPUTS")
print("="*65)
print("\n  outputs/figures/")
for f in sorted(OUT_FIGS.glob("*.png")):
    print(f"    {f.name}")
print("\n  outputs/tables/")
for f in sorted(OUT_TABLES.glob("*.csv")):
    print(f"    {f.name}")

print("\n" + "="*65)
print("  Step 4 COMPLETE - Project pipeline finished!")
print("  All figures are 300 DPI and ready for the Word report.")
print("="*65 + "\n")
