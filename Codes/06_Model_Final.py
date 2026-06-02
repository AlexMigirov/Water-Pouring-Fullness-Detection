import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "tables/preprocessed_dataset.csv"

N_SPLITS = 5
RANDOM_STATE = 42
TARGET_THRESHOLD = 0.3  # הסף המבוקש

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

FEATURE_COLS = [c for c in df.columns if c.startswith("MFCC")]

X = df[FEATURE_COLS].values
y = df["Is_Full"].values
groups = df["Group_KFold"].values

print(f"Loaded {len(df)} samples")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def get_model():
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )

# ─────────────────────────────────────────────
# CROSS-VALIDATION + METRICS PER fold
# ─────────────────────────────────────────────
gkf = GroupKFold(n_splits=N_SPLITS)

# רשימות לשמירת המטריקות מכל קיפול (Fold) עבור הגרף
fold_metrics = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "AUC-ROC": []
}

# רשימות לאיסוף גלובלי לצורך מטריצת הבלבול הסופית
all_probs = []
all_true = []

for train_idx, test_idx in gkf.split(X, y, groups):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = get_model()
    sample_weights = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr, y_tr, sample_weight=sample_weights)

    # חישוב הסתברויות עבור הקיפול הנוכחי
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs >= TARGET_THRESHOLD).astype(int)

    # שמירה לאוסף הכללי
    all_probs.extend(probs)
    all_true.extend(y_te)

    # חישוב מטריקות לקיפול הנוכחי (תחת סף 0.3)
    fold_metrics["Accuracy"].append(accuracy_score(y_te, preds))
    fold_metrics["Precision"].append(precision_score(y_te, preds, zero_division=0))
    fold_metrics["Recall"].append(recall_score(y_te, preds))
    fold_metrics["F1-Score"].append(f1_score(y_te, preds))
    
    # חישוב AUC-ROC מבוסס על ההסתברויות (probs)
    try:
        fold_metrics["AUC-ROC"].append(roc_auc_score(y_te, probs))
    except ValueError:
        # במקרה שאין מספיק דגימות מאחת המחלקות בטסט הספציפי
        fold_metrics["AUC-ROC"].append(0.5)

all_probs = np.array(all_probs)
all_true = np.array(all_true)

print("Collected predictions and calculated metrics per fold ✅")

# ─────────────────────────────────────────────
# COMPUTE MEAN AND STD FOR THE PLOT
# ─────────────────────────────────────────────
metric_names = list(fold_metrics.keys())
means = [np.mean(fold_metrics[m]) for m in metric_names]
stds = [np.std(fold_metrics[m]) for m in metric_names]

# ─────────────────────────────────────────────
# PLOT 1: THE REQUESTED BAR PLOT WITH ERROR BARS
# ─────────────────────────────────────────────
plt.figure(figsize=(9, 5))

# יצירת גרף המקלות עם קווי השגיאה (סטיית תקן)
bars = plt.bar(metric_names, means, yerr=stds, capsize=6, color="#e54b56", edgecolor="grey", width=0.4)

# הוספת הערכים המספריים מעל כל עמודה
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.3f}', 
             ha='center', va='bottom', fontsize=10, weight='bold')

plt.ylim(0, 1.15)
plt.ylabel("Score", fontsize=11)
plt.title(f"Gradient Boosting — Fullness Detection (Is_Full)\nGroupKFold (K=5) Mean +/- Std | Threshold = {TARGET_THRESHOLD}", 
          fontsize=12, pad=15, weight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# PLOT 2: CONFUSION MATRIX
# ─────────────────────────────────────────────
final_preds = (all_probs >= TARGET_THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(all_true, final_preds).ravel()
cm = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Not Full", "Pred Full"],
            yticklabels=["Actual Not Full", "Actual Full"])
plt.title(f"Confusion Matrix (threshold={TARGET_THRESHOLD:.2f})")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# PRINT RESULTS TO CONSOLE
# ─────────────────────────────────────────────
print("\n=== Mean Metrics (Threshold = 0.3) ===")
for m in metric_names:
    print(f"{m}: Mean = {np.mean(fold_metrics[m]):.3f} +/- Std = {np.std(fold_metrics[m]):.3f}")