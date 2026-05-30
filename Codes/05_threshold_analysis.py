import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "tables/preprocessed_dataset.csv"

N_SPLITS = 5
RANDOM_STATE = 42

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
# MODEL (same as training)
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
# CROSS-VALIDATION + PROB COLLECTION
# ─────────────────────────────────────────────
gkf = GroupKFold(n_splits=N_SPLITS)

all_probs = []
all_true  = []

for train_idx, test_idx in gkf.split(X, y, groups):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = get_model()

    # handle imbalance
    sample_weights = compute_sample_weight("balanced", y_tr)

    model.fit(X_tr, y_tr, sample_weight=sample_weights)

    probs = model.predict_proba(X_te)[:, 1]

    all_probs.extend(probs)
    all_true.extend(y_te)

all_probs = np.array(all_probs)
all_true = np.array(all_true)

print("Collected predictions from all folds ✅")

# ─────────────────────────────────────────────
# THRESHOLD SWEEP
# ─────────────────────────────────────────────
thresholds = np.arange(0.5, 0.19, -0.03)

results = []

for t in thresholds:
    y_pred = (all_probs >= t).astype(int)

    acc  = accuracy_score(all_true, y_pred)
    prec = precision_score(all_true, y_pred, zero_division=0)
    rec  = recall_score(all_true, y_pred)
    f1   = f1_score(all_true, y_pred)

    results.append((t, acc, prec, rec, f1))

# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────
print("\n=== Threshold Results ===")
for t, acc, prec, rec, f1 in results:
    print(f"t={t:.2f} | Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
thresholds_plot = [r[0] for r in results]
accuracy_list   = [r[1] for r in results]
precision_list  = [r[2] for r in results]
recall_list     = [r[3] for r in results]
f1_list         = [r[4] for r in results]

plt.figure(figsize=(8, 5))

plt.plot(thresholds_plot, recall_list, label="Recall")
plt.plot(thresholds_plot, precision_list, label="Precision")
plt.plot(thresholds_plot, f1_list, label="F1")
plt.plot(thresholds_plot, accuracy_list, label="Accuracy")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Sweep — Gradient Boosting")
plt.legend()
plt.grid(True)

plt.gca().invert_xaxis()  # threshold יורד משמאל לימין
plt.show()
