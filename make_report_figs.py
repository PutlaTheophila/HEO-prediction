"""Generate figures for the LaTeX report (4-class HEO predictor)."""
import os, pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier

SEED = 42
NAMES4 = ["Fluorite", "Rock-salt", "Spinel", "Perovskite"]
NAMES3 = ["Fluorite", "Rock-salt", "Spinel"]

# ---------------------------------------------------------------------------
# 1. Class-distribution comparison (real vs old-augmented vs new-augmented)
# ---------------------------------------------------------------------------
real_counts   = [451, 342, 2194, 0]
old_counts    = [951, 842, 2194, 0]
new_counts_df = pd.read_csv("heo_augmented_dataset.csv")["label"].value_counts().sort_index()
new_counts    = [int(new_counts_df.get(i, 0)) for i in range(4)]

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(4)
w = 0.27
ax.bar(x - w, real_counts,  w, label="Raw real data",      color="#888")
ax.bar(x,     old_counts,   w, label="Old augmentation",   color="#ffae42")
ax.bar(x + w, new_counts,   w, label="New augmentation",   color="#4caf50")
ax.set_xticks(x); ax.set_xticklabels(NAMES4)
ax.set_ylabel("Number of samples")
ax.set_title("Class distribution before vs. after rebalancing")
for i, vals in enumerate(zip(real_counts, old_counts, new_counts)):
    for k, v in enumerate(vals):
        if v > 0:
            ax.text(i + (k - 1) * w, v + 30, str(v), ha="center", fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig("fig_class_distribution.png", dpi=160)
plt.close()
print("✓ fig_class_distribution.png")

# ---------------------------------------------------------------------------
# 2. Train OLD 3-class model + NEW 4-class model and produce confusion matrices
# ---------------------------------------------------------------------------
def train_and_eval(X, y, names):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30,
                                          random_state=SEED, stratify=y)
    m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
                      eval_metric="mlogloss", random_state=SEED, verbosity=0,
                      n_jobs=-1)
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    return m, yte, yp

# OLD 3-class: real data only, raw imbalance ----------------------------------
df4 = pd.read_csv("heo_data_4comp.csv")
df5 = pd.read_csv("heo_data_5comp.csv")
df_real = pd.concat([df4, df5], ignore_index=True)
smap = {"aPbO2": 0, "bad": 1, "rut": 2}
df_real["y"] = df_real["Min_Crystal_rho"].map(smap)
df_real = df_real.dropna(subset=["y"])

from feature_engineering import calculate_all_features, features_to_array
Xo, yo = [], []
for _, row in df_real.iterrows():
    try:
        e = eval(row["Formula"])
        Xo.append(features_to_array(calculate_all_features(e)))
        yo.append(int(row["y"]))
    except Exception:
        pass
Xo = np.array(Xo); yo = np.array(yo)
m_old, yte_o, yp_o = train_and_eval(Xo, yo, NAMES3)

# NEW 4-class -----------------------------------------------------------------
df_aug = pd.read_csv("heo_augmented_dataset.csv")
Xn = df_aug.drop(columns=["label"]).values
yn = df_aug["label"].values
m_new, yte_n, yp_n = train_and_eval(Xn, yn, NAMES4)

# Side-by-side confusion matrices --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm_o = confusion_matrix(yte_o, yp_o)
cm_n = confusion_matrix(yte_n, yp_n)

sns.heatmap(cm_o, annot=True, fmt="d", cmap="Oranges",
            xticklabels=NAMES3, yticklabels=NAMES3, ax=axes[0], cbar=False)
axes[0].set_title(f"OLD 3-class on raw real data\nTest acc = {accuracy_score(yte_o, yp_o):.3f}")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

sns.heatmap(cm_n, annot=True, fmt="d", cmap="Greens",
            xticklabels=NAMES4, yticklabels=NAMES4, ax=axes[1], cbar=False)
axes[1].set_title(f"NEW 4-class on rebalanced data\nTest acc = {accuracy_score(yte_n, yp_n):.3f}")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.savefig("fig_confusion_compare.png", dpi=160)
plt.close()
print("✓ fig_confusion_compare.png")

# ---------------------------------------------------------------------------
# 3. Per-class F1 bar comparison
# ---------------------------------------------------------------------------
f1_old = f1_score(yte_o, yp_o, average=None)
f1_new = f1_score(yte_n, yp_n, average=None)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(4); w = 0.35
old_padded = list(f1_old) + [0.0]
ax.bar(x - w/2, old_padded,    w, label="Old (3-class, imbalanced)", color="#ffae42")
ax.bar(x + w/2, list(f1_new),  w, label="New (4-class, rebalanced)", color="#4caf50")
ax.set_xticks(x); ax.set_xticklabels(NAMES4)
ax.set_ylim(0, 1.05); ax.set_ylabel("Per-class F1-score")
ax.set_title("Per-class F1: old vs. new model")
for xi, v in enumerate(old_padded):
    if v > 0: ax.text(xi - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
for xi, v in enumerate(f1_new):
    ax.text(xi + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig("fig_f1_compare.png", dpi=160)
plt.close()
print("✓ fig_f1_compare.png")

# ---------------------------------------------------------------------------
# 4. Feature importance (new model)
# ---------------------------------------------------------------------------
feat_cols = [c for c in df_aug.columns if c != "label"]
imp = m_new.feature_importances_
order = np.argsort(imp)[::-1][:12]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([feat_cols[i] for i in order][::-1], imp[order][::-1], color="#3f51b5")
ax.set_title("Top-12 feature importances (new XGBoost model)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=160)
plt.close()
print("✓ fig_feature_importance.png")

# ---------------------------------------------------------------------------
# 5. Print a short summary the LaTeX file references
# ---------------------------------------------------------------------------
print("\n=== OLD 3-class report ===")
print(classification_report(yte_o, yp_o, target_names=NAMES3, digits=3))
print("=== NEW 4-class report ===")
print(classification_report(yte_n, yp_n, target_names=NAMES4, digits=3))
