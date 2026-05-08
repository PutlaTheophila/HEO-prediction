"""
HEO Crystal Structure Prediction — Multi-Model Comparison with ADASYN
======================================================================
- Balances classes using ADASYN (Adaptive Synthetic Sampling)
- Trains and evaluates many classifiers
- Reports test accuracy, macro-F1, per-class metrics, confusion matrices
- Saves the best model to heo_model_adasyn.pkl
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN, SMOTE

from feature_engineering import calculate_all_features, features_to_array


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("  HEO STRUCTURE PREDICTOR — MULTI-MODEL COMPARISON (with ADASYN)")
print("=" * 78)

print("\n[1/5] Loading datasets ...")
df4 = pd.read_csv("heo_data_4comp.csv")
df5 = pd.read_csv("heo_data_5comp.csv")
df = pd.concat([df4, df5], ignore_index=True)

structure_map = {"aPbO2": 0, "bad": 1, "rut": 2}
df["structure_label"] = df["Min_Crystal_rho"].map(structure_map)
df = df.dropna(subset=["structure_label"])

STRUCTURE_NAMES = {0: "Fluorite", 1: "Rock-salt", 2: "Spinel"}

print(f"   Total samples: {len(df)}")
print("   Class distribution (raw):")
for k, v in df["Min_Crystal_rho"].value_counts().items():
    print(f"     {STRUCTURE_NAMES[structure_map[k]]:10s} ({k}) : {v}")


# ---------------------------------------------------------------------------
# 2. Build feature matrix
# ---------------------------------------------------------------------------
print("\n[2/5] Computing composition-based features ...")

X_list, y_list = [], []
for _, row in df.iterrows():
    try:
        elements = eval(row["Formula"])
        feats = features_to_array(calculate_all_features(elements))
        X_list.append(feats)
        y_list.append(int(row["structure_label"]))
    except Exception:
        continue

X = np.array(X_list)
y = np.array(y_list)
print(f"   Feature matrix: {X.shape}")


# ---------------------------------------------------------------------------
# 3. Train / test split, then apply ADASYN to the training set only
# ---------------------------------------------------------------------------
print("\n[3/5] Splitting (70/30) and balancing training set with ADASYN ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("   Training class counts BEFORE ADASYN:")
for c in np.unique(y_train):
    print(f"     {STRUCTURE_NAMES[c]:10s} : {(y_train == c).sum()}")

try:
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_train_bal, y_train_bal = adasyn.fit_resample(X_train, y_train)
    sampler_name = "ADASYN"
except ValueError:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    sampler_name = "SMOTE (ADASYN fallback)"

print(f"\n   Sampler used: {sampler_name}")
print("   Training class counts AFTER balancing:")
for c in np.unique(y_train_bal):
    print(f"     {STRUCTURE_NAMES[c]:10s} : {(y_train_bal == c).sum()}")


# ---------------------------------------------------------------------------
# 4. Define candidate models
# ---------------------------------------------------------------------------
def make_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=600, max_depth=None, n_jobs=-1, random_state=42,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=600, n_jobs=-1, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=400, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance")),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]),
        "MLP (Neural Net)": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64),
                                   activation="relu", max_iter=500,
                                   random_state=42)),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("nb", GaussianNB()),
        ]),
    }


# ---------------------------------------------------------------------------
# 5. Train + evaluate every model
# ---------------------------------------------------------------------------
print("\n[4/5] Training and evaluating models ...")
print(f"   (training set after {sampler_name}: {X_train_bal.shape[0]} samples)\n")

results = []
trained = {}
class_names = [STRUCTURE_NAMES[i] for i in range(3)]

for name, model in make_models().items():
    try:
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        f1w = f1_score(y_test, y_pred, average="weighted")

        cm = confusion_matrix(y_test, y_pred)
        per_class_recall = cm.diagonal() / cm.sum(axis=1)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro-F1": f1m,
            "Weighted-F1": f1w,
            "Recall_Fluorite": per_class_recall[0],
            "Recall_RockSalt": per_class_recall[1],
            "Recall_Spinel":   per_class_recall[2],
        })
        trained[name] = (model, y_pred, cm)
        print(f"   OK  {name:25s}  acc={acc:.4f}  macro-F1={f1m:.4f}")
    except Exception as e:
        print(f"   FAIL {name:25s}  -> {e}")


# ---------------------------------------------------------------------------
# 6. Leaderboard
# ---------------------------------------------------------------------------
res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

print("\n" + "=" * 78)
print("  LEADERBOARD (sorted by test accuracy)")
print("=" * 78)
print(res_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_name = res_df.iloc[0]["Model"]
best_model, best_pred, best_cm = trained[best_name]

print("\n" + "=" * 78)
print(f"  BEST MODEL: {best_name}")
print("=" * 78)
print("\n  Classification report:")
print(classification_report(y_test, best_pred, target_names=class_names, digits=3))

print("  Confusion matrix:")
print(f"  {'':12s}" + "".join(f"{n[:9]:>10s}" for n in class_names))
for i, row in enumerate(best_cm):
    print(f"  Actual {class_names[i][:6]:6s}" + "".join(f"{v:>10d}" for v in row))


# ---------------------------------------------------------------------------
# 7. Save best model + leaderboard + confusion-matrix figure
# ---------------------------------------------------------------------------
print("\n[5/5] Saving artifacts ...")

with open("heo_model_adasyn.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"   Saved best model ({best_name}) -> heo_model_adasyn.pkl")

res_df.to_csv("model_comparison_results.csv", index=False)
print("   Saved leaderboard -> model_comparison_results.csv")

top4 = res_df.head(4)["Model"].tolist()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, name in zip(axes.ravel(), top4):
    _, _, cm = trained[name]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar=False)
    ax.set_title(f"{name}\nacc={accuracy_score(y_test, trained[name][1]):.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle(f"Top-4 models after {sampler_name} balancing", fontsize=14)
plt.tight_layout()
plt.savefig("model_comparison_confusion_matrices.png", dpi=130)
print("   Saved figure -> model_comparison_confusion_matrices.png")

print("\n" + "=" * 78)
print(f"  DONE. Best test accuracy: {res_df.iloc[0]['Accuracy']:.4f} "
      f"({best_name})")
print("=" * 78 + "\n")
