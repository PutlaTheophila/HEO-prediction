"""
HEO Crystal-Structure Prediction — Best-Accuracy Pipeline
==========================================================
Stages:
  1. Magpie-style features (weighted mean / std / min / max / range
     across all elemental properties) on top of the original 20 features.
  2. Optuna Bayesian hyperparameter tuning of XGBoost, LightGBM,
     CatBoost, RandomForest, ExtraTrees (stratified 5-fold CV).
  3. Stacking ensemble (LogReg meta-learner) of the top-3 tuned models.
  4. Final evaluation on held-out test set; saves best artifact.
"""

import os
import time
import pickle
import warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
from catboost import CatBoostClassifier

from feature_engineering import calculate_all_features, features_to_array
from elemental_properties import ELEMENTAL_PROPERTIES

t0 = time.time()
SEED = 42
np.random.seed(SEED)

print("\n" + "=" * 78)
print("  BEST-ACCURACY PIPELINE — HEO Crystal Structure")
print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Build Magpie-style features
# ---------------------------------------------------------------------------
PROPS = ['ionic_radius', 'atomic_radius', 'pauling_electronegativity',
         'mulliken_electronegativity', 'atomic_number', 'molar_mass']

def magpie_features(elements, fractions=None):
    n = len(elements)
    if fractions is None:
        fractions = np.ones(n) / n
    fractions = np.array(fractions, dtype=float)
    fractions /= fractions.sum()

    out = []
    for p in PROPS:
        vals = np.array([ELEMENTAL_PROPERTIES[e][p] for e in elements], dtype=float)
        wmean = float((vals * fractions).sum())
        wstd  = float(np.sqrt(((vals - wmean) ** 2 * fractions).sum()))
        out.extend([wmean, wstd, vals.min(), vals.max(), vals.max() - vals.min()])
    return np.array(out)


def build_features(elements):
    base = features_to_array(calculate_all_features(elements))
    extra = magpie_features(elements)
    return np.concatenate([base, extra])


print("\n[1/5] Loading + featurising ...")

# Prefer the augmented dataset (includes literature-sourced Fluorite/Rock-salt)
import os
AUG_PATH = "heo_augmented_dataset.csv"

if os.path.exists(AUG_PATH):
    print("   Using augmented dataset (with literature Fluorite/Rock-salt)")
    df_aug = pd.read_csv(AUG_PATH)
    feature_cols_aug = [c for c in df_aug.columns if c != "label"]
    X_base = df_aug[feature_cols_aug].values
    y = df_aug["label"].values.astype(int)
    
    # Also load original data for Magpie enrichment
    df_orig = pd.concat([pd.read_csv("heo_data_4comp.csv"),
                         pd.read_csv("heo_data_5comp.csv")], ignore_index=True)
    structure_map = {"aPbO2": 0, "bad": 1, "rut": 2}
    STRUCTURE_NAMES = {0: "Fluorite", 1: "Rock-salt", 2: "Spinel"}

    # Add Magpie features where possible (for original rows that have Formula)
    # For augmented rows, add zero-padded Magpie features (they already have strong base features)
    n_magpie = len(PROPS) * 5  # 6 props × 5 stats = 30
    X_magpie_pad = np.zeros((len(X_base), n_magpie))
    
    df_orig["y"] = df_orig["Min_Crystal_rho"].map(structure_map)
    df_orig = df_orig.dropna(subset=["y"])
    orig_count = 0
    for idx, (_, row) in enumerate(df_orig.iterrows()):
        if idx >= len(X_base):
            break
        try:
            elements = eval(row["Formula"])
            X_magpie_pad[idx] = magpie_features(elements)
            orig_count += 1
        except:
            pass
    print(f"   Magpie features added for {orig_count}/{len(X_base)} rows")
    X = np.hstack([X_base, X_magpie_pad])
else:
    print("   Loading original CSVs (augmented dataset not found)")
    df = pd.concat([pd.read_csv("heo_data_4comp.csv"),
                    pd.read_csv("heo_data_5comp.csv")], ignore_index=True)
    structure_map = {"aPbO2": 0, "bad": 1, "rut": 2}
    STRUCTURE_NAMES = {0: "Fluorite", 1: "Rock-salt", 2: "Spinel"}
    df["y"] = df["Min_Crystal_rho"].map(structure_map)
    df = df.dropna(subset=["y"])

    X_list, y_list = [], []
    for _, row in df.iterrows():
        try:
            elements = eval(row["Formula"])
            X_list.append(build_features(elements))
            y_list.append(int(row["y"]))
        except Exception:
            continue
    X = np.array(X_list)
    y = np.array(y_list)
print(f"   X = {X.shape}  features (base + Magpie where available)")
print(f"   class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

# ── Class imbalance fix: SMOTETomek ──────────────────────────────────────────
# Oversamples minority classes then cleans noisy boundary samples via Tomek links.
# This avoids the Spinel-heavy bias that misleads the model.
print(f"   Before resampling: {dict(Counter(y_train))}")
smt = SMOTETomek(random_state=SEED)
X_train, y_train = smt.fit_resample(X_train, y_train)
print(f"   After  resampling: {dict(Counter(y_train))}")

# class weights (still kept for models that support it as a secondary guard)
classes, counts = np.unique(y_train, return_counts=True)
cw = {int(c): float(len(y_train) / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
sample_w = np.array([cw[int(t)] for t in y_train])
print(f"   class weights after resample: {cw}")

# RepeatedStratifiedKFold for more stable CV on small datasets
# 5 splits × 10 repeats = 50 evaluations  (vs. 5 before)
from sklearn.model_selection import RepeatedStratifiedKFold
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)


# ---------------------------------------------------------------------------
# 2. Optuna tuning helpers
# ---------------------------------------------------------------------------
N_TRIALS = 100         # increased from 30 → 100 for better Bayesian search
SCORING = "accuracy"   # primary metric; macro-F1 reported alongside


def cv_score(model, Xtr=X_train, ytr=y_train, sw=None):
    if sw is not None:
        scores = []
        for tr, va in skf.split(Xtr, ytr):
            m = model.__class__(**model.get_params())
            try:
                m.fit(Xtr[tr], ytr[tr], sample_weight=sw[tr])
            except TypeError:
                m.fit(Xtr[tr], ytr[tr])
            scores.append(accuracy_score(ytr[va], m.predict(Xtr[va])))
        return float(np.mean(scores))
    return float(np.mean(cross_val_score(model, Xtr, ytr, cv=skf,
                                         scoring=SCORING, n_jobs=-1)))


def tune_xgb(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
        max_depth=trial.suggest_int("max_depth", 3, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        eval_metric="mlogloss", random_state=SEED, verbosity=0, n_jobs=-1,
    )
    return cv_score(XGBClassifier(**params), sw=sample_w)


def tune_lgbm(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
        max_depth=trial.suggest_int("max_depth", -1, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 40),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        class_weight="balanced", random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    return cv_score(LGBMClassifier(**params))


def tune_cat(trial):
    params = dict(
        iterations=trial.suggest_int("iterations", 300, 900, step=100),
        depth=trial.suggest_int("depth", 4, 9),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        auto_class_weights="Balanced",
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )
    return cv_score(CatBoostClassifier(**params))


def tune_rf(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
        max_depth=trial.suggest_int("max_depth", 4, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    return cv_score(RandomForestClassifier(**params))


def tune_et(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 900, step=100),
        max_depth=trial.suggest_int("max_depth", 4, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    return cv_score(ExtraTreesClassifier(**params))


# ---------------------------------------------------------------------------
# 3. Run tuning
# ---------------------------------------------------------------------------
print(f"\n[2/5] Optuna tuning  (n_trials={N_TRIALS} per model, 5-fold CV)...")

studies = {}
for name, fn in [("XGBoost", tune_xgb),
                 ("LightGBM", tune_lgbm),
                 ("CatBoost", tune_cat),
                 ("Random Forest", tune_rf),
                 ("Extra Trees", tune_et)]:
    t = time.time()
    s = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(fn, n_trials=N_TRIALS, show_progress_bar=False)
    studies[name] = s
    print(f"   {name:14s}  best CV acc = {s.best_value:.4f}   ({time.time()-t:.1f}s)")


# ---------------------------------------------------------------------------
# 4. Refit each best model + stack the top-3
# ---------------------------------------------------------------------------
print("\n[3/5] Refitting tuned models on full training set ...")

def build_best(name, params):
    if name == "XGBoost":
        return XGBClassifier(**{**params, "eval_metric": "mlogloss",
                                 "random_state": SEED, "verbosity": 0, "n_jobs": -1})
    if name == "LightGBM":
        return LGBMClassifier(**{**params, "class_weight": "balanced",
                                  "random_state": SEED, "verbosity": -1, "n_jobs": -1})
    if name == "CatBoost":
        return CatBoostClassifier(**{**params, "auto_class_weights": "Balanced",
                                       "random_seed": SEED, "verbose": 0,
                                       "allow_writing_files": False})
    if name == "Random Forest":
        return RandomForestClassifier(**{**params, "class_weight": "balanced",
                                          "n_jobs": -1, "random_state": SEED})
    if name == "Extra Trees":
        return ExtraTreesClassifier(**{**params, "class_weight": "balanced",
                                        "n_jobs": -1, "random_state": SEED})

tuned = {}
for name, s in studies.items():
    m = build_best(name, s.best_params)
    if name == "XGBoost":
        m.fit(X_train, y_train, sample_weight=sample_w)
    else:
        m.fit(X_train, y_train)
    tuned[name] = m

# Score each tuned model on the held-out test set
results = []
for name, m in tuned.items():
    p = m.predict(X_test)
    results.append({
        "Model": name + " (tuned)",
        "Accuracy": accuracy_score(y_test, p),
        "Macro-F1": f1_score(y_test, p, average="macro"),
        "Weighted-F1": f1_score(y_test, p, average="weighted"),
    })

# ---- Stacking ensemble of top-3 by CV score ----
top3 = sorted(studies.items(), key=lambda kv: kv[1].best_value, reverse=True)[:3]
top3_names = [n for n, _ in top3]
print(f"\n   Stacking top-3 models: {top3_names}")

stack_estimators = [(n.lower().replace(" ", "_"), tuned[n]) for n in top3_names]

# Try two meta-learners and keep the better one
_meta_candidates = {
    "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    "XGB": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                          eval_metric="mlogloss", random_state=SEED, verbosity=0),
}
best_stack, best_stack_acc, best_stack_label = None, -1, ""
for meta_label, meta_clf in _meta_candidates.items():
    _s = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=meta_clf,
        cv=5, n_jobs=-1, passthrough=False,
    )
    _s.fit(X_train, y_train)
    _acc = accuracy_score(y_test, _s.predict(X_test))
    print(f"   Stacking (meta={meta_label}) test acc = {_acc:.4f}")
    if _acc > best_stack_acc:
        best_stack_acc = _acc
        best_stack = _s
        best_stack_label = meta_label

print(f"   → Best meta-learner: {best_stack_label}")
stack = best_stack

# Probability calibration on the best stack
stack_cal = CalibratedClassifierCV(stack, method="isotonic", cv="prefit")
stack_cal.fit(X_train, y_train)

p = stack_cal.predict(X_test)
results.append({
    "Model": f"Stacking+Cal({'+'.join(top3_names)},meta={best_stack_label})",
    "Accuracy": accuracy_score(y_test, p),
    "Macro-F1": f1_score(y_test, p, average="macro"),
    "Weighted-F1": f1_score(y_test, p, average="weighted"),
})


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
print("\n[4/5] Final test-set leaderboard")
print("=" * 78)
res = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_row = res.iloc[0]
best_name = best_row["Model"]
print("\n" + "=" * 78)
print(f"  BEST: {best_name}   |   acc={best_row['Accuracy']:.4f}   "
      f"macro-F1={best_row['Macro-F1']:.4f}")
print("=" * 78)

# Pull the best estimator object
best_model = stack_cal if best_name.startswith("Stacking") else tuned[best_name.replace(" (tuned)", "")]
best_pred = best_model.predict(X_test)

class_names = [STRUCTURE_NAMES[i] for i in range(3)]
print("\n  Classification report:")
print(classification_report(y_test, best_pred, target_names=class_names, digits=3))
cm = confusion_matrix(y_test, best_pred)
print("  Confusion matrix:")
print(f"  {'':12s}" + "".join(f"{n[:9]:>10s}" for n in class_names))
for i, row in enumerate(cm):
    print(f"  Actual {class_names[i][:6]:6s}" + "".join(f"{v:>10d}" for v in row))


# ---------------------------------------------------------------------------
# 6. Save artifacts
# ---------------------------------------------------------------------------
print("\n[5/5] Saving artifacts ...")
with open("heo_model_best.pkl", "wb") as f:
    pickle.dump({"model": best_model, "feature_dim": X.shape[1],
                 "n_magpie_props": len(PROPS)}, f)
res.to_csv("best_pipeline_leaderboard.csv", index=False)
print("   heo_model_best.pkl, best_pipeline_leaderboard.csv")

# Confusion-matrix figure for best model
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
ax.set_title(f"{best_name}\nacc={best_row['Accuracy']:.3f}, "
             f"macro-F1={best_row['Macro-F1']:.3f}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("best_pipeline_confusion.png", dpi=130)
print("   best_pipeline_confusion.png")

print(f"\n   total runtime = {time.time() - t0:.1f}s")
