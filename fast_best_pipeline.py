"""
Fast HEO Crystal-Structure Pipeline — best accuracy without external data
==========================================================================
Tricks used:
  1. Magpie-style features (50 total).
  2. ADASYN to balance the training set (proven to help here).
  3. Two top models tuned with Optuna (15 trials each, 5-fold CV).
  4. Hierarchical classifier:  (Spinel vs not-Spinel)  ->  (Fluorite vs Rock-salt)
  5. Direct 3-class stacking ensemble.
  6. Probability calibration on the winner.
Reports leaderboard, picks the best, saves it.
"""

import os, time, pickle, warnings
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN

from feature_engineering import calculate_all_features, features_to_array
from elemental_properties import ELEMENTAL_PROPERTIES

t0 = time.time()
SEED = 42
np.random.seed(SEED)

print("=" * 78)
print("  FAST PIPELINE — hierarchical + stacking + calibration")
print("=" * 78)


# ---------- features ----------
PROPS = ['ionic_radius', 'atomic_radius', 'pauling_electronegativity',
         'mulliken_electronegativity', 'atomic_number', 'molar_mass']

def magpie(elements):
    n = len(elements); fr = np.full(n, 1/n)
    out = []
    for p in PROPS:
        v = np.array([ELEMENTAL_PROPERTIES[e][p] for e in elements], dtype=float)
        m = float((v*fr).sum()); s = float(np.sqrt(((v-m)**2*fr).sum()))
        out.extend([m, s, v.min(), v.max(), v.max()-v.min()])
    return np.array(out)

def feats(elements):
    return np.concatenate([features_to_array(calculate_all_features(elements)),
                            magpie(elements)])

print("\n[1] Loading data ...")
df = pd.concat([pd.read_csv("heo_data_4comp.csv"),
                pd.read_csv("heo_data_5comp.csv")], ignore_index=True)
SMAP = {"aPbO2": 0, "bad": 1, "rut": 2}
SNAMES = {0: "Fluorite", 1: "Rock-salt", 2: "Spinel"}
df["y"] = df["Min_Crystal_rho"].map(SMAP)
df = df.dropna(subset=["y"])

X, y = [], []
for _, row in df.iterrows():
    try:
        X.append(feats(eval(row["Formula"])))
        y.append(int(row["y"]))
    except Exception:
        pass
X = np.array(X); y = np.array(y)
print(f"   X={X.shape}, y counts={dict(zip(*np.unique(y, return_counts=True)))}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30,
                                            random_state=SEED, stratify=y)

# ADASYN balanced training set
ada = ADASYN(random_state=SEED, n_neighbors=5)
X_bal, y_bal = ada.fit_resample(X_tr, y_tr)
print(f"   after ADASYN: {dict(zip(*np.unique(y_bal, return_counts=True)))}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# ---------- Optuna tune XGBoost + LightGBM (15 trials each) ----------
N_TRIALS = 15

def cv_acc(model):
    return float(np.mean(cross_val_score(model, X_bal, y_bal, cv=skf,
                                          scoring="accuracy", n_jobs=-1)))

def xgb_obj(t):
    return cv_acc(XGBClassifier(
        n_estimators=t.suggest_int("n_estimators", 300, 800, step=100),
        max_depth=t.suggest_int("max_depth", 4, 8),
        learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        subsample=t.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.7, 1.0),
        min_child_weight=t.suggest_int("min_child_weight", 1, 6),
        eval_metric="mlogloss", random_state=SEED, verbosity=0, n_jobs=-1))

def lgb_obj(t):
    return cv_acc(LGBMClassifier(
        n_estimators=t.suggest_int("n_estimators", 300, 800, step=100),
        max_depth=t.suggest_int("max_depth", -1, 12),
        learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 100),
        subsample=t.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.7, 1.0),
        min_child_samples=t.suggest_int("min_child_samples", 5, 30),
        random_state=SEED, verbosity=-1, n_jobs=-1))

print(f"\n[2] Tuning (Optuna, {N_TRIALS} trials each) ...")
t = time.time()
sx = optuna.create_study(direction="maximize",
                         sampler=optuna.samplers.TPESampler(seed=SEED))
sx.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
print(f"   XGBoost  best CV acc = {sx.best_value:.4f}   ({time.time()-t:.1f}s)")

t = time.time()
sl = optuna.create_study(direction="maximize",
                         sampler=optuna.samplers.TPESampler(seed=SEED))
sl.optimize(lgb_obj, n_trials=N_TRIALS, show_progress_bar=False)
print(f"   LightGBM best CV acc = {sl.best_value:.4f}   ({time.time()-t:.1f}s)")


def make_xgb(extra=None):
    p = {**sx.best_params, "eval_metric": "mlogloss", "random_state": SEED,
         "verbosity": 0, "n_jobs": -1}
    if extra: p.update(extra)
    return XGBClassifier(**p)

def make_lgb(extra=None):
    p = {**sl.best_params, "random_state": SEED, "verbosity": -1, "n_jobs": -1}
    if extra: p.update(extra)
    return LGBMClassifier(**p)


# ---------- evaluate candidates on test ----------
results = []
trained = {}

print("\n[3] Fitting models on full ADASYN-balanced training set ...")

# A) Tuned XGBoost
xgb = make_xgb(); xgb.fit(X_bal, y_bal)
p = xgb.predict(X_te)
results.append(("Tuned XGBoost", accuracy_score(y_te, p),
                f1_score(y_te, p, average="macro"))); trained["Tuned XGBoost"] = (xgb, p)

# B) Tuned LightGBM
lgb = make_lgb(); lgb.fit(X_bal, y_bal)
p = lgb.predict(X_te)
results.append(("Tuned LightGBM", accuracy_score(y_te, p),
                f1_score(y_te, p, average="macro"))); trained["Tuned LightGBM"] = (lgb, p)

# C) Random Forest (cheap, often strong)
rf = RandomForestClassifier(n_estimators=800, n_jobs=-1, random_state=SEED,
                            class_weight="balanced_subsample")
rf.fit(X_bal, y_bal); p = rf.predict(X_te)
results.append(("Random Forest", accuracy_score(y_te, p),
                f1_score(y_te, p, average="macro"))); trained["Random Forest"] = (rf, p)

# D) Stacking (3-class direct)
stack = StackingClassifier(
    estimators=[("xgb", make_xgb()), ("lgb", make_lgb()), ("rf", rf)],
    final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
    cv=skf, n_jobs=-1)
stack.fit(X_bal, y_bal); p = stack.predict(X_te)
results.append(("Stacking (3-class)", accuracy_score(y_te, p),
                f1_score(y_te, p, average="macro"))); trained["Stacking (3-class)"] = (stack, p)


# E) Hierarchical: Stage1 = Spinel vs Not-Spinel (binary)
#                  Stage2 = Fluorite vs Rock-salt  (binary, only on not-Spinel)
print("\n[4] Hierarchical classifier (Spinel-vs-rest -> Fluorite-vs-Rocksalt) ...")

# Stage 1 binary labels: 1 if Spinel else 0
y_s1 = (y_tr == 2).astype(int)
ada1 = ADASYN(random_state=SEED, n_neighbors=5)
X_s1, ys1 = ada1.fit_resample(X_tr, y_s1)
clf1 = make_lgb({"is_unbalance": True})  # binary tuned LGBM
clf1.fit(X_s1, ys1)

# Stage 2: only train on non-Spinel rows
mask2 = (y_tr != 2)
X_tr2 = X_tr[mask2]; y_tr2 = (y_tr[mask2] == 0).astype(int)  # 1=Fluorite, 0=Rock-salt
ada2 = ADASYN(random_state=SEED, n_neighbors=5)
X_s2, ys2 = ada2.fit_resample(X_tr2, y_tr2)
clf2 = make_xgb()
clf2.fit(X_s2, ys2)

def hier_predict(Xq):
    s1 = clf1.predict(Xq)
    out = np.full(Xq.shape[0], 2)            # default Spinel
    not_s = np.where(s1 == 0)[0]
    if len(not_s) > 0:
        s2 = clf2.predict(Xq[not_s])         # 1=Fluorite, 0=Rock-salt
        out[not_s] = np.where(s2 == 1, 0, 1)
    return out

p = hier_predict(X_te)
results.append(("Hierarchical 2-stage", accuracy_score(y_te, p),
                f1_score(y_te, p, average="macro"))); trained["Hierarchical 2-stage"] = (None, p)


# F) Calibrated stacking
print("\n[5] Calibrating best model ...")
res_sorted = sorted(results, key=lambda x: x[1], reverse=True)
best_name = res_sorted[0][0]

if best_name != "Hierarchical 2-stage":
    base = trained[best_name][0]
    cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=skf)
    cal.fit(X_bal, y_bal); p = cal.predict(X_te)
    results.append((f"Calibrated {best_name}",
                    accuracy_score(y_te, p),
                    f1_score(y_te, p, average="macro")))
    trained[f"Calibrated {best_name}"] = (cal, p)


# ---------- leaderboard ----------
res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Macro-F1"])\
            .sort_values("Accuracy", ascending=False).reset_index(drop=True)
print("\n" + "=" * 78)
print("  LEADERBOARD (test set)")
print("=" * 78)
print(res_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_name = res_df.iloc[0]["Model"]
best_pred = trained[best_name][1]
class_names = [SNAMES[i] for i in range(3)]
print(f"\nBEST: {best_name}   acc={res_df.iloc[0]['Accuracy']:.4f}   "
      f"macro-F1={res_df.iloc[0]['Macro-F1']:.4f}")
print("\nClassification report:")
print(classification_report(y_te, best_pred, target_names=class_names, digits=3))
cm = confusion_matrix(y_te, best_pred)
print("Confusion matrix:")
print(f"{'':10s}" + "".join(f"{n[:9]:>10s}" for n in class_names))
for i, row in enumerate(cm):
    print(f"{class_names[i][:8]:10s}" + "".join(f"{v:>10d}" for v in row))


# ---------- save ----------
print("\n[6] Saving artifacts ...")
if trained[best_name][0] is not None:
    with open("heo_model_best.pkl", "wb") as f:
        pickle.dump({"model": trained[best_name][0], "feature_dim": X.shape[1]}, f)
    print("   heo_model_best.pkl")
res_df.to_csv("fast_pipeline_leaderboard.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
ax.set_title(f"{best_name}\nacc={res_df.iloc[0]['Accuracy']:.3f}")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("fast_pipeline_confusion.png", dpi=130)
print("   fast_pipeline_confusion.png")

print(f"\n   total runtime = {time.time()-t0:.1f}s")
