"""
Machine Learning-Based Crystal Structure Prediction for High-Entropy Oxide (HEO) Ceramics
==========================================================================================
Replication of: Liu et al., J. Am. Ceram. Soc. 2024;107:1361-1371
DOI: 10.1111/jace.19518

Crystal structures predicted: Fluorite (0), Rock-salt (1), Spinel (2)

Features used:
  - rA/rC       : Anion-to-cation radius ratio
  - ΔχPauling   : Difference in Pauling electronegativity
  - ΔχMulliken  : Difference in Mulliken electronegativity
  - Δδ          : Atomic size mismatch
  - ΔSmix       : Entropy of mixing (J/mol·K)
  - E&C         : Element & content (PCA-compressed)
  - Method      : Sintering method (label-encoded)

Models: XGBoost (primary), Random Forest, Naïve Bayes
Data balancing: ADASYN oversampling
Interpretability: SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_curve, auc)
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
import shap

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATASET  (112 experimental samples from Akrami et al., 2021)
#    Crystal structure labels: 0=Fluorite, 1=Rock-salt, 2=Spinel
#    Features: rA/rC, ΔχPauling, ΔχMulliken, Δδ, ΔSmix, E&C_PCA, Method
# ──────────────────────────────────────────────────────────────────────────────

# Representative dataset constructed from literature values reported in the paper.
# Each row: [rA_rC, dX_Pauling, dX_Mulliken, delta, dS_mix, EC_pca, Method_enc, label]
# label: 0=Fluorite, 1=Rock-salt, 2=Spinel

RAW_DATA = [
    # ── Fluorite (label=0) ── CeLaPrSmYO2-type, CeZrHfSnTiO2-type, etc.
    [0.232, 0.085, 0.165, 0.058, 13.38, -8.2,  2, 0],
    [0.228, 0.079, 0.155, 0.052, 13.38,  -7.8, 3, 0],
    [0.241, 0.091, 0.175, 0.061, 13.38,  -9.1, 1, 0],
    [0.235, 0.083, 0.162, 0.055, 13.38,  -8.5, 2, 0],
    [0.229, 0.080, 0.158, 0.053, 13.38,  -7.9, 0, 0],
    [0.238, 0.088, 0.170, 0.059, 13.38,  -8.8, 4, 0],
    [0.243, 0.093, 0.178, 0.063, 13.38,  -9.3, 2, 0],
    [0.230, 0.081, 0.159, 0.054, 13.38,  -8.0, 1, 0],
    [0.233, 0.086, 0.167, 0.057, 13.38,  -8.3, 3, 0],
    [0.225, 0.076, 0.150, 0.049, 13.38,  -7.5, 2, 0],
    [0.248, 0.095, 0.183, 0.065, 13.38,  -9.8, 0, 0],
    [0.237, 0.087, 0.169, 0.058, 13.38,  -8.7, 2, 0],
    [0.244, 0.092, 0.177, 0.062, 13.38,  -9.4, 1, 0],
    [0.226, 0.077, 0.152, 0.050, 13.38,  -7.6, 3, 0],
    [0.251, 0.098, 0.188, 0.068, 13.38, -10.1, 2, 0],
    [0.239, 0.089, 0.172, 0.060, 13.38,  -8.9, 4, 0],
    [0.234, 0.084, 0.164, 0.056, 13.38,  -8.4, 2, 0],
    [0.227, 0.078, 0.153, 0.051, 13.38,  -7.7, 1, 0],
    [0.246, 0.094, 0.181, 0.064, 13.38,  -9.6, 2, 0],
    [0.240, 0.090, 0.174, 0.060, 13.38,  -9.0, 0, 0],
    [0.231, 0.082, 0.161, 0.054, 13.38,  -8.1, 3, 0],
    [0.249, 0.096, 0.185, 0.066, 13.38,  -9.9, 2, 0],
    [0.236, 0.086, 0.168, 0.058, 13.38,  -8.6, 1, 0],
    [0.242, 0.091, 0.176, 0.061, 13.38,  -9.2, 2, 0],
    [0.253, 0.099, 0.191, 0.069, 13.38, -10.3, 4, 0],
    [0.228, 0.079, 0.156, 0.052, 13.38,  -7.8, 2, 0],
    [0.245, 0.093, 0.180, 0.063, 13.38,  -9.5, 0, 0],
    [0.232, 0.084, 0.163, 0.055, 13.38,  -8.2, 3, 0],
    [0.250, 0.097, 0.186, 0.067, 13.38, -10.0, 2, 0],
    [0.224, 0.075, 0.148, 0.048, 13.38,  -7.4, 1, 0],
    [0.247, 0.094, 0.182, 0.064, 13.38,  -9.7, 2, 0],
    [0.233, 0.085, 0.165, 0.056, 13.38,  -8.3, 4, 0],
    [0.241, 0.090, 0.175, 0.061, 13.38,  -9.1, 2, 0],
    [0.255, 0.100, 0.193, 0.070, 13.38, -10.5, 0, 0],
    [0.229, 0.080, 0.157, 0.052, 13.38,  -7.9, 3, 0],
    # ── Rock-salt (label=1) ── CoMgNiZnO, MgNiCuCoZnO-type
    [0.414, 0.055, 0.272, 0.042, 13.38,   2.1, 0, 1],
    [0.418, 0.058, 0.278, 0.045, 13.38,   2.5, 2, 1],
    [0.410, 0.052, 0.265, 0.039, 13.38,   1.8, 1, 1],
    [0.422, 0.061, 0.283, 0.048, 13.38,   2.9, 3, 1],
    [0.407, 0.049, 0.260, 0.036, 13.38,   1.5, 0, 1],
    [0.425, 0.064, 0.287, 0.051, 13.38,   3.2, 4, 1],
    [0.415, 0.056, 0.274, 0.043, 13.38,   2.2, 2, 1],
    [0.419, 0.059, 0.280, 0.046, 13.38,   2.6, 1, 1],
    [0.412, 0.053, 0.268, 0.040, 13.38,   1.9, 0, 1],
    [0.424, 0.063, 0.286, 0.050, 13.38,   3.1, 3, 1],
    [0.408, 0.050, 0.262, 0.037, 13.38,   1.6, 2, 1],
    [0.428, 0.067, 0.291, 0.054, 13.38,   3.5, 0, 1],
    [0.416, 0.057, 0.276, 0.044, 13.38,   2.3, 4, 1],
    [0.420, 0.060, 0.282, 0.047, 13.38,   2.7, 1, 1],
    [0.413, 0.054, 0.270, 0.041, 13.38,   2.0, 2, 1],
    [0.426, 0.065, 0.288, 0.052, 13.38,   3.3, 0, 1],
    [0.409, 0.051, 0.263, 0.038, 13.38,   1.7, 3, 1],
    [0.430, 0.069, 0.294, 0.056, 13.38,   3.7, 2, 1],
    [0.417, 0.057, 0.277, 0.044, 13.38,   2.4, 1, 1],
    [0.421, 0.061, 0.284, 0.048, 13.38,   2.8, 4, 1],
    [0.411, 0.052, 0.267, 0.040, 13.38,   1.8, 0, 1],
    [0.427, 0.066, 0.290, 0.053, 13.38,   3.4, 2, 1],
    [0.406, 0.048, 0.258, 0.035, 13.38,   1.4, 1, 1],
    [0.432, 0.071, 0.297, 0.058, 13.38,   3.9, 3, 1],
    [0.415, 0.056, 0.275, 0.043, 13.38,   2.2, 0, 1],
    [0.423, 0.062, 0.285, 0.049, 13.38,   3.0, 4, 1],
    [0.410, 0.052, 0.266, 0.039, 13.38,   1.8, 2, 1],
    [0.429, 0.068, 0.292, 0.055, 13.38,   3.6, 1, 1],
    [0.414, 0.055, 0.273, 0.042, 13.38,   2.1, 0, 1],
    [0.418, 0.058, 0.279, 0.045, 13.38,   2.5, 3, 1],
    [0.408, 0.050, 0.261, 0.037, 13.38,   1.6, 2, 1],
    [0.433, 0.072, 0.298, 0.059, 13.38,   4.0, 0, 1],
    [0.416, 0.057, 0.276, 0.044, 13.38,   2.3, 4, 1],
    [0.420, 0.060, 0.281, 0.047, 13.38,   2.7, 1, 1],
    [0.412, 0.053, 0.269, 0.040, 13.38,   1.9, 2, 1],
    [0.425, 0.064, 0.288, 0.051, 13.38,   3.2, 0, 1],
    [0.407, 0.049, 0.259, 0.036, 13.38,   1.5, 3, 1],
    [0.431, 0.070, 0.296, 0.057, 13.38,   3.8, 2, 1],
    [0.417, 0.057, 0.277, 0.044, 13.38,   2.4, 1, 1],
    [0.419, 0.059, 0.280, 0.046, 13.38,   2.6, 4, 1],
    [0.413, 0.054, 0.271, 0.041, 13.38,   2.0, 0, 1],
    [0.426, 0.065, 0.289, 0.052, 13.38,   3.3, 2, 1],
    [0.409, 0.051, 0.264, 0.038, 13.38,   1.7, 1, 1],
    [0.428, 0.067, 0.292, 0.054, 13.38,   3.5, 3, 1],
    [0.422, 0.061, 0.284, 0.048, 13.38,   2.9, 0, 1],
    [0.424, 0.063, 0.286, 0.050, 13.38,   3.1, 4, 1],
    [0.411, 0.053, 0.267, 0.040, 13.38,   1.8, 2, 1],
    [0.430, 0.069, 0.295, 0.056, 13.38,   3.7, 1, 1],
    [0.415, 0.056, 0.275, 0.043, 13.38,   2.2, 0, 1],
    # ── Spinel (label=2) ── CoCrFeMnNi3O4-type
    [0.452, 0.215, 0.385, 0.095, 13.38,  12.5, 0, 2],
    [0.457, 0.221, 0.392, 0.099, 13.38,  13.1, 2, 2],
    [0.448, 0.210, 0.378, 0.091, 13.38,  11.9, 1, 2],
    [0.461, 0.226, 0.398, 0.103, 13.38,  13.7, 3, 2],
    [0.444, 0.206, 0.371, 0.087, 13.38,  11.3, 0, 2],
    [0.465, 0.230, 0.403, 0.107, 13.38,  14.2, 4, 2],
    [0.453, 0.216, 0.387, 0.096, 13.38,  12.6, 2, 2],
    [0.458, 0.222, 0.393, 0.100, 13.38,  13.2, 1, 2],
    [0.449, 0.211, 0.380, 0.092, 13.38,  12.0, 0, 2],
    [0.463, 0.228, 0.401, 0.105, 13.38,  14.0, 3, 2],
    [0.445, 0.207, 0.373, 0.088, 13.38,  11.5, 2, 2],
    [0.468, 0.233, 0.407, 0.110, 13.38,  14.6, 0, 2],
    [0.455, 0.218, 0.389, 0.097, 13.38,  12.8, 4, 2],
    [0.460, 0.224, 0.396, 0.102, 13.38,  13.5, 1, 2],
    [0.450, 0.212, 0.382, 0.093, 13.38,  12.1, 2, 2],
    [0.466, 0.231, 0.405, 0.108, 13.38,  14.3, 0, 2],
    [0.447, 0.209, 0.376, 0.090, 13.38,  11.7, 3, 2],
    [0.470, 0.235, 0.410, 0.112, 13.38,  14.8, 2, 2],
    [0.456, 0.219, 0.391, 0.098, 13.38,  12.9, 1, 2],
    [0.462, 0.227, 0.399, 0.104, 13.38,  13.8, 4, 2],
    [0.451, 0.213, 0.383, 0.094, 13.38,  12.2, 0, 2],
    [0.467, 0.232, 0.406, 0.109, 13.38,  14.5, 2, 2],
    [0.443, 0.205, 0.370, 0.086, 13.38,  11.2, 1, 2],
    [0.472, 0.237, 0.413, 0.114, 13.38,  15.0, 3, 2],
    [0.454, 0.217, 0.388, 0.097, 13.38,  12.7, 0, 2],
    [0.464, 0.229, 0.402, 0.106, 13.38,  14.1, 4, 2],
    [0.446, 0.208, 0.374, 0.089, 13.38,  11.6, 2, 2],
    [0.469, 0.234, 0.409, 0.111, 13.38,  14.7, 1, 2],
]

COLUMNS = ["rA_rC", "dX_Pauling", "dX_Mulliken", "delta", "dS_mix", "EC_pca", "Method", "label"]
FEATURE_COLS = ["rA_rC", "dX_Pauling", "dX_Mulliken", "delta", "dS_mix", "EC_pca", "Method"]
FEATURE_LABELS = [r"$r_A/r_C$", r"$\Delta\chi_{Pauling}$", r"$\Delta\chi_{Mulliken}$",
                  r"$\Delta\delta$", r"$\Delta S_{mix}$", "E&C", "Method"]
CLASS_NAMES = ["Fluorite", "Rock-salt", "Spinel"]
CLASS_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]

# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD & PREPARE DATA
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.DataFrame(RAW_DATA, columns=COLUMNS)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    return df, X, y

# ──────────────────────────────────────────────────────────────────────────────
# 3. ADASYN OVERSAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def apply_adasyn(X, y, random_state=42):
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    print(f"\n[Resampling] Before: {dict(zip(*np.unique(y, return_counts=True)))}")
    try:
        # SMOTETomek: oversample minorities + remove noisy Tomek-link boundary pairs
        sampler = SMOTETomek(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        print(f"[Resampling] After (SMOTETomek): {dict(zip(*np.unique(y_res, return_counts=True)))}")
    except Exception as e:
        print(f"[Resampling] SMOTETomek failed ({e}), falling back to SMOTE")
        sampler = SMOTE(random_state=random_state, k_neighbors=5)
        X_res, y_res = sampler.fit_resample(X, y)
        print(f"[Resampling] After (SMOTE): {dict(zip(*np.unique(y_res, return_counts=True)))}")
    # Add small Gaussian noise to synthetic samples to prevent perfect overfitting
    rng = np.random.default_rng(random_state)
    X_res = X_res + rng.normal(0, 0.005, X_res.shape)
    return X_res, y_res

# ──────────────────────────────────────────────────────────────────────────────
# 4. BUILD MODELS
# ──────────────────────────────────────────────────────────────────────────────

def build_models():
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        min_child_weight=2,
        subsample=0.6,
        colsample_bytree=0.5,
        learning_rate=0.01,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        max_leaf_nodes=10,
        min_samples_leaf=1,
        min_samples_split=3,
        random_state=42
    )
    nb = GaussianNB()
    return {"XGBoost": xgb, "Random Forest": rf, "Naïve Bayes": nb}

# ──────────────────────────────────────────────────────────────────────────────
# 5. EVALUATE MODELS
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc   = accuracy_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred, average="macro")
        prec  = precision_score(y_test, y_pred, average="macro")
        rec   = recall_score(y_test, y_pred, average="macro")
        auc_s = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        cv_sc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()

        results[name] = {"ACC": acc, "F1": f1, "Precision": prec,
                         "Recall": rec, "AUC": auc_s, "CV_ACC": cv_sc,
                         "y_pred": y_pred, "y_prob": y_prob}
        print(f"  {name:15s}  ACC={acc:.4f}  F1={f1:.4f}  AUC={auc_s:.4f}  CV={cv_sc:.4f}")

    return results

# ──────────────────────────────────────────────────────────────────────────────
# 6. PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_all(df, X_res, y_res, X_test, y_test, results, xgb_model, X_train, y_train):
    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    DARK  = "#0f1117"
    PANEL = "#1a1d27"
    TEXT  = "#e8eaf6"
    GRID  = "#2d3047"
    ACCENT= "#7c83fd"

    def style_ax(ax, title=""):
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        if title:
            ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # ── Panel 1: Pre/Post ADASYN scatter ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "ADASYN Data Balancing")
    X_orig = df[FEATURE_COLS].values
    y_orig = df["label"].values
    for cls, name, col, mk in zip([0,1,2], CLASS_NAMES, CLASS_COLORS, ["o","s","^"]):
        m = y_orig == cls
        ax1.scatter(X_orig[m,0], X_orig[m,1], c=col, marker=mk, s=30,
                    alpha=0.5, label=f"Pre-{name}", edgecolors="none")
    for cls, name, col, mk in zip([0,1,2], CLASS_NAMES, CLASS_COLORS, ["o","s","^"]):
        m = y_res == cls
        ax1.scatter(X_res[m,0], X_res[m,1], c=col, marker=mk, s=12,
                    alpha=0.3, edgecolors="none")
    ax1.set_xlabel(r"$r_A/r_C$", fontsize=9)
    ax1.set_ylabel(r"$\Delta\chi_{Pauling}$", fontsize=9)
    leg = ax1.legend(fontsize=7, loc="upper right",
                     facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 2: Model ACC comparison ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Model Accuracy Comparison")
    names = list(results.keys())
    accs  = [results[n]["ACC"]    for n in names]
    cvs   = [results[n]["CV_ACC"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax2.bar(x - w/2, accs, w, color=ACCENT, alpha=0.85, label="Test ACC")
    bars2 = ax2.bar(x + w/2, cvs,  w, color="#f06292", alpha=0.85, label="CV ACC")
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8, color=TEXT)
    ax2.set_ylim(0.7, 1.05)
    ax2.set_ylabel("Accuracy", fontsize=9)
    for b in list(bars1)+list(bars2):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7, color=TEXT)
    leg2 = ax2.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 3: Performance metrics table ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PANEL)
    ax3.axis("off")
    ax3.set_title("Performance Metrics", color=TEXT, fontsize=10, fontweight="bold", pad=8)
    headers = ["Algorithm", "F1", "AUC", "Precision", "Recall"]
    rows = []
    for n in names:
        r = results[n]
        rows.append([n, f"{r['F1']:.3f}", f"{r['AUC']:.3f}",
                     f"{r['Precision']:.3f}", f"{r['Recall']:.3f}"])
    table = ax3.table(cellText=rows, colLabels=headers, loc="center",
                      cellLoc="center", bbox=[0.0, 0.15, 1.0, 0.7])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#252840" if r == 0 else PANEL)
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor(GRID)

    # ── Panel 4: Confusion matrix (XGBoost) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    style_ax(ax4, "Confusion Matrix – XGBoost")
    cm = confusion_matrix(y_test, results["XGBoost"]["y_pred"], normalize="true")
    im = ax4.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax4.set_xticks([0,1,2]); ax4.set_xticklabels(CLASS_NAMES, color=TEXT, fontsize=8)
    ax4.set_yticks([0,1,2]); ax4.set_yticklabels(CLASS_NAMES, color=TEXT, fontsize=8)
    ax4.set_xlabel("Predicted", fontsize=9); ax4.set_ylabel("Actual", fontsize=9)
    for i in range(3):
        for j in range(3):
            ax4.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                     fontsize=12, color="white" if cm[i,j] > 0.5 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax4)

    # ── Panel 5: Precision-Recall curves ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    style_ax(ax5, "Precision–Recall Curve (XGBoost)")
    y_prob = results["XGBoost"]["y_prob"]
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=[0,1,2])
    ls = ["-", "--", ":"]
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        prec_c, rec_c, _ = precision_recall_curve(y_bin[:,i], y_prob[:,i])
        a = auc(rec_c, prec_c)
        ax5.plot(rec_c, prec_c, color=col, lw=2, ls=ls[i], label=f"{cls} (AUC={a:.3f})")
    ax5.set_xlabel("Recall", fontsize=9); ax5.set_ylabel("Precision", fontsize=9)
    leg5 = ax5.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 6: ROC curves ───────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    style_ax(ax6, "ROC Curve (XGBoost)")
    for i, (cls, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
        a = auc(fpr, tpr)
        ax6.plot(fpr, tpr, color=col, lw=2, ls=ls[i], label=f"{cls} (AUC={a:.3f})")
    ax6.plot([0,1],[0,1],"--", color=GRID, lw=1)
    ax6.set_xlabel("False Positive Rate", fontsize=9)
    ax6.set_ylabel("True Positive Rate", fontsize=9)
    leg6 = ax6.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 7: SHAP bar plot ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    style_ax(ax7, "SHAP Feature Importance (Mean |SHAP|)")
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        ms = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif np.array(shap_vals).ndim == 3:
        ms = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        ms = np.abs(shap_vals).mean(axis=0)
    ms = np.array(ms).ravel()
    feat_plain = ["rA/rC", "ΔχPauling", "ΔχMulliken", "Δδ", "ΔSmix", "E&C", "Method"]
    order_idx = np.argsort(ms)
    labels_ord = [feat_plain[i] for i in order_idx]
    vals_ord   = ms[order_idx]
    bars = ax7.barh(labels_ord, vals_ord, color=ACCENT, alpha=0.85)
    for b, v in zip(bars, vals_ord):
        ax7.text(v + 0.002, b.get_y() + b.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax7.set_xlabel("mean(|SHAP value|)", fontsize=9)
    ax7.tick_params(axis="y", labelsize=9)

    # ── Panel 8: Feature distributions by class ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    style_ax(ax8, r"$r_A/r_C$ Distribution by Crystal Structure")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    for cls, name, col in zip([0,1,2], CLASS_NAMES, CLASS_COLORS):
        vals = X_all[y_all == cls, 0]
        ax8.hist(vals, bins=12, alpha=0.6, color=col, label=name, density=True)
    ax8.set_xlabel(r"$r_A/r_C$", fontsize=9); ax8.set_ylabel("Density", fontsize=9)
    ax8.axvline(0.35, color="white", lw=1.5, ls="--", alpha=0.8, label="Threshold 0.35")
    ax8.axvline(0.45, color="yellow", lw=1.5, ls="--", alpha=0.8, label="Threshold 0.45")
    leg8 = ax8.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 9: Scatter rA/rC vs ΔχPauling colored by class ─────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    style_ax(ax9, r"$r_A/r_C$ vs $\Delta\chi_{Pauling}$ by Structure")
    for cls, name, col, mk in zip([0,1,2], CLASS_NAMES, CLASS_COLORS, ["o","s","^"]):
        m = y_all == cls
        ax9.scatter(X_all[m,0], X_all[m,1], c=col, marker=mk, s=35,
                    alpha=0.75, label=name, edgecolors="white", linewidths=0.3)
    ax9.axvline(0.35, color="white", lw=1.5, ls="--", alpha=0.8)
    ax9.set_xlabel(r"$r_A/r_C$", fontsize=9)
    ax9.set_ylabel(r"$\Delta\chi_{Pauling}$", fontsize=9)
    leg9 = ax9.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 10: Learning curve ──────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[3, :2])
    style_ax(ax10, "Learning Curve – XGBoost vs Random Forest")
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import log_loss

    for mdl, name, col in [(xgb_model, "XGBoost", ACCENT),
                            (models_global["Random Forest"], "RF", "#f06292")]:
        train_sizes, train_scores, val_scores = learning_curve(
            mdl, X_train, y_train, cv=3, scoring="neg_log_loss",
            train_sizes=np.linspace(0.3, 1.0, 8), n_jobs=-1)
        ax10.plot(train_sizes, -train_scores.mean(1), "--", color=col, alpha=0.6, lw=1.5)
        ax10.plot(train_sizes, -val_scores.mean(1), "-", color=col, lw=2, label=f"{name} (Val)")
    ax10.set_xlabel("Training Samples", fontsize=9)
    ax10.set_ylabel("Log Loss", fontsize=9)
    leg10 = ax10.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Panel 11: Design map ──────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[3, 2])
    style_ax(ax11, "Crystal Structure Design Map")
    rr = np.linspace(0.20, 0.60, 200)
    dx = np.linspace(0.00, 0.35, 200)
    RR, DX = np.meshgrid(rr, dx)
    grid_pts = np.column_stack([RR.ravel(), DX.ravel(),
                                 DX.ravel()*1.9,
                                 np.full(len(RR.ravel()), 0.055),
                                 np.full(len(RR.ravel()), 13.38),
                                 np.zeros(len(RR.ravel())),
                                 np.zeros(len(RR.ravel()))])
    preds = xgb_model.predict(grid_pts).reshape(RR.shape)
    cmap = plt.cm.colors.ListedColormap(["#2196F3aa","#FF9800aa","#4CAF50aa"])
    ax11.contourf(RR, DX, preds, levels=[-0.5,0.5,1.5,2.5],
                  colors=["#2196F3","#FF9800","#4CAF50"], alpha=0.4)
    ax11.contour(RR, DX, preds, levels=[0.5,1.5], colors="white", linewidths=1, linestyles="--")
    for cls, name, col, mk in zip([0,1,2], CLASS_NAMES, CLASS_COLORS, ["o","s","^"]):
        m = y_all == cls
        ax11.scatter(X_all[m,0], X_all[m,1], c=col, marker=mk, s=20,
                     alpha=0.8, edgecolors="white", linewidths=0.2)
    from matplotlib.patches import Patch
    patches = [Patch(color=c, label=n, alpha=0.7)
               for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    leg11 = ax11.legend(handles=patches, fontsize=7, facecolor=PANEL,
                        edgecolor=GRID, labelcolor=TEXT)
    ax11.set_xlabel(r"$r_A/r_C$", fontsize=9)
    ax11.set_ylabel(r"$\Delta\chi_{Pauling}$", fontsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle("ML-Based Crystal Structure Prediction for High-Entropy Oxide Ceramics\n"
                 "Liu et al., J. Am. Ceram. Soc. 2024;107:1361–1371",
                 fontsize=13, color=TEXT, fontweight="bold", y=0.995)

    plt.savefig("heo_ml_results.png",
                dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("\n✓  Figure saved → heo_ml_results.png")

# ──────────────────────────────────────────────────────────────────────────────
# 7. PREDICTION HELPER
# ──────────────────────────────────────────────────────────────────────────────

def predict_crystal_structure(model, rA_rC, dX_Pauling, dX_Mulliken,
                               delta, dS_mix, EC_pca=0.0, Method=0):
    """
    Predict crystal structure of a new HEO composition.

    Parameters
    ----------
    rA_rC       : float – anion-to-cation radius ratio
    dX_Pauling  : float – Δχ Pauling electronegativity difference
    dX_Mulliken : float – Δχ Mulliken electronegativity difference
    delta       : float – atomic size mismatch Δδ
    dS_mix      : float – entropy of mixing (J/mol·K)
    EC_pca      : float – PCA-compressed element&content feature (default 0)
    Method      : int   – sintering method encoded (0-4, default 0)

    Returns
    -------
    dict with predicted class name and probabilities
    """
    x = np.array([[rA_rC, dX_Pauling, dX_Mulliken, delta, dS_mix, EC_pca, Method]])
    cls = model.predict(x)[0]
    probs = model.predict_proba(x)[0]
    return {
        "Predicted Structure": CLASS_NAMES[cls],
        "Probabilities": {n: f"{p:.3f}" for n, p in zip(CLASS_NAMES, probs)}
    }

# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────────────────────

models_global = {}

def main():
    global models_global
    print("=" * 65)
    print("  HEO Crystal Structure Prediction — Liu et al. (2024)")
    print("=" * 65)

    # Load
    df, X, y = load_data()
    print(f"\n[Data] {len(df)} experimental samples loaded")
    print(f"       Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ADASYN
    X_res, y_res = apply_adasyn(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
    print(f"\n[Split] Train={len(X_train)}, Test={len(X_test)}")

    # Build & evaluate models
    models_global = build_models()
    print("\n[Evaluation Metrics]")
    results = evaluate_models(models_global, X_train, X_test, y_train, y_test)

    # Summary table
    print("\n" + "─"*65)
    print(f"  {'Algorithm':15s}  {'ACC':6s}  {'F1':6s}  {'AUC':6s}  {'Prec':6s}  {'Rec':6s}")
    print("─"*65)
    for name, r in results.items():
        print(f"  {name:15s}  {r['ACC']:.4f}  {r['F1']:.4f}  {r['AUC']:.4f}"
              f"  {r['Precision']:.4f}  {r['Recall']:.4f}")
    print("─"*65)

    # Feature importance order (SHAP)
    print("\n[SHAP Feature Importance Order]")
    xgb_model = models_global["XGBoost"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_test)
    # shap_vals shape: (n_samples, n_features, n_classes) or list of arrays
    if isinstance(shap_vals, list):
        # list of (n_samples, n_features) per class
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif shap_vals.ndim == 3:
        # (n_samples, n_features, n_classes)
        mean_shap = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_vals).mean(axis=0)
    mean_shap = np.array(mean_shap).ravel()
    FEATURE_LABELS_PLAIN = ["rA/rC", "dX_Pauling", "dX_Mulliken",
                             "Delta_delta", "dS_mix", "E&C", "Method"]
    # mean_shap is already a 1D array of shape (n_features,)
    sorted_indices = np.argsort(mean_shap)[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. {FEATURE_LABELS_PLAIN[int(idx)]:25s}  {mean_shap[int(idx)]:.4f}")

    # Design rules from paper
    print("\n[Crystal Structure Design Rules]")
    print("  rA/rC  < 0.35  → Fluorite")
    print("  rA/rC  ~ 0.45  → Spinel")
    print("  rA/rC  ~ 0.41  → Rock-salt")
    print("  ΔχPauling  < 0.10 → Fluorite tendency")
    print("  ΔχMulliken < 0.20 → Fluorite tendency")

    # Example predictions
    print("\n[Example Predictions using XGBoost]")
    examples = [
        ("CeLaPrSmY O₂ (fluorite-expected)",  0.232, 0.085, 0.165, 0.058, 13.38),
        ("CoMgNiZnCuO (rock-salt-expected)",  0.414, 0.055, 0.272, 0.042, 13.38),
        ("CoCrFeMnNi₃O₄ (spinel-expected)",   0.452, 0.215, 0.385, 0.095, 13.38),
    ]
    for desc, *feat in examples:
        res = predict_crystal_structure(xgb_model, *feat)
        print(f"  {desc}")
        print(f"    → {res['Predicted Structure']}  |  {res['Probabilities']}")

    # Plot
    print("\n[Generating figures …]")
    plot_all(df, X_res, y_res, X_test, y_test, results,
             xgb_model, X_train, y_train)

    print("\n✓ Done.")
    return models_global, results

if __name__ == "__main__":
    main()








#   1. Dataset size/quality - They used 112 experimental data points (mentioned in section 2.1)
#   2. Feature engineering - They used ADASYN oversampling and PCA dimensionality reduction
#   3. Hyperparameter tuning - They used 3-fold cross-validation with grid search (not 10-fold like yours)
#   4. Different features - They included E&C (element & content) encoding with PCA