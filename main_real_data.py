"""
Machine Learning-Based Crystal Structure Prediction for High-Entropy Oxide (HEO) Ceramics
==========================================================================================
UPDATED VERSION WITH REAL DATA FROM:
- Computational HEO database (od-qmul/HEO_search GitHub)
- Features extracted from DFT calculations and experimental measurements

Crystal structures: alpha-PbO2, baddeleyite, rutile (mapped to Fluorite/Rock-salt/Spinel equivalents)

Models: XGBoost (primary), Random Forest, Naïve Bayes
Interpretability: SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
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
# 1. LOAD REAL DATA FROM COMPUTATIONAL DATABASE
# ──────────────────────────────────────────────────────────────────────────────

def load_real_heo_data():
    """
    Load real HEO data from computational DFT calculations.
    Data source: od-qmul/HEO_search GitHub repository
    """
    print("\n[Loading Real HEO Data]")

    # Load 4-component HEO data
    df4 = pd.read_csv("heo_data_4comp.csv")
    print(f"  Loaded {len(df4)} 4-component HEO samples")

    # Load 5-component HEO data
    df5 = pd.read_csv("heo_data_5comp.csv")
    print(f"  Loaded {len(df5)} 5-component HEO samples")

    # Combine datasets
    df = pd.concat([df4, df5], ignore_index=True)

    # Map crystal structures to standard labels
    # alpha-PbO2 → Fluorite-like (0)
    # baddeleyite → Rock-salt-like (1)
    # rutile → Spinel-like (2)
    structure_map = {
        'aPbO2': 0,  # Fluorite-like
        'bad': 1,    # Rock-salt-like (baddeleyite)
        'rut': 2     # Spinel-like (rutile)
    }

    df['structure_label'] = df['Min_Crystal_rho'].map(structure_map)

    # Remove samples without clear structure classification
    df = df.dropna(subset=['structure_label'])

    # Calculate equivalent features to Liu et al. paper
    # Note: These are approximations based on available computational data

    # Use cation sigma as proxy for size mismatch (Δδ)
    df['delta_size'] = df[['Cation sigma_aPbO2', 'Cation sigma_bad', 'Cation sigma_rut']].mean(axis=1)

    # Use bond length sigma as proxy for radius ratio variation
    df['bond_length_variation'] = df[['sigma bond length_aPbO2', 'sigma bond length_bad', 'sigma bond length_rut']].mean(axis=1)

    # Formation enthalpy as stability indicator
    df['delta_H_min'] = df[['DeltaH_aPbO2', 'DeltaH_bad', 'DeltaH_rut']].min(axis=1)

    # Density as a structural descriptor
    df['rho_min'] = df['Min_rho']

    # Extract number of components from formula
    df['n_components'] = df['Formula'].apply(lambda x: len(eval(x)))

    # Create feature matrix
    feature_cols = ['bond_length_variation', 'delta_size', 'delta_H_min', 'rho_min', 'n_components']

    # Add structure-specific features
    for struct in ['aPbO2', 'bad', 'rut']:
        if f'DeltaH_{struct}' in df.columns:
            df[f'DeltaH_{struct}_normalized'] = df[f'DeltaH_{struct}'].fillna(df[f'DeltaH_{struct}'].mean())

    feature_cols_extended = feature_cols + [
        'DeltaH_aPbO2_normalized', 'DeltaH_bad_normalized', 'DeltaH_rut_normalized'
    ]

    # Remove rows with NaN in critical columns
    df_clean = df.dropna(subset=feature_cols_extended + ['structure_label'])

    X = df_clean[feature_cols_extended].values
    y = df_clean['structure_label'].values.astype(int)

    print(f"\n[Data Statistics]")
    print(f"  Total samples: {len(df_clean)}")
    print(f"  Structure distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Features: {len(feature_cols_extended)}")
    print(f"  Feature names: {feature_cols_extended}")

    return df_clean, X, y, feature_cols_extended

CLASS_NAMES = ["Fluorite-like (α-PbO2)", "Rock-salt-like (bad)", "Spinel-like (rutile)"]
CLASS_COLORS = ["#2196F3", "#FF9800", "#4CAF50"]

# ──────────────────────────────────────────────────────────────────────────────
# 2. ADASYN OVERSAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def apply_adasyn(X, y, random_state=42):
    from imblearn.over_sampling import SMOTE
    print(f"\n[ADASYN] Before: {dict(zip(*np.unique(y, return_counts=True)))}")
    try:
        sampler = ADASYN(random_state=random_state, n_neighbors=3)
        X_res, y_res = sampler.fit_resample(X, y)
        print(f"[ADASYN] After (ADASYN): {dict(zip(*np.unique(y_res, return_counts=True)))}")
    except (RuntimeError, ValueError) as e:
        print(f"[ADASYN] Fallback to SMOTE: {str(e)[:50]}")
        sampler = SMOTE(random_state=random_state, k_neighbors=min(5, len(X)//10))
        X_res, y_res = sampler.fit_resample(X, y)
        print(f"[ADASYN] After (SMOTE): {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

# ──────────────────────────────────────────────────────────────────────────────
# 3. BUILD MODELS
# ──────────────────────────────────────────────────────────────────────────────

def build_models():
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        learning_rate=0.02,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42
    )
    nb = GaussianNB()
    return {"XGBoost": xgb, "Random Forest": rf, "Naïve Bayes": nb}

# ──────────────────────────────────────────────────────────────────────────────
# 4. EVALUATE MODELS
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
        prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec   = recall_score(y_test, y_pred, average="macro")
        auc_s = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        cv_sc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()

        results[name] = {"ACC": acc, "F1": f1, "Precision": prec,
                         "Recall": rec, "AUC": auc_s, "CV_ACC": cv_sc,
                         "y_pred": y_pred, "y_prob": y_prob}
        print(f"  {name:15s}  ACC={acc:.4f}  F1={f1:.4f}  AUC={auc_s:.4f}  CV={cv_sc:.4f}")

    return results

# ──────────────────────────────────────────────────────────────────────────────
# 5. PLOTTING (Simplified for real data)
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(X_test, y_test, results, xgb_model, feature_names):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

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
            ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # Panel 1: Model Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Model Accuracy Comparison (Real HEO Data)")
    names = list(results.keys())
    accs  = [results[n]["ACC"] for n in names]
    cvs   = [results[n]["CV_ACC"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax1.bar(x - w/2, accs, w, color=ACCENT, alpha=0.85, label="Test ACC")
    bars2 = ax1.bar(x + w/2, cvs,  w, color="#f06292", alpha=0.85, label="CV ACC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9, color=TEXT, rotation=15)
    ax1.set_ylim(0.3, 1.0)
    ax1.set_ylabel("Accuracy", fontsize=10)
    for b in list(bars1)+list(bars2):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax1.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # Panel 2: Confusion Matrix (XGBoost)
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Confusion Matrix – XGBoost (Real Data)")
    cm = confusion_matrix(y_test, results["XGBoost"]["y_pred"], normalize="true")
    im = ax2.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks([0,1,2])
    ax2.set_xticklabels(["Fluorite", "Rock-salt", "Spinel"], color=TEXT, fontsize=8)
    ax2.set_yticks([0,1,2])
    ax2.set_yticklabels(["Fluorite", "Rock-salt", "Spinel"], color=TEXT, fontsize=8)
    ax2.set_xlabel("Predicted", fontsize=10)
    ax2.set_ylabel("Actual", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                     fontsize=12, color="white" if cm[i,j] > 0.5 else "black", fontweight="bold")
    plt.colorbar(im, ax=ax2)

    # Panel 3: SHAP Feature Importance
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "SHAP Feature Importance")
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        ms = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    elif np.array(shap_vals).ndim == 3:
        ms = np.abs(shap_vals).mean(axis=(0, 2))
    else:
        ms = np.abs(shap_vals).mean(axis=0)
    ms = np.array(ms).ravel()

    order_idx = np.argsort(ms)
    labels_ord = [feature_names[i] for i in order_idx]
    vals_ord   = ms[order_idx]
    bars = ax3.barh(labels_ord, vals_ord, color=ACCENT, alpha=0.85)
    for b, v in zip(bars, vals_ord):
        ax3.text(v + 0.002, b.get_y() + b.get_height()/2,
                 f"{v:.3f}", va="center", fontsize=8, color=TEXT)
    ax3.set_xlabel("mean(|SHAP value|)", fontsize=10)
    ax3.tick_params(axis="y", labelsize=8)

    # Panel 4: ROC Curves
    ax4 = fig.add_subplot(gs[1, :2])
    style_ax(ax4, "ROC Curves – XGBoost (Real Data)")
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_test, classes=[0,1,2])
    y_prob = results["XGBoost"]["y_prob"]
    for i, (cls, col) in enumerate(zip(["Fluorite", "Rock-salt", "Spinel"], CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC={roc_auc:.3f})")
    ax4.plot([0,1],[0,1],"--", color=GRID, lw=1)
    ax4.set_xlabel("False Positive Rate", fontsize=10)
    ax4.set_ylabel("True Positive Rate", fontsize=10)
    ax4.legend(fontsize=9, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # Panel 5: Metrics Table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PANEL)
    ax5.axis("off")
    ax5.set_title("Performance Metrics (Real Data)", color=TEXT, fontsize=11, fontweight="bold", pad=8)
    headers = ["Model", "F1", "AUC", "Prec", "Rec"]
    rows = []
    for n in names:
        r = results[n]
        rows.append([n, f"{r['F1']:.3f}", f"{r['AUC']:.3f}",
                     f"{r['Precision']:.3f}", f"{r['Recall']:.3f}"])
    table = ax5.table(cellText=rows, colLabels=headers, loc="center",
                      cellLoc="center", bbox=[0.0, 0.2, 1.0, 0.65])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor("#252840" if r == 0 else PANEL)
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor(GRID)

    fig.suptitle("ML-Based HEO Crystal Structure Prediction\nUsing Real Computational Data (od-qmul/HEO_search)",
                 fontsize=14, color=TEXT, fontweight="bold", y=0.98)

    plt.savefig("heo_ml_results_REAL_DATA.png",
                dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close()
    print("\n✓  Figure saved → heo_ml_results_REAL_DATA.png")

# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────────

models_global = {}

def main():
    global models_global
    print("=" * 80)
    print("  HEO Crystal Structure Prediction — REAL DATA VERSION")
    print("  Data source: od-qmul/HEO_search (computational DFT)")
    print("=" * 80)

    # Load real data
    df, X, y, feature_names = load_real_heo_data()

    # Check if we have enough samples for ADASYN
    min_samples = min(np.unique(y, return_counts=True)[1])
    if min_samples < 6:
        print(f"\n[WARNING] Minimum class has only {min_samples} samples. Adjusting sampling strategy...")
        X_res, y_res = X, y  # Skip oversampling
    else:
        # ADASYN
        X_res, y_res = apply_adasyn(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.25, random_state=42, stratify=y_res)
    print(f"\n[Split] Train={len(X_train)}, Test={len(X_test)}")

    # Build & evaluate models
    models_global = build_models()
    print("\n[Evaluation Metrics]")
    results = evaluate_models(models_global, X_train, X_test, y_train, y_test)

    # Summary table
    print("\n" + "─"*80)
    print(f"  {'Algorithm':15s}  {'ACC':6s}  {'F1':6s}  {'AUC':6s}  {'Prec':6s}  {'Rec':6s}")
    print("─"*80)
    for name, r in results.items():
        print(f"  {name:15s}  {r['ACC']:.4f}  {r['F1']:.4f}  {r['AUC']:.4f}"
              f"  {r['Precision']:.4f}  {r['Recall']:.4f}")
    print("─"*80)

    # Plot
    print("\n[Generating figures …]")
    plot_results(X_test, y_test, results, models_global["XGBoost"], feature_names)

    print("\n✓ Done. This model uses REAL computational HEO data!")
    print("\nData sources:")
    print("  • GitHub: od-qmul/HEO_search")
    print("  • Paper: Expanding the search space of high entropy oxides (arXiv:2508.13389)")

    return models_global, results

if __name__ == "__main__":
    main()
