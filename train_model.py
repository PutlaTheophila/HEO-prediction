"""
Train HEO Crystal Structure Prediction Model
=============================================
Uses composition-based features only (NO data leakage).
Based on Liu et al. (2023) methodology.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import calculate_all_features, features_to_array

print("\n" + "=" * 70)
print("  🎓 TRAINING HEO CRYSTAL STRUCTURE PREDICTOR (3 classes)")
print("  (Composition-based features - NO data leakage)")
print("=" * 70)

# Load augmented dataset and drop the synthetic perovskite class (label==3).
# We keep the literature-sourced Fluorite/Rock-salt augmentations to balance
# against the dominant Spinel class, but exclude perovskite because it's
# trivially separable by Pb/Ce fingerprint → leakage-driven 100% accuracy.
print("\n[1/5] Loading datasets...")
AUG_PATH = "heo_augmented_dataset.csv"

if os.path.exists(AUG_PATH):
    print(f"   Loading {AUG_PATH} and dropping perovskite (label==3)")
    df_aug = pd.read_csv(AUG_PATH)
    df_aug = df_aug[df_aug['label'] != 3].reset_index(drop=True)
    feature_cols = [c for c in df_aug.columns if c != 'label']
    X = df_aug[feature_cols].values
    y = df_aug['label'].values.astype(int)
    print(f"   Total samples: {len(df_aug)}")
    print(f"   Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")
else:
    print("   Augmented dataset not found — falling back to raw 3-class CSVs")
    df4 = pd.read_csv("heo_data_4comp.csv")
    df5 = pd.read_csv("heo_data_5comp.csv")
    df = pd.concat([df4, df5], ignore_index=True)
    structure_map = {'aPbO2': 0, 'bad': 1, 'rut': 2}
    df['structure_label'] = df['Min_Crystal_rho'].map(structure_map)
    df = df.dropna(subset=['structure_label'])

    print(f"   Total samples: {len(df)}")
    print(f"   Structures: {df['Min_Crystal_rho'].value_counts().to_dict()}")

    features_list, labels_list = [], []
    for idx, row in df.iterrows():
        try:
            elements = eval(row['Formula'])
            features_dict = calculate_all_features(elements)
            features_list.append(features_to_array(features_dict))
            labels_list.append(int(row['structure_label']))
            if (idx + 1) % 500 == 0:
                print(f"   Processed {idx + 1}/{len(df)} samples...")
        except Exception as e:
            print(f"   Warning: Could not process {row['Formula']}: {e}")
            continue
    X = np.array(features_list)
    y = np.array(labels_list)

# Skip the redundant feature step block below — features already loaded above.

print(f"\n   ✓ Features ready: {X.shape}")
print(f"   Feature dimension: {X.shape[1]}")
print(f"   Number of classes: {len(np.unique(y))}")

# Split data
print("\n[3/5] Splitting data (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Perform 10-fold cross-validation
print("\n[4/6] Performing 10-fold cross-validation...")
print("   (This evaluates model generalization)")

model_cv = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)

cv_scores = cross_val_score(model_cv, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)

print(f"\n   Cross-Validation Results (10-fold):")
print(f"   Mean CV Accuracy:    {cv_scores.mean():.3f}")
print(f"   Std Dev:             {cv_scores.std():.3f}")
print(f"   Min CV Accuracy:     {cv_scores.min():.3f}")
print(f"   Max CV Accuracy:     {cv_scores.max():.3f}")
print(f"\n   Individual fold scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i:2d}: {score:.3f}")

# Train model
print("\n[5/6] Training XGBoost model on full training set...")
print("   (Using hyperparameters from paper)")

model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train)
print("   ✓ Model trained!")

# Evaluate
print("\n[6/7] Evaluating model on hold-out test set...")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n   Training Accuracy:   {train_acc:.3f}")
print(f"   Test Accuracy:       {test_acc:.3f}")

# Detailed metrics — supports 3 or 4 classes
STRUCTURE_NAMES_FULL = {0: "Fluorite", 1: "Rock-salt", 2: "Spinel"}
n_classes = int(np.unique(y).max()) + 1
STRUCTURE_NAMES = {i: STRUCTURE_NAMES_FULL[i] for i in range(n_classes)}

print("\n   Classification Report (Test Set):")
print("   " + "-" * 60)
report = classification_report(
    y_test, y_test_pred,
    target_names=[STRUCTURE_NAMES[i] for i in range(n_classes)],
    digits=3
)
for line in report.split('\n'):
    print("   " + line)

print("\n   Confusion Matrix (Test Set):")
print("   " + "-" * 60)
cm = confusion_matrix(y_test, y_test_pred)
header = "                  " + "  ".join(f"{STRUCTURE_NAMES[i][:4]:>4s}" for i in range(n_classes))
print("                  Predicted")
print(header)
for i, row in enumerate(cm):
    struct_name = STRUCTURE_NAMES[i][:4]
    cells = "  ".join(f"{v:4d}" for v in row)
    print(f"   Actual {struct_name:4s}   {cells}")

# Feature importance
print("\n   Top 10 Most Important Features:")
print("   " + "-" * 60)
feature_names = [
    'r_A/r_C', 'Δχ_Pauling', 'Δχ_Mulliken', 'Δδ_size',
    'ΔS_mix', 'n_comp', 'VEC', 'ΔH_mix', 'Ω'
] + [f'E_{i}' for i in range(14)]
# Pad/truncate to actual feature dimension to be safe
if len(feature_names) < X.shape[1]:
    feature_names += [f'F_{i}' for i in range(len(feature_names), X.shape[1])]
feature_names = feature_names[:X.shape[1]]

importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:10]

for i, idx in enumerate(indices, 1):
    print(f"   {i:2d}. {feature_names[idx]:15s} : {importance[idx]:.4f}")

# Save model
print("\n[7/7] Saving model...")
with open('heo_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("   ✓ Model saved to: heo_model.pkl")

print("\n" + "=" * 70)
print("  ✅ TRAINING COMPLETE!")
print("=" * 70)
print("\n  This model can now predict structures for ANY composition")
print("  using only elemental properties - no DFT calculations needed!\n")
