"""
Train HEO Crystal Structure Prediction Model
=============================================
Uses composition-based features only (NO data leakage).
Based on Liu et al. (2023) methodology.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import calculate_all_features, features_to_array

print("\n" + "=" * 70)
print("  🎓 TRAINING HEO CRYSTAL STRUCTURE PREDICTOR")
print("  (Composition-based features - NO data leakage)")
print("=" * 70)

# Load datasets
print("\n[1/5] Loading datasets...")
df4 = pd.read_csv("heo_data_4comp.csv")
df5 = pd.read_csv("heo_data_5comp.csv")
df = pd.concat([df4, df5], ignore_index=True)

# Map structures to labels
structure_map = {'aPbO2': 0, 'bad': 1, 'rut': 2}
df['structure_label'] = df['Min_Crystal_rho'].map(structure_map)
df = df.dropna(subset=['structure_label'])

print(f"   Total samples: {len(df)}")
print(f"   Structures: {df['Min_Crystal_rho'].value_counts().to_dict()}")

# Calculate composition-based features
print("\n[2/5] Calculating composition-based features...")
print("   (This may take a minute...)")

features_list = []
labels_list = []
formulas_list = []

for idx, row in df.iterrows():
    try:
        # Parse formula
        elements = eval(row['Formula'])

        # Calculate features from composition alone
        features_dict = calculate_all_features(elements)
        features_array = features_to_array(features_dict)

        features_list.append(features_array)
        labels_list.append(int(row['structure_label']))
        formulas_list.append(elements)

        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1}/{len(df)} samples...")

    except Exception as e:
        print(f"   Warning: Could not process {row['Formula']}: {e}")
        continue

X = np.array(features_list)
y = np.array(labels_list)

print(f"\n   ✓ Features calculated: {X.shape}")
print(f"   Feature dimension: {X.shape[1]}")

# Split data
print("\n[3/5] Splitting data (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Train model
print("\n[4/5] Training XGBoost model...")
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
print("\n[5/5] Evaluating model...")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n   Training Accuracy:   {train_acc:.3f}")
print(f"   Test Accuracy:       {test_acc:.3f}")

# Detailed metrics
STRUCTURE_NAMES = {0: "Fluorite", 1: "Rock-salt", 2: "Spinel"}

print("\n   Classification Report (Test Set):")
print("   " + "-" * 60)
report = classification_report(
    y_test, y_test_pred,
    target_names=[STRUCTURE_NAMES[i] for i in range(3)],
    digits=3
)
for line in report.split('\n'):
    print("   " + line)

print("\n   Confusion Matrix (Test Set):")
print("   " + "-" * 60)
cm = confusion_matrix(y_test, y_test_pred)
print("                  Predicted")
print("                  Flu   R-s   Spi")
for i, row in enumerate(cm):
    struct_name = STRUCTURE_NAMES[i][:3]
    print(f"   Actual {struct_name}    {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")

# Feature importance
print("\n   Top 10 Most Important Features:")
print("   " + "-" * 60)
feature_names = [
    'r_A/r_C', 'Δχ_Pauling', 'Δχ_Mulliken', 'Δδ_size',
    'ΔS_mix', 'n_comp'
] + [f'E_{i}' for i in range(14)]

importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:10]

for i, idx in enumerate(indices, 1):
    print(f"   {i:2d}. {feature_names[idx]:15s} : {importance[idx]:.4f}")

# Save model
print("\n[6/6] Saving model...")
with open('heo_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("   ✓ Model saved to: heo_model.pkl")

print("\n" + "=" * 70)
print("  ✅ TRAINING COMPLETE!")
print("=" * 70)
print("\n  This model can now predict structures for ANY composition")
print("  using only elemental properties - no DFT calculations needed!\n")
