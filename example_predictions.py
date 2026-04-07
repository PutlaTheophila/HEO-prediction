"""
Example: How to use the trained HEO model to predict crystal structures
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle

# Load the real data and train the model
print("=" * 70)
print("  HEO Crystal Structure Prediction - Example Usage")
print("=" * 70)

# Load data
df4 = pd.read_csv("heo_data_4comp.csv")
df5 = pd.read_csv("heo_data_5comp.csv")
df = pd.concat([df4, df5], ignore_index=True)

# Prepare data (same as main_real_data.py)
structure_map = {'aPbO2': 0, 'bad': 1, 'rut': 2}
df['structure_label'] = df['Min_Crystal_rho'].map(structure_map)
df = df.dropna(subset=['structure_label'])

df['delta_size'] = df[['Cation sigma_aPbO2', 'Cation sigma_bad', 'Cation sigma_rut']].mean(axis=1)
df['bond_length_variation'] = df[['sigma bond length_aPbO2', 'sigma bond length_bad', 'sigma bond length_rut']].mean(axis=1)
df['delta_H_min'] = df[['DeltaH_aPbO2', 'DeltaH_bad', 'DeltaH_rut']].min(axis=1)
df['rho_min'] = df['Min_rho']
df['n_components'] = df['Formula'].apply(lambda x: len(eval(x)))

for struct in ['aPbO2', 'bad', 'rut']:
    if f'DeltaH_{struct}' in df.columns:
        df[f'DeltaH_{struct}_normalized'] = df[f'DeltaH_{struct}'].fillna(df[f'DeltaH_{struct}'].mean())

feature_cols = ['bond_length_variation', 'delta_size', 'delta_H_min', 'rho_min',
                'n_components', 'DeltaH_aPbO2_normalized', 'DeltaH_bad_normalized',
                'DeltaH_rut_normalized']

df_clean = df.dropna(subset=feature_cols + ['structure_label'])
X = df_clean[feature_cols].values
y = df_clean['structure_label'].values.astype(int)

# Train model
print("\n[Training Model...]")
model = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, verbosity=0
)
model.fit(X, y)
print("✓ Model trained successfully!")

# Define structure names
STRUCTURE_NAMES = {
    0: "Fluorite-like (α-PbO2)",
    1: "Rock-salt-like (Baddeleyite)",
    2: "Spinel-like (Rutile)"
}

STRUCTURE_COLORS = {
    0: "🔵 Fluorite",
    1: "🟠 Rock-salt",
    2: "🟢 Spinel"
}

print("\n" + "=" * 70)
print("  EXAMPLE PREDICTIONS")
print("=" * 70)

# Get some real examples from the dataset
n_examples = 10
example_indices = np.random.choice(len(df_clean), n_examples, replace=False)

for i, idx in enumerate(example_indices):
    sample = df_clean.iloc[idx]
    composition = sample['Formula']
    true_structure = int(sample['structure_label'])

    # Extract features
    features = sample[feature_cols].values.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    print(f"\n{'─' * 70}")
    print(f"Example {i+1}: {composition}")
    print(f"{'─' * 70}")
    print(f"  Formation Energy: {sample['delta_H_min']:.4f} eV")
    print(f"  Density:          {sample['rho_min']:.4f} g/cm³")
    print(f"  Components:       {int(sample['n_components'])}")
    print(f"\n  Predictions:")
    for class_id, prob in enumerate(probabilities):
        marker = "✓" if class_id == prediction else " "
        star = "★" if class_id == true_structure else " "
        print(f"    {marker} {STRUCTURE_COLORS[class_id]}: {prob*100:5.1f}%  {star}")

    print(f"\n  → Predicted: {STRUCTURE_NAMES[prediction]}")
    print(f"  → Actual:    {STRUCTURE_NAMES[true_structure]}")

    if prediction == true_structure:
        print(f"  ✓ CORRECT! 🎯")
    else:
        print(f"  ✗ Wrong (but learning from this!)")

print("\n" + "=" * 70)
print("  HOW TO USE FOR NEW COMPOSITIONS")
print("=" * 70)
print("""
To predict a NEW HEO composition:

1. Calculate/obtain these features for your composition:
   - Bond length variation (from crystal structure prediction)
   - Cation size mismatch (from ionic radii tables)
   - Formation enthalpy (from DFT or thermodynamic databases)
   - Density (from composition and structure)
   - Number of components (count elements)
   - Phase-specific energies (from DFT calculations)

2. Create feature vector:
   new_composition = [0.075, 0.045, -0.12, 1.35, 4, -0.08, -0.11, -0.13]

3. Predict:
   prediction = model.predict([new_composition])
   probabilities = model.predict_proba([new_composition])

4. Interpret results and plan synthesis accordingly!
""")

print("\n" + "=" * 70)
print("  PRACTICAL APPLICATIONS")
print("=" * 70)
print("""
Use this model to:

1. 🔬 DESIGN: Screen 1000s of compositions computationally
2. 💰 SAVE: Reduce expensive trial-and-error experiments
3. ⚡ FAST: Get predictions in milliseconds vs months
4. 🎯 TARGET: Design HEOs with specific crystal structures
5. 📊 DISCOVER: Find unexpected structure-composition relationships

Example workflow:
  → Generate 10,000 candidate compositions
  → Model predicts structure for each (takes ~1 second)
  → Filter for desired structures (e.g., only spinel)
  → Synthesize top 10 candidates (save 99.9% of experiments!)
""")

print("=" * 70)
