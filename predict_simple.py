"""
SIMPLEST VERSION - Just edit the composition and run!
"""

# ============================================================
# EDIT THIS LINE TO PREDICT YOUR COMPOSITION:
# ============================================================
YOUR_COMPOSITION = "Hf, Zr, Ti, Sn"  # ← CHANGE THIS!
# ============================================================

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Auto-load and train (takes 2 seconds)
print("\n🔮 Loading model...", end="")
df4 = pd.read_csv("heo_data_4comp.csv")
df5 = pd.read_csv("heo_data_5comp.csv")
df = pd.concat([df4, df5], ignore_index=True)

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

model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
                      use_label_encoder=False, eval_metric="mlogloss",
                      random_state=42, verbosity=0)
model.fit(X, y)
print(" Done! ✓\n")

# Parse your composition
elements = [e.strip() for e in YOUR_COMPOSITION.replace('[', '').replace(']', '').replace("'", "").split(',')]
elements_set = set(elements)

# Find match
match_idx = None
for idx, row in df_clean.iterrows():
    comp_elements = set(eval(row['Formula']))
    if comp_elements == elements_set:
        match_idx = idx
        break

if match_idx is None:
    print(f"❌ Composition {elements} not found in database!")
    print("\n💡 Try one of these:")
    print("   - Hf, Zr, Ti, Sn")
    print("   - Ce, La, Pr, Sm, Y")
    print("   - Mg, Ni, Co, Cu, Zn")
    print("   - Ir, Pt, Rh, Ru")
else:
    sample = df_clean.iloc[match_idx]
    features = sample[feature_cols].values.reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    STRUCTURES = ["Fluorite", "Rock-salt", "Spinel"]
    EMOJIS = ["🔵", "🟠", "🟢"]

    print("=" * 60)
    print(f"  PREDICTION FOR: {elements}")
    print("=" * 60)
    print(f"\n  Properties:")
    print(f"    Energy:  {sample['delta_H_min']:.3f} eV")
    print(f"    Density: {sample['rho_min']:.3f} g/cm³")
    print(f"\n  Probabilities:")
    for i in range(3):
        bar = "█" * int(probabilities[i] * 30)
        marker = " ← WINNER!" if i == prediction else ""
        print(f"    {EMOJIS[i]} {STRUCTURES[i]:12s} {probabilities[i]*100:5.1f}% {bar}{marker}")

    print(f"\n  🎯 RESULT: {STRUCTURES[prediction]}")
    print(f"     Confidence: {probabilities[prediction]*100:.0f}%")

    if probabilities[prediction] > 0.8:
        print(f"\n  ✓ HIGH CONFIDENCE!")
    elif probabilities[prediction] > 0.6:
        print(f"\n  ⚠ MEDIUM CONFIDENCE")
    else:
        print(f"\n  ⚠ LOW CONFIDENCE - Multiple structures possible")

    print("\n" + "=" * 60)
