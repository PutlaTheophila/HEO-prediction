"""
SIMPLE INTERFACE: Predict Crystal Structure for Your HEO Composition
=====================================================================
Just run this script and follow the prompts!
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

print("\n" + "=" * 70)
print("  🔮 HEO CRYSTAL STRUCTURE PREDICTOR")
print("=" * 70)

# Load and train model (this happens automatically)
print("\n[Loading data and training model...]")

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

model = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, verbosity=0
)
model.fit(X, y)

print("✓ Model ready!\n")

# Structure names
STRUCTURES = {
    0: "Fluorite-like (α-PbO2)",
    1: "Rock-salt-like (Baddeleyite)",
    2: "Spinel-like (Rutile)"
}

EMOJIS = {0: "🔵", 1: "🟠", 2: "🟢"}

def search_composition(elements):
    """Search for a composition in the dataset"""
    elements_set = set([e.strip() for e in elements])

    matches = []
    for idx, row in df_clean.iterrows():
        comp_elements = set(eval(row['Formula']))
        if comp_elements == elements_set:
            matches.append(idx)

    return matches

def predict_from_dataset(composition_str):
    """Predict structure for a composition from the dataset"""
    elements = [e.strip() for e in composition_str.replace('[', '').replace(']', '').replace("'", "").split(',')]

    matches = search_composition(elements)

    if not matches:
        print(f"\n❌ Composition {elements} not found in dataset.")
        print("   Try one of these example compositions:")
        print("   - Hf, Zr, Ti, Sn")
        print("   - Ce, La, Pr, Sm, Y")
        print("   - Mg, Ni, Co, Cu, Zn")
        return

    # Use first match
    idx = matches[0]
    sample = df_clean.iloc[idx]
    features = sample[feature_cols].values.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    print("\n" + "=" * 70)
    print(f"  PREDICTION FOR: {elements}")
    print("=" * 70)
    print(f"\n  Composition Properties:")
    print(f"    • Formation Energy: {sample['delta_H_min']:.4f} eV")
    print(f"    • Density:          {sample['rho_min']:.4f} g/cm³")
    print(f"    • Components:       {int(sample['n_components'])} elements")

    print(f"\n  Structure Probabilities:")
    for i in range(3):
        marker = "→" if i == prediction else " "
        print(f"    {marker} {EMOJIS[i]} {STRUCTURES[i]:30s} : {probabilities[i]*100:5.1f}%")

    print(f"\n  🎯 PREDICTED STRUCTURE: {STRUCTURES[prediction]}")
    print(f"     Confidence: {probabilities[prediction]*100:.1f}%")

    if probabilities[prediction] > 0.8:
        print(f"\n  ✓ HIGH CONFIDENCE - Very likely to form this structure!")
    elif probabilities[prediction] > 0.6:
        print(f"\n  ⚠ MEDIUM CONFIDENCE - Likely but not certain")
    else:
        print(f"\n  ⚠ LOW CONFIDENCE - Multiple structures possible")

    print("\n" + "=" * 70)

def predict_from_features(features_dict):
    """Predict from manually entered features"""
    features_array = np.array([[
        features_dict['bond_length_variation'],
        features_dict['delta_size'],
        features_dict['delta_H_min'],
        features_dict['rho_min'],
        features_dict['n_components'],
        features_dict.get('DeltaH_aPbO2', -0.1),
        features_dict.get('DeltaH_bad', -0.1),
        features_dict.get('DeltaH_rut', -0.1)
    ]])

    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]

    print("\n" + "=" * 70)
    print(f"  PREDICTION FROM CUSTOM FEATURES")
    print("=" * 70)
    print(f"\n  Structure Probabilities:")
    for i in range(3):
        marker = "→" if i == prediction else " "
        print(f"    {marker} {EMOJIS[i]} {STRUCTURES[i]:30s} : {probabilities[i]*100:5.1f}%")

    print(f"\n  🎯 PREDICTED STRUCTURE: {STRUCTURES[prediction]}")
    print(f"     Confidence: {probabilities[prediction]*100:.1f}%")
    print("\n" + "=" * 70)

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def show_menu():
    print("\n" + "=" * 70)
    print("  HOW DO YOU WANT TO PREDICT?")
    print("=" * 70)
    print("\n  1. 📋 Choose from example compositions (easiest!)")
    print("  2. ⌨️  Enter element composition (e.g., Hf, Zr, Ti, Sn)")
    print("  3. 🔧 Enter custom features (advanced)")
    print("  4. 📊 Show available elements in database")
    print("  5. ❌ Exit")
    print("\n" + "=" * 70)

def show_examples():
    print("\n" + "=" * 70)
    print("  EXAMPLE COMPOSITIONS (Select one)")
    print("=" * 70)

    # Get some interesting examples
    examples = [
        "['Hf', 'Zr', 'Ti', 'Sn']",
        "['Ce', 'La', 'Pr', 'Sm', 'Y']",
        "['Mn', 'Fe', 'Co', 'Ni', 'Cu']",
        "['Hf', 'Pb', 'Sn', 'Zr']",
        "['Ir', 'Pt', 'Rh', 'Ru']",
        "['Ge', 'Hf', 'Sn', 'Ti', 'Zr']",
    ]

    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {ex}")

    choice = input("\n  Select example (1-6): ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            predict_from_dataset(examples[idx])
        else:
            print("  Invalid selection!")
    except:
        print("  Invalid input!")

def show_available_elements():
    print("\n" + "=" * 70)
    print("  AVAILABLE ELEMENTS IN DATABASE")
    print("=" * 70)

    all_elements = set()
    for formula in df_clean['Formula']:
        all_elements.update(eval(formula))

    elements_sorted = sorted(list(all_elements))
    print("\n  ", end="")
    for i, elem in enumerate(elements_sorted):
        print(f"{elem:4s}", end="")
        if (i + 1) % 15 == 0:
            print("\n  ", end="")
    print("\n")

# Main loop
while True:
    show_menu()
    choice = input("\n  Your choice (1-5): ").strip()

    if choice == '1':
        show_examples()

    elif choice == '2':
        print("\n  Enter elements separated by commas (e.g., Hf, Zr, Ti, Sn)")
        comp = input("  Elements: ").strip()
        predict_from_dataset(comp)

    elif choice == '3':
        print("\n  Enter feature values:")
        try:
            features = {
                'bond_length_variation': float(input("    Bond length variation: ")),
                'delta_size': float(input("    Cation size mismatch: ")),
                'delta_H_min': float(input("    Formation enthalpy (eV): ")),
                'rho_min': float(input("    Density (g/cm³): ")),
                'n_components': int(input("    Number of components: "))
            }
            predict_from_features(features)
        except:
            print("  ❌ Invalid input!")

    elif choice == '4':
        show_available_elements()

    elif choice == '5':
        print("\n  👋 Thank you for using the HEO predictor!")
        print("=" * 70)
        break

    else:
        print("  ❌ Invalid choice! Please select 1-5.")

    input("\n  Press Enter to continue...")
