"""
HEO CRYSTAL STRUCTURE PREDICTOR v2.0
=====================================
TRUE PREDICTION from composition alone!

This version can predict crystal structures for ANY composition,
even ones not in the training dataset.

Based on Liu et al. (2023) methodology.
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import calculate_all_features, features_to_array
from elemental_properties import get_available_elements, validate_composition

print("\n" + "=" * 70)
print("  🔮 HEO CRYSTAL STRUCTURE PREDICTOR v2.0")
print("  (TRUE PREDICTION - Works for ANY composition!)")
print("=" * 70)

# Load model
print("\n[Loading trained model...]")
try:
    with open('heo_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model ready!\n")
except FileNotFoundError:
    print("❌ Model file not found!")
    print("   Please run 'python train_model.py' first to train the model.")
    exit(1)

# Structure names
STRUCTURES = {
    0: "Fluorite-like (α-PbO2)",
    1: "Rock-salt-like (Baddeleyite)",
    2: "Spinel-like (Rutile)"
}

EMOJIS = {0: "🔵", 1: "🟠", 2: "🟢"}

AVAILABLE_ELEMENTS = get_available_elements()


def predict_structure(elements):
    """
    Predict crystal structure from element composition.

    Args:
        elements: List of element symbols (e.g., ['Hf', 'Zr', 'Ti', 'Sn'])

    Returns:
        prediction, probabilities, features
    """
    # Validate composition
    valid, missing = validate_composition(elements)
    if not valid:
        return None, None, None, missing

    # Calculate features
    features_dict = calculate_all_features(elements)
    features_array = features_to_array(features_dict).reshape(1, -1)

    # Predict
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]

    return prediction, probabilities, features_dict, None


def display_prediction(elements, prediction, probabilities, features):
    """Display prediction results"""
    print("\n" + "=" * 70)
    print(f"  PREDICTION FOR: {', '.join(elements)}")
    print("=" * 70)

    print(f"\n  📊 Calculated Features:")
    print(f"    • r_A/r_C ratio:           {features['r_A_r_C']:.4f}")
    print(f"    • Δχ (Pauling):            {features['delta_chi_pauling']:.4f}")
    print(f"    • Δχ (Mulliken):           {features['delta_chi_mulliken']:.4f}")
    print(f"    • Size mismatch (Δδ):      {features['delta_size']:.4f}")
    print(f"    • Mixing entropy (ΔS):     {features['entropy_mixing']:.4f} J/(mol·K)")
    print(f"    • Number of components:    {int(features['n_components'])}")

    print(f"\n  🎯 Structure Probabilities:")
    for i in range(3):
        marker = "→" if i == prediction else " "
        bar_length = int(probabilities[i] * 40)
        bar = "█" * bar_length
        print(f"    {marker} {EMOJIS[i]} {STRUCTURES[i]:30s} : {probabilities[i]*100:5.1f}% {bar}")

    print(f"\n  ✨ PREDICTED STRUCTURE: {STRUCTURES[prediction]}")
    print(f"     Confidence: {probabilities[prediction]*100:.1f}%")

    if probabilities[prediction] > 0.8:
        print(f"\n  ✅ HIGH CONFIDENCE - Very likely to form this structure!")
    elif probabilities[prediction] > 0.6:
        print(f"\n  ⚠️  MEDIUM CONFIDENCE - Likely but not certain")
    else:
        print(f"\n  ⚠️  LOW CONFIDENCE - Multiple structures possible")

    # Physical insights
    print(f"\n  💡 Physical Insights:")
    r_A_r_C = features['r_A_r_C']
    if r_A_r_C < 0.45:
        print(f"     Low r_A/r_C ({r_A_r_C:.3f}) → Favors Spinel structure")
    elif r_A_r_C > 0.35 and r_A_r_C < 0.55:
        print(f"     Medium r_A/r_C ({r_A_r_C:.3f}) → Rock-salt or Spinel possible")
    else:
        print(f"     High r_A/r_C ({r_A_r_C:.3f}) → Favors Fluorite structure")

    if features['delta_chi_pauling'] > 0.3:
        print(f"     High electronegativity difference → May affect stability")

    print("\n" + "=" * 70)


def show_menu():
    print("\n" + "=" * 70)
    print("  HOW DO YOU WANT TO PREDICT?")
    print("=" * 70)
    print("\n  1. 📋 Choose from example compositions (easiest!)")
    print("  2. ⌨️  Enter your own composition (ANY elements!)")
    print("  3. 🧪 Try a custom composition")
    print("  4. 📊 Show available elements")
    print("  5. ❌ Exit")
    print("\n" + "=" * 70)


def show_examples():
    print("\n" + "=" * 70)
    print("  EXAMPLE COMPOSITIONS (Select one)")
    print("=" * 70)

    examples = [
        ['Hf', 'Zr', 'Ti', 'Sn'],
        ['Ce', 'Hf', 'Ti', 'Zr'],
        ['Mn', 'Ir', 'Pt', 'Rh', 'Ru'],
        ['Hf', 'Pb', 'Sn', 'Zr'],
        ['Ge', 'Hf', 'Sn', 'Ti', 'Zr'],
        ['Ti', 'V', 'Nb', 'Zr'],  # New composition!
    ]

    for i, ex in enumerate(examples, 1):
        print(f"  {i}. {', '.join(ex)}")

    choice = input("\n  Select example (1-6): ").strip()

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            elements = examples[idx]
            prediction, probabilities, features, missing = predict_structure(elements)

            if prediction is not None:
                display_prediction(elements, prediction, probabilities, features)
            else:
                print(f"  ❌ Error: Elements {missing} not available")
        else:
            print("  Invalid selection!")
    except:
        print("  Invalid input!")


def show_available_elements():
    print("\n" + "=" * 70)
    print("  AVAILABLE ELEMENTS IN DATABASE")
    print("=" * 70)
    print("\n  ", end="")
    for i, elem in enumerate(AVAILABLE_ELEMENTS):
        print(f"{elem:4s}", end="")
        if (i + 1) % 10 == 0:
            print("\n  ", end="")
    print("\n")


# Main loop
while True:
    show_menu()
    choice = input("\n  Your choice (1-5): ").strip()

    if choice == '1':
        show_examples()

    elif choice == '2':
        print("\n  Enter elements separated by commas")
        print("  Example: Hf, Zr, Ti, Sn")
        print("  Example: Ru, Ti, V, Nb  (even if not in training data!)")
        comp = input("\n  Elements: ").strip()

        # Parse input
        elements = [e.strip() for e in comp.split(',')]
        elements = [e for e in elements if e]  # Remove empty strings

        if len(elements) < 3:
            print("  ❌ Please enter at least 3 elements")
            continue

        prediction, probabilities, features, missing = predict_structure(elements)

        if prediction is not None:
            display_prediction(elements, prediction, probabilities, features)
        else:
            print(f"\n  ❌ Error: Elements {missing} not available in database")
            print(f"  Available elements: {', '.join(AVAILABLE_ELEMENTS)}")

    elif choice == '3':
        print("\n  🧪 Custom Composition Mode")
        print("  Enter elements and their molar fractions")
        print("\n  This is currently under development.")
        print("  For now, please use equal molar fractions (option 2).")

    elif choice == '4':
        show_available_elements()

    elif choice == '5':
        print("\n  👋 Thank you for using the HEO predictor!")
        print("=" * 70)
        break

    else:
        print("  ❌ Invalid choice! Please select 1-5.")

    input("\n  Press Enter to continue...")
