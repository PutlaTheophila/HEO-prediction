#!/usr/bin/env python
"""
Command-line HEO Structure Predictor
=====================================
Quick predictions without web interface
"""

import sys
import pickle
import numpy as np
from feature_engineering import calculate_all_features, features_to_array
from elemental_properties import get_available_elements

STRUCTURES = {
    0: 'Fluorite (α-PbO₂)',
    1: 'Rock-salt (Baddeleyite)',
    2: 'Spinel (Rutile)'
}

def load_model():
    """Load trained model"""
    try:
        with open('heo_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ Error: Model file 'heo_model.pkl' not found!")
        print("Please train the model first: python train_model.py")
        sys.exit(1)

def print_header():
    """Print header"""
    print("=" * 70)
    print("  🔬 HEO CRYSTAL STRUCTURE PREDICTOR (CLI)")
    print("=" * 70)
    print()

def print_available_elements():
    """Print available elements"""
    elements = get_available_elements()
    print("Available cations:")
    print("  " + ", ".join(elements))
    print()

def get_composition():
    """Get composition from user"""
    print("Enter composition:")
    print("  Format: Element1:Fraction1 Element2:Fraction2 ...")
    print("  Example: Hf:0.25 Zr:0.25 Ti:0.25 Sn:0.25")
    print("  (or just elements for equal fractions: Hf Zr Ti Sn)")
    print()

    comp_input = input("Composition: ").strip()

    if not comp_input:
        print("❌ No composition entered!")
        sys.exit(1)

    # Parse composition
    parts = comp_input.split()
    elements = []
    fractions = []

    for part in parts:
        if ':' in part:
            elem, frac = part.split(':')
            elements.append(elem.strip())
            fractions.append(float(frac))
        else:
            elements.append(part.strip())

    # If no fractions given, use equal fractions
    if not fractions:
        fractions = [1.0/len(elements)] * len(elements)

    # Normalize fractions
    total = sum(fractions)
    fractions = [f/total for f in fractions]

    return elements, fractions

def predict(model, elements, fractions):
    """Make prediction"""
    # Calculate features
    features_dict = calculate_all_features(elements, fractions)
    features_array = features_to_array(features_dict)

    # Predict
    prediction = model.predict([features_array])[0]
    probabilities = model.predict_proba([features_array])[0]

    return prediction, probabilities, features_dict

def print_results(elements, fractions, prediction, probabilities, features):
    """Print results"""
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print()

    # Composition
    print("📋 Composition:")
    for elem, frac in zip(elements, fractions):
        print(f"  {elem:3s}: {frac:.4f} ({frac*100:.2f}%)")
    print()

    # Prediction
    print("🎯 Predicted Structure:")
    print(f"  {STRUCTURES[prediction]}")
    print()

    # Confidence
    confidence = probabilities[prediction] * 100
    if confidence > 70:
        status = "✅ HIGH"
    elif confidence > 50:
        status = "⚠️  MEDIUM"
    else:
        status = "❌ LOW"

    print(f"Confidence: {status} ({confidence:.1f}%)")
    print()

    # Probabilities
    print("📊 Structure Probabilities:")
    for i, (struct_name, prob) in enumerate(zip(
        ['Fluorite', 'Rock-salt', 'Spinel'],
        probabilities
    )):
        bar = "█" * int(prob * 40)
        print(f"  {struct_name:12s} [{bar:<40}] {prob*100:5.1f}%")
    print()

    # Features
    print("🔍 Key Features:")
    print(f"  r_A/r_C          : {features['r_A_r_C']:.4f}")
    print(f"  Δχ (Pauling)     : {features['delta_chi_pauling']:.4f}")
    print(f"  Δχ (Mulliken)    : {features['delta_chi_mulliken']:.4f}")
    print(f"  Δδ (size)        : {features['delta_size']:.4f}")
    print(f"  ΔS_mix (J/mol·K) : {features['entropy_mixing']:.2f}")
    print(f"  n_components     : {features['n_components']}")
    print()

    # Design rules
    print("📐 Design Rule Check:")
    r_ratio = features['r_A_r_C']
    if r_ratio < 0.35:
        print("  ✓ r_A/r_C < 0.35 → Fluorite favorable")
    elif r_ratio > 0.45:
        print("  ✓ r_A/r_C > 0.45 → Spinel favorable")
    else:
        print("  ✓ r_A/r_C ~ 0.35-0.45 → Rock-salt favorable")
    print()

    print("=" * 70)

def interactive_mode(model):
    """Interactive prediction mode"""
    print_header()
    print_available_elements()

    while True:
        try:
            elements, fractions = get_composition()
            prediction, probabilities, features = predict(model, elements, fractions)
            print_results(elements, fractions, prediction, probabilities, features)

            print()
            again = input("Predict another composition? (y/n): ").strip().lower()
            if again != 'y':
                break
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print()

def batch_mode(model, composition_list):
    """Batch prediction mode"""
    print_header()
    print(f"Processing {len(composition_list)} compositions...")
    print()

    results = []

    for comp in composition_list:
        try:
            # Parse composition
            parts = comp.split()
            elements = []
            fractions = []

            for part in parts:
                if ':' in part:
                    elem, frac = part.split(':')
                    elements.append(elem.strip())
                    fractions.append(float(frac))
                else:
                    elements.append(part.strip())

            if not fractions:
                fractions = [1.0/len(elements)] * len(elements)

            total = sum(fractions)
            fractions = [f/total for f in fractions]

            # Predict
            prediction, probabilities, features = predict(model, elements, fractions)

            results.append({
                'composition': comp,
                'structure': STRUCTURES[prediction],
                'confidence': probabilities[prediction] * 100,
                'r_A_r_C': features['r_A_r_C']
            })

        except Exception as e:
            print(f"⚠️  Skipped '{comp}': {e}")

    # Print results
    print()
    print("=" * 90)
    print(f"{'Composition':<40} {'Structure':<20} {'Confidence':>12} {'r_A/r_C':>10}")
    print("=" * 90)

    for res in results:
        print(f"{res['composition']:<40} {res['structure']:<20} "
              f"{res['confidence']:>11.1f}% {res['r_A_r_C']:>10.4f}")

    print("=" * 90)
    print(f"Total: {len(results)} compositions processed")

def main():
    """Main function"""
    model = load_model()

    if len(sys.argv) > 1:
        # Batch mode
        compositions = sys.argv[1:]
        batch_mode(model, compositions)
    else:
        # Interactive mode
        interactive_mode(model)

if __name__ == "__main__":
    main()
