#!/usr/bin/env python3
"""
Find Optimal HEO Composition (CLI Tool)
========================================
Given elements, find the optimal proportions for maximum stability
"""

import pickle
import argparse
import sys
from optimize_composition import optimize_composition, multi_structure_optimization

# Structure names
STRUCTURES = {0: 'Fluorite', 1: 'Rock-salt', 2: 'Spinel'}


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal HEO composition given elements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find optimal composition for any stable structure
  python find_optimal_composition.py Ce La Pr Sm

  # Optimize for specific structure (Fluorite)
  python find_optimal_composition.py Ce La Pr Sm --target fluorite

  # Find optimal for all three structures
  python find_optimal_composition.py Ce La Pr Sm --all

  # Use evolutionary algorithm (better for complex searches)
  python find_optimal_composition.py Ce Zr Hf Ti Sn --method evolutionary
        """
    )

    parser.add_argument('elements', nargs='+', help='Element symbols (e.g., Ce La Pr Sm)')
    parser.add_argument(
        '--target', '-t',
        choices=['fluorite', 'rock-salt', 'spinel'],
        help='Target structure (if not specified, finds highest confidence)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Find optimal compositions for ALL structures'
    )
    parser.add_argument(
        '--method', '-m',
        choices=['auto', 'gradient', 'evolutionary'],
        default='auto',
        help='Optimization method (default: auto)'
    )
    parser.add_argument(
        '--model', '-M',
        default='heo_model.pkl',
        help='Path to trained model (default: heo_model.pkl)'
    )

    args = parser.parse_args()

    # Load model
    print("\n" + "=" * 70)
    print("  🔬 HEO OPTIMAL COMPOSITION FINDER")
    print("=" * 70)

    try:
        print(f"\n[1/3] Loading model from {args.model}...")
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        print("   ✓ Model loaded successfully")
    except FileNotFoundError:
        print(f"\n   ❌ Error: Model file '{args.model}' not found!")
        print("   Please train the model first using: python train_model.py")
        sys.exit(1)

    # Validate elements
    elements = [e.capitalize() for e in args.elements]
    if len(elements) < 2:
        print("\n   ❌ Error: Need at least 2 elements!")
        sys.exit(1)

    if len(elements) != len(set(elements)):
        print("\n   ❌ Error: Duplicate elements detected!")
        sys.exit(1)

    print(f"\n[2/3] Selected elements: {', '.join(elements)}")

    # Determine target structure
    target_structure = None
    if args.target:
        target_map = {'fluorite': 0, 'rock-salt': 1, 'spinel': 2}
        target_structure = target_map[args.target]
        print(f"   Target structure: {args.target.title()}")
    else:
        print("   Target: Any stable structure (highest confidence)")

    print(f"   Optimization method: {args.method}")

    # Run optimization
    if args.all:
        print("\n[3/3] Finding optimal compositions for ALL structures...")
        print("   (This may take a minute...)\n")

        results = multi_structure_optimization(model, elements)

        print("=" * 70)
        print("  RESULTS: Optimal Compositions for Each Structure")
        print("=" * 70)

        for struct_name, result in results.items():
            print(f"\n  {struct_name.upper()}")
            print("  " + "-" * 68)

            # Composition
            print("  Optimal Composition:")
            for elem, frac in zip(elements, result['optimal_fractions']):
                print(f"     {elem:3s}: {frac:.4f}  ({frac*100:5.2f}%)")

            # Formula
            formula = " + ".join([
                f"{frac:.3f}{elem}"
                for elem, frac in zip(elements, result['optimal_fractions'])
            ])
            print(f"\n  Formula: {formula}")

            # Confidence
            confidence = result['confidence'] * 100
            print(f"\n  Confidence: {confidence:.1f}%", end="")
            if confidence > 80:
                print("  ✅ (Very High)")
            elif confidence > 70:
                print("  ✅ (High)")
            elif confidence > 50:
                print("  ⚠️  (Medium)")
            else:
                print("  ❌ (Low)")

            # Probabilities
            print("\n  Structure Probabilities:")
            for i, prob in enumerate(result['structure_probabilities']):
                bar_len = int(prob * 40)
                bar = "█" * bar_len + "░" * (40 - bar_len)
                print(f"     {STRUCTURES[i]:12s} {bar} {prob*100:5.1f}%")

            # Key features
            features = result['features']
            print(f"\n  Key Features:")
            print(f"     r_A/r_C: {features['r_A_r_C']:.4f}")
            print(f"     Δχ (Pauling): {features['delta_chi_pauling']:.4f}")
            print(f"     ΔS_mix: {features['entropy_mixing']:.2f} J/mol·K")

    else:
        print("\n[3/3] Running optimization...")
        print("   (This may take a moment...)\n")

        result = optimize_composition(
            model,
            elements,
            target_structure=target_structure,
            method=args.method
        )

        print("=" * 70)
        print("  OPTIMAL COMPOSITION FOUND")
        print("=" * 70)

        # Composition
        print("\n  Optimal Proportions:")
        for elem, frac in zip(elements, result['optimal_fractions']):
            print(f"     {elem:3s}: {frac:.4f}  ({frac*100:5.2f}%)")

        # Formula
        formula = " + ".join([
            f"{frac:.3f}{elem}"
            for elem, frac in zip(elements, result['optimal_fractions'])
        ])
        print(f"\n  Formula: {formula}")

        # Predicted structure
        predicted_struct = STRUCTURES[result['predicted_structure']]
        print(f"\n  Predicted Structure: {predicted_struct}")

        # Confidence
        confidence = result['confidence'] * 100
        print(f"  Confidence: {confidence:.1f}%", end="")
        if confidence > 80:
            print("  ✅ (Very High)")
        elif confidence > 70:
            print("  ✅ (High)")
        elif confidence > 50:
            print("  ⚠️  (Medium)")
        else:
            print("  ❌ (Low)")

        # Structure probabilities
        print("\n  Structure Probabilities:")
        for i, prob in enumerate(result['structure_probabilities']):
            bar_len = int(prob * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"     {STRUCTURES[i]:12s} {bar} {prob*100:5.1f}%")

        # Key features
        features = result['features']
        print(f"\n  Calculated Features:")
        print(f"     r_A/r_C:          {features['r_A_r_C']:.4f}")
        print(f"     Δχ (Pauling):     {features['delta_chi_pauling']:.4f}")
        print(f"     Δχ (Mulliken):    {features['delta_chi_mulliken']:.4f}")
        print(f"     Δδ (size):        {features['delta_size']:.4f}")
        print(f"     ΔS_mix:           {features['entropy_mixing']:.2f} J/mol·K")
        print(f"     Components:       {features['n_components']}")

        # Design rules
        print("\n  Design Rules Check:")
        r_ratio = features['r_A_r_C']
        if r_ratio < 0.35:
            print("     ✓ r_A/r_C < 0.35 → Fluorite favorable")
        elif r_ratio > 0.45:
            print("     ✓ r_A/r_C > 0.45 → Spinel favorable")
        else:
            print("     ✓ r_A/r_C ~ 0.35-0.45 → Rock-salt favorable")

    print("\n" + "=" * 70)
    print("  ✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
