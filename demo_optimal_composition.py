#!/usr/bin/env python3
"""
Demonstration: Find Optimal HEO Compositions
=============================================
Shows the new capability: Elements → Optimal Proportions
(Instead of: Proportions → Structure prediction)
"""

import pickle
from optimize_composition import optimize_composition, multi_structure_optimization

# Load model
print("\n" + "="*70)
print("  DEMONSTRATION: OPTIMAL COMPOSITION FINDER")
print("  From: Elements → Find Optimal Proportions + Predicted Structure")
print("="*70)

with open('heo_model.pkl', 'rb') as f:
    model = pickle.load(f)

structures = {0: 'Fluorite', 1: 'Rock-salt', 2: 'Spinel'}

# ============================================================================
# SCENARIO 1: Materials Scientist wants to make a Fluorite HEO
# ============================================================================
print("\n" + "-"*70)
print("SCENARIO 1: Design a Fluorite structure from Ce, Zr, Hf, Ti")
print("-"*70)
print("Question: What proportions will form the most stable Fluorite?")

result = optimize_composition(
    model,
    elements=['Ce', 'Zr', 'Hf', 'Ti'],
    target_structure=0,  # 0 = Fluorite
    method='evolutionary'
)

print("\nAnswer:")
print(f"  Optimal composition: ", end="")
elements = ['Ce', 'Zr', 'Hf', 'Ti']
for elem, frac in zip(elements, result['optimal_fractions']):
    print(f"{elem}={frac:.3f} ", end="")
print(f"\n  Fluorite probability: {result['structure_probabilities'][0]*100:.1f}%")
print(f"  Overall confidence: {result['confidence']*100:.1f}%")
print(f"  r_A/r_C ratio: {result['features']['r_A_r_C']:.4f}")

# ============================================================================
# SCENARIO 2: Explore what structures are possible with given elements
# ============================================================================
print("\n" + "-"*70)
print("SCENARIO 2: What can we make with Ce, Ti, and Zr?")
print("-"*70)
print("Question: What are the optimal compositions for each structure?")

multi_results = multi_structure_optimization(model, ['Ce', 'Ti', 'Zr'])

print("\nAnswer:")
for struct_name, res in multi_results.items():
    formula = ""
    for elem, frac in zip(['Ce', 'Ti', 'Zr'], res['optimal_fractions']):
        formula += f"{frac:.3f}{elem} + "
    formula = formula.rstrip(" + ")

    print(f"\n  {struct_name}:")
    print(f"    Composition: {formula}")
    print(f"    Confidence:  {res['confidence']*100:.1f}%")
    print(f"    r_A/r_C:     {res['features']['r_A_r_C']:.4f}")

# ============================================================================
# SCENARIO 3: Find the most stable structure overall
# ============================================================================
print("\n" + "-"*70)
print("SCENARIO 3: Maximum stability with Zr, Ti, Hf")
print("-"*70)
print("Question: What composition gives highest overall stability?")

result3 = optimize_composition(
    model,
    elements=['Zr', 'Ti', 'Hf'],
    target_structure=None,  # Find any stable structure
    method='evolutionary'
)

print("\nAnswer:")
print(f"  Predicted structure: {structures[result3['predicted_structure']]}")
print(f"  Optimal composition: ", end="")
for elem, frac in zip(['Zr', 'Ti', 'Hf'], result3['optimal_fractions']):
    print(f"{elem}={frac:.3f} ", end="")
print(f"\n  Confidence: {result3['confidence']*100:.1f}%")
print(f"  ΔS_mix: {result3['features']['entropy_mixing']:.2f} J/mol·K")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("  SUMMARY: What This Tool Does")
print("="*70)
print("""
OLD APPROACH (Manual):
  Input:  User guesses proportions (e.g., 25% Ce + 25% Zr + 25% Hf + 25% Ti)
  Output: Predicts which structure will form

NEW APPROACH (Optimized):
  Input:  Elements only (e.g., Ce, Zr, Hf, Ti)
  Output: OPTIMAL proportions that maximize stability
          + Predicted structure
          + Confidence level
          + Key features (r_A/r_C, ΔS_mix, etc.)

ADVANTAGES:
  ✓ No trial-and-error needed
  ✓ Finds compositions you might not guess
  ✓ Can target specific structures OR find most stable
  ✓ Saves experimental time and materials
  ✓ Explores full composition space systematically

USAGE:
  CLI:       python find_optimal_composition.py Ce Zr Hf Ti --all
  Web UI:    streamlit run interactive_predictor.py
  Python:    from optimize_composition import optimize_composition
""")

print("="*70)
print("  ✅ DEMONSTRATION COMPLETE")
print("="*70)
print()
