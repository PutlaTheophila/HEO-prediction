"""
Optimize HEO Composition for Maximum Stability
===============================================
Given elements, find the optimal proportions that form the most stable structure
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from feature_engineering import calculate_all_features, features_to_array


def optimize_composition(model, elements, target_structure=None, method='auto'):
    """
    Find optimal element proportions for maximum stability.

    Parameters
    ----------
    model : trained model
        The XGBoost model for structure prediction
    elements : list of str
        List of element symbols (e.g., ['Ce', 'La', 'Pr', 'Sm'])
    target_structure : int, optional
        If specified (0=Fluorite, 1=Rock-salt, 2=Spinel), optimize for that structure.
        If None, optimize for highest overall confidence.
    method : str
        'auto', 'gradient', or 'evolutionary'

    Returns
    -------
    dict with:
        - optimal_fractions: optimized element proportions
        - predicted_structure: predicted structure (0, 1, or 2)
        - confidence: prediction confidence (0-1)
        - structure_probabilities: probabilities for each structure
    """

    n_elements = len(elements)

    if n_elements < 2:
        raise ValueError("Need at least 2 elements")

    # Define objective function
    def objective(fractions):
        """Return negative confidence (for minimization)"""
        try:
            # Ensure non-negative and normalized
            fractions = np.abs(fractions)
            fractions = fractions / fractions.sum()

            # Calculate features
            features_dict = calculate_all_features(elements, fractions.tolist())
            features_array = features_to_array(features_dict)

            # Get prediction probabilities
            probabilities = model.predict_proba([features_array])[0]

            if target_structure is not None:
                # Optimize for specific structure
                confidence = probabilities[target_structure]
            else:
                # Optimize for highest overall confidence
                confidence = probabilities.max()

            # Return negative (since we minimize)
            return -confidence

        except Exception as e:
            # If calculation fails, return high penalty
            return 1000.0

    # Constraints: fractions sum to 1
    def constraint_sum(fractions):
        return np.sum(fractions) - 1.0

    # Initial guess: equal proportions
    x0 = np.ones(n_elements) / n_elements

    # Bounds: each fraction between 0.01 and 0.95
    bounds = [(0.01, 0.95) for _ in range(n_elements)]

    # Choose optimization method
    if method == 'gradient' or (method == 'auto' and n_elements <= 4):
        # Use gradient-based optimization for small problems
        constraints = {'type': 'eq', 'fun': constraint_sum}

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )

        optimal_fractions = result.x

    else:
        # Use evolutionary algorithm for better global search
        # Convert constraint to bounds that sum to 1
        def objective_with_normalization(fractions):
            # Normalize before evaluation
            fractions = np.abs(fractions)
            fractions = fractions / fractions.sum()
            return objective(fractions)

        result = differential_evolution(
            objective_with_normalization,
            bounds,
            maxiter=300,
            seed=42,
            atol=1e-8,
            tol=1e-8,
            workers=1
        )

        optimal_fractions = result.x

    # Normalize final result
    optimal_fractions = np.abs(optimal_fractions)
    optimal_fractions = optimal_fractions / optimal_fractions.sum()

    # Get final prediction
    features_dict = calculate_all_features(elements, optimal_fractions.tolist())
    features_array = features_to_array(features_dict)

    prediction = model.predict([features_array])[0]
    probabilities = model.predict_proba([features_array])[0]
    confidence = probabilities[prediction]

    return {
        'optimal_fractions': optimal_fractions,
        'predicted_structure': int(prediction),
        'confidence': float(confidence),
        'structure_probabilities': probabilities.tolist(),
        'features': features_dict
    }


def multi_structure_optimization(model, elements):
    """
    Find optimal compositions for ALL three structure types.

    Returns the best composition for forming Fluorite, Rock-salt, and Spinel.
    """

    structures = {
        0: 'Fluorite',
        1: 'Rock-salt',
        2: 'Spinel'
    }

    results = {}

    for struct_idx, struct_name in structures.items():
        result = optimize_composition(
            model,
            elements,
            target_structure=struct_idx,
            method='evolutionary'
        )
        results[struct_name] = result

    return results


def composition_stability_landscape(model, elements, n_samples=1000):
    """
    Sample the composition space to understand stability landscape.

    Returns compositions and their predicted structures for visualization.
    """

    n_elements = len(elements)
    results = []

    # Generate random compositions using Dirichlet distribution
    for _ in range(n_samples):
        fractions = np.random.dirichlet(np.ones(n_elements))

        try:
            features_dict = calculate_all_features(elements, fractions.tolist())
            features_array = features_to_array(features_dict)

            prediction = model.predict([features_array])[0]
            probabilities = model.predict_proba([features_array])[0]
            confidence = probabilities[prediction]

            results.append({
                'fractions': fractions,
                'structure': int(prediction),
                'confidence': float(confidence),
                'probabilities': probabilities,
                'features': features_dict
            })

        except:
            continue

    return results


if __name__ == "__main__":
    # Example usage
    import pickle

    print("=" * 70)
    print("  COMPOSITION OPTIMIZATION EXAMPLE")
    print("=" * 70)

    # Load model
    print("\n[1] Loading model...")
    with open('heo_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("   ✓ Model loaded")

    # Example 1: Find optimal composition for any stable structure
    print("\n[2] Finding optimal composition (any structure)...")
    elements = ['Ce', 'La', 'Pr', 'Sm']

    result = optimize_composition(model, elements, method='evolutionary')

    print(f"\n   Elements: {elements}")
    print(f"   Optimal proportions:")
    for elem, frac in zip(elements, result['optimal_fractions']):
        print(f"      {elem}: {frac:.3f} ({frac*100:.1f}%)")

    structures = {0: 'Fluorite', 1: 'Rock-salt', 2: 'Spinel'}
    print(f"\n   Predicted structure: {structures[result['predicted_structure']]}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"\n   Probabilities:")
    for i, prob in enumerate(result['structure_probabilities']):
        print(f"      {structures[i]}: {prob*100:.1f}%")

    # Example 2: Find optimal compositions for each structure
    print("\n[3] Finding optimal compositions for EACH structure...")
    multi_results = multi_structure_optimization(model, elements)

    for struct_name, res in multi_results.items():
        print(f"\n   {struct_name}:")
        print(f"      Composition: " + " + ".join([
            f"{frac:.2f}{elem}"
            for elem, frac in zip(elements, res['optimal_fractions'])
        ]))
        print(f"      Confidence: {res['confidence']*100:.1f}%")

    print("\n" + "=" * 70)
    print("  ✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
