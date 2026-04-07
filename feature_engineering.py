"""
Feature Engineering for HEO Crystal Structure Prediction
==========================================================
Calculates composition-based features from element lists alone.
Based on Liu et al. (2023) methodology.
"""

import numpy as np
from elemental_properties import ELEMENTAL_PROPERTIES, OXYGEN_PROPERTIES


def calculate_r_A_r_C(elements, fractions=None):
    """
    Calculate anion-to-cation radius ratio (r_A/r_C)
    This is the most important feature according to the paper.

    Args:
        elements: List of cation elements
        fractions: Molar fractions (if None, assumes equal fractions)

    Returns:
        r_A/r_C ratio
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)

    # Calculate average cation radius (weighted by fractions)
    avg_cation_radius = sum(
        ELEMENTAL_PROPERTIES[elem]['ionic_radius'] * frac
        for elem, frac in zip(elements, fractions)
    )

    # Anion radius (oxygen)
    anion_radius = OXYGEN_PROPERTIES['ionic_radius']

    return anion_radius / avg_cation_radius


def calculate_delta_chi_pauling(elements, fractions=None):
    """
    Calculate difference in Pauling electronegativity (ΔχPauling)
    Using standard deviation as measure of variation.
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)

    chi_values = [ELEMENTAL_PROPERTIES[elem]['pauling_electronegativity'] for elem in elements]

    # Weighted mean
    chi_mean = sum(chi * frac for chi, frac in zip(chi_values, fractions))

    # Weighted standard deviation
    variance = sum(frac * (chi - chi_mean)**2 for chi, frac in zip(chi_values, fractions))

    return np.sqrt(variance)


def calculate_delta_chi_mulliken(elements, fractions=None):
    """
    Calculate difference in Mulliken electronegativity (ΔχMulliken)
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)

    chi_values = [ELEMENTAL_PROPERTIES[elem]['mulliken_electronegativity'] for elem in elements]

    # Weighted mean
    chi_mean = sum(chi * frac for chi, frac in zip(chi_values, fractions))

    # Weighted standard deviation
    variance = sum(frac * (chi - chi_mean)**2 for chi, frac in zip(chi_values, fractions))

    return np.sqrt(variance)


def calculate_delta_size(elements, fractions=None):
    """
    Calculate cation size mismatch (Δδ)
    Measures the variation in ionic radii.
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)

    radii = [ELEMENTAL_PROPERTIES[elem]['ionic_radius'] for elem in elements]

    # Average radius
    r_mean = sum(r * frac for r, frac in zip(radii, fractions))

    # Size mismatch parameter (from Hume-Rothery rules)
    delta = np.sqrt(sum(frac * (1 - r/r_mean)**2 for r, frac in zip(radii, fractions)))

    return delta


def calculate_entropy_mixing(elements, fractions=None):
    """
    Calculate configurational entropy of mixing (ΔSmix)

    ΔSmix = -R Σ(ci * ln(ci))
    where R is gas constant and ci are molar fractions
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)

    R = 8.314  # J/(mol·K) - gas constant

    # Configurational entropy
    entropy = -R * sum(c * np.log(c) if c > 0 else 0 for c in fractions)

    return entropy


def encode_elements(elements, fractions=None):
    """
    Encode element composition using label encoding approach from paper.
    Returns a feature vector representing the composition.

    For simplicity, we'll use a binary encoding of which elements are present.
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)

    # All possible elements in order
    all_elements = ['Ce', 'Ge', 'Hf', 'Ir', 'Mn', 'Nb', 'Pb', 'Pt', 'Rh', 'Ru', 'Sn', 'Ti', 'V', 'Zr']

    # Create encoding vector with fractions
    encoding = np.zeros(len(all_elements))
    for elem, frac in zip(elements, fractions):
        if elem in all_elements:
            idx = all_elements.index(elem)
            encoding[idx] = frac

    return encoding


def calculate_all_features(elements, fractions=None):
    """
    Calculate all features for a composition.

    Args:
        elements: List of element symbols (e.g., ['Hf', 'Zr', 'Ti', 'Sn'])
        fractions: Molar fractions (if None, assumes equimolar)

    Returns:
        Dictionary of features
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)

    # Normalize fractions to sum to 1
    total = sum(fractions)
    fractions = [f/total for f in fractions]

    features = {
        'r_A_r_C': calculate_r_A_r_C(elements, fractions),
        'delta_chi_pauling': calculate_delta_chi_pauling(elements, fractions),
        'delta_chi_mulliken': calculate_delta_chi_mulliken(elements, fractions),
        'delta_size': calculate_delta_size(elements, fractions),
        'entropy_mixing': calculate_entropy_mixing(elements, fractions),
        'n_components': len(elements),
    }

    # Add element encoding
    elem_encoding = encode_elements(elements, fractions)
    for i, val in enumerate(elem_encoding):
        features[f'elem_{i}'] = val

    return features


def features_to_array(features_dict):
    """Convert features dictionary to numpy array in consistent order"""
    # Define feature order
    feature_order = [
        'r_A_r_C',
        'delta_chi_pauling',
        'delta_chi_mulliken',
        'delta_size',
        'entropy_mixing',
        'n_components',
    ]

    # Add element encoding features
    for i in range(14):  # 14 elements
        feature_order.append(f'elem_{i}')

    return np.array([features_dict[f] for f in feature_order])


if __name__ == "__main__":
    # Test with example composition
    test_elements = ['Hf', 'Zr', 'Ti', 'Sn']
    features = calculate_all_features(test_elements)

    print("Test composition:", test_elements)
    print("\nCalculated features:")
    for key, value in features.items():
        if not key.startswith('elem_'):
            print(f"  {key}: {value:.4f}")
