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


def calculate_vec(elements, fractions=None):
    """
    Calculate Valence Electron Concentration (VEC).
    VEC = Σ(ci * VECi)  where VECi is the valence electron count of element i.
    High VEC (>8) tends toward FCC/Fluorite; low VEC (<6.87) tends toward BCC/Rock-salt.
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)
    return sum(
        ELEMENTAL_PROPERTIES[elem]['valence_electron_count'] * frac
        for elem, frac in zip(elements, fractions)
    )


def calculate_delta_hmix(elements, fractions=None):
    """
    Estimate enthalpy of mixing (ΔHmix) using a Miedema-inspired pairwise model.
    ΔHmix ≈ Σ_i≠j [ 4 * ΔHij * ci * cj ]
    where ΔHij is approximated from the electronegativity difference:
        ΔHij ≈ -2 * (χi - χj)^2  (kJ/mol, empirical)
    This is a simplified but widely-used estimate when full Miedema tables
    are unavailable.
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)
    fractions = list(fractions)
    n = len(elements)
    h_mix = 0.0
    for i in range(n):
        chi_i = ELEMENTAL_PROPERTIES[elements[i]]['pauling_electronegativity']
        for j in range(i + 1, n):
            chi_j = ELEMENTAL_PROPERTIES[elements[j]]['pauling_electronegativity']
            delta_h_ij = -2.0 * (chi_i - chi_j) ** 2   # kJ/mol, simplified
            h_mix += 4.0 * delta_h_ij * fractions[i] * fractions[j]
    return h_mix


def calculate_omega(elements, fractions=None):
    """
    Calculate the Omega stability criterion (Liu & Zhang, 2012).
    Omega = T_m * ΔSmix / |ΔHmix|
    Omega >= 1  →  entropy dominates  →  solid-solution (single-phase) favoured.
    Returns 10.0 (capped) when |ΔHmix| ≈ 0 (fully ideal mixing).
    """
    if fractions is None:
        fractions = [1.0/len(elements)] * len(elements)
    fractions = list(fractions)

    # Weighted average melting point
    t_m = sum(
        ELEMENTAL_PROPERTIES[elem]['melting_point'] * frac
        for elem, frac in zip(elements, fractions)
    )
    ds_mix = calculate_entropy_mixing(elements, fractions)   # J/(mol·K)
    dh_mix = calculate_delta_hmix(elements, fractions)        # kJ/mol
    dh_mix_j = abs(dh_mix) * 1000.0                          # convert to J/mol

    if dh_mix_j < 1e-6:          # near-ideal mixing
        return 10.0               # cap at 10 (large positive value)
    return (t_m * ds_mix) / dh_mix_j


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
        # ── New thermodynamic descriptors ──────────────────────────────────
        'vec': calculate_vec(elements, fractions),
        'delta_hmix': calculate_delta_hmix(elements, fractions),
        'omega': calculate_omega(elements, fractions),
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
        # New thermodynamic descriptors
        'vec',
        'delta_hmix',
        'omega',
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
