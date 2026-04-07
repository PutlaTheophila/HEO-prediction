"""
Elemental Properties Database for HEO Crystal Structure Prediction
====================================================================
Contains atomic radii, electronegativity, and other properties needed
for feature engineering from composition alone.
"""

import pandas as pd
import numpy as np

# Elemental properties for all elements in the HEO dataset
# Sources:
# - Shannon ionic radii (for 6-fold coordination, M4+ for most)
# - Pauling electronegativity
# - Atomic radii from various sources

ELEMENTAL_PROPERTIES = {
    'Ce': {
        'ionic_radius': 0.97,   # Å (Ce4+, 6-fold)
        'atomic_radius': 1.825,  # Å
        'pauling_electronegativity': 1.12,
        'mulliken_electronegativity': 1.17,
        'atomic_number': 58,
        'molar_mass': 140.116,
    },
    'Ge': {
        'ionic_radius': 0.53,   # Å (Ge4+, 6-fold)
        'atomic_radius': 1.225,
        'pauling_electronegativity': 2.01,
        'mulliken_electronegativity': 2.31,
        'atomic_number': 32,
        'molar_mass': 72.630,
    },
    'Hf': {
        'ionic_radius': 0.71,   # Å (Hf4+, 6-fold)
        'atomic_radius': 1.575,
        'pauling_electronegativity': 1.3,
        'mulliken_electronegativity': 1.44,
        'atomic_number': 72,
        'molar_mass': 178.49,
    },
    'Ir': {
        'ionic_radius': 0.625,  # Å (Ir4+, 6-fold)
        'atomic_radius': 1.355,
        'pauling_electronegativity': 2.20,
        'mulliken_electronegativity': 2.25,
        'atomic_number': 77,
        'molar_mass': 192.217,
    },
    'Mn': {
        'ionic_radius': 0.53,   # Å (Mn4+, 6-fold)
        'atomic_radius': 1.40,
        'pauling_electronegativity': 1.55,
        'mulliken_electronegativity': 1.72,
        'atomic_number': 25,
        'molar_mass': 54.938,
    },
    'Nb': {
        'ionic_radius': 0.68,   # Å (Nb4+, 6-fold)
        'atomic_radius': 1.45,
        'pauling_electronegativity': 1.6,
        'mulliken_electronegativity': 1.89,
        'atomic_number': 41,
        'molar_mass': 92.906,
    },
    'Pb': {
        'ionic_radius': 0.775,  # Å (Pb4+, 6-fold)
        'atomic_radius': 1.80,
        'pauling_electronegativity': 2.33,
        'mulliken_electronegativity': 1.87,
        'atomic_number': 82,
        'molar_mass': 207.2,
    },
    'Pt': {
        'ionic_radius': 0.625,  # Å (Pt4+, 6-fold)
        'atomic_radius': 1.385,
        'pauling_electronegativity': 2.28,
        'mulliken_electronegativity': 2.21,
        'atomic_number': 78,
        'molar_mass': 195.084,
    },
    'Rh': {
        'ionic_radius': 0.60,   # Å (Rh4+, 6-fold)
        'atomic_radius': 1.345,
        'pauling_electronegativity': 2.28,
        'mulliken_electronegativity': 2.15,
        'atomic_number': 45,
        'molar_mass': 102.906,
    },
    'Ru': {
        'ionic_radius': 0.62,   # Å (Ru4+, 6-fold)
        'atomic_radius': 1.34,
        'pauling_electronegativity': 2.2,
        'mulliken_electronegativity': 2.25,
        'atomic_number': 44,
        'molar_mass': 101.07,
    },
    'Sn': {
        'ionic_radius': 0.69,   # Å (Sn4+, 6-fold)
        'atomic_radius': 1.45,
        'pauling_electronegativity': 1.96,
        'mulliken_electronegativity': 2.12,
        'atomic_number': 50,
        'molar_mass': 118.710,
    },
    'Ti': {
        'ionic_radius': 0.605,  # Å (Ti4+, 6-fold)
        'atomic_radius': 1.475,
        'pauling_electronegativity': 1.54,
        'mulliken_electronegativity': 1.75,
        'atomic_number': 22,
        'molar_mass': 47.867,
    },
    'V': {
        'ionic_radius': 0.58,   # Å (V4+, 6-fold)
        'atomic_radius': 1.35,
        'pauling_electronegativity': 1.63,
        'mulliken_electronegativity': 1.66,
        'atomic_number': 23,
        'molar_mass': 50.942,
    },
    'Zr': {
        'ionic_radius': 0.72,   # Å (Zr4+, 6-fold)
        'atomic_radius': 1.60,
        'pauling_electronegativity': 1.33,
        'mulliken_electronegativity': 1.42,
        'atomic_number': 40,
        'molar_mass': 91.224,
    },
}

# Oxygen properties (anion)
OXYGEN_PROPERTIES = {
    'ionic_radius': 1.40,  # Å (O2-, 6-fold coordination)
    'pauling_electronegativity': 3.44,
    'mulliken_electronegativity': 3.22,
    'atomic_number': 8,
    'molar_mass': 15.999,
}


def get_element_property(element, property_name):
    """Get a property value for an element"""
    if element not in ELEMENTAL_PROPERTIES:
        raise ValueError(f"Element {element} not in database. Available: {list(ELEMENTAL_PROPERTIES.keys())}")
    return ELEMENTAL_PROPERTIES[element][property_name]


def get_available_elements():
    """Return list of available elements"""
    return sorted(list(ELEMENTAL_PROPERTIES.keys()))


def validate_composition(elements):
    """Check if all elements in composition are available"""
    available = set(ELEMENTAL_PROPERTIES.keys())
    elements_set = set(elements)
    missing = elements_set - available

    if missing:
        return False, list(missing)
    return True, []
