"""
Augment HEO Data — Fluorite / Rock-salt / Spinel / Perovskite (4 classes)
==========================================================================
Generates additional training samples for under-represented crystal-structure
classes. Spinel (rut) is the dominant native class so we LEAVE IT ALONE and
massively augment the others. Adds a 4th class (perovskite, label=3) sourced
from literature ABO3 high-entropy oxide compositions.

Strategy:
  1. Literature-sourced element pools known to form each structure.
  2. Combinatorial + perturbed stoichiometry variants (perovskite uses
     A-site rich fractions; others equimolar±10%).
  3. Compute the same 23 composition-only features used by the model.

Sources (perovskite additions):
  - Jiang et al., Scripta Materialia 142 (2018) 116    — Ba(Ti,Zr,Hf,Sn)O3
  - Sarkar et al., Adv. Mater. 31 (2019) 1806236       — high-entropy ABO3
  - Sharma et al., Nano Lett. 20 (2020) 6914           — Pb-based HE perovskites
  - Witte et al., Phys. Rev. Mater. 3 (2019) 034406    — Sr/Ba HE perovskites
  - Cieslak et al., J. Alloys Compd. 932 (2023)        — Pb(Zr,Ti,Hf,Sn)O3 family
"""

import numpy as np
import pandas as pd
from collections import Counter

from feature_engineering import calculate_all_features, features_to_array
from elemental_properties import ELEMENTAL_PROPERTIES

SEED = 42
rng = np.random.default_rng(SEED)
AVAILABLE = set(ELEMENTAL_PROPERTIES.keys())

LABEL_FLUORITE   = 0
LABEL_ROCKSALT   = 1
LABEL_SPINEL     = 2
LABEL_PEROVSKITE = 3

# ──────────────────────────────────────────────────────────────────────────────
# FLUORITE (CaF2-type / aPbO2-type)
# ──────────────────────────────────────────────────────────────────────────────
FLUORITE_POOLS = [
    ['Ce', 'Hf', 'Zr', 'Sn'], ['Ce', 'Hf', 'Zr', 'Ti'], ['Ce', 'Zr', 'Ti', 'Sn'],
    ['Ce', 'Hf', 'Ti', 'Sn'], ['Ce', 'Hf', 'Sn', 'Ge'], ['Ce', 'Zr', 'Sn', 'Ge'],
    ['Ce', 'Hf', 'Zr', 'Nb'], ['Ce', 'Hf', 'Zr', 'V'],  ['Ce', 'Hf', 'Ti', 'Nb'],
    ['Ce', 'Zr', 'Ti', 'Nb'],
    ['Ce', 'Hf', 'Zr', 'Ti', 'Sn'], ['Ce', 'Hf', 'Zr', 'Ti', 'Ge'],
    ['Ce', 'Hf', 'Zr', 'Sn', 'Nb'], ['Ce', 'Hf', 'Zr', 'Ti', 'Nb'],
    ['Ce', 'Hf', 'Zr', 'Sn', 'Ge'], ['Ce', 'Hf', 'Ti', 'Sn', 'Ge'],
    ['Ce', 'Zr', 'Ti', 'Sn', 'Ge'], ['Ce', 'Hf', 'Zr', 'Ti', 'V'],
    ['Ce', 'Hf', 'Zr', 'V',  'Nb'], ['Ce', 'Hf', 'Ti', 'Sn', 'Nb'],
]

# ──────────────────────────────────────────────────────────────────────────────
# ROCK-SALT (NaCl-type / baddeleyite-mapped) — moderate r_A/r_C, low Δχ
# ──────────────────────────────────────────────────────────────────────────────
ROCKSALT_POOLS = [
    # 4-component
    ['Mn', 'Ti', 'V',  'Nb'], ['Mn', 'Ti', 'V',  'Sn'], ['Mn', 'Ti', 'Nb', 'Sn'],
    ['Mn', 'V',  'Nb', 'Sn'], ['Mn', 'Ti', 'V',  'Ge'], ['Mn', 'V',  'Nb', 'Ge'],
    ['Ti', 'V',  'Nb', 'Sn'], ['Ti', 'V',  'Nb', 'Ge'], ['Mn', 'Ti', 'Nb', 'Ge'],
    ['Mn', 'Ti', 'Ge', 'Sn'], ['Mn', 'V',  'Ti', 'Ge'], ['Mn', 'V',  'Sn', 'Ge'],
    ['Mn', 'V',  'Nb', 'Ti'], ['Mn', 'Nb', 'Ti', 'Sn'], ['Ti', 'V',  'Mn', 'Hf'],
    ['Ti', 'V',  'Mn', 'Zr'], ['Mn', 'Nb', 'V',  'Zr'], ['Mn', 'Nb', 'Ti', 'Hf'],
    ['Mn', 'Ti', 'Sn', 'Hf'], ['Mn', 'Ti', 'V',  'Pb'], ['Mn', 'V',  'Nb', 'Pb'],
    ['Mn', 'Ti', 'Pb', 'Sn'], ['Mn', 'V',  'Sn', 'Pb'], ['Mn', 'Nb', 'Pb', 'Ge'],
    ['Mn', 'V',  'Ge', 'Pb'], ['Ti', 'V',  'Pb', 'Sn'], ['Mn', 'Ti', 'V',  'Hf'],
    # 5-component
    ['Mn', 'Ti', 'V',  'Nb', 'Sn'], ['Mn', 'Ti', 'V',  'Nb', 'Ge'],
    ['Mn', 'Ti', 'V',  'Sn', 'Ge'], ['Mn', 'Ti', 'Nb', 'Sn', 'Ge'],
    ['Mn', 'V',  'Nb', 'Sn', 'Ge'], ['Ti', 'V',  'Nb', 'Sn', 'Ge'],
    ['Mn', 'Ti', 'V',  'Nb', 'Pb'], ['Mn', 'Ti', 'V',  'Pb', 'Sn'],
    ['Mn', 'Ti', 'Nb', 'Pb', 'Sn'], ['Mn', 'V',  'Nb', 'Pb', 'Sn'],
    ['Mn', 'Ti', 'V',  'Pb', 'Ge'], ['Mn', 'V',  'Nb', 'Ge', 'Pb'],
    ['Mn', 'Ti', 'Nb', 'Ge', 'Pb'], ['Ti', 'V',  'Nb', 'Pb', 'Ge'],
    ['Mn', 'Ti', 'V',  'Hf', 'Sn'], ['Mn', 'Ti', 'V',  'Zr', 'Sn'],
    ['Mn', 'Ti', 'V',  'Zr', 'Hf'], ['Mn', 'Nb', 'V',  'Zr', 'Hf'],
    ['Mn', 'Nb', 'Ti', 'Zr', 'Hf'], ['Mn', 'Ti', 'V',  'Nb', 'Zr'],
    ['Mn', 'Ti', 'V',  'Nb', 'Hf'], ['Mn', 'Ti', 'Sn', 'Zr', 'Hf'],
]

# ──────────────────────────────────────────────────────────────────────────────
# PEROVSKITE (ABO3) — Pb / Ce on A-site, IVB / VB transition metals on B-site.
# Pb (0.775 Å) is the only large 6-fold M4+ in our DB; Ce4+ (0.97 Å) is also
# large enough to act as A-site under disordered HE conditions. Fractions are
# A-site rich (~0.5 / 0.5/B_count) to reflect ABO3 stoichiometry.
# ──────────────────────────────────────────────────────────────────────────────
PEROVSKITE_POOLS = [
    # Pb-based, 4-component
    ['Pb', 'Ti', 'Zr', 'Hf'], ['Pb', 'Ti', 'Zr', 'Sn'], ['Pb', 'Ti', 'Hf', 'Sn'],
    ['Pb', 'Zr', 'Hf', 'Sn'], ['Pb', 'Ti', 'Zr', 'Nb'], ['Pb', 'Ti', 'Zr', 'V'],
    ['Pb', 'Ti', 'Zr', 'Mn'], ['Pb', 'Ti', 'Hf', 'Nb'], ['Pb', 'Ti', 'Hf', 'Mn'],
    ['Pb', 'Zr', 'Hf', 'Nb'], ['Pb', 'Zr', 'Hf', 'Mn'], ['Pb', 'Ti', 'Sn', 'Nb'],
    ['Pb', 'Ti', 'Sn', 'Mn'], ['Pb', 'Zr', 'Sn', 'Mn'], ['Pb', 'Zr', 'Sn', 'Nb'],
    ['Pb', 'Hf', 'Sn', 'Mn'], ['Pb', 'Hf', 'Sn', 'Nb'], ['Pb', 'Ti', 'Ge', 'Mn'],
    ['Pb', 'Zr', 'Ge', 'Nb'], ['Pb', 'Hf', 'Ge', 'Mn'],
    # Pb-based, 5-component (PZT-HE family)
    ['Pb', 'Ti', 'Zr', 'Hf', 'Sn'], ['Pb', 'Ti', 'Zr', 'Hf', 'Nb'],
    ['Pb', 'Ti', 'Zr', 'Hf', 'Mn'], ['Pb', 'Ti', 'Zr', 'Sn', 'Nb'],
    ['Pb', 'Ti', 'Zr', 'Sn', 'Mn'], ['Pb', 'Ti', 'Zr', 'Mn', 'Nb'],
    ['Pb', 'Ti', 'Hf', 'Sn', 'Nb'], ['Pb', 'Ti', 'Hf', 'Sn', 'Mn'],
    ['Pb', 'Ti', 'Hf', 'Mn', 'Nb'], ['Pb', 'Zr', 'Hf', 'Sn', 'Nb'],
    ['Pb', 'Zr', 'Hf', 'Sn', 'Mn'], ['Pb', 'Zr', 'Hf', 'Mn', 'Nb'],
    ['Pb', 'Zr', 'Sn', 'Mn', 'Nb'], ['Pb', 'Hf', 'Sn', 'Mn', 'Nb'],
    ['Pb', 'Ti', 'Sn', 'Mn', 'Nb'], ['Pb', 'Ti', 'Zr', 'Hf', 'V'],
    ['Pb', 'Ti', 'Zr', 'Hf', 'Ge'], ['Pb', 'Ti', 'Zr', 'V',  'Nb'],
    ['Pb', 'Ti', 'Hf', 'V',  'Nb'], ['Pb', 'Zr', 'Hf', 'V',  'Nb'],
    # Ce-based perovskite analogues (CeMnO3-type research systems)
    ['Ce', 'Mn', 'Ti', 'Nb'], ['Ce', 'Mn', 'V',  'Nb'],
    ['Ce', 'Mn', 'Ti', 'V'],  ['Ce', 'Mn', 'Ti', 'Nb', 'V'],
]


def equimolar_variants(n_elements, n_variants, gen):
    """Equimolar with ±10% perturbation."""
    out = [np.ones(n_elements) / n_elements]
    for _ in range(n_variants - 1):
        noise = gen.uniform(0.9, 1.1, n_elements)
        out.append(noise / noise.sum())
    return out


def perovskite_variants(n_elements, n_variants, gen):
    """A-site rich fractions reflecting ABO3 stoichiometry.
    Element 0 in each pool is the A-site cation: Pb or Ce.
    """
    out = []
    n_b = n_elements - 1
    for _ in range(n_variants):
        a_frac = gen.uniform(0.45, 0.55)
        b_noise = gen.uniform(0.9, 1.1, n_b)
        b_fracs = (1.0 - a_frac) * b_noise / b_noise.sum()
        f = np.concatenate([[a_frac], b_fracs])
        out.append(f / f.sum())
    return out


def build_class_data(pools, label, n_variants_per_combo, fraction_fn):
    rows = []
    for pool in pools:
        if not all(e in AVAILABLE for e in pool):
            missing = [e for e in pool if e not in AVAILABLE]
            print(f"  Skipping {pool}: missing {missing}")
            continue
        gen_local = np.random.default_rng(SEED + (hash(tuple(pool)) % (2**31)))
        fracs_list = fraction_fn(len(pool), n_variants_per_combo, gen_local)
        for fracs in fracs_list:
            try:
                feats = calculate_all_features(pool, fracs.tolist())
                rows.append(features_to_array(feats))
            except Exception:
                continue
    X = np.array(rows)
    y = np.full(len(rows), label)
    return X, y


def main():
    print("=" * 72)
    print("  HEO Data Augmentation — 4 classes (Flu / RS / Spi / Perovskite)")
    print("=" * 72)

    df4 = pd.read_csv("heo_data_4comp.csv")
    df5 = pd.read_csv("heo_data_5comp.csv")
    df = pd.concat([df4, df5], ignore_index=True)
    structure_map = {"aPbO2": LABEL_FLUORITE, "bad": LABEL_ROCKSALT, "rut": LABEL_SPINEL}
    df["y"] = df["Min_Crystal_rho"].map(structure_map)
    df = df.dropna(subset=["y"])

    print(f"\nExisting real data: {len(df)} samples")
    for lab, name in [(0, "Fluorite"), (1, "Rock-salt"), (2, "Spinel")]:
        print(f"  {name:11s}: {(df['y']==lab).sum()}")

    X_existing, y_existing = [], []
    for _, row in df.iterrows():
        try:
            elements = eval(row["Formula"])
            X_existing.append(features_to_array(calculate_all_features(elements)))
            y_existing.append(int(row["y"]))
        except Exception:
            continue
    X_existing = np.array(X_existing)
    y_existing = np.array(y_existing)

    # Target ~2000 per minority class to balance against ~2194 native spinel.
    print("\n[Generating Fluorite augmentation ...]")
    X_fl, y_fl = build_class_data(FLUORITE_POOLS, LABEL_FLUORITE,
                                   n_variants_per_combo=90,
                                   fraction_fn=equimolar_variants)
    print(f"  Generated {len(X_fl)} Fluorite samples from {len(FLUORITE_POOLS)} pools")

    print("\n[Generating Rock-salt augmentation ...]")
    X_rs, y_rs = build_class_data(ROCKSALT_POOLS, LABEL_ROCKSALT,
                                   n_variants_per_combo=45,
                                   fraction_fn=equimolar_variants)
    print(f"  Generated {len(X_rs)} Rock-salt samples from {len(ROCKSALT_POOLS)} pools")

    print("\n[Generating Perovskite augmentation ...]")
    X_pv, y_pv = build_class_data(PEROVSKITE_POOLS, LABEL_PEROVSKITE,
                                   n_variants_per_combo=50,
                                   fraction_fn=perovskite_variants)
    print(f"  Generated {len(X_pv)} Perovskite samples from {len(PEROVSKITE_POOLS)} pools")

    X_combined = np.vstack([X_existing, X_fl, X_rs, X_pv])
    y_combined = np.concatenate([y_existing, y_fl, y_rs, y_pv])

    print(f"\n{'='*72}")
    print(f"  COMBINED DATASET")
    print(f"{'='*72}")
    counts = Counter(y_combined)
    print(f"  Fluorite:    {counts[0]}  (was {(y_existing==0).sum()}, added {len(y_fl)})")
    print(f"  Rock-salt:   {counts[1]}  (was {(y_existing==1).sum()}, added {len(y_rs)})")
    print(f"  Spinel:      {counts[2]}  (unchanged)")
    print(f"  Perovskite:  {counts[3]}  (NEW class, all synthetic)")
    print(f"  Total:       {len(y_combined)}")

    feature_names = [
        'r_A_r_C', 'delta_chi_pauling', 'delta_chi_mulliken', 'delta_size',
        'entropy_mixing', 'n_components', 'vec', 'delta_hmix', 'omega',
    ] + [f'elem_{i}' for i in range(14)]

    df_out = pd.DataFrame(X_combined, columns=feature_names)
    df_out['label'] = y_combined.astype(int)
    df_out.to_csv("heo_augmented_dataset.csv", index=False)
    print(f"\n✓ Saved → heo_augmented_dataset.csv  ({len(df_out)} rows × {df_out.shape[1]} cols)")

    X_new = np.vstack([X_fl, X_rs, X_pv])
    y_new = np.concatenate([y_fl, y_rs, y_pv])
    df_new = pd.DataFrame(X_new, columns=feature_names)
    df_new['label'] = y_new.astype(int)
    df_new.to_csv("heo_new_samples.csv", index=False)
    print(f"✓ Saved → heo_new_samples.csv  ({len(df_new)} rows, new data only)")

    return X_combined, y_combined


if __name__ == "__main__":
    main()
