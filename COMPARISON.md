# HEO Crystal Structure Predictor - Version Comparison

## 📊 Two Versions Available

### ❌ Version 1 (predict_structure.py) - DATABASE LOOKUP
**Problem: Data Leakage - NOT True Prediction**

- **How it works**: Searches for composition in database, uses pre-calculated DFT features
- **Limitation**: Can ONLY work with compositions already in the dataset
- **Features used**: DeltaH_aPbO2, DeltaH_bad, DeltaH_rut (requires DFT calculations)
- **Accuracy**: 83.8% (artificially high due to data leakage)
- **Use case**: Post-hoc analysis of known compositions

**Example failure:**
```
Input: Ru, Sc, Ti, Nd, V
Output: ❌ Composition not found in dataset
```

---

### ✅ Version 2 (predict_structure_v2.py) - TRUE PREDICTION
**Solution: Composition-based features - Matches Liu et al. (2023) paper**

- **How it works**: Calculates features from elemental properties alone, predicts structure
- **Capability**: Works for ANY composition (even new, untested ones!)
- **Features used**: r_A/r_C, Δχ_Pauling, Δχ_Mulliken, Δδ, ΔS_mix, element composition
- **Accuracy**: 76.9% (true predictive power, no data leakage)
- **Use case**: Discovering new materials before expensive DFT calculations

**Example success:**
```
Input: Ru, Ti, V, Nb (not in training data)
Output: ✅ Predicted: Spinel structure (82.0% confidence)
```

---

## 🔬 Technical Comparison

| Aspect | Version 1 (OLD) | Version 2 (NEW) |
|--------|----------------|-----------------|
| **Prediction Method** | Database lookup | True ML prediction |
| **Works for new compositions** | ❌ No | ✅ Yes |
| **Data leakage** | ❌ Yes | ✅ No |
| **Requires DFT calculations** | ✅ Yes | ❌ No |
| **Test Accuracy** | 83.8% (inflated) | 76.9% (real) |
| **Matches paper methodology** | ❌ No | ✅ Yes |
| **Scientific validity** | ❌ Limited | ✅ High |

---

## 📈 Performance Comparison

### Version 1 (with data leakage):
- Training: 100.0%
- Testing: 83.8%
- **Problem**: Model learns "pick structure with lowest DeltaH" - not useful!

### Version 2 (no data leakage):
- Training: 85.6%
- Testing: 76.9%
- **Advantage**: True predictive power for new materials!

The 7% accuracy difference is the "cost" of doing real prediction vs. database lookup.

---

## 🎯 Which Version Should You Use?

### Use Version 1 if:
- You already have DFT data for all three structures
- You want to compare your DFT results with ML predictions
- You're doing post-hoc analysis

### Use Version 2 if:
- You want to discover NEW materials
- You want to screen compositions BEFORE expensive DFT
- You want scientifically valid predictions
- **You want to match the Liu et al. (2023) paper methodology**

---

## 🚀 Getting Started with Version 2

1. **Train the model** (one-time setup):
   ```bash
   source venv/bin/activate
   python train_model.py
   ```

2. **Run predictions**:
   ```bash
   python predict_structure_v2.py
   ```

3. **Try ANY composition**:
   - Enter elements: `Ru, Ti, V, Nb`
   - Get instant prediction without DFT!

---

## 📚 References

This implementation follows:
- **Liu et al. (2023)** - "Machine learning-based crystal structure prediction for high-entropy oxide ceramics"
- Journal of the American Ceramic Society, DOI: 10.1111/jace.19318

Key methodology:
- Composition-based features (r_A/r_C, electronegativity, etc.)
- XGBoost classifier
- No DFT-calculated properties in features
- True materials discovery capability
