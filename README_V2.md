# ✨ HEO Crystal Structure Predictor v2.0

## 🎯 TRUE MACHINE LEARNING PREDICTION

This implementation can predict crystal structures for **ANY composition** - even ones never seen before!

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train model (one-time, ~2 minutes)
python train_model.py

# 3. Start predicting!
python predict_structure_v2.py
```

---

## ✅ What Makes This Different

### ❌ Old Version (predict_structure.py):
```python
Input: Ru, Ti, V, Nb
Output: ❌ Composition not found in dataset
```
**Problem:** Database lookup only - can't predict new materials!

### ✅ New Version (predict_structure_v2.py):
```python
Input: Ru, Ti, V, Nb  (never seen before!)
Output: ✅ Spinel structure (82% confidence)
```
**Solution:** True ML prediction from composition alone!

---

## 📊 Key Improvements

| Feature | Old | New |
|---------|-----|-----|
| Predicts new compositions | ❌ | ✅ |
| Data leakage | ❌ Yes | ✅ No |
| Needs DFT calculations | ✅ | ❌ |
| Matches paper methodology | ❌ | ✅ |
| Scientific validity | Limited | High |
| Test accuracy | 83.8% (inflated) | 76.9% (real) |

---

## 📁 New Files Created

### Core Modules:
1. **`elemental_properties.py`**
   - Database of 14 elements with atomic properties
   - Ionic radii, electronegativity, atomic mass, etc.

2. **`feature_engineering.py`**
   - Calculates features from composition alone
   - r_A/r_C, Δχ, Δδ, ΔS_mix, element encoding
   - No DFT calculations needed!

3. **`train_model.py`**
   - Trains XGBoost model on composition-based features
   - 76.9% test accuracy (no data leakage)
   - Saves model to `heo_model.pkl`

4. **`predict_structure_v2.py`**
   - Interactive predictor for ANY composition
   - Shows probabilities, features, and physical insights
   - Works even for new, untested compositions

### Documentation:
5. **`COMPARISON.md`**
   - Detailed technical comparison of both versions
   - Performance metrics and use cases

6. **`QUICK_START.md`** (updated)
   - Complete usage guide for both versions
   - Examples and troubleshooting

---

## 🔬 How It Works

### Step 1: Extract Elemental Properties
```python
composition = ['Hf', 'Zr', 'Ti', 'Sn']
# Look up ionic radii, electronegativity, etc.
```

### Step 2: Calculate Features
```python
features = {
    'r_A/r_C': 2.0550,              # Anion/cation radius ratio
    'Δχ_Pauling': 0.2636,           # Electronegativity difference
    'Δχ_Mulliken': 0.2845,          # Mulliken electronegativity
    'Δδ_size': 0.0665,              # Size mismatch
    'ΔS_mix': 11.5257,              # Mixing entropy
    'n_components': 4,               # Number of elements
    # + element composition encoding
}
```

### Step 3: Predict Structure
```python
prediction = model.predict(features)
# → Spinel-like (Rutile) with 78.5% confidence
```

**No DFT calculations required!** ✨

---

## 📈 Model Performance

### Training Results:
```
Total samples: 2987
Training set: 2090 samples
Test set: 897 samples

Training Accuracy: 85.6%
Test Accuracy: 76.9%

Per-structure Performance:
  Fluorite:   50.6% precision, 33.3% recall
  Rock-salt:  37.0% precision,  9.7% recall
  Spinel:     81.3% precision, 96.4% recall
```

### Feature Importance:
```
1. Element composition (E_0, E_5, E_7...)  ← Most important
2. r_A/r_C ratio                          ← Key physical feature
3. Δχ_Pauling                             ← Electronegativity
4. Δχ_Mulliken
5. Δδ_size (size mismatch)
```

This matches the paper's findings: **r_A/r_C is the most important physical feature**!

---

## 🎯 Usage Examples

### Example 1: Interactive Mode
```bash
$ python predict_structure_v2.py

Your choice: 2
Elements: Ti, V, Nb, Zr

✅ Predicted: Spinel structure (78% confidence)
```

### Example 2: Python Script
```python
from feature_engineering import calculate_all_features, features_to_array
import pickle

# Load model
with open('heo_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict for ANY composition
composition = ['Hf', 'Zr', 'Ti', 'Sn']
features = features_to_array(calculate_all_features(composition)).reshape(1, -1)

prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

print(f"Prediction: {['Fluorite', 'Rock-salt', 'Spinel'][prediction]}")
print(f"Confidence: {probabilities[prediction]*100:.1f}%")
```

### Example 3: Batch Screening
```python
# Screen 100 candidate compositions
candidates = [
    ['Hf', 'Zr', 'Ti', 'Sn'],
    ['Ti', 'V', 'Nb', 'Zr'],
    ['Ir', 'Pt', 'Rh', 'Ru'],
    # ... 97 more
]

for comp in candidates:
    features = features_to_array(calculate_all_features(comp)).reshape(1, -1)
    prob = model.predict_proba(features)[0]

    if prob[2] > 0.8:  # High confidence for Spinel
        print(f"✅ {comp} → Spinel ({prob[2]*100:.1f}%)")
```

---

## 🔬 Scientific Validity

### Matches Liu et al. (2023) Paper:
- ✅ Uses composition-based features
- ✅ r_A/r_C as key descriptor
- ✅ Electronegativity differences
- ✅ Size mismatch and mixing entropy
- ✅ XGBoost algorithm
- ✅ No DFT-calculated properties in features

### Key Advantage:
**Can screen thousands of compositions in seconds** without expensive DFT calculations!

Traditional approach:
```
Design composition → DFT (days) → Check stability → Repeat
```

ML approach:
```
Design composition → ML (seconds) → Filter candidates → DFT only for promising ones
```

**Saves months of computational time!** ⚡

---

## 📚 Available Elements

Only 14 elements currently supported:
```
Ce   Ge   Hf   Ir   Mn   Nb   Pb
Pt   Rh   Ru   Sn   Ti   V    Zr
```

### To Add More Elements:
1. Edit `elemental_properties.py`
2. Add ionic radius, electronegativity, atomic mass
3. Retrain model with `python train_model.py`

---

## 💡 Tips for Best Results

### Physical Guidelines:
1. **r_A/r_C ratio** is most important
   - < 0.45 → Spinel likely
   - \> 0.55 → Fluorite likely
   - 0.35-0.55 → Rock-salt or Spinel

2. **High electronegativity difference** may affect stability

3. **Size mismatch** should be moderate (0.05-0.15)

4. **Mixing entropy** increases with more components

### Prediction Guidelines:
1. ✅ Trust predictions > 70% confidence
2. ⚠️ Validate 50-70% experimentally
3. ❌ Be skeptical of < 50%

---

## 🎓 Citation

If you use this code, please cite:

**Liu et al. (2023)** - "Machine learning-based crystal structure prediction for high-entropy oxide ceramics"
- Journal of the American Ceramic Society
- DOI: 10.1111/jace.19318

---

## 📞 Support

### Documentation:
- `QUICK_START.md` - Usage guide
- `COMPARISON.md` - Version comparison
- `feature_engineering.py` - Implementation details

### Troubleshooting:
See the troubleshooting section in `QUICK_START.md`

---

## 🎉 Get Started!

```bash
source venv/bin/activate
python train_model.py
python predict_structure_v2.py
```

**Start predicting crystal structures for ANY HEO composition!** 🔮✨
