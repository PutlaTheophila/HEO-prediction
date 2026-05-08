# Optimal Composition Finder - User Guide

## Overview

**NEW FEATURE**: The system now predicts **optimal element proportions** given a set of elements, rather than just predicting structure from user-provided proportions.

### What Changed?

**Before (Old System):**
- Input: Element proportions (e.g., 25% Ce, 25% La, 25% Pr, 25% Sm)
- Output: Predicted crystal structure (Fluorite, Rock-salt, or Spinel)

**Now (New System):**
- Input: Elements only (e.g., Ce, Zr, Hf, Ti)
- Output: **Optimal proportions** that form the most stable structure
- Plus: Can find optimal compositions for ALL three structures

---

## How It Works

The system uses **numerical optimization** to search the composition space and find the proportions that maximize prediction confidence (stability).

### Optimization Methods

1. **Auto** (recommended): Gradient-based for ≤4 elements, evolutionary for 5 elements
2. **Gradient**: Fast, works well for simple problems
3. **Evolutionary**: Slower but more thorough global search

---

## Usage

### 1. Command Line Interface (CLI)

#### Find Optimal for Any Stable Structure
```bash
python find_optimal_composition.py Ce Zr Hf Ti
```

#### Target Specific Structure
```bash
python find_optimal_composition.py Ce Zr Hf Ti --target fluorite
python find_optimal_composition.py Ce Zr Hf Ti --target rock-salt
python find_optimal_composition.py Ce Zr Hf Ti --target spinel
```

#### Find Optimal for ALL Three Structures
```bash
python find_optimal_composition.py Ce Zr Hf Ti --all
```

#### Use Evolutionary Algorithm (Better for 5 Elements)
```bash
python find_optimal_composition.py Ce Zr Hf Ti Sn --method evolutionary
```

### 2. Interactive Web Interface (Streamlit)

```bash
streamlit run interactive_predictor.py
```

Then select **"Find Optimal Composition"** mode from the sidebar.

Features:
- Select elements from dropdown
- Choose optimization target (any structure or specific)
- One-click optimization
- Visual results with confidence levels
- Option to find optimal for all structures


### 3. Python API
```python
import pickle
from optimize_composition import optimize_composition, multi_structure_optimization

# Load model
with open('heo_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Find optimal for any stable structure
result = optimize_composition(
    model,
    elements=['Ce', 'Zr', 'Hf', 'Ti'],
    target_structure=None,  # or 0=Fluorite, 1=Rock-salt, 2=Spinel
    method='evolutionary'
)

print(f"Optimal: {result['optimal_fractions']}")
print(f"Structure: {result['predicted_structure']}")
print(f"Confidence: {result['confidence']}")

# Find optimal for ALL structures
results = multi_structure_optimization(model, ['Ce', 'Zr', 'Hf', 'Ti'])
for struct_name, res in results.items():
    print(f"{struct_name}: {res['optimal_fractions']}")
```
---
## Available Elements

The current database includes:
- **Ce, Ge, Hf, Ir, Mn, Nb, Pb, Pt, Rh, Ru, Sn, Ti, V, Zr**

(You can use 2-5 elements at a time)

---

## Example Results

### Example 1: Ce-Zr-Hf-Ti System

**Finding optimal for highest confidence:**
```bash
$ python find_optimal_composition.py Ce Zr Hf Ti

Optimal Proportions:
   Ce : 0.0696  ( 6.96%)
   Zr : 0.6936  (69.36%)
   Hf : 0.2284  (22.84%)
   Ti : 0.0084  ( 0.84%)

Predicted Structure: Fluorite
Confidence: 98.2%  ✅ (Very High)
```

**Finding optimal for each structure:**
```bash
$ python find_optimal_composition.py Ce Zr Hf Ti --all

FLUORITE:   0.069Ce + 0.714Zr + 0.208Hf + 0.009Ti  (98.2% confidence)
ROCK-SALT:  0.347Ce + 0.351Zr + 0.075Hf + 0.227Ti  (50.4% confidence)
SPINEL:     0.010Ce + 0.015Zr + 0.011Hf + 0.964Ti  (88.5% confidence)
```

### Interpretation

- **Fluorite** is most stable with Zr-rich composition (98.2% confidence)
- **Spinel** is highly stable with Ti-rich composition (88.5% confidence)
- **Rock-salt** is difficult to stabilize in this system (only 50.4% confidence)

---

## Old Functionality Still Available

The original "Quick Predict" mode is still available in the Streamlit app:
- You can still manually set proportions and predict the structure
- Useful for testing specific compositions
- Good for educational purposes

---

## Technical Details

### How the Algorithm Finds Optimal Compositions

#### High-Level Approach

The optimization system works by **searching the composition space** to find proportions that maximize the ML model's prediction confidence:

```
1. Start with initial guess (usually equal proportions)
2. Evaluate ML model → get structure probabilities
3. Calculate "goodness" score (confidence)
4. Adjust proportions to improve score
5. Repeat steps 2-4 until convergence
6. Return optimal proportions + predicted structure
```

#### Detailed Algorithm Steps

**Step 1: Problem Formulation**
```
Given:
  - Elements: [Ce, Zr, Hf, Ti]
  - ML Model: trained XGBoost classifier
  - Target: Fluorite (or any structure)

Find:
  - Proportions: [x_Ce, x_Zr, x_Hf, x_Ti]

Such that:
  - x_Ce + x_Zr + x_Hf + x_Ti = 1.0  (constraint)
  - Each x_i ≥ 0.01, ≤ 0.95         (bounds)
  - P(Fluorite | composition) is MAXIMIZED
```

**Step 2: Objective Function**
```python
def objective(proportions):
    # Calculate features from composition
    features = calculate_features(elements, proportions)

    # Get ML model prediction probabilities
    probabilities = model.predict_proba([features])[0]
    # probabilities = [P(Fluorite), P(Rock-salt), P(Spinel)]

    # If targeting specific structure:
    score = probabilities[target_structure]

    # If finding any stable structure:
    score = max(probabilities)

    # Return negative (since we minimize)
    return -score
```

**Step 3: Optimization Methods**

**Method A: Gradient-Based (SLSQP)**
- Sequential Least Squares Programming
- Uses gradient information (derivatives)
- Fast convergence (5-20 iterations typically)
- Works well for smooth landscapes
- Algorithm:
  ```
  1. Start: x = [0.25, 0.25, 0.25, 0.25]  (equal proportions)
  2. Compute gradient: ∇f(x)
  3. Find search direction that satisfies constraints
  4. Take step: x_new = x + α * direction
  5. If |f(x_new) - f(x)| < tolerance → STOP
  6. Else: x = x_new, go to step 2
  ```

**Method B: Evolutionary (Differential Evolution)**
- Population-based global search
- Doesn't require gradients
- More robust for complex landscapes
- Better at finding global optimum
- Algorithm:
  ```
  1. Generate population of 15*N compositions (random)
     Example for 4 elements: 60 random compositions
  2. For each generation:
     a. Select random compositions (mutation)
     b. Create new candidate (crossover)
     c. Evaluate fitness (ML model confidence)
     d. Keep better solutions (selection)
  3. Repeat for 300 generations
  4. Return best solution found
  ```

**Step 4: Constraint Handling**

The proportions must sum to 1.0. This is handled differently:

- **Gradient method**: Explicit constraint in optimizer
  ```python
  constraint = {'type': 'eq', 'fun': lambda x: sum(x) - 1.0}
  ```

- **Evolutionary method**: Normalization after each evaluation
  ```python
  proportions = proportions / sum(proportions)  # Always sum to 1
  ```

**Step 5: Convergence**

Optimization stops when:
- Gradient method: Change in objective < 1e-9
- Evolutionary: Max iterations reached (300) OR no improvement for 10 generations

#### Mathematical Formulation

**Optimization Problem**:
```
maximize  P(target_structure | x₁, x₂, ..., xₙ)

subject to:
  x₁ + x₂ + ... + xₙ = 1
  0.01 ≤ xᵢ ≤ 0.95  for all i
```

Where:
- xᵢ = fraction of element i
- P(...) = ML model prediction probability
- n = number of elements (2-5)

**Feature Calculation**:
```
Given proportions → Calculate features:
  r_A/r_C = (Σ r_i × xᵢ) / r_O
  Δχ = sqrt(Σ xᵢ × (χᵢ - χ̄)²)
  Δδ = sqrt(Σ xᵢ × (rᵢ - r̄)²) / r̄
  ΔS_mix = -R × Σ xᵢ × ln(xᵢ)
  ... etc
```

**ML Model**:
```
Features → XGBoost → [P(Fluorite), P(Rock-salt), P(Spinel)]
```

#### Example: Finding Optimal Fluorite

**Initial State**:
```
Elements: [Ce, Zr, Hf, Ti]
Start:    [0.25, 0.25, 0.25, 0.25]
Target:   Fluorite (structure 0)

Evaluate: P(Fluorite) = 0.867 (86.7%)
```

**Iteration 1**:
```
Try:      [0.30, 0.30, 0.30, 0.10]  (less Ti)
Evaluate: P(Fluorite) = 0.921 (92.1%)  ← Better!
```

**Iteration 5**:
```
Try:      [0.10, 0.60, 0.25, 0.05]  (more Zr)
Evaluate: P(Fluorite) = 0.965 (96.5%)  ← Better!
```

**Iteration 15** (Converged):
```
Final:    [0.069, 0.714, 0.208, 0.009]
Evaluate: P(Fluorite) = 0.982 (98.2%)  ← Optimal!
```

**Interpretation**: Zr-rich composition (71%) with moderate Hf (21%) and minimal Ce/Ti strongly favors Fluorite structure.

### Optimization Process

1. **Objective Function**: Maximize prediction probability (confidence)
2. **Constraints**: Proportions sum to 1.0, each ≥ 0.01
3. **Search Space**: All valid compositions in the simplex
4. **Algorithm**:
   - Gradient: Sequential Least Squares Programming (SLSQP)
   - Evolutionary: Differential Evolution

#### Algorithm Flowchart

```
START
  ↓
┌─────────────────────────┐
│ Input: Elements         │
│ [Ce, Zr, Hf, Ti]       │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ Initialize Proportions  │
│ x = [0.25, 0.25,       │
│      0.25, 0.25]       │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ Calculate Features      │
│ - r_A/r_C              │
│ - Δχ, Δδ, ΔS_mix      │
│ - Element encoding     │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ ML Model Prediction     │
│ XGBoost Classifier      │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ Get Probabilities       │
│ [P(F), P(R), P(S)]     │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ Compute Objective       │
│ score = -max(P)        │
└────────┬────────────────┘
         ↓
     ┌───┴───┐
     │ Optimizer│
     │ (SLSQP or│
     │  DiffEv) │
     └───┬───┘
         ↓
    Converged? ────No───┐
         │              │
        Yes             │
         ↓              │
┌─────────────────┐     │
│ Return Optimal  │     │
│ Proportions +   │     │
│ Predicted       │     │
│ Structure       │     │
└─────────────────┘     │
         ↓              │
       STOP             │
                        │
         ↑              │
         │              │
         └──────────────┘
      Adjust proportions
```

#### Why This Works

1. **ML Model as Oracle**: The trained model has learned patterns from experimental data
   - Knows which compositions → which structures
   - Encoded as complex nonlinear function

2. **Optimization Searches Smartly**: Instead of random sampling
   - Explores promising regions more
   - Converges to optimal solution
   - Much faster than brute force

3. **Constraints Ensure Validity**: All solutions are chemically valid
   - Proportions sum to 100%
   - No negative amounts
   - No pure elements (min 1% each)

4. **Multiple Runs for Robustness**: Evolutionary method especially
   - Population-based → explores multiple regions
   - Less likely to get stuck in local optima
   - More reliable for complex systems

### Performance

- 2-4 elements: ~5-10 seconds per structure (gradient method)
- 5 elements: ~20-30 seconds per structure (evolutionary method)
- Finding all 3 structures: 3× the time

**Why Different Times?**
- More elements → larger search space → longer optimization
- Gradient method faster but may find local optima
- Evolutionary method slower but more thorough

---

## ⚠️ Temperature Effects & Limitations

### Important: Temperature Considerations

**Current Status**: The model **does NOT include temperature** as an input feature. Predictions are based on composition and intrinsic properties only.

#### Why Temperature Matters for HEOs

High-entropy oxides can exhibit different crystal structures at different temperatures:

**Temperature-Dependent Phase Behavior**:
```
Synthesis Flow:
  Room Temp  →  Heating (1000-1600°C)  →  Sintering  →  Cooling
      ↓              ↓                       ↓             ↓
  Unreacted    Phase formation      Stable phase   Retained/Transformed?
```

**Common Scenarios**:
- **Fluorite**: Often stable at high temperature (1400-1600°C), may retain on quenching
- **Spinel**: Stable at intermediate temps (800-1300°C), can be metastable at room temp
- **Rock-salt**: Temperature-sensitive, may form mixed phases

#### What the Model Actually Predicts

The model predicts **thermodynamic favorability** based on:
- ✅ Composition (element ratios)
- ✅ Ionic radii (r_A/r_C ratio)
- ✅ Electronegativity differences
- ✅ Entropy of mixing
- ❌ **NOT** synthesis temperature
- ❌ **NOT** cooling rate
- ❌ **NOT** kinetic barriers

**Interpretation**:
```
High Confidence (>80%)
  → Structure is thermodynamically favored
  → Likely forms at typical HEO sintering temps (1200-1500°C)
  → Should be stable across reasonable temperature ranges

Medium Confidence (50-80%)
  → Temperature-sensitive
  → May have competing phases
  → Careful temperature control needed

Low Confidence (<50%)
  → Multiple structures possible
  → Highly dependent on synthesis conditions
  → Requires experimental optimization
```

#### Practical Synthesis Guidelines

**Temperature Ranges for HEO Structures**:

| Structure | Typical Formation Temp | Stability | Notes |
|-----------|----------------------|-----------|-------|
| Fluorite | 1400-1600°C | High-T stable | Fast quench to retain |
| Rock-salt | 1000-1400°C | Intermediate-T | May phase separate |
| Spinel | 800-1300°C | Lower-T stable | Common metastable phase |

**Example: Same Composition, Different Temperatures**
```
Composition: 0.25Ce + 0.25Zr + 0.25Hf + 0.25Ti
Model prediction: Fluorite (86.7% confidence)

Synthesis at different temperatures:
  @ 1500°C: Fluorite forms ✅ (as predicted)
  @ 1200°C: Mixed Fluorite + other phases possible ⚠️
  @ 900°C:  May not form single phase ❌ (kinetic limitations)

Cooling:
  Fast quench (>100°C/min):  Retain high-T Fluorite ✅
  Slow cool (<10°C/min):     Possible phase transitions ⚠️
  Furnace cool:              May decompose or transform ⚠️
```

#### Best Practices

1. **Use High Confidence Predictions**:
   - >80% confidence → More forgiving of temperature variations
   - <50% confidence → Need precise temperature control

2. **Consider Synthesis Temperature**:
   - Start with typical range for predicted structure
   - Fluorite → 1400-1600°C
   - Rock-salt → 1200-1400°C
   - Spinel → 1000-1300°C

3. **Control Cooling**:
   - High confidence phases: Normal cooling OK
   - Low confidence phases: Try quenching to retain structure

4. **Validate with XRD**:
   - Check phase purity after synthesis
   - Compare with model prediction
   - Adjust temperature if needed

#### Current Limitations

**What's Missing**:
1. **Temperature input**: Cannot specify synthesis temperature
2. **Kinetics**: No information about formation rates
3. **Phase transitions**: No prediction of T-dependent transformations
4. **Cooling effects**: No modeling of quench vs. slow cool
5. **Pressure**: No pressure-dependent predictions
6. **Metastability**: Cannot predict if phases are kinetically trapped

**Future Enhancement**: Temperature-dependent optimization could be added by:
- Training model with temperature as additional feature
- Using thermodynamic databases (Gibbs free energy vs. T)
- Incorporating phase diagram data

#### How to Use This Tool for Real Synthesis

**Recommended Workflow**:

1. **Find Optimal Composition**:
   ```bash
   python find_optimal_composition.py Ce Zr Hf Ti --all
   ```

2. **Choose Based on Confidence**:
   - Pick structure with >80% confidence if possible
   - Note the optimal proportions

3. **Plan Synthesis**:
   - Use typical temperature for that structure
   - Plan quenching if needed (especially for Fluorite)
   - Prepare for possible mixed phases if confidence <80%

4. **Synthesize & Characterize**:
   - Perform XRD to verify structure
   - If different from prediction → adjust T or composition

5. **Iterate if Needed**:
   - If wrong structure forms, try:
     - Adjust temperature ±100-200°C
     - Try quenching vs. slow cooling
     - Fine-tune composition slightly

**Example**:
```bash
# Find optimal Fluorite
python find_optimal_composition.py Ce Zr Hf Ti --target fluorite

# Result: 0.07Ce + 0.71Zr + 0.21Hf + 0.01Ti (98.2% confidence)

# Synthesis plan:
# - Mix oxides in these proportions
# - Sinter at 1500°C (high temp for Fluorite)
# - Hold 4-6 hours
# - QUENCH in air or water to retain structure
# - Verify with XRD
```

---

## Tips for Best Results

1. **Start with 4 elements** - Good balance of complexity and optimization speed
2. **Use evolutionary method for 5 elements** - Better at finding global optima
3. **Check all structures** - Sometimes unexpected structures have high stability
4. **Validate experimentally** - These are predictions, not guarantees
5. **Consider practical constraints** - Optimal may not be practically feasible

---

## Files Added

- `optimize_composition.py` - Core optimization algorithms
- `find_optimal_composition.py` - CLI tool
- Updated `interactive_predictor.py` - Added "Find Optimal Composition" mode

---

## Future Enhancements

Potential improvements:
- [ ] Add more elements to database
- [ ] Multi-objective optimization (stability + cost + properties)
- [ ] Composition range constraints (e.g., min 10% of element X)
- [ ] Temperature-dependent predictions
- [ ] Visualization of composition-structure landscape

---

## Questions?

This tool finds **stable compositions** (optimal proportions for stability) rather than just predicting structures from user-chosen proportions.

For support, check the main README or run:
```bash
python find_optimal_composition.py --help
```
