# 🔬 Interactive HEO Crystal Structure Predictor

An interactive web application to predict High-Entropy Oxide (HEO) crystal structures based on cation concentrations.

## Features

### 🎯 Quick Predict Mode
- Select cations and their concentrations
- Get instant structure predictions (Fluorite, Rock-salt, or Spinel)
- View confidence levels and probabilities
- See calculated features (r_A/r_C, Δχ, Δδ, ΔS_mix)
- Understand design rules

### 🔬 Concentration Sweep Mode
- Analyze how varying one element's concentration affects the structure
- Visualize structure probability vs. concentration
- See r_A/r_C ratio changes
- Find optimal concentrations for target structures
- **Understand which concentrations give more stable structures**

### ⚗️ Batch Screening Mode
- Test hundreds of random compositions
- Filter by target structure
- Visualize structure distribution map
- Download results as CSV
- Identify high-confidence compositions

## Installation

1. Install required packages:
```bash
pip install -r requirements_streamlit.txt
```

Or install individually:
```bash
pip install streamlit plotly
```

2. Make sure you have trained the model first:
```bash
python train_model.py
```

This will create `heo_model.pkl` which is required for predictions.

## Running the App

### Option 1: Using the launcher script
```bash
./run_app.sh
```

### Option 2: Direct command
```bash
streamlit run interactive_predictor.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage Guide

### Quick Predict
1. Select the number of cations (2-5)
2. Choose each cation from the dropdown
3. Enter the fraction/concentration for each (should sum to 1.0)
4. Click "Predict Structure"
5. View the predicted structure, confidence, and detailed features

### Finding Stable Concentrations
1. Switch to "Concentration Sweep" mode
2. Set up your base composition
3. Select which element to vary
4. Click "Run Sweep"
5. The graphs will show:
   - Which concentrations favor which structure
   - Where each structure is most stable
   - How r_A/r_C ratio changes

**Example**: If you want to know what concentration of Ti gives the most stable Fluorite structure in a Hf-Zr-Ti-Sn system:
- Set up the base composition with all 4 elements
- Select Ti as the element to vary
- Run the sweep
- Look for the peak in the Fluorite probability curve

### Batch Screening
1. Switch to "Batch Screening" mode
2. Select 3-5 elements to explore
3. Choose target structure (or "All")
4. Set number of compositions to test (50-500)
5. Click "Run Screening"
6. View top candidates and download results

## Available Elements

The following cations are available:
- Ce, Ge, Hf, Ir, Mn, Nb, Pb, Pt, Rh, Ru, Sn, Ti, V, Zr

## Structure Types

| Structure | Typical r_A/r_C | Characteristics |
|-----------|----------------|-----------------|
| **Fluorite** | < 0.35 | Low electronegativity difference, high coordination |
| **Rock-salt** | ~ 0.41 | Moderate properties, 6-fold coordination |
| **Spinel** | ~ 0.45 | Higher electronegativity difference, mixed coordination |

## Interpreting Results

### Confidence Levels
- **High (>70%)**: Structure is very likely to form
- **Medium (50-70%)**: Structure may form, consider synthesis conditions
- **Low (<50%)**: Structure formation uncertain

### Stability Analysis
In Concentration Sweep mode:
- **Peak probability**: Most stable concentration for that structure
- **Plateau regions**: Stable across concentration range
- **Sharp transitions**: Sensitive to composition changes

### Key Features
- **r_A/r_C**: Most important predictor (anion-to-cation radius ratio)
- **Δχ**: Electronegativity differences (Pauling and Mulliken)
- **Δδ**: Atomic size mismatch
- **ΔS_mix**: Configurational entropy of mixing

## Tips

1. **Equal fractions**: For quick tests, use equal concentrations (auto-calculated)
2. **Normalize**: Make sure fractions sum to 1.0
3. **Explore**: Use Concentration Sweep to understand sensitivity
4. **Screen**: Use Batch mode to discover unexpected compositions
5. **Design rules**: Check if your composition follows the design rules

## Troubleshooting

**Model not found error**:
```bash
python train_model.py
```

**Import errors**:
```bash
pip install -r requirements_streamlit.txt
```

**Port already in use**:
```bash
streamlit run interactive_predictor.py --server.port 8502
```

## Citation

Based on methodology from:
- Liu et al., J. Am. Ceram. Soc. 2024;107:1361-1371
- DOI: 10.1111/jace.19518

## Questions?

The app includes built-in help text and expandable sections for detailed information.
