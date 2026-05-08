# 🚀 Quick Start Guide

## What I Created For You

An interactive web app where you can:
1. **Choose cation concentrations** - Select elements and their fractions
2. **Predict structures** - See which structure forms (Fluorite/Rock-salt/Spinel)
3. **Find stable concentrations** - Discover which concentrations give the most stable structures
4. **Analyze sensitivity** - See how changing concentrations affects the structure

## How to Run

### Step 1: Activate Virtual Environment
```bash
cd /Users/putla_theophila/Desktop/UGP
source venv/bin/activate
```

### Step 2: Launch the App
```bash
streamlit run interactive_predictor.py
```

Or use the launcher script:
```bash
./run_app.sh
```

### Step 3: Open in Browser
The app will automatically open at: **http://localhost:8501**

## Three Modes

### 1. 🎯 Quick Predict
- **Use for**: Testing a specific composition
- **Example**: "What structure forms with 25% Hf, 25% Zr, 25% Ti, 25% Sn?"
- **Steps**:
  1. Select number of cations
  2. Choose each element
  3. Enter fractions (auto-normalized)
  4. Click "Predict"
  5. See structure, confidence, and features

### 2. 🔬 Concentration Sweep
- **Use for**: Finding optimal concentrations
- **Example**: "What Ti concentration gives the most stable Fluorite?"
- **Steps**:
  1. Set up base composition (e.g., Hf-Zr-Ti-Sn)
  2. Select element to vary (e.g., Ti)
  3. Click "Run Sweep"
  4. **See graphs showing**:
     - Structure probability vs. concentration
     - Where each structure is most stable
     - r_A/r_C ratio changes

### 3. ⚗️ Batch Screening
- **Use for**: Exploring many compositions
- **Example**: "Find all compositions of Ce, Hf, Zr, Ti that form Fluorite"
- **Steps**:
  1. Select 3-5 elements
  2. Choose target structure (or "All")
  3. Test 50-500 random compositions
  4. Download results as CSV

## Example Workflow: Finding Stable Concentrations

**Question**: "Which Hf-Zr-Ti concentration gives the most stable Fluorite structure?"

1. Launch app: `streamlit run interactive_predictor.py`
2. Go to **"Concentration Sweep"** mode
3. Set base composition:
   - Cation 1: Hf (0.33)
   - Cation 2: Zr (0.33)
   - Cation 3: Ti (0.34)
4. Select "Vary concentration of: Hf"
5. Click "Run Sweep"
6. **Results show**:
   - Graph with Fluorite probability vs. Hf concentration
   - Peak shows optimal Hf concentration for Fluorite
   - Green success message: "Fluorite most stable at Hf = 0.45 (probability: 85%)"

## Available Cations

Ce, Ge, Hf, Ir, Mn, Nb, Pb, Pt, Rh, Ru, Sn, Ti, V, Zr

## Understanding Results

### Confidence Levels
- ✅ **High (>70%)**: Very likely to form
- ⚠️ **Medium (50-70%)**: May form
- ❌ **Low (<50%)**: Unlikely to form

### Key Features
- **r_A/r_C < 0.35** → Fluorite
- **r_A/r_C ~ 0.41** → Rock-salt
- **r_A/r_C > 0.45** → Spinel

## Troubleshooting

**"Model not found"**:
```bash
python train_model.py
```

**"Module not found"**:
```bash
source venv/bin/activate
pip install streamlit plotly
```

**Port in use**:
```bash
streamlit run interactive_predictor.py --server.port 8502
```

## What Makes This Special

This app answers your original question: **"What concentrations give more stable structures?"**

The **Concentration Sweep** mode specifically:
- Tests 30 different concentrations for your chosen element
- Shows probability curves for all three structures
- Highlights where each structure is most stable
- Gives you specific concentration values for maximum stability

Enjoy exploring! 🔬
