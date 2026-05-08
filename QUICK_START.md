# Quick Start: Optimal Composition Finder

## What This Does

**Input**: Elements (e.g., Ce, Zr, Hf, Ti)
**Output**: Optimal proportions that maximize stability + predicted structure

---

## 30-Second Start

```bash
# 1. Find optimal composition
python find_optimal_composition.py Ce Zr Hf Ti

# 2. Find optimal for all 3 structures
python find_optimal_composition.py Ce Zr Hf Ti --all

# 3. Target specific structure
python find_optimal_composition.py Ce Zr Hf Ti --target fluorite
```

---

## How It Works (Simple Explanation)

1. **You provide**: Elements you want to use
2. **Algorithm searches**: All possible proportions (e.g., 10% Ce, 70% Zr, 15% Hf, 5% Ti)
3. **ML model evaluates**: Each composition → structure probability
4. **Optimization finds**: Proportions with highest stability
5. **You get**: Optimal recipe + predicted structure + confidence

---

## Algorithm in 3 Steps

```
Step 1: Start with equal proportions (25%, 25%, 25%, 25%)
        ↓
Step 2: Try adjusting proportions
        Calculate features → ML model → Get confidence score
        If score improves → keep adjustment
        ↓
Step 3: Repeat until no more improvement
        ↓
        Return optimal proportions!
```

**Methods**:
- **Gradient** (SLSQP): Follows "uphill" direction, fast
- **Evolutionary**: Tests many candidates, more thorough

---

## Available Elements

Ce, Ge, Hf, Ir, Mn, Nb, Pb, Pt, Rh, Ru, Sn, Ti, V, Zr
(Use 2-5 at a time)

---

## Example Result

**Input**:
```bash
python find_optimal_composition.py Ce Zr Hf Ti --all
```

**Output**:
```
FLUORITE:   0.069Ce + 0.714Zr + 0.208Hf + 0.009Ti  (98.2% confidence) ✅
ROCK-SALT:  0.347Ce + 0.351Zr + 0.075Hf + 0.227Ti  (50.4% confidence) ⚠️
SPINEL:     0.010Ce + 0.015Zr + 0.011Hf + 0.964Ti  (88.5% confidence) ✅
```

**Interpretation**:
- Zr-rich favors Fluorite (very stable)
- Ti-rich favors Spinel (stable)
- Rock-salt hard to form (less stable)

---

## Files & Documentation

- **OPTIMAL_COMPOSITION_GUIDE.md** - Complete guide with algorithm details
- **IMPLEMENTATION_SUMMARY.md** - Technical summary
- **demo_optimal_composition.py** - Example use cases

---

## Quick Reference

| Task | Command |
|------|---------|
| Find any stable | `python find_optimal_composition.py Ce Zr Hf Ti` |
| Target Fluorite | `python find_optimal_composition.py Ce Zr Hf Ti --target fluorite` |
| All structures | `python find_optimal_composition.py Ce Zr Hf Ti --all` |
| 5 elements | `python find_optimal_composition.py Ce Zr Hf Ti Sn --method evolutionary` |
| Web interface | `streamlit run interactive_predictor.py` |

---

## Old vs New

| Old System | New System |
|------------|------------|
| You guess proportions | Algorithm finds optimal |
| Trial and error | One calculation |
| Many experiments | Minimal experiments |
| → Predict structure | → Predict structure + proportions |

---

**Ready to use!** 🚀
