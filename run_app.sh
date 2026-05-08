#!/bin/bash

echo "🔬 Starting HEO Crystal Structure Predictor..."
echo ""

# Check if model exists
if [ ! -f "heo_model.pkl" ]; then
    echo "⚠️  Model file not found!"
    echo "Please train the model first by running:"
    echo "  python train_model.py"
    echo ""
    exit 1
fi

# Launch Streamlit app
streamlit run interactive_predictor.py
