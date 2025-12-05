#!/bin/bash
# Quick start script for Traffic Prediction Dashboard

echo "🚗 Traffic Flow Prediction Dashboard"
echo "===================================="
echo ""

# Check if model exists
if [ ! -f "checkpoints/best_model.pt" ] && [ ! -f "checkpoints_colab/best_model.pt" ]; then
    echo "❌ No trained model found!"
    echo ""
    echo "Please train the model first:"
    echo "  python3 scripts/train_colab_safe.py"
    echo ""
    exit 1
fi

# Check if data exists
if [ ! -f "data/pems_bay_processed.npz" ]; then
    echo "❌ Processed data not found!"
    echo ""
    echo "Please run data preprocessing first."
    echo "See notebooks/01_data_exploration.ipynb"
    echo ""
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import streamlit, plotly, torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies!"
    echo ""
    echo "Installing required packages..."
    pip install streamlit plotly scipy
    echo ""
fi

echo "✅ All checks passed!"
echo ""
echo "🚀 Starting dashboard..."
echo "   URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Launch Streamlit
streamlit run app.py
