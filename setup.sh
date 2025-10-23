#!/bin/bash

echo "🚀 Setting up Improved ParkRun Predictor..."
echo "=========================================="

# Create directories
mkdir -p models output

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Make script executable
chmod +x park_run_speed_predict.py

echo "✅ Setup completed!"
echo ""
echo "Usage examples:"
echo "  python park_run_speed_predict.py M 5 30"
echo "  python park_run_speed_predict.py F 1 25 --retrain"
echo ""
echo "Happy predicting! 🏃‍♂️"
