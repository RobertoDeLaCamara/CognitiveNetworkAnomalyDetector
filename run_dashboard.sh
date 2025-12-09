#!/bin/bash
# Launch script for the Cognitive Anomaly Detector Dashboard

echo "🚀 Starting Cognitive Anomaly Detector Dashboard..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    source venv/bin/activate
else
    echo "⚠ Virtual environment not found. Using system Python."
fi

# Check if Streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not installed. Installing dependencies..."
    pip install -r requirements.txt
fi

# Parse configuration from environment variables or use defaults
PORT="${STREAMLIT_SERVER_PORT:-8501}"
ADDRESS="${STREAMLIT_SERVER_ADDRESS:-localhost}"

# Launch dashboard
echo ""
echo "🎯 Dashboard will be available at: http://${ADDRESS}:${PORT}"
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run dashboard.py \
    --server.port "${PORT}" \
    --server.address "${ADDRESS}" \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#ff4b4b"
