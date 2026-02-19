#!/bin/bash
# Auto-train and update LSTM model
# Usage: sudo ./auto_train_lstm.sh

# Ensure we are in the project root directory
cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found in $(pwd)"
    exit 1
fi

# Configuration
DURATION=300  # 5 minutes
LOG_FILE="logs/auto_train.log"
mkdir -p logs

echo "==================================================" >> "$LOG_FILE"
echo "Starting scheduled LSTM training at $(date)" >> "$LOG_FILE"

# Run training with live traffic capture
# This expects to be run with sudo
python scripts/train_lstm_model.py --duration "$DURATION" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully." >> "$LOG_FILE"
    echo "Model files updated in models/" >> "$LOG_FILE"
    echo "The detector should automatically reload the new model within 60 seconds." >> "$LOG_FILE"
else
    echo "❌ Training failed. Check logs above." >> "$LOG_FILE"
fi

echo "Finished at $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"
