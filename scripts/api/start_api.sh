#!/bin/bash

# TE-AI Evolutionary API Startup Script

echo "Starting TE-AI Evolutionary Algorithm API..."

# Activate virtual environment if it exists
if [ -f "../../venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../../venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Check for required dependencies
echo "Checking dependencies..."
python -c "import fastapi" 2>/dev/null || { echo "Installing fastapi..."; pip install fastapi; }
python -c "import uvicorn" 2>/dev/null || { echo "Installing uvicorn..."; pip install uvicorn; }
python -c "import websockets" 2>/dev/null || { echo "Installing websockets..."; pip install websockets; }

# Set Python path to include project root
export PYTHONPATH="${PYTHONPATH}:../.."

# Check for GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "No GPU detected, will run on CPU (slower performance)"
fi

# Start the API server
echo "Starting API server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

# Run with uvicorn for better performance
uvicorn evolutionary_api:app --host 0.0.0.0 --port 8000 --reload