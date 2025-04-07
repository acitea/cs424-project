#!/bin/bash

# Download the U2NET model
echo "Downloading U2NET model if needed..."
python download_model.py

# Start the FastAPI server
echo "Starting the API server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000