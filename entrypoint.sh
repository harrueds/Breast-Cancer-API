#!/bin/bash
set -e

echo "===================================="
echo "Running training..."
echo "===================================="

uv run python -m src.api_breast_cancer_prediction.training

echo "===================================="
echo "Training complete. Starting API..."
echo "===================================="

uv run python -m src.api_breast_cancer_prediction.app
