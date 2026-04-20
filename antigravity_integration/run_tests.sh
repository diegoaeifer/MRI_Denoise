#!/bin/bash
echo "==========================================="
echo "Running Antigravity Local GPU Test Suite"
echo "Target HW: 16GB VRAM / 32GB RAM"
echo "==========================================="

# Ensure pytest is installed
if ! command -v pytest &> /dev/null
then
    echo "pytest could not be found, installing..."
    pip install pytest
fi

# Run the isolated test suite
PYTHONPATH=.. pytest test_antigravity.py -v --tb=short
