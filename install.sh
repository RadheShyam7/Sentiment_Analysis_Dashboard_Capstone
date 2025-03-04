#!/bin/bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip within the venv
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

echo "Installation complete. To use the environment, run:"
echo "source venv/bin/activate"