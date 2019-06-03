#!/bin/bash

echo "Creating virtual environment.."
python3 -m venv --p Python3 .venv

echo "Installing requirements.."
source .venv/bin/activate
pip install -r requirements.txt

echo "Installing virtual environment as ipykernel..."
python -m ipykernel install --user --name .venv --display-name "Python (DP Memory Example)"

echo "Installation complete, kernel should be installed with name 'Python (DP Memory Example)'"
