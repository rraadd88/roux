#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Installing uv with pipx ---"
pipx install uv

echo "--- Installing project dependencies ---"
# Installs all extras plus the `dev` group (tox, pytest, papermill, ipykernel etc.),
# matching the CI pipeline.
uv sync --all-extras

echo "--- Installing Jupyter kernel for notebooks ---"
# This creates a kernel named 'roux' that will be available in VS Code
# and points to the Python interpreter inside the .venv created by uv.
uv run python -m ipykernel install --user --name "roux" --display-name "Python (roux)"

# //"postCreateCommand": "
sudo npm install -g @google/gemini-cli
# ",
# //"postStartCommand": "
gemini --version
# "  

echo "--- Dev container setup complete ---"

