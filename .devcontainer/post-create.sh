#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Installing uv with pipx ---"
pipx install uv

echo "--- Installing project dependencies ---"
# This single command replicates the `uv sync` and `uv pip install` steps
# by installing all dependencies from your pyproject.toml and locking them.
uv sync --all-extras --dev

echo "--- Ensuring testing and notebook tools are installed ---"
# While `uv sync` should handle this, we explicitly install them
# to perfectly match the CI pipeline and add ipykernel.
uv pip install tox pytest pytest-cov papermill ipykernel

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

