#!/bin/bash

# Install nvsubq first (if it exists)
if [ -d "/workspaces/nvSubquadratic" ]; then
    echo "Installing nvSubquadtratic in editable mode..."
    cd /workspaces/nvSubquadratic
    pip install --no-build-isolation --editable .
    git config --global --add safe.directory /workspaces/nvSubquadratic
    cd /workspaces/nvSubquadratic-private
fi

# Install the project in editable mode (with dependencies)
pip install --no-build-isolation --editable .

# Set up git safe directory
git config --global --add safe.directory /workspaces/nvSubquadratic-private

# Configure bash history search (up/down arrows)
echo 'Setting up bash history search...'
cat >> ~/.bashrc << 'EOF'

# Enhanced history search with up/down arrows
bind '"\e[A": history-search-backward'
bind '"\e[B": history-search-forward'
EOF

# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type pre-push
else
    echo "No .pre-commit-config.yaml found, skipping pre-commit installation"
fi

exit 0
