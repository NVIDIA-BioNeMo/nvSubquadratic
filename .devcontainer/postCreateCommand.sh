#!/bin/bash

# Install the project in editable mode (with dependencies)
pip install --no-build-isolation --editable .

# Install subquadratic_ops wheel file if not already installed (from Dockerfile)
# GITLAB_TOKEN is required for this installation
echo "GITLAB_TOKEN value: ${GITLAB_TOKEN:-not_set}"
if ! python -c "import subquadratic_ops" 2>/dev/null; then
    if [ -n "${GITLAB_TOKEN}" ] && [ "${GITLAB_TOKEN}" != "not_set" ]; then
        echo "Installing subquadratic-ops with token..."
        pip install subquadratic-ops==v0.0.1+cuda12.9 --index-url https://__token__:${GITLAB_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple
    else
        echo "ERROR: GITLAB_TOKEN is required but not available. Please set GITLAB_TOKEN environment variable."
        exit 1
    fi
else
    echo "subquadratic-ops already installed (from Dockerfile)"
fi

# Set up git safe directory
git config --global --add safe.directory /workspace

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
