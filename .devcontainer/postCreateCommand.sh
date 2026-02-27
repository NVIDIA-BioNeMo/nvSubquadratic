#!/bin/bash

# Install nvsubq first (if it exists)
#if [ -d "/workspaces/nvSubquadratic" ]; then
#    echo "Installing nvSubquadtratic in editable mode..."
#    cd /workspaces/nvSubquadratic
#    pip install --no-build-isolation --editable .
#    git config --global --add safe.directory /workspaces/nvSubquadratic
#    cd /workspaces/nvSubquadratic-private
#fi

# Install the project in editable mode first so all nvsubq_paper deps are satisfied
# (avoids "dependency conflicts" when nvSubquadratic is installed before this)
echo "Installing nvsubq_paper in editable mode (with dependencies)..."
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

# Install nvSubquadratic (nvsubq) if not already installed (code uses: from nvsubq import ...)
if ! python -c "import nvsubq" 2>/dev/null; then
    echo "nvsubq (nvSubquadratic) not found, attempting to install..."
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | grep GITHUB_TOKEN | xargs)
    fi
    if [ -n "${GITHUB_TOKEN}" ]; then
        echo "Installing nvSubquadratic with token from .env..."
        pip install --no-cache-dir "git+https://${GITHUB_TOKEN}@github.com/NVIDIA-Digital-Bio/nvSubquadratic.git" && echo "nvSubquadratic installed successfully"
    else
        echo "No GITHUB_TOKEN in .env; trying public clone..."
        pip install --no-cache-dir "git+https://github.com/NVIDIA-Digital-Bio/nvSubquadratic.git" && echo "nvSubquadratic installed successfully" || echo "WARNING: Install failed. For a private repo, add GITHUB_TOKEN to .env"
    fi
else
    echo "nvsubq (nvSubquadratic) already installed"
fi


# Install pre-commit hooks if .pre-commit-config.yaml exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type pre-push
else
    echo "No .pre-commit-config.yaml found, skipping pre-commit installation"
fi

exit 0
