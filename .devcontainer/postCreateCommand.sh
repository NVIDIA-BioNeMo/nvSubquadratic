#!/bin/bash

# Install the project in editable mode
pip install --no-deps --no-build-isolation --editable .

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

exit 0
