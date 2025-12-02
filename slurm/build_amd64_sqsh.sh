#!/bin/bash

# Check if JET_GITLAB_TOKEN is set
if [ -z "$GITLAB_TOKEN" ]; then
    echo "Error: GITLAB_TOKEN environment variable is not set"
    echo "Please set it with: export GITLAB_TOKEN=your_token_here"
    exit 1
fi

# Build the subquadratic_ops wheel file for amd64
docker buildx build --no-cache \
    --build-arg GITLAB_TOKEN=$GITLAB_TOKEN \
    -t nvsubquadratic-amd64 \
    -f ../Dockerfile ..

# Import the Docker image to enroot sqsh format
enroot import -o nvsubquadratic-amd64.sqsh dockerd://nvsubquadratic-amd64