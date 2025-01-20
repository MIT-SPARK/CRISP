#!/bin/bash

# Helper script to build the docker image
# Optional arguments
DOCKERFILE=${1:-'pytorch1.13.1-cuda11.6'}
IMAGE_NAME=${2:-'hydra_objects'}
# Directory containing the script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Building Docker image..."
echo "Script director: $SCRIPT_DIR"
echo "Dockerfile: $DOCKERFILE"
echo "Image name to use: $IMAGE_NAME"

cd "$SCRIPT_DIR/../$DOCKERFILE"
docker build . -t $IMAGE_NAME
