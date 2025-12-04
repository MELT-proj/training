#!/bin/bash
# local_setup.sh - Run this on your local machine with internet connection
# This script creates and packages a PyTorch environment for HPC deployment

set -e  # Exit on error

source /etc/profile.d/02-lmod.sh
module load cuda

# Configuration
ENV_NAME="py310_cuda12.4"
CUDA_VERSION="12.4"  # Change this to match your HPC cluster's CUDA version
OUTPUT_FILE="${ENV_NAME}.tar.gz"

echo "================================================"
echo "Creating PyTorch environment for HPC deployment"
echo "================================================"
echo "Environment name: ${ENV_NAME}"
echo "CUDA version: ${CUDA_VERSION}"
echo "Output file: ${OUTPUT_FILE}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if CUDA is available (optional but recommended)
if [ -z "${CUDA_HOME}" ]; then
    echo "Warning: CUDA_HOME is not set or empty"
    echo "This is OK if you're building on a machine without CUDA,"
    echo "but make sure the HPC cluster has CUDA ${CUDA_VERSION} available."
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
else
    echo "CUDA_HOME detected: ${CUDA_HOME}"
    if command -v nvcc &> /dev/null; then
        CUDA_LOCAL_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        echo "Local CUDA version: ${CUDA_LOCAL_VERSION}"
        echo "Target CUDA version: ${CUDA_VERSION}"
    fi
    echo ""
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n ${ENV_NAME} -y
fi

# Create new environment
echo "Creating new conda environment..."
conda create -n ${ENV_NAME} python=3.10 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
conda install torchvision torchaudio pytorch-cuda=${CUDA_VERSION} -c pytorch -c nvidia -y

# Install additional dependencies from conda-forge
echo "Installing additional packages from conda-forge..."
conda install -c conda-forge \
    numpy pandas scikit-learn matplotlib jupyter \
    transformers \
    wandb datasets tyro librosa pysoundfile sentencepiece accelerate -y

# Install packages only available via pip
echo "Installing packages from pip (deepspeed)..."
pip install --no-cache-dir deepspeed

echo ""
echo "Environment creation completed successfully!"
echo ""
echo "Do you want to pack this environment for HPC deployment?"
echo "This will install conda-pack and create a ${OUTPUT_FILE} file."
read -p "Pack environment? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Install conda-pack if not already installed
    echo "Installing conda-pack..."
    conda install conda-pack -c conda-forge -y

    # Pack the environment
    echo "Packing environment (this may take a few minutes)..."
    conda pack -n ${ENV_NAME} -o ${OUTPUT_FILE}

    # Get file size
    FILE_SIZE=$(du -h ${OUTPUT_FILE} | cut -f1)

    echo ""
    echo "================================================"
    echo "SUCCESS! Environment packed successfully"
    echo "================================================"
    echo "Output file: ${OUTPUT_FILE}"
    echo "File size: ${FILE_SIZE}"
    echo ""
    echo "Next steps:"
    echo "1. Transfer the file to your HPC cluster:"
    echo "   scp ${OUTPUT_FILE} username@hpc-cluster:/path/to/destination/"
    echo ""
    echo "2. Run remote_setup.sh on the HPC cluster"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "SUCCESS! Environment created successfully"
    echo "================================================"
    echo "Environment name: ${ENV_NAME}"
    echo ""
    echo "The environment is ready to use locally."
    echo "If you want to pack it later for HPC deployment, you can run:"
    echo "conda activate ${ENV_NAME}"
    echo "conda install conda-pack -c conda-forge"
    echo "conda pack -n ${ENV_NAME} -o ${OUTPUT_FILE}"
    echo "================================================"
fi