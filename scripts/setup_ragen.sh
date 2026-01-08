#!/bin/bash

# Exit on error
set -e

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "No CUDA GPU detected"
        return 1
    fi
}

# Function to check if micromamba is available
check_micromamba() {
    if command -v micromamba &> /dev/null; then
        echo "Micromamba is available"
        return 0
    else
        echo "Micromamba is not installed. Please install Micromamba first."
        return 1
    fi
}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
main() {
    # Check prerequisites
    check_micromamba || exit 1
    
    # Create and activate micromamba environment
    # if not exists, create it
    if ! micromamba env list | grep -q "ragen"; then
        print_step "Creating micromamba environment 'ragen' with Python 3.12..."
        micromamba create -n ragen python=3.12 -y
    else
        print_step "Micromamba environment 'ragen' already exists"
    fi
    
    # Need to source micromamba for script environment
    eval "$(micromamba shell hook -s bash)"
    micromamba activate ragen

    pip install -U pip setuptools wheel
    pip install numpy ninja packaging psutil

    if check_cuda; then
        print_step "CUDA detected, checking CUDA version..."
        
        if command -v nvcc &> /dev/null; then
            nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            nvcc_major=$(echo $nvcc_version | cut -d. -f1)
            nvcc_minor=$(echo $nvcc_version | cut -d. -f2)
            
            print_step "Found NVCC version: $nvcc_version"
            
            if [[ "$nvcc_major" -gt 12 || ("$nvcc_major" -eq 12 && "$nvcc_minor" -ge 1) ]]; then
                print_step "CUDA $nvcc_version is already installed and meets requirements (>=12.4)"
                export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
            else
                print_step "CUDA version < 12.4, installing CUDA toolkit 12.4..."
                micromamba install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
                export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/ragen
            fi
        else
            print_step "NVCC not found, installing CUDA toolkit 12.4..."
            micromamba install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
            export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/ragen
        fi
        
        print_step "Installing PyTorch with CUDA support..."
        pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
        
        print_step "Installing flash-attention..."
        # Try to install prebuilt wheel first, fallback to building from source if it fails
        pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl || \
        pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
    else
        print_step "Installing PyTorch without CUDA support..."
        pip install torch==2.5.0
    fi
    
    # Install package in editable mode
    print_step "setting up verl..."
    git submodule init
    git submodule update
    cd verl
    pip install -e . --no-dependencies # we put dependencies in requirements.txt
    cd ..
    
    # Install package in editable mode (without dependencies to avoid reinstalling flash-attn)
    print_step "Installing ragen package..."
    pip install -e . --no-dependencies

    # Install spatial environment dependencies
    print_step "Installing spatial environment dependencies..."
    pip install -e ragen/env/spatial/Base
    
    # installing webshop
    print_step "Installing webshop dependencies..."
    micromamba install -c pytorch -c conda-forge faiss-cpu -y
    sudo apt update
    sudo apt install default-jdk -y
    micromamba install -c conda-forge openjdk=21 maven -y

    # Install remaining requirements
    print_step "Installing additional requirements..."
    pip install -r requirements.txt

    # webshop installation, model loading
    pip install -e external/webshop-minimal/ --no-dependencies
    python -m spacy download en_core_web_sm
    python -m spacy download en_core_web_lg

    print_step "Downloading data..."
    python scripts/download_data.py

    # Optional: download full data set
    print_step "Downloading full data set..."
    micromamba install conda-forge::gdown -y
    mkdir -p external/webshop-minimal/webshop_minimal/data/full
    cd external/webshop-minimal/webshop_minimal/data/full
    gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB # items_shuffle
    gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi # items_ins_v2
    cd ../../../../..

    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "To activate the environment, run: micromamba activate ragen"


}

# Run main installation
main
