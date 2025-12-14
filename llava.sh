#!/bin/bash

# Script to run training experiments with different llava noise configurations
# Usage: ./llava.sh [GPU_ID]
# Example: ./llava.sh 3

# Default GPU if not specified
GPU=${1:-3}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cll


echo "=========================================="
echo "Running LLaVA experiments on GPU ${GPU}"
echo "=========================================="

# Loop through nrand values from 2 to 5
for nrand in {8..10}; do
    echo ""
    echo "=========================================="
    echo "Running with nrand=${nrand}"
    echo "=========================================="
    
    CLL_TYPE="llava_noise=True-nrand=${nrand}"
    
    echo ""
    echo ">>> Running cpe-f with ${CLL_TYPE}"
    python train.py \
        --algo=cpe-f \
        --dataset_name CIFAR100 \
        --setup_type Dbar[prompt]_T \
        --cll_type "${CLL_TYPE}" \
        --gpu ${GPU}
    
    echo ""
    echo ">>> Running fwd-int with ${CLL_TYPE}"
    python train.py \
        --algo=fwd-int \
        --dataset_name CIFAR100 \
        --setup_type Dbar[prompt]_T \
        --cll_type "${CLL_TYPE}" \
        --gpu ${GPU}
    
    echo ""
    echo "Completed nrand=${nrand}"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
