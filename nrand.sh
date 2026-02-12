#!/bin/bash

# Script to run training experiments with different llava noise configurations
# Usage: ./llava.sh [GPU_ID]
# Example: ./llava.sh 3

# Default GPU if not specified
GPU=${1:-0}
DATASET="CIFAR10"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cll


echo "=========================================="
echo "Running LLaVA experiments on GPU ${GPU}"
echo "=========================================="

# Loop through nrand values from 2 to 5
for iter in {2..4}; do
    # echo ""
    # echo "=========================================="
    # echo "Running with kmean=${KMEAN}"
    # echo "=========================================="
    
    CLL_TYPE="llava_random-noise=True-nrand=${iter}"
    
    echo ""
    echo ">>> Running cpe-f with ${CLL_TYPE}"
    python train.py \
        --algo=cpe-f \
        --dataset_name ${DATASET} \
        --setup_type Dbar[prompt]_T \
        --cll_type "${CLL_TYPE}" \
        --gpu ${GPU} \
        --batch_size 512
    
    # echo ""
    # echo ">>> Running fwd-int with ${CLL_TYPE}"
    # python train.py \
    #     --algo=fwd-int \
    #     --dataset_name ${DATASET} \
    #     --setup_type Dbar[prompt]_T \
    #     --cll_type "${CLL_TYPE}" \
    #     --gpu ${GPU} \
    #     --batch_size 512
    
    echo ""
    echo "Completed nrand=${nrand}"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
