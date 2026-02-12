#!/bin/bash

# Script to run comb-oc experiments with different gamma values
# Usage: ./run_comb_oc.sh <dataset> <percentage> <gpu>
# Example: ./run_comb_oc.sh CIFAR10 5 0

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <dataset> <percentage> <gpu>"
    echo "Example: $0 CIFAR10 5 0"
    exit 1
fi

DATASET=$1
PERCENTAGE=$2
GPU=$3

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cll

# Validate percentage
if ! [[ "$PERCENTAGE" =~ ^[0-9]+$ ]] || [ "$PERCENTAGE" -le 0 ] || [ "$PERCENTAGE" -ge 100 ]; then
    echo "Error: Percentage must be a positive integer between 1 and 99"
    exit 1
fi

# Define dataset properties (num_classes, total_samples)
declare -A DATASET_INFO
DATASET_INFO["CIFAR10"]="10:50000"
DATASET_INFO["CIFAR20"]="20:50000"
DATASET_INFO["CIFAR100"]="100:50000"
DATASET_INFO["MNIST"]="10:60000"
DATASET_INFO["FashionMNIST"]="10:60000"
DATASET_INFO["KMNIST"]="10:60000"
DATASET_INFO["Tiny200"]="200:100000"

# Get dataset info
if [ -z "${DATASET_INFO[$DATASET]}" ]; then
    echo "Error: Unknown dataset '$DATASET'"
    echo "Supported datasets: ${!DATASET_INFO[@]}"
    exit 1
fi

IFS=':' read -r NUM_CLASSES TOTAL_SAMPLES <<< "${DATASET_INFO[$DATASET]}"

# Calculate ord_num per class
SAMPLES_PER_CLASS=$((TOTAL_SAMPLES / NUM_CLASSES))
ORD_NUM=$(awk "BEGIN {printf \"%.0f\", $SAMPLES_PER_CLASS * $PERCENTAGE / 100}")

echo "========================================"
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Percentage: $PERCENTAGE%"
echo "  GPU: $GPU"
echo "  Total samples: $TOTAL_SAMPLES"
echo "  Classes: $NUM_CLASSES"
echo "  Samples per class: $SAMPLES_PER_CLASS"
echo "  ord_num (ordinary samples per class): $ORD_NUM"
echo "========================================"
echo ""

# Array to store PIDs
PIDS=()

# Function to run training with specific gamma
run_training() {
    local gamma=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training with gamma=$gamma"
    python train.py \
        --algo=comb-oc \
        --dataset_name=$DATASET \
        --setup_type=Dbar_T \
        --ord_num=$ORD_NUM \
        --gamma=$gamma \
        --cll_type=random \
        --gpu=$GPU
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed training with gamma=$gamma"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Spawn 3 processes for different gamma values
for gamma in 0 0.5 1; do
    run_training $gamma &
    PIDS+=($!)
    echo "Spawned process with PID ${PIDS[-1]} for gamma=$gamma"
    # Small delay to avoid race conditions
    sleep 5
done

echo ""
echo "All 3 processes spawned:"
echo "  Gamma 0.0: PID ${PIDS[0]}"
echo "  Gamma 0.5: PID ${PIDS[1]}"
echo "  Gamma 1.0: PID ${PIDS[2]}"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background processes
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Process ${PIDS[$i]} (gamma ${gammas[$i]}) completed successfully"
    else
        echo "Process ${PIDS[$i]} (gamma ${gammas[$i]}) failed with exit code $exit_code"
    fi
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
