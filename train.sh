#!/bin/bash

# Check if imb_factor argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide imb_factor as an argument"
    echo "Usage: bash train_scl_nl.sh <imb_factor>"
    echo "Example: bash train_scl_nl.sh 0.01"
    exit 1
fi

# Get imb_factor from command line argument
IMB_FACTOR=$1

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cll

# # Run mising run 
# python train.py \
#         --algo "fwd-int" \
#         --dataset_name MNIST \
#         --model resnet18 \
#         --imb_type exp \
#         --imb_factor ${IMB_FACTOR} \
#         --mixup true \
#         --alpha 0.2 \
#         --k_cluster 50 \
#         --new_data_aug none \
#         --data_aug true \
#         --aug_type flipflop \
#         --gpu 1


# Define parameter arrays
# ALGOS=("scl-nl" "fwd-int" "scl-exp" "lw")
# NEW_DATA_AUGS=("icm" "none")
ALGOS=("scl-nl" "fwd-int")
# ALGOS=("scl-exp" "lw")
NEW_DATA_AUGS=("none" "icm")

echo "Using imb_factor=${IMB_FACTOR}"

# Loop through all combinations
for algo in "${ALGOS[@]}"; do
    for new_data_aug in "${NEW_DATA_AUGS[@]}"; do
        echo "Running configuration: imb_factor=${IMB_FACTOR}, algo=${algo}, new_data_aug=${new_data_aug}"

        # Run each configuration 3 times
        for run in {1..3}; do
            echo "Run ${run}/3 for imb_factor=${IMB_FACTOR}, algo=${algo}, new_data_aug=${new_data_aug}"
            python train.py \
            --algo=${algo} \
            --dataset_name PCLCIFAR20 \
            --model resnet18 \
            --imb_type exp \
            --imb_factor ${IMB_FACTOR} \
            --mixup true \
            --alpha 0.2 \
            --k_cluster 50 \
            --new_data_aug ${new_data_aug} \
            --data_aug true \
            --aug_type flipflop \
            --gpu 1
        done
    done
done