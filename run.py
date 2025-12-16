import subprocess
import glob
import os

print("Starting training processes...")

# Get list of llava_v2-[8:11] to llava_v2-[16:19] files from transition_matrix/cifar10
cll_types = []
for i in range(8, 17):
    for j in range(i+3, min(i+4, 20)):
        cll_type = f'llava_as-[{i}:{j}]'
        cll_types.append(cll_type)
print(f"Found CLL types: {cll_types}")

# GPUs to cycle through
gpus = [2, 3]

# List to hold processes
processes = []
gpu_index = 0

# For each cll_type, run with both algos
for cll_type in cll_types:
    for algo in ['cpe-f', 'fwd-int']:
        gpu = gpus[gpu_index % len(gpus)]
        cmd = f"python train.py --algo={algo} --dataset_name CIFAR20 --setup_type Dbar[prompt]_T --cll_type {cll_type} --gpu {gpu}"
        processes.append(subprocess.Popen(cmd, shell=True))
        gpu_index += 1

# Wait for all processes to complete
for p in processes:
    p.wait()