#!/bin/bash

# Define target combinations as single strings that mimic a list structure
declare -a target_combinations=(
    "metal"
    "X3_ligand"
    "E_ligand"
)

# Loop through each target combination and run the training script
for targets in "${target_combinations[@]}"; do
    echo "Running model training for targets: $targets"
    # Pass targets as a single comma-separated string, mimicking list-like input
    python scripts/training/one_target.py --target "$targets" --max_evals 2
    echo "Training completed for $targets"
done

echo "All training processes have completed."
