#!/bin/bash

# Define target combinations as single strings that mimic a list structure
declare -a target_combinations=(
    "metal E_ligand"
    "metal X3_ligand"
    "X3_ligand E_ligand"
    "metal E_ligand X3_ligand"
)

# Loop through each target combination and run the training script
for targets in "${target_combinations[@]}"; do
    echo "Running model training for targets: $targets"
    # Pass targets as a single comma-separated string, mimicking list-like input
    python scripts/training/multi_targets.py --target "$targets" --max_evals 50 --include_structural False
    echo "Training completed for $targets"
done

echo "All training processes have completed."
