#!/bin/bash
# Grid search on R0, R1, R2, and sampler using GNU parallel
# R0: 0.0001 to 0.5 (steps)
# R1: 0.1 to 2 (steps)
# R2: 0.5 to 3 (steps)
# sampler: normal or topological

# Record the start time
start_time=$(date +%s)

# Function to generate a linspace using Python
linspace() {
    python3 -c "import numpy as np; print('\n'.join(map(str, np.linspace($1, $2, $3))))"
}

# Number of steps for each parameter
steps=5
echo "steps: $steps"

# Generate arrays of parameter values
r0_values=($(linspace 0.0001 0.5 "$steps"))
r1_values=($(linspace 0.1 2 "$steps"))
r2_values=($(linspace 0.5 3 "$steps"))
sampler_values=("normal" "topological")

# Generate a list of all parameter combinations and pipe to GNU parallel
for sampler in "${sampler_values[@]}"; do
    for r0 in "${r0_values[@]}"; do
        for r1 in "${r1_values[@]}"; do
            for r2 in "${r2_values[@]}"; do
                # Output parameters separated by spaces
                echo "$r0 $r1 $r2 $sampler"
            done
        done
    done
done | parallel -j 10 --colsep ' ' "python ./ter.py --R0 {1} --R1 {2} --R2 {3} --loss TB --growth_parameter 0.1 --n_trajectories 200000 --sampler {4}"
# Record the end time and compute elapsed time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Script finished at $(date)"
echo "Total elapsed time: ${elapsed} seconds"