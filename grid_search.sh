#!/bin/bash
# Grid search on R0, R1, R2, lr, and sampler using GNU parallel
# sampler: normal or topological

# Record the start time
start_time=$(date +%s)
# Generate arrays of parameter values
lr_value=("0.005" "0.01" "0.05")
r0_values=("0.00001" "0.00005" "0.0001" "0.0005" "0.001" "0.005" "0.01" "0.05" "0.1")
r1=("1.0")
r2=("3.0")
sampler_values=("normal" "topological")
ndim_values=("2" "3")
height_values=("8" "64")
# Generate a list of all parameter combinations and pipe to GNU parallel
for ndim in "${ndim_values[@]}"; do
    for height in "${height_values[@]}"; do
        for sampler in "${sampler_values[@]}"; do
            for r0 in "${r0_values[@]}"; do
                # Output parameters separated by spaces
                echo "$r0 $r1 $r2 $sampler $ndim $height"
            done
        done
    done
done | parallel -j 6 --joblog job.log --colsep ' ' "python ./ter.py --R0 {1} --R1 {2} --R2 {3} --loss TB --growth_parameter 0.1 --n_trajectories 100000 --sampler {4} --ndim {5} --height {6} --wandb_project topological-sampler"
# Record the end time and compute elapsed time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Script finished at $(date)"
echo "Total elapsed time: ${elapsed} seconds"