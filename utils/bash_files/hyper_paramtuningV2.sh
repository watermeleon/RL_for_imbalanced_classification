#!/bin/bash

#SBATCH --job-name=exp1
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:59:00
#SBATCH --output=./slurmfiles/slurm_output_%A_%a.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
cd $HOME/RL_for_genderbias/

# Activate your environment
source activate rl_bias

# Define the values for each hyperparameter to vary
hyp_lr=(1e-4 3e-4 5e-4)  # Learning rate
hyp_bs=(128 256 512)  # Batch size
hyp_entropyc=(0.01 0.05 0.1 0.2)  # Entropy coefficient
hyp_epsclip=(0.1 0.2 0.3)  # Epsilon clip
hyp_Kepochs=(5 10 15)  # K epochs
hyp_norm_rew=(True False)  # Normalize rewards

# Initialize a counter
counter=1


# Function to run the Python script with the given hyperparameter and all variations of normalize_rewards
run_script() {
    for norm_rew in "${hyp_norm_rew[@]}"; do
        srun --exclusive -n 1 -c 2 python -u train_imbalanced_classification.py \
        --model_name "stokstaartje_${counter}_normrew_$norm_rew" $1 \
        --normalize_rewards $norm_rew &
        ((counter++))
    done
}

# Loop over each hyperparameter while keeping the others at their default values
# Learning rate variations
for lr in "${hyp_lr[@]}"; do
    run_script "--lr $lr"
done

# Batch size variations
for bs in "${hyp_bs[@]}"; do
    run_script "--batch_size $bs"
done

# Entropy coefficient variations
for entropyc in "${hyp_entropyc[@]}"; do
    run_script "--entropy_coef $entropyc"
done

# Epsilon clip variations
for epsclip in "${hyp_epsclip[@]}"; do
    run_script "--eps_clip $epsclip"
done

# K epochs variations
for Kepochs in "${hyp_Kepochs[@]}"; do
    run_script "--K_epochs $Kepochs"
done

# Wait for all background processes to finish
wait