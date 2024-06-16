#!/bin/bash

#SBATCH --job-name=exp1
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
hyp_lr=(3e-4 5e-4)  # Learning rate
hyp_Kepochs=(5 10)  # K epochs
hyp_norm_rew=(True False)  # Normalize rewards
# hyp_global_double_loss=(True False)  # Global double loss

# Initialize a counter
counter=1

for lr in "${hyp_lr[@]}"; do
    for Kepochs in "${hyp_Kepochs[@]}"; do
        for norm_rew in "${hyp_norm_rew[@]}"; do
            # Run the Python script with the current combination of hyperparameters and a unique name in the background
            srun --exclusive -n 1 -c 2 python -u train_imbalanced_classification.py \
            --model_name "orca8_imb_DoubleLoss_$counter" \
            --train_type "ppo" \
            --env_imb True \
            --rs_relative_perc True \
            --scale_reward_prof_imb False \
            --lr $lr \
            --K_epochs $Kepochs \
            --eps_clip $epsclip \
            --normalize_rewards $norm_rew \
            --global_double_loss True \
            --num_epoch 10 &
            # Increment the counter
            ((counter++))
        done
    done
done
wait