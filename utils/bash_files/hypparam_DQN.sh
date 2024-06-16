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
hyp_decay_min=(0.1 0.01)  # Learning rate
hyp_decay_stop_perc=(1.0 0.2)  # K epochs
hyp_lr = (1e-4 3e-4)

# hyp_norm_rew=(True False)  # Normalize rewards
# hyp_global_double_loss=(True False)  # Global double loss

# Initialize a counter
counter=1

for decay_min in "${hyp_decay_min[@]}"; do
    for decay_stop in "${hyp_decay_stopperc[@]}"; do
        for lr in "${hyp_lr[@]}"; do
            # for norm_rew in "${hyp_norm_rew[@]}"; do
            # Run the Python script with the current combination of hyperparameters and a unique name in the background
            srun --exclusive -n 1 -c 2 python -u train_imbalanced_classification.py \
            --model_name "orca8_imb_DQN_$counter" \
            --train_type "dqb" \
            --env_imb False \
            --rs_relative_perc True \
            --scale_reward_prof_imb False \
            --lr $lr \
            --num_epoch 4 \
            --EPS_END $decay_min \
            --EPS_DECAY_FACTOR $decay_stop&
            # Increment the counter
            ((counter++))
        done
    done
done
wait