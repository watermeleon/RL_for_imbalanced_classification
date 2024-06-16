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

# Define the values for each hyperparameter
hyp_lr=(1e-4, 3e-4, 5e-4)  # Replace these with your actual values for hyperparameter 1
hyp_bs=(128, 256, 512)  # Replace these with your actual values for hyperparameter 2
hyp_entropyc=(0.01, 0.05, 0.1, 0.2)  # Replace these with your actual values for hyperparameter 3
hyp_epsclip=(0.1, 0.2, 0.3)  # Replace these with your actual values for hyperparameter 4
hyp_Kepochs=(5, 10, 15)  # Replace these with your actual values for hyperparameter 5
hyp_norm_rew=(True, False)  # Replace these with your actual values for hyperparameter 6
hyp_global_double_loss=(True, False)  # Replace these with your actual values for hyperparameter 7


# Initialize a counter
counter=1

# Nested loop to iterate over all combinations of hyperparameter values
for lr in "${hyp_lr[@]}"; do
    for bs in "${hyp_bs[@]}"; do
        for entropyc in "${hyp_entropyc[@]}"; do
            for epsclip in "${hyp_epsclip[@]}"; do
                for Kepochs in "${hyp_Kepochs[@]}"; do
                    for norm_rew in "${hyp_norm_rew[@]}"; do
                        for global_double_loss in "${hyp_global_double_loss[@]}"; do
                            # Run the Python script with the current combination of hyperparameters and a unique name in the background
                            srun --exclusive -n 1 -c 2 python -u train_imbalanced_classification.py \
                            --model_name "orca8_imb_RS2_$counter" \
                            --train_type "ppo" \
                            --env_imb True \
                            --rs_relative_perc True \
                            --scale_reward_prof_imb False \
                            --lr $lr \
                            --batch_size $bs \
                            --entropy_coef $entropyc \
                            --K_epochs $Kepochs \
                            --eps_clip $epsclip \
                            --normalize_rewards $norm_rew \
                            --global_double_loss $global_double_loss &
                            # Increment the counter
                            ((counter++))
                        done
                    done
                done
            done
        done
    done
done


wait
