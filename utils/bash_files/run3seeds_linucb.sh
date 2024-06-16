#!/bin/bash

#SBATCH --job-name=exp_28C
#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:59:00
#SBATCH --output=./slurmfiles/slurm_output_%A_%a.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
cd $HOME/RL_for_genderbias/

# Activate your environment
source activate rl_bias

# Define array of seeds
seeds=(1 42 83)
env_imb_values=(False True)
most_common="True"

# Loop over env_imb values
for env_imb in "${env_imb_values[@]}"; do
    # Decide model name prefix based on env_imb value
    if [ "$env_imb" = "False" ]; then
        infix="const"
    else
        infix="imb"
    fi

    # Loop over seeds
    for seed in "${seeds[@]}"; do
        # Construct the model name
        if [ "$most_common" == "True" ]; then
            class_string="8C"
        else
            class_string="28C"
        fi

        model_name="PPO_${infix}_${class_string}_Skew_S${seed}"

        # Run the command with the constructed model name and other parameters
        srun --exclusive -n 1 -c 4 python -u train_imbalanced_classification.py \
            --model_name "$model_name" \
            --train_type "linucb" \
            --linucb_alpha 1.5 \
            --t_eval 30000 --num_epoch 3 \
            --env_imb $env_imb \
            --use_most_common_classes True \
            --random_seed $seed \
            --wandb_name "occup_class_Top8_Skew" \
            --store_tpr_wandb True \
            --skew_data False &
    done
done

wait