
#SBATCH --job-name=ppobiossl_bios_sweep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=18
#SBATCH --output=./slurmfiles/slurm_output_%A_%a.out

module purge
module load 2021
module load Anaconda3/2021.05

# Your job starts in the directory where you call sbatch
cd $HOME/RL_for_genderbias/

# Activate your environment
source activate rl_bias

# ppo emoji
#wandb_id=watermelontology/sweeps_demo/ofcj1tc1
num_count_per_run=1


srun wandb sweep --project sweeps_3seed_2dts utils/final_params/final_params_supervised_bios.yml


source $HOME/wandbid_from_slurmid.sh

wandbid=$(extract_wandbid_id $SLURM_JOB_ID)


# emoji ratio supervised
#wandbid=nxjealhq

wandb_id=watermelontology/sweeps_3seed_2dts/"$wandbid"

srun wandb agent --count $num_count_per_run $wandb_id &
#srun wandb agent --count $num_count_per_run $wandb_id &
#srun wandb agent --count $num_count_per_run $wandb_id &

#srun wandb agent --count $num_count_per_run $wandb_id &
#srun wandb agent --count $num_count_per_run $wandb_id &
#srun wandb agent --count $num_count_per_run $wandb_id &

#srun wandb agent --count $num_count_per_run $wandb_id &
#srun wandb agent --count $num_count_per_run $wandb_id &
#srun wandb agent --count $num_count_per_run $wandb_id &

wait








