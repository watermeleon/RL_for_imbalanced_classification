# RL for imbalanced classification


## Install the environment:
`pip install -r requirements.txt`


## Data
To download the data follow the instructions from : https://github.com/shauli-ravfogel/nullspace_projection/tree/master
Specificall follow their steps to:
1. Run the `download_data.sh`, however, we do not use the files stored in `data/embeddings` or the `_avg` embeddings, so you can remove those lines before dowloading.
2. For Emoji dataset, process the downloaded files using the file `run_deepmoji_debiasing.sh`.
3. For the BiasBios dataset, go to the folder `imbalanced_classification` and run the function `process_biasbios_data.py`, make sure to set the input path from the folder of ravfogel, and output to our data folder

## Run experiments
To rerun the experiments all the relevant code is included, see imbalanced_classification for explanation of the parameters.
Each experiment for each algorithm is executed using wandb sweeps, the different seeds and hyper parameters are placed in their config files, see the folder `./imbalanced_classification/configs/`


As an example, to run the experiment of PPO on the Emoji datset under varying stereotypical ratios run:
```
wandb sweep --project rl_imb ./imbalanced_classification/configs/ratios_ppo_emoji.yml
wandb agent --count 1 <sweep-ID>
```
Afterwards results can be viewed from wandb
