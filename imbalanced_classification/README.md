

# Train Imbalanced Classification README

This code trains imbalanced classification models using various algorithms and datasets. The main algorithms used are PPO, Supervised Learning, DQN, Thompson Sampling, and LinUCB. The datasets supported are "biasbios" and "emoji".

## Configuration Parameters

### General Parameters

| Parameter | Description |
|-----------|-------------|
| `random_seed` | Random seed for reproducibility |
| `num_workers` | Number of worker processes for data loading |
| `datapath` | Path to the dataset directory |
| `dataset` | Dataset to use, either "biasbios" or "emoji" |
| `gender_bool` | Whether to add gender information to the input data |
| `create_data` | Whether to create new data or use existing data |
| `num_epoch` | Number of epochs to train for |
| `exp_name` | Experiment name for logging and saving results |
| `wandb_name` | Wandb project name for logging |
| `store_tpr_wandb` | Whether to store TPR (True Positive Rate) metrics in Wandb |
| `use_wandbid_name` | Whether to use Wandb run ID in the model name |
| `wandb_username` | Wandb username for logging |
| `train_type` | Training algorithm to use, one of "ppo", "supervised", "dqn", "linucb" |
| `reward_scale_type` | Reward scaling type, one of "constant", "EO", "imb_ratio_plus", "imb_ratio_neg", "gender_and_prof" |
| `test_only` | Whether to only run inference on the test set |
| `lr` | Learning rate for the policy network |

### Dataset-specific Parameters

#### Biasbios Dataset

| Parameter | Description |
|-----------|-------------|
| `use_most_common_classes` | Whether to use only the most common classes |
| `skew_data` | Whether to skew the data distribution |

#### Emoji Dataset

| Parameter | Description |
|-----------|-------------|
| `emoji_ratio` | Ratio of emoji data to use |

### Algorithm-specific Parameters

#### PPO (Proximal Policy Optimization)

| Parameter | Description |
|-----------|-------------|
| `n_hidden` | Number of hidden units in the policy network |
| `critic_lr` | Learning rate for the critic network |
| `batch_size` | Batch size for training |
| `entropy_coef` | Coefficient for entropy regularization |
| `K_epochs` | Number of epochs to train the policy network |
| `eps_clip` | Clipping parameter for PPO |
| `use_critic` | Whether to use a critic network |
| `use_rl_scheduler` | Whether to use a learning rate scheduler |
| `patience` | Patience for early stopping |
| `normalize_rewards` | Whether to normalize rewards |

#### DQN (Deep Q-Network)

| Parameter | Description |
|-----------|-------------|
| `EPS_START` | Initial value of epsilon for exploration |
| `EPS_END` | Final value of epsilon for exploration |
| `EPS_DECAY_FACTOR` | Decay factor for epsilon |
| `t_eval` | Number of steps between evaluations |

#### LinUCB (Linear Upper Confidence Bound)

| Parameter | Description |
|-----------|-------------|
| `linucb_alpha` | Exploration parameter for LinUCB |

#### Supervised Learning

| Parameter | Description |
|-----------|-------------|
| `lr` | Learning rate for supervised learning |

#### Debiasing Methods

| Parameter | Description |
|-----------|-------------|
| `debias_embs` | Debiasing method to use, one of "mp", "inlp", or "none" |
| `debiase_posthoc` | Whether to apply debiasing post-hoc |
| `inlp_num_classifiers` | Number of classifiers to use for INLP debiasing |

