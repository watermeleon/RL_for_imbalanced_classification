

# Train Imbalanced Classification README

This code trains imbalanced classification models using various algorithms and datasets. The main algorithms used are PPO, Supervised Learning, DQN, Thompson Sampling, and LinUCB. The datasets supported are "biasbios" and "emoji".

## Configuration Parameters

### General Parameters

#### Experiment Settings
| Parameter | Description |
|-----------|-------------|
| `random_seed` | Random seed for reproducibility |
| `exp_name` | Experiment name for logging and saving results |
| `create_data` | Whether to create new data or use existing data |
| `num_epoch` | Number of epochs to train for |
| `test_only` | Whether to only run inference on the test set |
| `t_eval` | Number of steps between evaluations |

#### Dataset and Data Loading
| Parameter | Description |
|-----------|-------------|
| `num_workers` | Number of worker processes for data loading |
| `datapath` | Path to the dataset directory |
| `dataset` | Dataset to use, either "biasbios" or "emoji" |
| `use_most_common_classes` | For BiasBios whether to use only 8 most common classes |
| `emoji_ratio` | For Emoji the stereotyping ratio to use |


#### Logging and Wandb
| Parameter | Description |
|-----------|-------------|
| `wandb_name` | Wandb project name for logging |
| `store_tpr_wandb` | Whether to store TPR (True Positive Rate) metrics in Wandb |
| `use_wandbid_name` | Whether to use Wandb run ID in the model name |
| `wandb_username` | Wandb username for logging |

#### Training Settings
| Parameter | Description |
|-----------|-------------|
| `train_type` | Training algorithm to use, one of "ppo", "supervised", "dqn", "linucb" |
| `reward_scale_type` | Reward scaling type, one of "constant", "EO", "imb_ratio_plus", "imb_ratio_neg", "gender_and_prof" |


### Algorithm-specific Parameters

#### Neural Network parameters (used by: PPO, DQN, and Supervised Learning)
| Parameter | Description |
|-----------|-------------|
| `lr` | Learning rate for the policy network (PPO), Q-network (DQN), or supervised learning |
| `n_hidden` | Number of hidden units in the policy network (PPO), Q-network (DQN), or supervised learning model |
| `patience` | Patience for early stopping |

#### PPO and Supervised Learning
| Parameter | Description |
|-----------|-------------|
| `batch_size` | Batch size for training |
| `use_rl_scheduler` | Whether to use a learning rate scheduler |


#### PPO (Proximal Policy Optimization)
| Parameter | Description |
|-----------|-------------|
| `critic_lr` | Learning rate for the critic network |
| `entropy_coef` | Coefficient for entropy regularization |
| `K_epochs` | Number of epochs to train the policy network |
| `eps_clip` | Clipping parameter for PPO |
| `use_critic` | Whether to use a critic network |
| `normalize_rewards` | Whether to normalize rewards |

#### DQN (Deep Q-Network)
| Parameter | Description |
|-----------|-------------|
| `EPS_START` | Initial value of epsilon for exploration |
| `EPS_END` | Final value of epsilon for exploration |
| `EPS_DECAY_FACTOR` | Decay factor for epsilon |

#### LinUCB (Linear Upper Confidence Bound)
| Parameter | Description |
|-----------|-------------|
| `linucb_alpha` | Exploration parameter for LinUCB |

#### Gender Information Strength experiments
| Parameter | Description |
|-----------|-------------|
| `debias_embs` | Debiasing method to use, one of "mp", "inlp", or "none" |
| `debiase_posthoc` | Whether to apply debiasing post-hoc |
| `inlp_num_classifiers` | Number of classifiers to use for INLP debiasing |
| `inlp_by_class` | Whether to apply INLP debiasing per class |
| `gender_bool` | Whether to add gender information to the input data |

