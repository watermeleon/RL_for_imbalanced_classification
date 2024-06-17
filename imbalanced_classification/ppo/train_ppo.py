import math
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from imbalanced_classification.ppo.ppo_agent import *
from utils import evaluate_on_validation_set
from utils.metrics_and_stat_functions import   get_best_timestep


def load_ppo_model(env, state_dim, action_dim, modelname, config):
    """ Load the PPO model for testing"""
    K_epochs = config["K_epochs"]        # update policy for K epochs, used to be 40
    eps_clip = 0.2                       # clip parameter for PPO
    lr_actor = config["lr"]              # learning rate for actor network, used to be: 0.0003
    lr_critic = config["critic_lr"]      # learning rate for critic network, used to be 0.001
    random_seed = config["random_seed"]          # set random seed if required (0 = no random seed)

    checkpoint_path = "./results/" + "PPO_{}_{}.pth".format(random_seed, modelname)
    print("save checkpoint path : " + checkpoint_path)


    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, K_epochs, eps_clip, config)
    print("loading model at : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)

    return ppo_agent




def scale_list(scales, scale_min = 1.0, scale_max = 10):
    min_value = np.min(scales[np.nonzero(scales)])
    max_value = np.max(scales[np.nonzero(scales)])

    # Scale between 0.1 and 10
    new_scales = [((x - min_value) / (max_value - min_value) * (scale_max - scale_min)) + scale_min for x in scales]
    return np.array(new_scales)


# log in logging file  + wandb log metrics
def log_metrics(i_episode, time_step, log_running_reward, log_running_episodes, log_f, correct_batch_buffer, avg_loss, data_iter, wandb):
    # log average reward till last episode
    log_avg_reward = log_running_reward / log_running_episodes
    log_avg_reward = round(log_avg_reward, 3)

    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
    log_f.flush()

    log_running_reward = 0
    log_running_episodes = 0

    accuracy = round(np.mean(correct_batch_buffer), 3)
    mean_loss = np.round(np.mean(avg_loss), 5)
    ep_metrics = {"time_step": time_step, "reward": log_avg_reward, "accuracy": accuracy, "mean_loss": mean_loss}
    wandb.log(ep_metrics)
    data_iter.set_description("Timestep : {}   Avg Reward : {:.3f}  Accuracy : {:.3f}   Avg Loss {:.3f}".format(time_step, log_avg_reward, accuracy, mean_loss))
            
    avg_loss = []
    correct_batch_buffer = []

    return log_running_reward, log_running_episodes, avg_loss, correct_batch_buffer


def train_ppo(env, dataloader_train, dataloader_val, state_dim, action_dim, modelname, config, wandb):
    ####### initialize environment hyperparameters ######
    len_train_data = len(env.X_train)   # should be 255710
    batch_size = config["batch_size"]                    # max timesteps in one episode
    num_epochs = config["num_epoch"]                      # number of epochs
    t_eval = config["t_eval"]                             # number of timesteps to evaluate the model
    
    # find a multiplication of batch_size closest to t_eval
    eval_freq = int(t_eval/batch_size) * batch_size
    max_training_timesteps = int(len_train_data*num_epochs)   # break training loop if timeteps > max_training_timesteps
    print("max_training_timesteps:", max_training_timesteps)


    ################ PPO hyperparameters ################

    K_epochs = config["K_epochs"]                # update policy for K epochs, used to be 40
    eps_clip = config["eps_clip"]              # clip parameter for PPO
    lr_actor = config["lr"]              # learning rate for actor network, used to be: 0.0003
    lr_critic = config["critic_lr"]      # learning rate for critic network, used to be 0.001

    random_seed = config["random_seed"]         # set random seed if required (0 = no random seed)
    result_path = config["result_path"]
    checkpoint_path = result_path + "PPO_checkpoint_TimeStep{}.pth"
    best_checkpoint_path = result_path + "PPO_BEST_checkpoint_TimeStep{}.pth"
    log_f_name = "./results/" + "log_PPO_{}_{}.pth".format(random_seed, modelname)

    print("result_path:", result_path)
    print("save checkpoint path : " + checkpoint_path)


    ##################### initialize agent and start training ################################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, K_epochs, eps_clip, config)

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')


    # printing and logging variables
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0
    avg_loss = []
    correct_batch_buffer = []

    state_list = []
    true_class_list = []
    gender_list = []
    for data in dataloader_val:
        inputs, labels, gender = data
        state_list.append(inputs)
        true_class_list.append(labels)
        gender_list.append(gender)

    # Stack all inputs and labels
    state_list = torch.cat(state_list, dim=0)
    true_class_list = torch.cat(true_class_list, dim=0)
    gender_list = torch.cat(gender_list, dim=0)
    dataloader_val_split = state_list, true_class_list, gender_list

    all_eval_metrics = {}
    for epoch_i in range(num_epochs):
        print("epoch_i:", epoch_i)
        data_iter = tqdm(dataloader_train)
        for data in data_iter:

            state_batch, label_batch, gender_batch, imb_penalty_batch = data
            current_ep_reward = 0

            # select action with policy
            action_batch = ppo_agent.select_action_batch(state_batch)

            # element-wise comparison between action and label
            correct_batch = (action_batch == label_batch).float()*2 - 1
            # imb_penalty_batch = env.reward_scale[label_batch.cpu(), gender_batch.cpu()] # reward scale can be retrieved from dataloader and Env

            reward_batch = correct_batch * imb_penalty_batch


            # saving reward and is_terminals
            ppo_agent.buffer.rewards.extend(reward_batch)
            ppo_agent.buffer.class_labels.extend(label_batch)
            ppo_agent.buffer.genders.extend(gender_batch)
            correct_batch_buffer.extend(correct_batch.tolist())

            # update running metrics
            time_step += batch_size
            current_ep_reward += reward_batch.sum().item()
            log_running_reward += current_ep_reward
            log_running_episodes += batch_size
            i_episode +=batch_size

            # update PPO agent
            if time_step % batch_size == 0:
                avg_loss.append(ppo_agent.update(wandb))


            if time_step % eval_freq == 0:
                # log metrics local and to wandb
                log_running_reward, log_running_episodes, avg_loss, correct_batch_buffer = log_metrics(i_episode, time_step, log_running_reward, log_running_episodes, log_f, correct_batch_buffer, avg_loss, data_iter, wandb)
                
                eval_acc, tpr_gap_pp, tpr_gap_rms = evaluate_on_validation_set(ppo_agent.policy.actor, dataloader_val_split, supervised = False, val_presplit=True)
                print("Eval acc:", eval_acc, "tpr_gap_rms:", tpr_gap_rms )

                if config["use_rl_scheduler"]:
                    ppo_agent.lr_scheduler_actor.step(eval_acc)

                tpr_gap_pp_list = np.array([tpr_gap_pp[i] if i in tpr_gap_pp else 0.0  for i in range(ppo_agent.action_dim)])
                tpr_gap_plot = weight_plot = None


                # store eval metrics to dict and save to wandb
                fairness_metric = 1-tpr_gap_rms
                dto_dist_heurstic_eval = math.sqrt((1-eval_acc)**2 + (1-fairness_metric)**2)
                all_eval_metrics[time_step] = {"performance": eval_acc, "fairness": fairness_metric, "eval_dto_dist": dto_dist_heurstic_eval}
                eval_metrics = {"reward_scales": weight_plot , "TPRgap_plot": tpr_gap_plot,"eval_acc": eval_acc, "time_step": time_step, "tpr_gap_rms_eval": tpr_gap_rms, "lr_scheduler": ppo_agent.optim_actor.param_groups[0]['lr']}
                wandb.log(eval_metrics)

                # save model weights
                checkpoint_path_epoch = checkpoint_path.format(time_step)
                print("saving model at : " + checkpoint_path_epoch)
                ppo_agent.save(checkpoint_path_epoch)


        
    # Apply DTO and save network
    best_timestep = get_best_timestep(all_eval_metrics, selection_criterion = "DTO")
    best_checkpoint_path = checkpoint_path.format(best_timestep)

    # load best agent
    ppo_agent.load(best_checkpoint_path)

    # resave best model at best checkpoint
    best_checkpoint_path = best_checkpoint_path.format(best_timestep)
    ppo_agent.save(checkpoint_path_epoch)

    log_f.close()
    env.close()

    return ppo_agent, all_eval_metrics


