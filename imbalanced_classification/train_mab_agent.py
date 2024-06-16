import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import math
from tqdm import tqdm
import random

from utils.metrics_and_stat_functions import get_best_timestep
from utils.dataloader_eval_test import evaluate_on_validation_set, evaluate_on_validation_set_batchwise
from imbalanced_classification.linucb.cmab_agents import linucb_policy


import cProfile
import pstats
from io import StringIO
from torchrl.data import ReplayBuffer, ListStorage, SamplerWithoutReplacement, RandomSampler

class NeuralNetwork2(nn.Module):
    """ Neural network for the bandit problem - input are BERT embeddings"""
    def __init__(self, n_features, n_actions, n_hidden=128):
        super(NeuralNetwork2, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_features, n_hidden), 
            nn.ReLU(), 
            nn.Linear(n_hidden, n_actions)
        )

    def forward(self, x):
        return    self.layer(x)
    
    def __getitem__(self, index):
        return self.layer[index]

class NeuralBandit2:
    def __init__(self, n_actions, n_features, learning_rate=0.01, n_steps=10000, config = None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.device = config["device"]
        self.batch_size = config["batch_size"]

        # Initialize the neural network model for each action
        self.model = NeuralNetwork2(n_features, n_actions, n_hidden=config["n_hidden"]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.HuberLoss()

        # initialize epsilon greedy variables
        self.EPS_START = config['EPS_START']
        self.EPS_END = config['EPS_END']
        self.EPS_DECAY = int(config['EPS_DECAY_FACTOR'] * n_steps)
        self.steps_done = 0
        
        self.replay_buffer =  ReplayBuffer(storage=ListStorage(max_size=10000), batch_size=self.batch_size,     sampler=SamplerWithoutReplacement())

    def predict(self, context):
        with torch.no_grad():
            return self.model(context)
    
    def select_action(self, context):
        # global steps_done 
        sample = random.random()
        # eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            # math.exp(-1. * self.steps_done / self.EPS_DECAY)
        eps_threshold = max(self.EPS_END, self.EPS_START - (self.steps_done * (self.EPS_START - self.EPS_END) / self.EPS_DECAY))

        self.steps_done += 1

        # take action using epsilong greedy policy
        if sample > eps_threshold:
            with torch.no_grad():
                probs = self.model(context)
            action = probs.max(0).indices.item()
        else:
            action = torch.tensor(np.random.randint(0,self.n_actions,(1))[0]).item()
            
        return action, None
    

    def update_dqn(self, context_tensor, train_env):
        self.optimizer.zero_grad()

        # select action from model
        probs = self.model(context_tensor)
        action, _ = self.select_action(context_tensor)

        # get reward and next state from env
        next_state, reward, _, _ = train_env.step(action)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)  # Convert to tensor

        # don't use log prob but use prob
        model_output_action_i = probs[action]
        loss = self.criterion(model_output_action_i, reward_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item(), next_state, action, reward

    def update_dqn_batch(self, context_tensor, train_env):
        # select action and add to buffer         
        action, _ = self.select_action(context_tensor)
        next_state, reward, _, _ = train_env.step(action)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)

        self.replay_buffer.extend([(context_tensor, action, reward)])

        state_batch, action_batch, reward_batch = self.replay_buffer.sample()[0]

        # compute Q values
        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # compute loss
        loss = self.criterion(state_action_values, reward_batch)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), next_state, action, reward
    



    def train_dqn_agent(self, train_env, n_steps, dataloader_val, config, wandb):
        reward_list = []
        state = train_env.reset()
        pbar = range(n_steps)

        running_loss, running_acc, running_reward= 0.0, 0.0, 0.0
        t_eval = config["t_eval"]
        previous_eval_acc = 0
        patience = 0
        max_patience = 100

        optimizer_path = config["result_path"] + "checkpoint_optimizer_TimeStep{}.pt"
        checkpoint_path = config["result_path"]  + "checkpoint_TimeStep{}.pth"
        best_checkpoint_path = config["result_path"]  + "BEST_checkpoint_TimeStep{}.pth"

        all_eval_metrics = {}

        for t in tqdm(pbar):
            if len(self.replay_buffer) < self.batch_size:
                # if buffer is not full enough, select random action
                action = np.random.randint(0, train_env.num_professions,(1))[0]
                next_state, reward, _, _ = train_env.step(action)
                reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
                action = torch.tensor(action, dtype=torch.int64).to(self.device)
                self.replay_buffer.extend([(state, action, reward)])
                continue

            loss_i, next_state, action, reward = self.update_dqn_batch(state, train_env)

            # update running metrics
            reward = reward.item()
            running_reward += reward
            running_loss += loss_i
            good = 1.0 if reward > 0 else 0.0
            running_acc += good
            state = next_state
        
            # log metrics
            if t % t_eval == (t_eval-1):
                #  eval on validation set
                eval_acc, _, tpr_gap_rms = evaluate_on_validation_set(self.model, dataloader_val, supervised = False)

                # store metrics
                time_step = t+1
                fairness_metric = 1-tpr_gap_rms
                dto_dist_heurstic_eval = math.sqrt((1-eval_acc)**2 + (1-fairness_metric)**2)
                ep_metrics = {"loss": round(running_loss / t_eval, 5) ,"reward": round(running_reward/t_eval, 5), "accuracy": round(running_acc/t_eval, 5), "eval_acc": round(eval_acc, 5), "time_step":time_step, "tpr_gap_rms_eval": round(tpr_gap_rms,4)}
                all_eval_metrics[time_step] = {"performance": eval_acc, "fairness": fairness_metric, "eval_dto_dist": dto_dist_heurstic_eval}
                
                print(f'[{t + 1:5d}, ' + str(ep_metrics), "patience", patience)
                running_loss, running_acc, running_reward= 0.0, 0.0, 0.0
                wandb.log(ep_metrics)

                # save model if eval acc is better than previous
                torch.save(self.model.state_dict(), checkpoint_path.format(time_step))
                torch.save(self.optimizer.state_dict(), optimizer_path.format(time_step))
                if eval_acc > previous_eval_acc:
                    previous_eval_acc = eval_acc
                    patience = 0
                else:
                    patience +=1
                    print("Eval acc did not improve", patience)
                    if patience > max_patience:
                        print("Early stopping")
                        break

        # load best model
        best_timestep = get_best_timestep(all_eval_metrics, selection_criterion = "DTO")
        self.model.load_state_dict(torch.load(checkpoint_path.format(best_timestep)))
                                   
        # resave best model under best checkpoint
        best_checkpoint_path = best_checkpoint_path.format(best_timestep)
        torch.save(self.model.state_dict(), best_checkpoint_path)

        return all_eval_metrics


    def load_linucb_agent(self, train_env, n_steps, dataloader_val, config, wandb):
        alpha = config["linucb_alpha"]
        linucb_agent = linucb_policy(K_arms = train_env.num_professions, d = self.n_features, alpha = alpha, device=config["device"])

        reward_scale_type = config["reward_scale_type"]
        if reward_scale_type == "constant":
            result_folder_name = "linucb_S83_28C_constant/"
        elif reward_scale_type == "EO":
            result_folder_name = "linucb_S83_28C_EO/"

        import os
        # get the name of the first file in the folder
        result_folder = "./results/" + result_folder_name
        result_file =  next(iter(os.listdir(result_folder)))
        print("Loading checkpoint from file", result_folder +result_file)
        linucb_agent = linucb_agent.load(result_folder + result_file)

        return linucb_agent.select_arm_batch, {1:{"eval_dto_dist": 0.0}, 2:{"eval_dto_dist": 0.0}}

    def train_linucb_agent(self, train_env, n_steps, dataloader_val, config, wandb):
        state = train_env.reset()
        pbar = range(n_steps)

        running_loss, running_acc, running_reward= 0.0, 0.0, 0.0
        t_eval = config["t_eval"]
        previous_eval_acc = 0
        patience = 0
        max_patience = config["patience"]

        alpha = config["linucb_alpha"]
        linucb_agent = linucb_policy(K_arms = train_env.num_professions, d = self.n_features, alpha = alpha, device=config["device"])

        checkpoint_path = config["result_path"]  + "checkpoint_TimeStep{}.pth"
        best_checkpoint_path = config["result_path"]  + "BEST_checkpoint_TimeStep{}.pth"

        all_eval_metrics = {}
        for t in tqdm(pbar):
            state_np = state
            action, _ = linucb_agent.select_arm(state_np)
            next_state, reward, _, _ = train_env.step(action)

            # Update the model with the chosen arm and reward
            linucb_agent.linucb_arms[action].reward_update(reward, state_np)

            # update running metrics
            running_reward += reward
            running_loss += 0.1
            good = 1.0 if reward > 0 else 0.0
            running_acc += good
            state = next_state

        
            # log metrics
            if t % t_eval == (t_eval-1):
                #  eval on validation set
                eval_acc, _, tpr_gap_rms = evaluate_on_validation_set_batchwise(linucb_agent.select_arm_batch, dataloader_val, supervised = False)
                
                # store metrics
                time_step = t+1
                fairness_metric = 1-tpr_gap_rms
                dto_dist_heurstic_eval = math.sqrt((1-eval_acc)**2 + (1-fairness_metric)**2)
                ep_metrics = {"loss": round(running_loss / t_eval, 5) ,"reward": round(running_reward/t_eval, 5), "accuracy": round(running_acc/t_eval, 5), "eval_acc": round(eval_acc, 5), "time_step":time_step, "tpr_gap_rms_eval": round(tpr_gap_rms,4)}
                all_eval_metrics[time_step] = {"performance": eval_acc, "fairness": fairness_metric, "eval_dto_dist": dto_dist_heurstic_eval}

                print(f'[{t + 1:5d}, ' + str(ep_metrics), "patience", patience)
                running_loss, running_acc, running_reward= 0.0, 0.0, 0.0
                wandb.log(ep_metrics)

                linucb_agent.save(checkpoint_path.format(time_step))  
                # save model if eval acc is better than previous
                if eval_acc > previous_eval_acc:
                    previous_eval_acc = eval_acc
                    print("Eval Acc improved")
                    patience = 0
                else:
                    patience +=1
                    print("Eval acc did not improve", patience)
                    if patience > max_patience:
                        print("Early stopping - Using last saved model")
                        break


        # load best model
        print("All eval metrics", all_eval_metrics)
        best_timestep = get_best_timestep(all_eval_metrics, selection_criterion = "DTO")
        linucb_agent = linucb_agent.load(checkpoint_path.format(best_timestep))
                                   
        # resave best model under best checkpoint
        best_checkpoint_path = best_checkpoint_path.format(best_timestep)
        linucb_agent.save(best_checkpoint_path)


        return linucb_agent.select_arm_batch, all_eval_metrics
    

    def train_supervised(self, dataloader, dataloader_val, config, wandb):
        use_reward_scale = config["reward_scale_type"] != "constant"
        reward_list = []
        previous_eval_acc = 0
        patience = 0
        max_patience = 20 
        epoch_size = config["num_epoch"]


        optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])
        if config["use_rl_scheduler"]:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=config["patience"], verbose=False)

        criterion = nn.CrossEntropyLoss(reduction="none")

        len_dataset = len(dataloader) * dataloader.batch_size
        print("Length of dataset", len_dataset)
        print("len dataloader", len(dataloader))
        print("batch size", config["batch_size"])

        optimizer_path = config["result_path"] + "checkpoint_optimizer_TimeStep{}.pt"
        checkpoint_path = config["result_path"]  + "checkpoint_TimeStep{}.pth"
        best_checkpoint_path = config["result_path"]  + "BEST_checkpoint_TimeStep{}.pth"


        all_eval_metrics = {}

        for epoch in range(epoch_size):
            self.model.train()
            all_labels = []
            all_predictions = []

            running_loss = 0.0
            for i, data in enumerate(tqdm(dataloader), 0):
                # get the inputs; data is a list of [inputs, labels]
                if use_reward_scale:
                    inputs, labels, imb_penalty = data
                else:
                    inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                if use_reward_scale:
                    loss = torch.mean(loss * imb_penalty)
                else:
                    loss = torch.mean(loss)
                loss.backward()
                optimizer.step()

                # store predicted labels and true labels
                running_loss += loss.item()
                all_labels.extend(labels.argmax(dim=1).tolist())
                all_predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())

            # get training and eval accuracy
            train_acc = np.mean((np.array(all_labels) == np.array(all_predictions)).astype(float))
            eval_acc, _, tpr_gap_rms = evaluate_on_validation_set(self.model, dataloader_val, supervised = True)

            if config["use_rl_scheduler"]:
                lr_scheduler.step(eval_acc)
            
            # store metrics
            time_step =  (epoch + 1)*len_dataset
            # calculate the DTO with the utopian values as 1 for both performance and fairness
            fairness_metric = 1-tpr_gap_rms
            dto_dist_heurstic_eval = math.sqrt((1-eval_acc)**2 + (1-fairness_metric)**2)
            ep_metrics = {"loss": round(running_loss/len_dataset , 5) , "accuracy": round(train_acc, 5), "eval_acc": round(eval_acc, 5), "time_step":time_step, 
                          "tpr_gap_rms_eval": tpr_gap_rms, "lr_scheduler": optimizer.param_groups[0]['lr'], "eval_dto": dto_dist_heurstic_eval}

            all_eval_metrics[time_step] = {"performance": eval_acc, "fairness": fairness_metric, "eval_dto_dist": dto_dist_heurstic_eval}
            print(f'[{epoch:5d}, ' + str(ep_metrics))
            wandb.log(ep_metrics)
            self.model.train()


            # save model but set patience if eval acc does not improve
            torch.save(self.model.state_dict(), checkpoint_path.format(time_step))
            torch.save(self.optimizer.state_dict(), optimizer_path.format(time_step))
            if eval_acc > previous_eval_acc:
                previous_eval_acc = eval_acc
                patience = 0
            else:
                patience +=1
                print("Eval acc did not improve", patience)
                if patience > max_patience:
                    print("Early stopping")
                    break


        # load best model
        best_timestep = get_best_timestep(all_eval_metrics, selection_criterion = "DTO")
        self.model.load_state_dict(torch.load(checkpoint_path.format(best_timestep)))
                                   
        # resave best model under best checkpoint
        best_checkpoint_path = best_checkpoint_path.format(best_timestep)
        torch.save(self.model.state_dict(), best_checkpoint_path)


        return all_eval_metrics

