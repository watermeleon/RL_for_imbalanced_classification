"""
code from : https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
"""
import numpy as np
import torch
import torch.nn as nn
from imbalanced_classification.ppo.actor_critic import *
from utils.metrics_and_stat_functions import calc_tpr_gap, get_tpr
from sklearn.metrics import f1_score


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.class_labels = []
        self.genders = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.class_labels[:]
        del self.genders[:]



class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, K_epochs, eps_clip, config):
        self.device = config["device"]

        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entropy_coef = config["entropy_coef"]
        self.use_critic = config["use_critic"]

        self.normalize_rewards = config["normalize_rewards"]
        self.global_double_loss = config["global_double_loss"]
        self.scale_tpr = config["scale_tpr"]
        self.rm_scale_range = config["rm_scale_range"]


        self.buffer = RolloutBuffer()
        self.policy = ActorCritic_classifier(state_dim, action_dim).to(self.device)


        self.optim_actor = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=lr_critic)
        if config["use_rl_scheduler"]:
            self.lr_scheduler_actor = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim_actor, mode='max', factor=0.5, patience=config["patience"], verbose=False, min_lr=1e-6, cooldown=1)

        self.policy_old = ActorCritic_classifier(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()
    
    def select_action_batch(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state, batch_data=True)
        
        self.buffer.states.extend(state)
        self.buffer.actions.extend(action)
        self.buffer.logprobs.extend(action_logprob)
        self.buffer.state_values.extend(state_val)

        return action
          


    def update(self, wandb):
        # Monte Carlo estimate of returns
        rewards = self.buffer.rewards
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if self.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)


        # calculate advantages
        if self.use_critic:
            advantages = rewards.detach() - old_state_values.detach()
        else:
            advantages = rewards.detach() 

        # Optimize policy for K epochs
        loss_list = []
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            if self.use_critic is True:
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_coef * dist_entropy # used to be 0.01
            else:
                loss = - torch.min(surr1, surr2) - self.entropy_coef * dist_entropy # self.entropy_coef used to be 0.01
            
            # take gradient step
            self.optim_actor.zero_grad()
            self.optim_critic.zero_grad()
            loss.mean().backward()
            self.optim_actor.step()
            self.optim_critic.step()

            loss_list.append(loss.mean().item())
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        return np.mean(loss_list)
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       

