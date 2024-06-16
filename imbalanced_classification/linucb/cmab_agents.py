# Code from : https://github.com/kfoofw/bandit_simulations/blob/master/python/contextual_bandits/notebooks/LinUCB_disjoint.ipynb
import numpy as np
import torch
from numba import jit, float64, int64
import pickle

import torch
import torch.nn as nn

class linucb_disjoint_arm(nn.Module):
    def __init__(self, arm_index, d, alpha, device):
        super(linucb_disjoint_arm, self).__init__()
        self.arm_index = arm_index
        self.alpha = alpha
        self.A = nn.Parameter(torch.eye(d, device=device), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(d, device=device), requires_grad=False)
        self.A_inv = nn.Parameter(torch.eye(d, device=device), requires_grad=False)

    def to(self, device):
        self.A = self.A.to(device)
        self.b = self.b.to(device)
        self.A_inv = self.A_inv.to(device)
        return self

    def calc_UCB(self, x):
        # Directly use stored A_inv
        self.theta = torch.matmul(self.A_inv, self.b)
        # Compute UCB using vectorized operations
        p = torch.matmul(self.theta, x) + self.alpha * torch.sqrt(torch.matmul(x.T, torch.matmul(self.A_inv, x)))
        return p

    def calc_UCB_batch(self, X):
        # Calculate theta for the arm
        self.theta = torch.matmul(self.A_inv, self.b)
        # Calculate the predicted payoff
        predicted_payoff = torch.matmul(X, self.theta)
        # Calculate the uncertainty
        uncertainty = self.alpha * torch.sqrt(torch.sum(torch.matmul(X, self.A_inv) * X, dim=1))
        # The total UCB is the sum of the predicted payoff and the uncertainty
        ucb = predicted_payoff + uncertainty
        return ucb

    def reward_update(self, reward, x):
        # Use Sherman-Morrison formula for updating A_inv efficiently
        x = x.view(-1, 1)  # Reshape x for the outer product
        self.A += torch.matmul(x, x.T)
        self.b += reward * x.view(-1)  # Update b directly

        # Update A_inv using Sherman-Morrison
        A_inv_x = torch.matmul(self.A_inv, x)
        self.A_inv -= torch.matmul(A_inv_x, A_inv_x.T) / (1 + torch.matmul(x.T, A_inv_x))


class linucb_policy(nn.Module):
    def __init__(self, K_arms, d, alpha, device):
        super(linucb_policy, self).__init__()
        self.K_arms = K_arms
        self.d = d
        self.alpha = alpha
        self.device = device
        self.linucb_arms = nn.ModuleList([linucb_disjoint_arm(i, d, alpha, device) for i in range(K_arms)])

    def to(self, device):
        self.linucb_arms = self.linucb_arms.to(device)
        return self

    def select_arm(self, x_array):
        highest_ucb = float('-inf')
        candidate_arms = []

        for arm_index, arm in enumerate(self.linucb_arms):
            arm_ucb = arm.calc_UCB(x_array)

            if arm_ucb > highest_ucb:
                highest_ucb = arm_ucb
                candidate_arms = [arm_index]
            elif arm_ucb == highest_ucb:
                candidate_arms.append(arm_index)

        # randomly select 1 element of the list chosen_arm, but not like way above. Should be 1 int
        chosen_arm = torch.as_tensor(candidate_arms)[torch.randint(len(candidate_arms), (1,))].to(x_array.device)
        
        return chosen_arm, None

    def select_arm_batch(self, X):
        X = torch.as_tensor(X, device=self.linucb_arms[0].A.device)
        all_ucbs = torch.zeros((X.shape[0], self.K_arms), device=X.device)

        for arm_index, arm in enumerate(self.linucb_arms):
            all_ucbs[:, arm_index] = arm.calc_UCB_batch(X)

        chosen_arms = torch.argmax(all_ucbs, dim=1)
        return chosen_arms, None

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, K_arms=None, d=None, alpha=None, device=None):
        if K_arms is None or d is None or alpha is None or device is None:
            print("Loading parameters from existing model")
            model = linucb_policy(self.K_arms, self.d, self.alpha, self.device)
        else:
            model = linucb_policy(K_arms, d, alpha, device)
        model.load_state_dict(torch.load(filename))
        return model