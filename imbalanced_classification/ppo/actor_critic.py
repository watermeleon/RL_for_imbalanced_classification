import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class NeuralNetwork2(nn.Module):
    """ Neural network for the bandit problem - input are BERT embeddings"""
    def __init__(self, n_features, n_actions):
        super(NeuralNetwork2, self).__init__()

        self.fc1 = nn.Linear(n_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, n_actions)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

class ActorCritic_classifier(nn.Module):
    def __init__(self, state_dim, action_dim, model_type = "NN"):
        super(ActorCritic_classifier, self).__init__()
    
        if model_type == "NN":
            actor_network = NeuralNetwork2(state_dim, action_dim)
            self.actor = nn.Sequential(actor_network, actor_network.fc2, nn.Softmax(dim=-1)) 
            critic_network = NeuralNetwork2(state_dim, action_dim)
            self.critic = nn.Sequential(critic_network, critic_network.fc3)


    def set_action_std(self, new_action_std):
        print("--------------------------------------------------------------------------------------------")
        print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    
    def act(self, state, batch_data=False):

        if batch_data is False: 
            state = state.unsqueeze(0)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def act2(self, state):
        """This is used for the evaluation of the model, where we do not want to sample actions"""
        # state = state.unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy    
    
    def store_model(self, result_path):
        torch.save(self.actor.state_dict(), result_path + "_params_actor")
        torch.save(self.critic.state_dict(), result_path + "_params_critic")

        
    def load_model(self, result_path):
        self.actor.load_state_dict(torch.load(result_path + "_params_actor"))
        self.critic.load_state_dict(torch.load(result_path + "_params_critic"))

