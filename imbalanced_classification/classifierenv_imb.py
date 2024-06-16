import numpy as np
#from tf_agents.specs.array_spec import ArraySpec, BoundedArraySpec
import gym
from gym import spaces
import numpy as np
import pickle
from utils.reward_scales import get_reward_scale_matrix

class ClassifierEnvGym_Imb(gym.Env):
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, genders: np.ndarray, prof2fem_indx: dict, reward_scale_type: str):
        """
            reward_scale_type: "constant", "gender_imb", "learned_scale"
        """
        super(ClassifierEnvGym_Imb, self).__init__()
        self._episode_ended = False

        self.num_professions = len(prof2fem_indx)
        self.num_genders = 2

        self.X_train = X_train
        self.y_train = y_train
        genders2 = np.copy(genders)
        self.prof2fem_indx = prof2fem_indx
        
        self.data_size = self.X_train.shape[0]
        self.id = np.arange(self.data_size)  # List of IDs to connect X and y data
        self.group_data = [[self.X_train[i], self.y_train[i], genders2[i]] for i in range(self.data_size)]


        self.episode_step = 0  # Episode step, resets every episode
        self._state = self.X_train[self.id[self.episode_step]]
        self.action_space = spaces.Discrete(self.num_professions)  # Assuming there are 2 classes
        self.observation_space = spaces.Box(low=self.X_train.min().item(), high=self.X_train.max().item(), shape=self.X_train.shape[1:], dtype=np.float32)

        self.reward_scale = get_reward_scale_matrix(reward_scale_type,  prof2fem_indx)



    def sample_N_datapoints(self, N):
        """sample N datapoints from the self.group_data and return this"""
        idx = np.random.randint(0, self.data_size, N)
        return [self.group_data[i] for i in idx]


    def reset(self):
        np.random.shuffle(self.id)
        self.episode_step = 0
        self._state = self.X_train[self.id[self.episode_step]]
        self._episode_ended = False
        return self._state
    

    def step(self, action: int):
        _, class_label, gender_bool = self.group_data[self.id[self.episode_step]]
        self.episode_step += 1

        reward = self.reward_scale[class_label, gender_bool]
        reward_correct = 1 if action == class_label else -1
        reward = reward * reward_correct

        if self.episode_step == self.data_size - 1:
            self.reset()

        next_state = self.X_train[self.id[self.episode_step]]

        return next_state, reward, class_label, gender_bool 

    def step_batch(self, action: int, batch_size: int):
        _, class_label, gender_bool = self.group_data[self.id[self.episode_step]]
        self.episode_step += 1

        reward_scales = self.reward_scale[class_label, gender_bool]

        if self.episode_step >= self.data_size - 1:
            self.reset()

        next_state = self.X_train[self.id[self.episode_step]]

        return next_state, reward_scales, class_label, gender_bool    

    def render(self, mode='human'):
        # Implement if you want to visualize the environment
        pass

    def close(self):
        # Implement if you need to perform any cleanup
        pass
