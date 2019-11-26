
import tensorflow as tf
import numpy as np
import gym
from itertools import product
import random


class SARSA(object):
    """
    Class for SARSA Reinforcement Learning algorithm

    params 
        - env: the environment we want model to learn

    hyperparameters
        - fourier_dim - order of fourier appoximation (start from 1)
        - epsilon - exploring probability
        - gamma - future discount factor
        - lambda_, alpha - gradient descend learning rate
    """

    def __init__(self, env):
        
        
        # game environment
        self.env = env

        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.high.size
        self.bound = np.vstack((env.observation_space.low, env.observation_space.high))

        # hyper parameter
        self.fourier_dim = 4
        self.epsilon = 1e-1
        self.gamma = 0.99
        self.lambda_ = 0.5
        self.alpha = 5e-4
        
        # game state
        self.state = self.env.reset()

        # get all fourier wave number  (num_of_wave_number, dim_of_wave_number)
        self.wave_num = np.array(list(product(*[range(self.fourier_dim)]*self.state_dim)))   
        
        # Parameter_space
        self.w = np.zeros([int(self.fourier_dim**self.state_dim), self.action_dim])

    """
    Calculate fourier basis at a state

    params:
        - state - position in state space
    return:
        - fourier component at a point in state space
    """
    def basis (self, state):
        norm_state = np.clip((state - self.bound[0,:])/((self.bound[1, :] - self.bound[0, :])), 0,1)
        return np.cos(np.math.pi * (self.wave_num@norm_state))
    
    """
    Calculate Q function at a state and an action

    params:
        - state - position in state space
        - action - an action or a list of action
    return:
        Q-function approximated by fourier basis
    """
    def Q_func(self, state, action):
        fourier_basis = self.basis(state)
        return fourier_basis@self.w[:, action]

    """
    Calculate an policy for a state

    params:
        - state - position in state space
        - is_test - a boolean
    return:
        an action with maximum Q_value (random exploration with probability for training)
    """
    def policy(self, state, is_test = False):
        if (random.random() < 1. - self.epsilon) or is_test:
            return np.argmax(self.Q_func(state, np.arange(self.action_dim)))
        else:
            return self.env.action_space.sample()

    """
    Update parameter according to gradient descend
    params:
        - action - an action
        - next_state - state follow from action
        - reward - reward from such action
        - e - eligibility traces
    return:
        updated state
        updated e
    """
    def update(self, action, next_state, reward,  e):

        next_action = self.policy(next_state)
        
        delta = reward + self.gamma * self.Q_func(next_state, next_action) - self.Q_func(self.state, action)
        e = self.gamma * self.lambda_ * e
        e[:,action] += self.basis(self.state)

        self.w += self.alpha * delta * e

        return  next_state, e

    """
    Train the model for 1 game
    params:
        - num_step - maximum allowed number of step
        - render - boolean tell whether to render the animation or not
    return:
        sum of rewarded
    """
    def train(self, num_step = 800, render = False):
        e = np.zeros(self.w.shape)
        reward_sum = 0
        
        for _ in range(num_step):
            
            if render:
                self.env.render()
            
            action = self.policy(self.state)
            next_state, reward, done, _ = self.env.step(action)
               
            self.state, e = self.update(action, next_state, reward,  e)
            reward_sum += reward

            if done:
                break
        

        return reward_sum


    """
    Test the model for 1 game
    params:
        - num_step - maximum allowed number of step
        - render - boolean tell whether to render the animation or not
    return:
        sum of rewarded
    """
    def test(self, num_step = 800, render = False):
        reward_sum = 0
        
        for _ in range(num_step):
            
            if render:
                self.env.render()

            action = self.policy(self.state)
            self.state, reward, done, _ = self.env.step(action)
            reward_sum += reward

            if done:
                break
            
        
        return reward_sum
    
    """
    Reset all weight to 0
    """
    def reset_weight(self):
        self.w = np.zeros([int(self.fourier_dim**self.action_dim), self.action_dim])
    
    """
    Reset all state to the start of the game
    """
    def reset_state(self):
        self.state = self.env.reset()
    
    """
    Save weight  to file
    
    params:
        - file - file name (.npy)
    """
    def save(self, file):
        np.save(file, self.w)
    
    """
    Load  weight from file
    
    params:
        - file - file name (.npy)
    """
    def load(self, file):
        self.w = np.load(file)



if __name__ == '__main__':
    """
    Define parameters
    """
    num_episodes = 4000  # 1000
    num_test_episodes = 100
    num_timesteps = 800  # 200
    
    """
    Create environment
    """
    gym.envs.register(
        id="MountainCarLongerEpisodeLength-v0",
        entry_point="gym.envs.classic_control:MountainCarEnv",
        max_episode_steps=num_timesteps,  # MountainCar-v0 uses 200
        reward_threshold=-110.0,
    )
    env = gym.make("MountainCarLongerEpisodeLength-v0")
    
    """
    Instantiate model
    """
    model = SARSA(env)

    """
    Train model
    """
    
    for i in range(num_episodes):
        reward = model.train(num_timesteps, render = False)
        print('train episode: {}/{} reward: {}'.format(i+1, num_episodes, reward), end = '\r')
        
        if ((i+1)%int(num_episodes/10)==0):
            print()
        
        model.reset_state()
    print('Training Reward:{}'.format(reward))
    
    """
    Save model for later use
    """
    model.save('weight.npy')

    """
    Test model
    """
    for i in range(num_test_episodes):
        reward = model.test(num_timesteps, render =  True)
        print('test episode: {}/{} reward: {}'.format(i+1, num_test_episodes, reward), end = '\r')
        if ((i+1)%int(num_test_episodes/10)==0):
            print()
        model.reset_state()







