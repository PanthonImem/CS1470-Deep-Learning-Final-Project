import numpy as np
from itertools import product
import random
from env import stage_1, animate_game, stage_2
from Action import Action, get_action_dict
import matplotlib.pyplot as plt


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

        self.action_dim = env.num_action
        self.state_dim_continuous = env.state_dim
        self.state_dim_discrete = len(env.possible_holding)

        self.bound = np.vstack(([0,0], [env.height, env.width]))

        # hyper parameter
        self.fourier_dim = 5
        self.epsilon = 1e-1
        self.gamma = 0.99
        self.lambda_ = 0.5
        self.alpha = 5e-4
        
        # game state
        self.state = self.env.reset()

        # get all fourier wave number  (num_of_wave_number, dim_of_wave_number)
        self.wave_num = np.array(list(product(*[range(self.fourier_dim)]*self.state_dim_continuous)))   
        
        # Parameter_space
        self.w = np.zeros([int(self.fourier_dim**self.state_dim_continuous), self.state_dim_discrete, self.action_dim])

        self.x = []
        self.rewards = []

    """
    Calculate fourier basis at a state

    params:
        - cont_state - position in state space (continuous)
    return:
        - fourier component at a point in state space
    """
    def basis (self, state):
        cont_state, _ = state
        norm_state = np.clip((cont_state - self.bound[0,:])/((self.bound[1, :] - self.bound[0, :])), 0,1)
        return np.cos(np.math.pi * (self.wave_num@norm_state))
    
    
    """
    Calculate Q function at a state and an action

    params:
        - state - state which consists of continuous state and discrete state
        - action - an action or a list of action
    return:
        Q-function approximated by fourier basis
    """
    def Q_func(self, state, action_int):
        _, disc_state = state
        idx = np.where(np.array(disc_state) == 1)[0][0]
        fourier_basis = self.basis(state)
        return fourier_basis@self.w[:, idx ,action_int]

    """
    Calculate an policy for a state

    params:
        - state - state which consists of continuous state and discrete state
        - is_test - a boolean
    return:
        an integer representation action with maximum Q_value (random exploration with probability for training)
    """
    def policy(self, state , is_test):
        if (random.random() < 1. - self.epsilon) or is_test:
            action_int =  np.argmax(self.Q_func(state, np.arange(self.action_dim)))
        else:
            action_int =  np.random.choice(self.state_dim_discrete)
        return action_int

    """
    Update parameter according to gradient descend
    params:
        - action_int - an integer representation of the action
        - next_state - state follow from action
        - reward - reward from such action
        - e - eligibility traces
    return:
        updated state
        updated e
    """
    def update(self, action_int, next_state, reward,  e, is_test = False): 

        next_action_int = self.policy(next_state, is_test)
        _, disc_state = self.state
        idx = np.where(np.array(disc_state) == 1)[0][0]
        delta = reward + self.gamma * self.Q_func(next_state, next_action_int) - self.Q_func(self.state, action_int)
        e = self.gamma * self.lambda_ * e
        
        e[:,idx, action_int] += self.basis(self.state)

        self.w += self.alpha * delta * e

        return next_state, e

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
            

            action_int = self.policy(self.state, is_test = False)
            next_state, reward, done= self.env.step(action_int)
            self.x.append(next_state[0])
            self.state, e = self.update(action_int, next_state, reward,  e)
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

            action_int = self.policy(self.state, is_test = True)
            self.state, reward, done = self.env.step(action_int)
            reward_sum += reward

            if done:
                break
            
        
        return reward_sum
    
    """
    Reset all weight to 0
    """
    def reset_weight(self):
        self.w = np.zeros([int(self.fourier_dim**self.state_dim_continuous), self.state_dim_discrete, self.action_dim])
    
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
    num_episodes = 2000  # 1000
    num_test_episodes = 100
    num_timesteps = 210  # 200
    
    """
    Create environment
    """
    env = stage_2()
    
    """
    Instantiate model
    """
    model = SARSA(env)

    """
    load model
    """
    
    # model.load('weight.npy')

    """
    Train model
    """

    for i in range(num_episodes):
        model.reset_state()
        reward = model.train(num_timesteps)
        print('train episode: {:5d}/{:5d} reward: {:8d}'.format(i+1, num_episodes, reward), end = '\r')

        if ((i+1)%1 == 0):
            model.rewards.append(reward)
        
        if ((i+1)%int(num_episodes/10)==0):
            print()

    # model.reset_state()
    print('Training Reward:{}'.format(reward))

    plt.plot(model.rewards)
    plt.show()
    
    


    """
    Test model
    """
    # for i in range(num_test_episodes):
    #     model.reset_state()
    #     reward = model.test(num_timesteps, render =  False)
    #     print('test episode: {}/{} reward: {}'.format(i+1, num_test_episodes, reward), end = '\r')
    #     if ((i+1)%int(num_test_episodes/10)==0):
    #         print()

    # print('Training Reward:{}'.format(reward))
    
    
    
    animate_game(env)
    
    # """
    # Save model for later use
    # """
    model.save('weight2.npy')




