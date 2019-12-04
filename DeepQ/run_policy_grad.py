import os
import gym
import numpy as np
import tensorflow as tf
from pylab import *

from env import stage_1

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ReinforceWithBaseline(tf.keras.Model):
	def __init__(self, state_size, num_actions):
		super(ReinforceWithBaseline, self).__init__()
		self.num_actions = num_actions
		
		# Define actor network parameters, critic network parameters, and optimizer
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
		# actor network
		self.actor_hidden_size = 16
		self.actor_dense1 = tf.keras.layers.Dense(self.actor_hidden_size, activation='relu')
		self.actor_dense2 = tf.keras.layers.Dense(self.num_actions, activation='softmax')
		# critic network
		self.critic_hidden_size = 9
		self.critic_dense1 = tf.keras.layers.Dense(self.critic_hidden_size, activation='relu')
		self.critic_dense2 = tf.keras.layers.Dense(1)
	
	@tf.function
	def call(self, states):
		out = self.actor_dense1(states)
		out = self.actor_dense2(out)
		return out
	
	def value_function(self, states):
		out = self.critic_dense1(states)
		out = self.critic_dense2(out)
		return out
	
	def loss(self, states, actions, discounted_rewards):
		prbs = self.call(states)
		indices = tf.stack([tf.range(actions.shape[0]), actions], axis=1)
		prbs_act = tf.gather_nd(prbs, indices)
		neg_log_prbs_act = -tf.math.log(prbs_act)
		
		state_values = self.value_function(states)
		
		actor_loss = tf.reduce_sum(
			neg_log_prbs_act * tf.dtypes.cast(tf.stop_gradient(discounted_rewards - state_values), tf.float32))
		critic_loss = tf.reduce_sum((discounted_rewards - state_values) ** 2)
		
		return actor_loss + 0.5 * critic_loss


def visualize_data(total_rewards):
	"""
    Takes in array of rewards from each episode, visualizes reward over episodes.

    :param rewards: List of rewards from all episodes
    """
	
	x_values = arange(0, len(total_rewards), 1)
	y_values = total_rewards
	plot(x_values, y_values)
	xlabel('episodes')
	ylabel('cumulative rewards')
	title('Reward by Episode')
	grid(True)
	show()


def discount(rewards, discount_factor=.99):
	# Compute discounted rewards
	discounted_sum = 0
	rev_discounted_rewards = []
	for i in range(len(rewards) - 1, -1, -1):
		discounted_sum = discounted_sum * discount_factor + rewards[i]
		rev_discounted_rewards.append(discounted_sum)
	return list(reversed(rev_discounted_rewards))


def generate_trajectory(env, model):
	states = []
	actions = []
	rewards = []
	(pos, holding) = env.reset()
	state = pos + holding
	done = False
	
	while not done:
		# 1) use model to generate probability distribution over next actions
		prbs = model(np.asarray([state]))
		# 2) sample from this distribution to pick the next action
		action = np.random.choice(model.num_actions, p=prbs.numpy()[0])
		
		states.append(state)
		actions.append(action)
		(pos, holding), rwd, done = env.step(action)
		state = pos + holding
		rewards.append(rwd)
	
	return states, actions, rewards


def train(env, model):
	# 1) Use generate trajectory to run an episode and get states, actions, and rewards.
	with tf.GradientTape() as tape:
		# print('Gen trajectory')
		states, actions, rewards = generate_trajectory(env, model)
		# print('Done gen trajectory')
		# 2) Compute discounted rewards.
		discounted_rewards = discount(rewards)
		# 3) Compute the loss from the model and run backpropagation on the model
		loss = model.loss(np.asarray(states), np.asarray(actions), np.asarray(discounted_rewards))
	grads = tape.gradient(loss, model.trainable_variables)
	model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return np.sum(rewards)


def main():
	import time
	st = time.time()
	
	env = stage_1()  # environment
	state_size = 5
	num_actions = 9
	
	model = ReinforceWithBaseline(state_size, num_actions)
	
	total_rewards = []
	for i in range(2000):
		res = train(env, model)
		print(f'Episode {i}: {res}')
		total_rewards.append(res)
	# print(f'Episode {i}: reward = {res}')
	print(f'The average of the last 50 rewards = {np.mean(total_rewards[-50:])}')
	print(f'Time = {(time.time() - st) / 60.0}')
	
	# Visualize your rewards.
	visualize_data(total_rewards)


if __name__ == '__main__':
	main()