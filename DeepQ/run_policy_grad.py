import os
import gym
import numpy as np
import tensorflow as tf
from pylab import *

from env import stage_1, stage_2

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ReinforceWithBaseline(tf.keras.Model):
	def __init__(self, state_size, num_actions):
		super(ReinforceWithBaseline, self).__init__()
		self.num_actions = num_actions
		
		# Define actor network parameters, critic network parameters, and optimizer
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		# actor network
		self.actor_hidden_size = 1000
		self.actor_hidden_size2 = 100
		self.actor_hidden_size3 = 16
		self.actor_dense1 = tf.keras.layers.Dense(self.actor_hidden_size, activation='relu')
		self.actor_dense2 = tf.keras.layers.Dense(self.actor_hidden_size2, activation='relu')
		self.actor_dense3 = tf.keras.layers.Dense(self.actor_hidden_size3, activation='relu')
		self.actor_dense4 = tf.keras.layers.Dense(self.num_actions, activation='softmax')
		# critic network
		# self.critic_hidden_size = 16
		# self.critic_dense1 = tf.keras.layers.Dense(self.critic_hidden_size, activation='relu')
		self.critic_dense1 = tf.keras.layers.Dense(self.actor_hidden_size, activation='relu')
		self.critic_dense2 = tf.keras.layers.Dense(self.actor_hidden_size2, activation='relu')
		self.critic_dense3 = tf.keras.layers.Dense(self.actor_hidden_size3, activation='relu')
		self.critic_dense4 = tf.keras.layers.Dense(1)
	
	@tf.function
	def call(self, states):
		out = self.actor_dense1(states)
		out = self.actor_dense2(out)
		out = self.actor_dense3(out)
		out = self.actor_dense4(out)
		return out
	
	def value_function(self, states):
		out = self.critic_dense1(states)
		out = self.critic_dense2(out)
		out = self.critic_dense3(out)
		out = self.critic_dense4(out)
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


def generate_trajectory(env, model, verbose):
	states = []
	actions = []
	rewards = []
	(pos, holding) = env.reset()
	pos = [pos[0] / env.height, pos[1] / env.width]
	state = pos + holding
	done = False
	
	while not done:
		prbs = model(np.asarray([state]))
		action = np.random.choice(model.num_actions, p=prbs.numpy()[0])
		
		if verbose:
			print(pos, prbs, action)
		
		states.append(state)
		actions.append(action)
		(pos, holding), rwd, done = env.step(action)
		if rwd == 200 - 1:
			print("Get food")
		if rwd == 350 - 1:
			print("Cut")
		if rwd == 1000 - 1:
			print("Serve")
		pos = [pos[0] / env.height, pos[1] / env.width]
		state = pos + holding
		rewards.append(rwd)
	
	return states, actions, rewards


def train(env, model, verbose):
	with tf.GradientTape() as tape:
		states, actions, rewards = generate_trajectory(env, model, verbose)
		discounted_rewards = discount(rewards)
		loss = model.loss(np.asarray(states), np.asarray(actions), np.asarray(discounted_rewards))
	grads = tape.gradient(loss, model.trainable_variables)
	model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return np.sum(rewards)


def main():
	import time
	st = time.time()
	
	env = stage_2()  # environment
	state_size = 5
	num_actions = 9
	
	model = ReinforceWithBaseline(state_size, num_actions)
	
	total_rewards = []
	for i in range(2000):
		res = train(env, model, i % 1000 == 0)
		print(f'Episode {i}: {res}')
		total_rewards.append(res)
	# print(f'Episode {i}: reward = {res}')
	print(f'The average of the last 50 rewards = {np.mean(total_rewards[-50:])}')
	print(f'Time = {(time.time() - st) / 60.0}')
	
	# Visualize your rewards.
	visualize_data(total_rewards)


if __name__ == '__main__':
	main()