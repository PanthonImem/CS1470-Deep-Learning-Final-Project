import random
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files (x86)\\ffmpeg\\ffmpeg-20191206-b66a800-win64-static\\bin\\ffmpeg.exe'

from env import stage_1, stage_2, stage_3, render, visualize


class DeepQ(tf.keras.Model):
	def __init__(self, state_size, num_actions):
		"""Deep NN for predicting Q values

		Args:
			state_size: int, size of states
			num_action: int, number of actions
		"""
		super(DeepQ, self).__init__()
		self.state_size = state_size
		self.num_actions = num_actions
		
		self.hidden_size1 = 2000
		self.hidden_size2 = 200
		self.hidden_size3 = 32
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		
		self.dense1 = tf.keras.layers.Dense(self.hidden_size1, activation='relu')
		self.dense2 = tf.keras.layers.Dense(self.hidden_size2, activation='relu')
		self.dense3 = tf.keras.layers.Dense(self.hidden_size3, activation='relu')
		self.dense4 = tf.keras.layers.Dense(self.num_actions)
	
	@tf.function
	def call(self, states):
		""" Compute Q values for the states in the batch

		Args:
			states: ndarray of states, a batch of states

		Returns:
			a 2d tensor of each state's Q values
		"""
		out = self.dense1(states)
		out = self.dense2(out)
		out = self.dense3(out)
		out = self.dense4(out)
		return out


class DeepQSolver:
	def __init__(self, state_size, num_actions, num_memory, num_replay, gamma=0.99):
		""" provides API for the DQN model

		Args:
			state_size: int, size of a state
			num_actions: int, number of actions
			num_memory: int, size of the memory
			num_replay: int, number of times for each replay
			gamma: float, discount
		"""
		self.model = DeepQ(state_size, num_actions)
		self.num_memory = num_memory
		self.memory = []
		self.num_replay = num_replay
		self.gamma = gamma
	
	def best_action(self, state):
		""" gets the best action to perform at the current state

		Args:
			state: state

		Returns:
			the action in the state with the highest Q value
		"""
		Q_values = self.model(np.asarray([state]))
		action = tf.argmax(Q_values, 1)[0].numpy()
		# print(state, Q_values)
		return action
	
	def add_memory(self, tuple):
		""" add information to the memory

		Args:
			tuple: tuple, (state, next_state, action, rwd, finished)
		"""
		self.memory.append(tuple)
		if len(self.memory) > self.num_memory:
			self.memory = self.memory[1:]
	
	def experience_replay(self):
		"""
		replays previous episodes
		"""
		if len(self.memory) < self.num_replay:
			return
		batch = random.sample(self.memory, self.num_replay)
		states, next_states, actions, rwds, finished = zip(*batch)
		# print(actions)
		
		with tf.GradientTape() as tape:
			Q_values = self.model(tf.convert_to_tensor(states))
			Q_next_values = self.model(tf.convert_to_tensor(next_states))
			targetQ = tf.stop_gradient(tf.clip_by_value(rwds + self.gamma * tf.reduce_max(Q_next_values, axis=1), clip_value_min=-10000, clip_value_max=20000))
			relevant_Q = tf.gather_nd(Q_values, tf.stack([tf.range(len(actions)), actions], axis=1))
			loss = tf.reduce_mean(tf.square(relevant_Q - targetQ))
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def train(env, solver, epsilon=0.05):
	""" Train the model for one episode

	Args:
		env: gym environment
		model: DeepQ instance
		epsilon: double, probability of going random
		gamma: double, discount

	Returns:
		total step
	"""
	(pos, holding) = env.reset()
	pos = [pos[0] / env.height, pos[1] / env.width]
	state = pos + holding
	finished = False
	total_rwd = 0
	# print(state, solver.model(tf.convert_to_tensor([state])))
	while not finished:
		if np.random.rand(1) < epsilon:
			action = np.random.randint(9)
		else:
			action = solver.best_action(state)
		(next_pos, next_holding), rwd, finished = env.step(action)
		next_pos = [next_pos[0] / env.height, next_pos[1] / env.width]
		next_state = next_pos + next_holding
		if rwd == 200 - 1:
			print("Get food")
		if rwd == 350 - 1:
			print("Cut")
		if rwd == 1000 - 1:
			print("Serve")
		total_rwd += rwd
		solver.add_memory((state, next_state, action, rwd, finished))
		solver.experience_replay()
		state = next_state
	return total_rwd


def main():
	import time
	st = time.time()
	env = stage_2()
	state_size = 5
	num_actions = 9
	
	solver = DeepQSolver(state_size, num_actions, 2000, 100)
	epsilon = 1
	train_rewards = []
	for i in range(500):
		res = train(env, solver, epsilon)
		print("Train: Episode", i, "epsilon", epsilon, "time", (time.time() - st) / 60, ": Reward =", res)
		epsilon = max(epsilon * 0.99, 0.05)
		train_rewards.append(res)
	visualize(train_rewards, 'DeepQ', 'DeepQ_stage2.png')
	
	# st = time.time()
	# test_rewards = []
	# for i in range(100):
	# 	res = train(env, solver, 0)
	# 	print("Test: Episode", i, "time", (time.time() - st) / 60, ": Reward =", res)
	# 	test_rewards.append(res)
	# print(f'Test: average {np.mean(test_rewards)}')
	
	render(env, save_path='DeepQ_stage2.mp4')


if __name__ == '__main__':
	main()
