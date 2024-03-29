import random
import gym
import numpy as np
import tensorflow as tf


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

		self.hidden_size1 = 24
		self.hidden_size2 = 24
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

		self.dense1 = tf.keras.layers.Dense(self.hidden_size1, activation='relu')
		self.dense2 = tf.keras.layers.Dense(self.hidden_size2, activation='relu')
		self.dense3 = tf.keras.layers.Dense(self.num_actions)

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
		for (state, next_state, action, rwd, finished) in batch:
			with tf.GradientTape() as tape:
				Q_values = self.model(np.asarray([state]))
				targetQ = Q_values.numpy()
				if finished:
					targetQ[0][action] = -5
				else:
					targetQ[0][action] = rwd + self.gamma * tf.reduce_max(self.model(np.asarray([next_state]))).numpy()
				loss = tf.reduce_sum(tf.square(Q_values - targetQ))

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
	state = env.reset()
	finished = False
	total_step = 0
	while not finished:
		if np.random.rand(1) < epsilon:
			action = env.action_space.sample()
		else:
			action = solver.best_action(state)
		next_state, rwd, finished, _ = env.step(action)
		total_step += 1
		solver.add_memory((state, next_state, action, rwd, finished))
		solver.experience_replay()
		state = next_state
	return total_step


def main():
	import time
	st = time.time()
	env = gym.make("CartPole-v1")  # environment
	state_size = env.observation_space.shape[0]
	num_actions = env.action_space.n

	solver = DeepQSolver(state_size, num_actions, 100000, 30)
	epsilon = 1
	for i in range(5000):
		res = train(env, solver, epsilon)
		print(f'Episode {i} epsilon {epsilon} time {(time.time() - st)/60}: Reward = {res}')
		epsilon = max(epsilon * 0.95, 0.01)


if __name__ == '__main__':
	main()
