import random
import gym
import numpy as np
import tensorflow as tf

from env import stage_1

class SARSADeepQ(tf.keras.Model):
	def __init__(self, env):
		"""Deep NN for predicting Q values

		Args:
			state_size: int, size of states
			num_action: int, number of actions
		"""
		super(SARSADeepQ, self).__init__()
		self.env = env

		self.action_dim = env.num_action
		self.state_dim_continuous = env.state_dim
		self.state_dim_discrete = len(env.possible_holding)

		self.fourier_dim = 4

		# Hyperparameter and layer define here

		self.basis = np.pi * np.indices([self.fourier_dim] * self.state_dim_continuous)

		self.b = tf.Variable(tf.random.truncate_normal([self.num_actions], mean = 0.0, stddev = 0.02))

		self.lifting_disc = tf.constant(np.identity(self.state_dim_discrete))

		# potentially use lifting dimension as another hyper parameter
		self.w = tf.Variable(tf.random.truncate_normal([self.lifting_disc, self.fourier_dim * self.fourier_dim ,self.num_actions], mean = 0.0, stddev = 0.02))

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

		self.layers = []

		# continuous_state -> basis, discrete_state -> lifting vector
		# cont: [state_dim_cont] -> [fourier_dim, fourier_dim]
		# disc: [state_dim_disc] -> [state_dim_disc]
		self.layers.append(lambda inputs: (tf.tensordot(inputs[0], self.basis, 1), tf.matmul(inputs[1], self.lifting_disc)))

		# Flatten basis into 1D
		# cont: [fourier_dim, fourier_dim] -> [1, fourier_dim * fourier_dim]
		self.layers.append(lambda inputs: (tf.reshape(inputs[0], [-1, self.fourier_dim * self.fourier_dim]), inputs[1]))
		
		# add bias 
		# cont: [1, fourier_dim * fourier_dim] -> [1, fourier_dim * fourier_dim] 
		self.layers.append(lambda inputs: (inputs[0] + self.b, inputs[1]))
		
		# activate basis with cos function and create dense matrix depend on discrete_state
		# cont: [1, fourier_dim * fourier_dim] -> [1, fourier_dim * fourier_dim] 
		# disc: [state_dim_disc] -> [fourier_dim * fourier_dim, num_action]
		self.layers.append(lambda inputs: (tf.math.cos(inputs[0]), inputs[1]))
		
	
		self.layers.append(lambda inputs: (inputs[0], inputs[1]))
		
		


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
	(pos, holding) = env.reset()
	state = pos + holding
	finished = False
	total_rwd = 0
	while not finished:
		if np.random.rand(1) < epsilon:
			action = np.random.randint(9)
		else:
			action = solver.best_action(state)
		(next_pos, next_holding), rwd, finished = env.step(action)
		next_state = next_pos + next_holding
		total_rwd += rwd
		solver.add_memory((state, next_state, action, rwd, finished))
		solver.experience_replay()
		state = next_state
	return total_rwd


def main():
	import time
	st = time.time()
	env = stage_1()
	state_size = 5
	num_actions = 9
	
	solver = DeepQSolver(state_size, num_actions, 100, 5)
	epsilon = 1
	for i in range(1000):
		res = train(env, solver, epsilon)
		print("Episode", i, "epsilon", epsilon, "time", (time.time() - st) / 60, ": Reward =", res)
		epsilon = max(epsilon * 0.95, 0.01)
		

if __name__ == '__main__':
	main()
