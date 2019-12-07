import random
import gym
import numpy as np
import tensorflow as tf

from env import stage_1, stage_2, stage_3, animate_game, render

class SARSADeepQ(tf.keras.Model):
	def __init__(self, env):
		"""Deep NN for predicting Q values

		Args:
			state_size: int, size of states
			num_action: int, number of actions
		"""
		super(SARSADeepQ, self).__init__()
		self.env = env

		self.num_actions = env.num_action
		self.state_dim_continuous = env.state_dim
		self.state_dim_discrete = len(env.possible_holding)
		self.bound = tf.stack([[0,0], [env.height, env.width]])


		self.fourier_dim = 4

		# Hyperparameter and layer define here

		self.basis = tf.Variable(np.pi * np.indices([self.fourier_dim] * self.state_dim_continuous), dtype = tf.float32)

		self.b = tf.constant(tf.zeros([int(self.fourier_dim ** self.state_dim_continuous)]))

		self.lifting_disc = tf.Variable(np.identity(self.state_dim_discrete), dtype = tf.float32)

		# potentially use lifting dimension as another hyper parameter
		self.w = tf.Variable(tf.random.truncated_normal([self.state_dim_discrete, int(self.fourier_dim ** self.state_dim_continuous) ,self.num_actions], mean = 0.0, stddev = 0.02))

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.5e-3)

		print('initialized model')


		


	@tf.function
	def call(self, cont, disc):
		""" Compute Q values for the states in the batch

		Args:
			states: ndarray of states, a batch of states

		Returns:
			a 2d tensor of each state's Q values
		"""
		# continuous_state -> basis, discrete_state -> lifting vector
		# cont: [state_dim_cont] -> [fourier_dim, fourier_dim]
		# disc: [state_dim_disc] -> [state_dim_disc]
		cont  = tf.cast(tf.clip_by_value((cont - self.bound[0,:])/((self.bound[1, :] - self.bound[0, :])), 0.0,1.0), dtype = tf.float32)
		# disc = tf.convert_to_tensor(disc, dtype = tf.float32)
		idx = np.where(np.array(disc) == 1)[0][0]

		basis_state = tf.tensordot(cont, self.basis, 1)
		lifted =  tf.gather(self.lifting_disc, idx)

		# Flatten basis into 1D
		# cont: [fourier_dim, fourier_dim] -> [1, fourier_dim * fourier_dim]
		basis_state_reshape = tf.reshape(basis_state, [1, -1]) + self. b
		
		# activate basis with cos function and 
		# cont: [1, fourier_dim * fourier_dim] -> [1, fourier_dim * fourier_dim] 
		activated = tf.math.cos(basis_state_reshape)
		
		# create dense matrix depend on discrete_state
		# disc: [state_dim_disc] -> [fourier_dim * fourier_dim, num_action]
		dense = tf.tensordot(lifted, self.w, 1)
		
		# use the dense layer to apply to get Q value 
		# state -> [action_num]
		
		return tf.matmul(activated, dense)
	
	def save(self):
		print('saving model...')
		np.save('./DeepS-checkpoint/w.npy', self.w.numpy())
		np.save('./DeepS-checkpoint/lifting_disc.npy', self.lifting_disc.numpy())
		np.save('./DeepS-checkpoint/basis.npy', self.basis.numpy())
	
	def load(self):
		print('loading model...')
		self.w = tf.Variable(np.load('./DeepS-checkpoint/w.npy'))
		self.lifting_disc = tf.Variable(np.load('./DeepS-checkpoint/lifting_disc.npy'))
		self.basis = tf.Variable(np.load('./DeepS-checkpoint/basis.npy'))



class DeepQSolver:
	def __init__(self, env, state_size, num_actions, num_memory, num_replay, gamma=0.99):
		""" provides API for the DQN model

		Args:
			state_size: int, size of a state
			num_actions: int, number of actions
			num_memory: int, size of the memory
			num_replay: int, number of times for each replay
			gamma: float, discount
		"""
		self.env = env
		self.model = SARSADeepQ(env)
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
		cont, disc = state
		Q_values = self.model(cont, disc)
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
				cont, disc = state
				Q_values = self.model(cont, disc)
				targetQ = Q_values.numpy()
				next_cont, next_disc = next_state
				targetQ[0][action] = rwd + self.gamma * tf.reduce_max(self.model(next_cont, next_disc)).numpy()
				loss = tf.reduce_sum(tf.square(Q_values - targetQ))

			grads = tape.gradient(loss, self.model.trainable_variables)
			self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def train(solver, epsilon=0.05):
	""" Train the model for one episode

	Args:
		env: gym environment
		model: DeepQ instance
		epsilon: double, probability of going random
		gamma: double, discount

	Returns:
		total step
	"""
	state = solver.env.reset()
	finished = False
	total_rwd = 0
	while not finished:
		if np.random.rand(1) < epsilon:
			action = np.random.randint(9)
		else:
			action = solver.best_action(state)
		next_state, rwd, finished = solver.env.step(action)
		total_rwd += rwd
		solver.add_memory((state, next_state, action, rwd, finished))
		solver.experience_replay()
		state = next_state
		# print('reward: {}'.format(rwd), end = '\r')
	return total_rwd

def test(solver, lf, load = True, epsilon = 0.01):
	if load:
		solver.model.load()
	state = solver.env.reset()
	finished = False
	total_rwd = 0

	while not finished:
		if np.random.rand(1) < epsilon:
			action = np.random.randint(9)
		else:
			action = solver.best_action(state)
		next_state, rwd, finished = solver.env.step(action)
		total_rwd += rwd
		state = next_state
		# print('reward: {}'.format(rwd), end = '\r')
	return total_rwd
	
	


def main():
	env = stage_3()
	state_size = 5
	num_actions = 9
	
	solver = DeepQSolver(env, state_size, num_actions, 100, 5)

	
	# epsilon = 1.0
	# # solver.model.load()
	# for i in range(500):
	# 	res = train(solver, epsilon)
	# 	print("Episode :{:4d} Reward: {:6d}".format(i, res), end = '\r')
	# 	if ((i+1)%100 == 0):
	# 		print()
	# 		solver.model.save()

	# 	# epsilon = max(epsilon * 0.95, 0.05)
	
	test(solver, 0.05)
	# animate_game(env)
	render(env, True)

if __name__ == '__main__':
	main()
