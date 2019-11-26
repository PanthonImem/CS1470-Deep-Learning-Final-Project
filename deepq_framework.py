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

		self.hidden_size = 16
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

		self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
		self.dense2 = tf.keras.layers.Dense(self.num_actions)

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
		return out


def train(env, model, epsilon=0.05, gamma=0.99):
	""" Train the model for one episode

	Args:
		env: gym environment
		model: DeepQ instance
		epsilon: double, probability of going random
		gamma: double, discount

	Returns:
		total reward
	"""
	state = env.reset()
	finished = False
	total_rwd = 0
	while not finished:
		with tf.GradientTape() as tape:
			Q_values = model(np.asarray([state]))
			if np.random.rand(1) < epsilon:
				action = env.action_space.sample()
			else:
				action = tf.argmax(Q_values, 1)[0].numpy()
			state, rwd, finished, _ = env.step(action)
			total_rwd += rwd
			targetQ = Q_values.numpy()
			if finished:
				targetQ[0][action] = rwd
			else:
				targetQ[0][action] = rwd + gamma * tf.reduce_max(model(np.asarray([state]))).numpy()
			loss = tf.reduce_sum(tf.square(Q_values - targetQ))
		grads = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
		# print(action, Q_values, targetQ, model(np.asarray([old_state]))[0])
	return total_rwd


def main():
	env = gym.make("CartPole-v1")  # environment
	state_size = env.observation_space.shape[0]
	num_actions = env.action_space.n

	model = DeepQ(state_size, num_actions)
	for i in range(5000):
		res = train(env, model)
		if i % 50 == 0:
			print(f'Episode {i}: Reward = {res}')


if __name__ == '__main__':
	main()