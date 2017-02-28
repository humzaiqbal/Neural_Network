simport numpy as np 

def __relu(x):
	return np.maximum(x, 0, x)

def __relu_derivative(x):
	x[x <= 0] = 0
	return x

def relu(x, derivative=False):
	if derivative:
		return __relu_derivative(x)
	return __relu(x)


def __sigmoid(x):
	ex = np.exp(-x)
	y = ex / (1 + ex) ** 2
	return y
def __sigmoid_derivative(x):
	return __sigmoid(x) * (1 - __sigmoid(x))

def sigmoid(x, derivative=False):
	if derivative:
		return __sigmoid_derivative(x)
	else:
		return __sigmoid(x)

def softmax(x):
	# do this to avoid blow up
	x -= np.max(x)
	return np.exp(x)/ np.sum(np.exp(x))



class FullyConnectedLayer():
	def __init__(self, learning_rate, input_shape, output_shape, optimization_method,  mu=0.9, update='sgd'):
		self.learning_rate = learning_rate
		self.output_shape = output_shape
		self.W = np.random.randn(output_shape)
		self.input_shape = input_shape
		self.update = update
		#for momentum and nesterov
		self.v = 0
		self.mu = mu
		self.v_prev = 0
		# for adagrad and RMS
		self.cache = np.zeros(input_shape)
		self.eps = 1e-8
		# for RMS
		self.decay_rate = 0.99
		# for Adam
		self.beta1 = 0.9
		self.beta2 = 0.99
		self.m = 0

	def forward(self, data):
		if data.shape != self.input_shape:
			raise Exception('data shapes dont match')
		self.result = np.dot(data, self.W)
		self.data = data
		return self.result 
	def backward(prev_layer):
		self.prev_gradient = np.dot(self.W.T, prev_layer)
		self.W_gradient = np.dot(prev_layer, self.data.T)
		if self.update == 'SGD':
			self.W -= self.learning_rate * self.W_gradient
		elif self.update == 'Momentum':
			self.v = self.mu * self.v - self.learning_rate * self.W_gradient
			self.W += self.v
		elif self.update == 'Nesterov':
			self.v_prev= v
			self.v = self.mu * self.v - self.learning_rate * self.W_gradient
			self.W -= self.mu * self.v_prev + (1 + self.mu) * self.v 
		elif self.update == 'Adagrad':
			self.cache += self.W_gradient ** 2 
			self.W -= self.learning_rate * self.W_gradient / (np.sqrt(cache) + self.eps)
		elif self.update == 'RMS':
			self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * self.W_gradient ** 2
			self.W -=  self.learning_rate * self.W_gradient / (np.sqrt(self.cache) + self.eps)
		elif self.update == 'Adam':
			self.m = self.beta1 * self.m + (1 - self.beta1) * self.W_gradient
			self.v = self.beta2 * self.v + (1 - self.beta1) * (self.dx ** 2)
			self.W -=  self.learning_rate * self.m / (np.sqrt(self.v) + self.eps)
		else:
			raise Exception('update method not defined')
		return self.prev_gradient

class ActivationLayer():
	def __init__(self, activation_function, dropout, input_shape):
		self.activation_function = activation_function
		self.dropout = dropout
	def forward(self, data):
		if data.shape != input_shape:
			raise Exception('data shapes dont match')
		self.result = self.activation_function(data)
		self.dropout_matrix = (np.random.rand(*self.result.shape) < self.dropout) / self.dropout
		self.result = np.dot(self.result, self.dropout_matrix)
		return self.result 
	def backward(self, prev_layer):
		self.gradient = np.dot(prev_layer, self.activation_function(self.data, derivative=True))
		return self.gradient

class SoftmaxLayer():
	def __init__(self, input_shape):
		self.input_shape = input_shape
	def forward(self, data):
		if data.shape != input_shape:
			raise Exception('data shapes dont match')
		self.result = softmax(data)
		return self.result 
	def backward(prev_layer):
		num_classes = self.input_shape[0]
		diagonal = np.multiply(prev_layer, 1 - prev_layer)
		J = np.zeros((num_classes, num_classes))
		for i in range(num_classes):
			for j in range(num_classes):
				if i == j:
					continue
				else:
					J[i][j] = -1 * prev_layer[i] * prev_layer[j]
		self.gradient = np.dot(J, prev_layer)
		return self.gradient


