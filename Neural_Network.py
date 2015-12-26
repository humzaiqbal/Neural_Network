import math
import numpy as np 



"""
A flexible Neural Network class which allows the user to specify the number 
of hidden layers, the number of features, the number of output neurons, and the number
of neurons per hidden layer


Parameters
__________

loss: string (default = Cross Entropy)
      Determines which loss function the output layer should use. 
      Can choose between mean square loss and Cross Entropy loss 

num_features: int 
              Determines the number of neurons in the input layer 

num_outputs:  int 
              Determines the number of neurons in the output layer 

num_hidden_layers: int
                   Determines how many hidden layers for the network 

num_neurons:  int 
              Determines how many neurons per hidden layer   

learning_rate: float 
               Determines how much the weights are updated during the back propagation algorithm  
"""
class Neural_Network():
	def __init__(self, loss, num_features, num_outputs, num_hidden_layers, num_neurons, learning_rate):
		self.num_outputs = num_outputs
		self.num_hidden_layers = num_hidden_layers
		self.num_neurons = num_neurons
		self.weight_vector = []
		self.switcher = {}
		self.layer_matrix = []
		self.error_matrix = []
		self.one_hot_encode(self.switcher, num_outputs)

	def initialize_weights(self):
		#Initialize the weights depending in which layer it is
		for layer in range(len(self.num_hidden_layers + 2)):
			if layer == 1:
				self.weight_vector.append(0, 0.0001, (self.num_neurons, self.num_features))
			elif layer == self.num_hidden_layers + 1:
				self.weight_vector.append(0, 0.0001, (self.num_neurons, self.num_outputs))
			else:
				self.weight_vector.append(0, 0.0001, (self.num_neurons, self.num_neurons))



	def sigmoid(self, vector):
		lst = []
		for elem in vector:
			if elem >= 0:
				z = math.exp(-elem)
				lst.append(1 / (1 + z))
			else:
				z = math.exp(elem)
				lst.append(z / (1 + z))
		return np.array(lst)

	def sigmoid_prime(self, vector):
		return self.sigmoid(vector) * (1 - self.sigmoid(vector))

	def tanh(self, vector):
		return np.tanh(vector)

	def log(a):
		if a > 1e-20:
			return math.log(a)
		return math.log(1e-20)


	def tanh_prime(self, vector):
		val = self.tanh(vector)
		return np.subtract(1, np.multiply(val, val))

	def loss_function(self, loss='Mean Squares'):
		if loss == 'Mean Squares':
			return self.layer_matrix


	def forward_propagate(self, example):
		for layer_number in range(len(self.num_hidden_layers + 2)):
			if layer_number == 0:
				self.layer_matrix.append(self.tanh(np.dot(example, self.weight_vector[0])))
			elif layer_number == self.num_hidden_layers + 1:
				self.layer_matrix.append(self.sigmoid(np.dot(self.layer_matrix[-1], self.weight_vector[layer])))
			else:
				self.layer_matrix.append(self.tanh(np.dot(self.layer_matrix[-1], self.weight_vector[layer])))
			self.layer_matrix = np.array(self.layer_matrix)

		return self.get_encoder(self.switcher, np.argmax(self.layer_matrix[-1]))

	def train(self, images, labels, test_images, test_labels, loss ='Mean Squares'):
		counter = 0
		self.initialize_weights()
		while True:
			index = 0
			count = 0
			while index < len(labels):
				self.weight_vector = [0] * (self.num_hidden_layers + 2)
				counter = len(self.weight_vector)
				while counter >= 0:
					if counter == len(self.weight_vector):
						weight_updates = np.dot(self.layer_matrix[counter], self.sigmoid_prime(self.layer_matrix[counter]))

				if loss == 'Cross Entropy':
					a = self.get_encoder(self.switcher, labels[index][0])
					b = self.forward_propagate(images[index])
					l2_error = y / a - ((1 - y) / (1 - a))
				else:
					l2_error = self.get_encoder(self.switcher, labels[index][0]) - self.forward_propagate(images[index])
				training_image = images[index]
				l2_delta = l2_error * self.part2 * (1 - self.part2)

				l1_error = l2_delta.dot(self.second_weights.T)


				l1_delta = l1_error * self.tanh_prime(self.part1)

				self.second_weights += self.learning_rate * np.outer(self.part1, l2_delta)
				#print np.count_nonzero(dank_second - self.second_weights)
				dank_weights = self.first_weights
				self.first_weights += self.learning_rate * np.outer(training_image, l1_delta)
				#print np.count_nonzero(dank_weights - self.first_weights)
				#print count
				if index == len(labels) - 1:
					print self.predict(test_images, test_labels)
					index = 0
				else:
					index += 1



	def put_one_at_index(self, index, num_outputs):
		c = [0] * num_outputs
		c[index] = 1
		return np.array(c)

	def one_hot_encode(self, switcher, num_outputs):
		for i in range(num_outputs):
			switcher[i] = self.put_one_at_index(num_outputs)

	def get_encoder(self, switcher, number):
		return self.switcher[number]

	def predict(self, vector, actual):
		data = []
		accuracy = 0
		for index in range(len(vector)):
			#print list(self.forward_propagate(vector[index]))
			data.append(list(self.forward_propagate(vector[index])).index(1))
		for new_index in range(len(data)):
			if data[new_index] == actual[new_index]:
				accuracy += 1
		print accuracy / float(len(data))

def sigmoid_scalar(x):
	if x >= 0:
		z = math.exp(-x)
		return 1 / (1 + z)
	else:
		z = math.exp(x)
		return z / ( 1 + z)



