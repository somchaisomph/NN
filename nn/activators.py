import numpy as np

class ReLU:
	def forward(self,X):
		z = np.zeros(X.shape)
		return np.maximum(X,z)
	
	def backward(self,X):
		p1 = self.forward(X)	
		ones = np.ones(p1.shape)
		prime = np.minimum(p1,ones)
		return prime
		
class Sigmoid:
	def forward(self, X):
		return 1.0 / (1.0 + np.exp(-X))

	def backward(self, X, top_diff):
		output = self.forward(X)
		return (1.0 - output) * output * top_diff

class Tanh:
	def forward(self, X):
		return np.tanh(X)

	def backward(self, X, top_diff):
		output = self.forward(X)
		return (1.0 - np.square(output)) * top_diff	

class Softmax:
	def predict(self, X):
		exp_scores = np.exp(X)
		#return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		return exp_scores / np.sum(exp_scores, keepdims=True)

	def loss(self, X, y):
		num_examples = X.shape[0]
		probs = self.predict(X)
		corect_logprobs = -np.log(probs[range(num_examples), y])
		data_loss = np.sum(corect_logprobs)
		return 1./num_examples * data_loss

	def diff(self, X, y):
		# reference : https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
		num_examples = X.shape[0] # number of data records in train set
		probs = self.predict(X)
		probs[range(num_examples), y] -= 1
		return probs

