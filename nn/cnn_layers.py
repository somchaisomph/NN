import numpy as np
from nn.activators import ReLU, Sigmoid, Tanh, Softmax
from nn.gates import MultiplyGate, AddGate
from nn import helpers as helpers

# Layer
_activation = dict(
	relu=ReLU(),
	softmax=Softmax(),
	tanh=Tanh(),
	sigmoid=Sigmoid())

class Layer():
	def __init__(self):
		self._next = None
		self._prev = None
		
	def next_layer(self,layer):
		self._next = layer
		self._next.prev_layer(self)
		
	def prev_layer(self,layer):
		self._prev = layer
		
	def forward(self,data):
		pass
		
	def backward(self,data):
		pass	
		
class Data(Layer):
	def __init__(self,value):
		self._value = value
		
	def set_value(self,value):
		self._value = value
	
	def get_value(self):
		return self._value

class Conv(Layer):	
	
	def __init__(self,pad=2,stride=2,activation='relu',kernel_count=1,kernel_dim=3):
		self.activation = _activation[activation]
		self.W,self.b = self._gen_(kernel_count,kernel_dim)
		self.stride = stride
		self.pad = pad
		self.cache = None
		self.learning_rate = 0.01
		self.reg_lambda = 0.01
		
	def _gen_(self,count=1,size=3):
		W = np.zeros((count,size,size)) 
		b = np.zeros((count,1,1)) 
		for i in range(count):
			W[i,:,:] = np.random.randn(size,size)/np.sqrt(size)
			b[i,:,:] = np.random.randn(1,1)/np.sqrt(size)
		return W,b
		
	def single_step(self,a_slice_prev, W, b):
		"""
		Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
		of the previous layer.
	
		Arguments:
		a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
		W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
		b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
	
		Returns:
		Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
		"""

		# Element-wise product between a_slice and W. Do not add the bias yet.
		s = a_slice_prev * W
	
		# Sum over all entries of the volume s.
		Z = np.sum(s)
	
		# Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
		Z = float(Z + b)
		
		return Z

	#def forward(self,A_prev, W, b, hparameters):
	def forward(self,A_prev):
		"""
		Implements the forward propagation for a convolution function
	
		Arguments:
		A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
		where
		n_C_prev = number of channel of previous layer
		n_H_prev = number of vertical dimension of previous layer
		n_W_prev = number of horizontal dimension of previous layer
	
		W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
		b -- Biases, numpy array of shape (1, 1, 1, n_C)
		hparameters -- python dictionary containing "stride" and "pad"
		n_C -- Channel count for output (input of next layer)
		
		Returns:
		Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
		cache -- cache of values needed for the conv_backward() function
		"""
			
		# Retrieve dimensions from A_prev's shape   
		(m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
	
		# Retrieve dimensions from W's shape 
		(n_C,f, f) = np.shape(self.W)

		# Retrieve information from "hparameters" 	
		# Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. 
		n_H = int((n_H_prev - f + 2 * self.pad) / self.stride) + 1
		n_W = int((n_W_prev - f + 2 * self.pad) / self.stride) + 1
	
		# Initialize the output volume Z with zeros. (≈1 line)

		Z = np.zeros((m, n_H, n_W, n_C))
	
		# Create A_prev_pad by padding A_prev
		A_prev_pad = helpers.zero_pad(A_prev, self.pad)

		for i in range(m):	# loop over the batch of training examples
			a_prev_pad = A_prev_pad[i,:,:,:]# Select ith training example's padded activation
			for h in range(n_H):  # loop over vertical axis of the output volume
				for w in range(n_W): # loop over horizontal axis of the output volume
					for c in range(n_C): # loop over channels (= #filters) of the output volume					
						# Find the corners of the current "slice" (≈4 lines)
						vert_start = h * self.stride
						vert_end = h * self.stride+ f
						horiz_start = w * self.stride
						horiz_end = w * self.stride + f
					
						# Use the corners to define the (3D) slice of a_prev_pad. 						
						a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
					
						# Convolve the (3D) slice with the correct filter W and bias b, 
						# to get back one output neuron. 
						Z[i, h, w, c] = self.single_step(a_slice_prev,self.W[c,:,:], self.b[c,:,:])
						Z[i, h, w, c] = self.activation.forward(Z[i, h, w, c]) # activation
			 
		# Making sure your output shape is correct
		assert(Z.shape == (m, n_H, n_W, n_C))
	
		# Save information in "cache" for the backprop
		self.cache = A_prev
		return Z
		
	def backward(self,dZ):
		"""
		Implement the backward propagation for a convolution function
	
		Arguments:
		dZ -- gradient of the cost with respect to the output of the conv layer (Z), 
		numpy array of shape (m, n_H, n_W, n_C)
		cache -- cache of values needed for the conv_backward(), output of conv_forward()
	
		Returns:
		dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
			   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
		dW -- gradient of the cost with respect to the weights of the conv layer (W)
			numpy array of shape (f, f, n_C_prev, n_C)
		db -- gradient of the cost with respect to the biases of the conv layer (b)
			numpy array of shape (1, 1, 1, n_C)
		"""
		dZ = self.activation.backward(dZ)

		# Retrieve information from "cache"
		A_prev = self.cache
		W = self.W

		# Retrieve dimensions from A_prev's shape
		(m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
	
		# Retrieve dimensions from W's shape
		f = W.shape[1]
		W = np.reshape(W,(f,f,n_C_prev, W.shape[0]))

		# Retrieve information from "hparameters"
		stride = self.stride
		pad = self.pad
	
		# Retrieve dimensions from dZ's shape
		(m, n_H, n_W, n_C) = np.shape(dZ)
	
		# Initialize dA_prev, dW, db with the correct shapes
		dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))							
		dW = np.zeros((f, f, n_C_prev, n_C))
		db = np.zeros((1, 1, 1, n_C))

		# Pad A_prev and dA_prev
		A_prev_pad = helpers.zero_pad(A_prev, pad)
		dA_prev_pad = helpers.zero_pad(dA_prev, pad)
	
		for i in range(m):	# loop over the training examples
			# select ith training example from A_prev_pad and dA_prev_pad
			a_prev_pad = A_prev_pad[i,:,:,:]
			da_prev_pad = dA_prev_pad[i,:,:,:]
		
			for h in range(n_H): # loop over vertical axis of the output volume
				for w in range(n_W): # loop over horizontal axis of the output volume
					for c in range(n_C):# loop over the channels of the output volume
					
						# Find the corners of the current "slice"
						v_start = h * stride  #vertical start
						v_end = h * stride+ f #vertical end
						h_start = w * stride # horizon start
						h_end = w * stride + f # horizon end
					
						# Use the corners to define the slice from a_prev_pad
						a_slice = a_prev_pad[v_start:v_end,h_start:h_end,:]

						# Update gradients for the window and the 
						# filter's parameters 
						da_prev_pad[v_start:v_end, h_start:h_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
						dW[:,:,:,c] += a_slice * dZ[i, h, w, c] # convolution without padding and stride=1
						db[:,:,:,c] += dZ[i, h, w, c]
						
						
					
		# Set the ith training example's dA_prev to the unpaded da_prev_pad 
		#(Hint: use X[pad:-pad, pad:-pad, :])
		dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
		
		# update parameter
		self.update_param(dW.reshape((1,f,f)),db.reshape((n_C,1,1)))

		# Making sure your output shape is correct
		assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
		return dA_prev

		
	def update_param(self,dW,db):
		#self.W * self.reg_lambda
		self.W += -self.learning_rate * dW
		self.b += -self.learning_rate * db

class Pooling(Layer):
	def __init__(self,window_size=2,stride=2,mode='max'):
		self.stride = stride
		self.window_size = window_size
		self.mode = mode
		self.cache = None

		
	def forward(self,A_prev):
		# Retrieve dimensions from the input shape
		# m -- sample count
		# n_H_prev -- vertical dimension
		# n_W_prev -- horizontal dimension
		# n_C_prev -- channel count		
		(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
		
		# Define the dimensions of the output
		n_H = int(1 + (n_H_prev - self.window_size) / self.stride)
		n_W = int(1 + (n_W_prev - self.window_size) / self.stride)
		n_C = n_C_prev
		
		# Initialize output matrix A
		A = np.zeros((m, n_H, n_W, n_C))
		for i in range(m):	 # loop over the training examples
			for h in range(n_H):	# loop on the vertical axis of the output volume
				for w in range(n_W):	# loop on the horizontal axis of the output volume
					for c in range (n_C):	# loop over the channels of the output volume
					
						# Find the corners of the current "slice" (≈4 lines)
						vert_start = h * self.stride
						vert_end = h * self.stride + self.window_size
						horiz_start = w * self.stride
						horiz_end = w * self.stride + self.window_size
					
						# Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
						a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end,c]
					
						# Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
						if self.mode == "max":
							A[i, h, w, c] = np.max(a_prev_slice)
						elif self.mode == "average":
							A[i, h, w, c] = np.mean(a_prev_slice)
							

		self.cache = (A_prev,A.shape) # for backpropagation

		# Making sure your output shape is correct
		assert(A.shape == (m, n_H, n_W, n_C))

		return A
	
	def backward(self,dA,  mode = "max"):
		"""
		Implements the backward pass of the pooling layer
	
		Arguments:
		dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
		cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
		mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
	
		Returns:
		dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
		"""	
		# Retrieve information from cache 
		A_prev = self.cache[0]
		
		# if dA has not properly shape , reshape it
		dA = np.reshape(dA,self.cache[1])
	
		# Retrieve hyperparameters from "hparameters" 
		stride = self.stride
		f = self.window_size
		# Retrieve dimensions from A_prev's shape and dA's shape 
		m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
		m, n_H, n_W, n_C = dA.shape
	
		# Initialize dA_prev with zeros 
		dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
	
		for i in range(m):	# loop over the training examples
			# select training example from A_prev (≈1 line)
			a_prev = A_prev[i,:,:,:]		
			for h in range(n_H):# loop on the vertical axis
				for w in range(n_W):# loop on the horizontal axis
					for c in range(n_C):# loop over the channels (depth)				
						# Find the corners of the current "slice" (≈4 lines)
						vert_start = h * stride
						vert_end = h * stride+ f
						horiz_start = w * stride
						horiz_end = w * stride + f
						# Compute the backward propagation in both modes.						
						if mode == "max":						
							# Use the corners and "c" to define the current slice from a_prev (≈1 line)
							a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
							# Create the mask from a_prev_slice (≈1 line)
							mask = self.create_mask_from_window(a_prev_slice)
							# Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
							dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, h, w, c]						
						elif mode == "average":						
							# Get the value a from dA (≈1 line)
							da = dA[i, h, w, c]
							# Define the shape of the filter as fxf (≈1 line)
							shape = (f, f)
							# Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
							dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(da, shape)						
	
		# Making sure your output shape is correct
		assert(dA_prev.shape == A_prev.shape)
		return dA_prev

	
		
	def create_mask_from_window(self,x):
		"""
		Creates a mask from an input matrix x, to identify the max entry of x.
	
		Arguments:
		x -- Array of shape (f, f)
	
		Returns:
		mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
		"""	
		mask = (np.max(x) == x)
	
		return mask

	def distribute_value(self,dz, shape):
		"""
		Distributes the input value in the matrix of dimension shape
	
		Arguments:
		dz -- input scalar
		shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
	
		Returns:
		a -- Array of size (n_H, n_W) for which we distributed the value of dz
		"""
		# Retrieve dimensions from shape (≈1 line)
		(n_H, n_W) = shape
	
		# Compute the value to distribute on the matrix (≈1 line)
		average = dz / (n_H * n_W)
	
		# Create a matrix where every entry is the "average" value (≈1 line)
		a = np.ones(shape) * average
		### END CODE HERE ###
	
		return a



class FullyConnected(Layer):
	def __init__(self,activation='tanh',output_layer='softmax'):
		self.W = []
		self.b = []
		self.learning_rate = 0.01
		self.reg_lambda = 0.01
		self.output_layer = _activation[output_layer]
		self.activation=_activation[activation]
	
	def init_param(self,layers_dim):	
		for i in range(len(layers_dim)-1): # exclude output layer
			self.W.append(np.random.randn(layers_dim[i], layers_dim[i+1]) / np.sqrt(layers_dim[i]))
			self.b.append(np.random.randn(layers_dim[i+1]).reshape(1, layers_dim[i+1]))
	
	def predict(self, X):
		mul_gate = MultiplyGate()
		add_gate = AddGate()
		for i in range(len(self.W)):
			mul = mul_gate.forward(self.W[i], X)
			z_l = add_gate.forward(mul, self.b[i])
			X = self.activation.forward(z_l)

		prob = self.output_layer.predict(X)
		#return np.argmax(prob, axis=1)		
		return prob
		
	def forward(self,X):
		mul_gate = MultiplyGate()
		add_gate = AddGate()
		a_l = X # at input layer a_l is X 
		cache = [(None, None, X)]
		for l in range(len(self.W)): # l = 0,1,2,...,L
			mul = mul_gate.forward(self.W[l], a_l)
			z_l = add_gate.forward(mul,self.b[l])
			a_l = self.activation.forward(z_l) # do activation
			cache.append((mul, z_l, a_l)) #keep track
			
		return cache

	def backward(self,cache,target):
		mul_gate = MultiplyGate()
		add_gate = AddGate()
		dtanh = self.output_layer.diff(cache[len(cache)-1][2], target) 
		for i in range(len(cache)-1, 0, -1): # traverse back to input layer
			dadd = self.activation.backward(cache[i][1], dtanh)
			db, dmul = add_gate.backward(cache[i][0], self.b[i-1], dadd)
			dW, dtanh = mul_gate.backward(self.W[i-1], cache[i-1][2], dmul)
			# Add regularization terms (b1 and b2 don't have regularization terms)
			dW += self.reg_lambda * self.W[i-1]
			# Gradient descent parameter update
			self.b[i-1] += -self.learning_rate * db
			self.W[i-1] += -self.learning_rate * dW

		return dtanh
		
	def calculate_loss(self, X, y):
		mul_gate = MultiplyGate()
		add_gate = AddGate()
		softmaxOutput = self.output_layer
		for i in range(len(self.W)):
			mul = mul_gate.forward(self.W[i], X)
			z_l = add_gate.forward(mul, self.b[i])
			X = self.activation.forward(z_l)

		return self.output_layer.loss(X, y)
#--------------------------------------------		
# Network CNN

class Network():
	def __init__(self):
		self.layers = []	
	
	def insert(self,layer):
		self.layers.append(layer)
		
	def train(self,X,Y,num_class):
		"""
		X : training data set
		Y : labeled data set
		"""
		fw_output = self.forward(X,num_class)
		bw_output = self.backward(fw_output,Y)
		#print("Lost",self.layers[-1].calculate_loss(X,Y))
		return bw_output
		
	def forward(self,X,num_class=3):
		first_layer = self.layers[0]		
		_output = first_layer.forward(X)
		
		for i in range(1,len(self.layers),1):
			_input = _output
			_layer = self.layers[i]
			if type(_layer) == FullyConnected :
				L = np.ravel(_input[0,:,:,0])
				L = np.reshape(L,(1,len(L)))
				if _layer.W == [] or _layer.b == [] :
					_layer.init_param([L.shape[1],L.shape[1],num_class])
				_output = _layer.forward(L)
			else :
				_output = _layer.forward(_input)
		return _output	
		
	def backward(self,fw_output,Y):
		last_layer = self.layers[-1]
		_output = last_layer.backward(fw_output,Y)
		
		for l in range(len(self.layers)-2,-1,-1):
			_input = _output
			_layer = self.layers[l]
			_output= _layer.backward(_input)
		
		return _output
		
	def predict(self,X):
		"""
		X : query data set
		"""
		first_layer = self.layers[0]
		_output = first_layer.forward(X)
		for l in range(1,len(self.layers),1):
			_input = _output
			_layer = self.layers[l]
			if type(_layer) == FullyConnected :
				L = np.ravel(_input[0,:,:,0])
				L = np.reshape(L,(1,len(L)))
				_output = _layer.predict(L)
			else :
				_output = _layer.forward(_input)
		return _output
	
