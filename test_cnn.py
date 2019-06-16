import numpy as np
from nn.cnn_layers import Conv,Pooling,FullyConnected,Network

def test_run():
	from skimage import io,color	
	
	img = io.imread('./image/dog3.jpg',flatten=True) 
	img -= np.mean(img) #subtract mean
	img /= np.std(img)
	
	img = np.reshape(img,(img.shape[0],img.shape[1],1))
	data = []
	data.append(img)
	data = np.array(data)
	target = np.array([[0,0,0,0,0,0,0,7,0,0]])
	layers = []
	
	conv1 = Conv(pad=2,stride=2,activation='relu')
	conv2 = Conv(pad=2,stride=2,activation='relu')
	conv3 = Conv(pad=4,stride=4,activation='relu')
	pool = Pooling(window_size=4,stride=4,mode='max')
	fc = FullyConnected()
	network = Network()
	
	network.insert(conv1)
	network.insert(conv2)
	network.insert(conv3)
	network.insert(pool)
	network.insert(fc)
	
	for i in range(50):
		_ = network.train(data,target,target.shape[1])
	print(network.predict(data)*100)

if __name__ == "__main__":
	test_run()
	pass

