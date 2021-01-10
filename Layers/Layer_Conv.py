import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

class Conv2D :

	"""
	Convolution is a linear operation that involves the multiplication of a set of weights with the input, much like a traditional neural network.
	the technique was designed for two-dimensional input, the multiplication is performed between an array of input data and a two-dimensional 
	array of weights, called a filter or a kernel.
	The filter is smaller than the input data and the type of multiplication applied between a filter-sized patch of the input and the filter is 
	a dot product ( Element wise multiplication ).
	So this method is used to identify features using differnt types of random vectors.

	"""

	def __init__(self,NUM_FILTERS,KERNEL_SIZE,input_shape,STRIDES=1,PAD=0,ACTIVATION_FUNCTION='ReLU') :
		self.__Name__ = 'Conv2D'
		self.__type__ = 'convolving'
		if NUM_FILTERS <= 0 :
			raise ValueError('[*] Error : no of filters must be greater than 0')
		elif NUM_FILTERS >= 1 :
			self.NUM_FILTERS = NUM_FILTERS
		else :
			raise ValueError('[*] Error : no of filters must be passed and greater than 1')

		if KERNEL_SIZE <= 0 :
			raise ValueError('[*] Error : KERNEL SIZE must be greater than 0')
		elif KERNEL_SIZE >= 1 :
			self.KERNEL_SIZE = KERNEL_SIZE
		else :
			raise ValueError('[*] Error : KERNEL SIZE must be passed and greater than 1')

		if STRIDES < 0 :
			raise ValueError('[*] Error : STRIDES must be greater than 0')

		elif STRIDES >= 1 :
			self.STRIDES = STRIDES

		else :
			raise ValueError('[*] Error : STRIDES must be greater than 0')

		if PAD < 0 :
			raise ValueError('[*] Error : PAD must be greater than 0')

		elif PAD >= 1 :
			self.PAD = PAD

		if input_shape is None :
			raise ValueError('[*] Error : input shape must be pass')
		else :
			self.input_shape = input_shape

		if ACTIVATION_FUNCTION is None :
			self.ACTIVATION_FUNCTION = 'ReLU'
		else :
			self.ACTIVATION_FUNCTION = ACTIVATION_FUNCTION

		### Finding Output shape

		self.result = np.zeros((self.input_shape[1]-self.KERNEL_SIZE+1,self.input_shape[2]-self.KERNEL_SIZE+1,self.NUM_FILTERS))

		self.Filters = np.random.randint(-1,2,(self.NUM_FILTERS,self.input_shape[-1],self.KERNEL_SIZE,self.KERNEL_SIZE))

		self.output = []

		self.output_shape = tuple([None]+list(self.result.shape))

	def conv(self,input_batch,WEIGHTS) :

		"""
			adding WEIGHTS either trained or default to Filters for next epochs using filters size
		"""
		
		output_batch = []
		for x in range(len(input_batch)) :
			"""
				self.result initiated to zero again or else it gives same result again
			"""
			self.result = np.zeros((self.input_shape[1]-self.KERNEL_SIZE+1,self.input_shape[2]-self.KERNEL_SIZE+1,self.NUM_FILTERS))
			for i in range(0,self.output_shape[1],self.STRIDES) :
				for j in range(0,self.output_shape[1],self.STRIDES) :
					cur_reg = input_batch[x][i:i+self.KERNEL_SIZE,j:j+self.KERNEL_SIZE].T@self.Filters + WEIGHTS[x]
					for c in range(self.NUM_FILTERS) :
						self.result[i,j,c] = np.sum(cur_reg[c])

			output_batch.append(self.result)
			for o in output_batch :
				self.output.append(o)
		return output_batch

	def feed(self,X,WEIGHTS) :  
		"""
			Feed to Neural Network with length of batch_size 
			X is input batch to feed NN
		"""		
		return self.conv(X,WEIGHTS)

	def plotImg(self,X_train) :
		for X in X_train :
			Filter_SIZE = int(X.shape[-1]**(1/2))
			_, axs = plt.subplots(Filter_SIZE,Filter_SIZE, figsize=(8,8))
			axs = axs.flatten()

			print(X.shape)
			for i , ax in enumerate(axs) :
				img = X[:,:,i]
				ax.axis('off')
				ax.imshow(img)
			plt.show()
			break

	def Summary(self) :
		print(f'{self.__Name__}\t\t{self.input_shape}\t\t{self.output_shape}')


def main() :
	shape = (100,100,3)
	inp = Input(shape=shape)
	X_train = []
	Path = 'D:/Data/TestData'
	for i in os.listdir(Path) :
		img = cv2.imread(Path + '/' + i)
		img = cv2.resize(img,shape[:2])
		img = img/255.
		X_train.append(img)
	conv = Conv2D(NUM_FILTERS=16,KERNEL_SIZE=3,input_shape=inp.output_shape)
	X_train = np.array(X_train)
	output = conv.feed(X_train)
	conv.Summary()
	conv.plotImg(output)

if __name__ == '__main__':
	from Layer_Input import Input
	main()