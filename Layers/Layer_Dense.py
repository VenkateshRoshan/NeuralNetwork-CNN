import numpy as np
import os
import cv2

class Dense() :

	def __init__ (self,NUM_FILTERS,input_shape=None,ACTIVATION_FUNCTION='ReLU',STRIDES=1,PAD=0) :

		self.__Name__ = 'Dense'
		self.__type__ = 'dense'

		if NUM_FILTERS <= 0 :
			raise ValueError('[*] Error : no of filters must be greater than 0')
		elif NUM_FILTERS >= 1 :
			self.NUM_FILTERS = NUM_FILTERS
		else :
			raise ValueError('[*] Error : no of filters must be passed and greater than 1')

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

		self.output_shape = (input_shape[0],NUM_FILTERS)

		self.input = []

		self.WEIGHTS = []

	def dense(self,input_batch) :
		self.output_batch = []
		self.input = input_batch
		self.WEIGHTS = 0.01*np.random.randn(len(input_batch[0]),self.NUM_FILTERS) / np.sqrt(len(input_batch[0])+self.NUM_FILTERS)

		self.bias = np.random.randn(len(input_batch), self.NUM_FILTERS) / np.sqrt(len(input_batch[0])+self.NUM_FILTERS)
		
		self.output_batch = np.dot(np.array(input_batch),self.WEIGHTS) + self.bias
		return self.output_batch

	def feed(self,X_train) :
		return self.dense(X_train)

	def feed_backward(self,out_err,lr) :
		"""
			Error measuring and updating weights and bias to improve accuracy
		"""
		self.input = np.array(self.input)
		self.WEIGHTS = np.array(self.WEIGHTS)
		input_error = []
		for i in out_err :
			input_error.append(np.dot(i,self.WEIGHTS.T))
		weights_err = []
		for i in np.dot(self.input.T,out_err) :
			weights_err.append(i)
		self.WEIGHTS -= lr*np.array(weights_err)
		self.bias -= lr * out_err
		return np.array(input_error)

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
	maxpool = MaxPool2D(KERNEL_SIZE=3,STRIDES=3,input_shape=conv.output_shape)
	output = maxpool.feed(conv.feed(X_train))
	flat = Flatten(input_shape=maxpool.output_shape)
	output = flat.feed(output)
	dense = Dense(32,input_shape=flat.output_shape)
	output = dense.feed(output)
	print(output)

if __name__ == '__main__':
	from Layer_Input import Input
	from Layer_Conv import Conv2D
	from Layer_Pool import MaxPool2D
	from Layer_Flatten import Flatten
	main()