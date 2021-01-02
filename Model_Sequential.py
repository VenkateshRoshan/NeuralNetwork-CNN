import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

class Sequential() :

	def __init__ (self) :
		self.Layers = []
		self.input_shape = None

	def add(self,Layer) :
		self.Layers.append(Layer)
		if self.input_shape is None :
			self.input_shape = Layer.input_shape
		self.output_shape = Layer.output_shape

	def feed(self,X) :
		self.output_batch = X
		for Layer in self.Layers :
			"""
				Feeding data
			"""
			self.output_batch = Layer.feed(self.output_batch)
			"""
				Applying Activation functions to the output of data which is feeded
			"""
			if Layer.ACTIVATION_FUNCTION.lower() == 'Relu'.lower() :
				relu = ReLU()
				self.output_batch = relu.feed(self.output_batch)
			elif Layer.ACTIVATION_FUNCTION.lower() == 'Softmax'.lower() :
				softmax = Softmax()
				self.output_batch = softmax.feed(self.output_batch)
			elif Layer.ACTIVATION_FUNCTION.lower() == 'Sigmoid'.lower() :
				sigmoid = Sigmoid()
				self.output_batch = sigmoid.feed(self.output_batch)

		return self.output_batch


	def fit(self,train_data=None,validation_data=None,validation_split=0,batch_size=32,epochs=1) : 
		print('Model Fitting')

		"""
			Fitting model using train data
		"""

		if train_data is None :
			raise ValueError('[*] Error : train data must be passed to train Neural Network')
		else :
			self.train_data = train_data

		self.validation_data = validation_data

		self.validation_split = validation_split

		self.batch_size = batch_size

		self.epochs = epochs

		N = len(train_data[0]) # Length of train data
		print(N)

		self.output = []

		for ep in range(epochs) :
			print(f'\nEpoch {ep+1} : ')
			self.WEIGHTS = []
			"""
				every epoch train data and finding error rate
			"""
			co = 1
			self.accuracy = 0
			self.error = 0
			for ind in range(0,N,self.batch_size) :
				print(f'\r[','='*co,'>','.'*(int(400/self.batch_size)-co),']' , 'accuracy :',self.accuracy , 'error :' , self.error,end="")
				if ind+self.batch_size <= N :
					self.output_batch = self.feed(self.train_data[0][ind:ind+self.batch_size])
				elif ind+self.batch_size > N :
					self.output_batch = self.feed(self.train_data[0][ind:])
				co += 1
				for o in self.output_batch :
					self.output.append(o)
		print('\n')
		self.output = np.array(self.output)
		print(self.output.shape)


def main() :
	model = Sequential()
	shape = (32,32,3)
	input = Input(shape)
	model.add(Conv2D(NUM_FILTERS=16,KERNEL_SIZE=3,input_shape=input.output_shape,ACTIVATION_FUNCTION='Relu'))
	X_train = []
	Y_train = []
	Path = ['D:/Data/MultiDomain/Dataset/Animals/cats/','D:/Data/MultiDomain/Dataset/Animals/dogs/']
	for i in Path :
		for j in os.listdir(i) :
			img = cv2.imread(i+j)
			img = cv2.resize(img,(shape[:2]))
			X_train.append(img/255.)
			Y_train.append(Path.index(i))
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	print(f'X : {X_train.shape}, Y : {Y_train.shape}')
	model.fit(train_data=(X_train,Y_train),epochs=1)
	#print(model.Output)

if __name__ == '__main__':
	from Layers.Layer_Input import Input
	from Layers.Layer_Conv import Conv2D
	from Activations.Activation_ReLU import ReLU
	from Activations.Activation_Sigmoid import Sigmoid
	from Activations.Activation_Softmax import Softmax
	main()