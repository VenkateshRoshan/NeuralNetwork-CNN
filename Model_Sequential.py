import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

class Sequential() :

	"""
		arranging every layer to prepare a model
	"""

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
			#print(f'Feeding : {Layer.__Name__} ')
			"""
				Feeding data
			"""
			self.output_batch = Layer.feed(self.output_batch)

			"""
				Applying Activation functions to the output of data which is feeded
			"""
			if Layer.__type__.lower() != 'pool' and Layer.__type__.lower() != 'flatten' :
				if Layer.ACTIVATION_FUNCTION.lower() == 'Relu'.lower() :
					relu = ReLU()
					self.output_batch = relu.feed(np.array(self.output_batch,dtype=np.float64))
				elif Layer.ACTIVATION_FUNCTION.lower() == 'Softmax'.lower() :
					softmax = Softmax()
					self.output_batch = softmax.feed(np.array(self.output_batch,dtype=np.float64))
				elif Layer.ACTIVATION_FUNCTION.lower() == 'Sigmoid'.lower() :
					sigmoid = Sigmoid()
					self.output_batch = sigmoid.feed(np.array(self.output_batch,dtype=np.float64))

		return self.output_batch


	def fit(self,train_data=None,validation_data=None,validation_split=0,batch_size=32,epochs=1,lr=0.1) : 
		print('Model Fitting : ')
		self.lr = lr

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

		for ep in range(epochs) :
			self.output = []
			print(f'\nEpoch {ep+1} : ')
			"""
				every epoch train data and finding error rate
			"""
			self.accuracy = 0
			self.error = 0
			for ind in range(0,N,self.batch_size) :
				print(f'\r','[','='*int(ind/self.batch_size),'>','.'*(int(400/self.batch_size)-int(ind/self.batch_size)),']',end="")
				if ind+self.batch_size <= N :
					self.output_batch = self.feed(self.train_data[0][ind:ind+self.batch_size])
				elif ind+self.batch_size > N :
					self.output_batch = self.feed(self.train_data[0][ind:])
				for o in range(len(self.output_batch)) :
					self.output.append(self.output_batch[o])

			print('\n')
			self.output = np.array(self.output)
			"""
				Accuracy and Error Measuring
			"""
			co = 0
			for i in range(N) :
				pred_label = np.argmax(self.output[i])
				if self.train_data[1][i] ==  pred_label:
					co += 1
			
			self.accuracy = (co/N)*100
			self.error = self.mse(np.argmax(self.output,axis=1),self.train_data[1])
			out_err = self.mse_prime(np.argmax(self.output,axis=1),self.train_data[1])
			for Layer in reversed(self.Layers) :
				if Layer.__Name__ == 'Dense' :
					out_err = self.mse_prime(out_err,self.train_data[1])
			print('Accuracy :',self.accuracy , 'Error :' , self.error)
			print(self.output)

	def mse(self,y_true, y_pred):
		"""
			Mean Squared error
		"""
		return np.mean(np.power(y_true - y_pred, 2))

	def mse_prime(self,y_true, y_pred):
		return 2 * (y_pred - y_true) / y_pred.size

	def Summary(self) :
		print('Layer\t\t\tInput Shape\t\tOutput Shape')
		for Layer in self.Layers :
			Layer.Summary()

	def plotImg(self) :
		for Layer in self.Layers :
			if Layer.__type__ == 'pool' or Layer.__type__ == 'convolving' :
				Layer.plotImg(Layer.output)

def main() :
	model = Sequential()
	shape = (50,50,3)
	input = Input(shape)
	model.add(Conv2D(NUM_FILTERS=16,KERNEL_SIZE=3,input_shape=input.output_shape,ACTIVATION_FUNCTION='Relu')) # Conv layer feed
	model.add(Conv2D(NUM_FILTERS=32,KERNEL_SIZE=5,input_shape=model.output_shape)) # Conv layer feed
	model.add(MaxPool2D(KERNEL_SIZE=3,STRIDES=2,input_shape=model.output_shape)) # MaxPool layer feed
	model.add(Conv2D(NUM_FILTERS=64,KERNEL_SIZE=3,input_shape=model.output_shape)) # Conv layer feed
	model.add(MaxPool2D(KERNEL_SIZE=5,STRIDES=2,input_shape=model.output_shape)) # MaxPool layer feed
	model.add(Flatten(input_shape=model.output_shape))
	model.add(Dense(32,ACTIVATION_FUNCTION='ReLU',input_shape=model.output_shape))
	model.add(Dense(2,ACTIVATION_FUNCTION='Sigmoid',input_shape=model.output_shape))
	model.Summary()
	X_train = []
	Y_train = []
	Path = ['D:/Data/MultiDomain/Dataset/Animals/cats/','D:/Data/MultiDomain/Dataset/Animals/dogs/']#,'D:/Data/MultiDomain/Dataset/Animals/fox/']
	for i in Path :
		co = 0
		for j in os.listdir(i) :
			img = cv2.imread(i+j)
			img = cv2.resize(img,(shape[:2]))
			X_train.append(img/255.)
			Y_train.append(Path.index(i))
			co += 1
			if co == 1 :
				break
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	print(f'X : {X_train.shape}, Y : {Y_train.shape}')
	model.fit(train_data=(X_train,Y_train),epochs=10)
	#model.plotImg()
	#print(model.output)

if __name__ == '__main__':
	from Layers.Layer_Input import Input
	from Layers.Layer_Conv import Conv2D
	from Layers.Layer_Pool import MaxPool2D
	from Layers.Layer_Flatten import Flatten
	from Layers.Layer_Dense import Dense
	from Activations.Activation_ReLU import ReLU
	from Activations.Activation_Sigmoid import Sigmoid
	from Activations.Activation_Softmax import Softmax
	main()