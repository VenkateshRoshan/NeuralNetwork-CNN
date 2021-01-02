import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

class MaxPool2D :

	"""
		Pooling is to progressively reduce the spatial size of the representation to reduce the amount of parameters
		and computation in network and to control overfitting and it operates independently on every depth layer slice 
		of the input and resizes it spatially, using the MAX operation.
		Accepts a volume of size W1×H1×D1
		Requires two hyperparameters:
			their spatial extent F,
			the stride S,
		Produces a volume of size W2×H2×D2 where:
			W2=(W1−F)/S+1
			H2=(H1−F)/S+1
			D2=D1
	"""

	def __init__ (self,KERNEL_SIZE=2,STRIDES=1,input_shape=None) :
		self.__Name__ = 'MaxPool2D'
		self.__type__ = 'pool'
		self.ACTIVATION_FUNCTION = None
		if KERNEL_SIZE < 1 :
			raise ValueError('[*] Error : KERNEL_SIZE must be greater than or equal to 1')
		else :
			self.KERNEL_SIZE = KERNEL_SIZE

		if STRIDES < 1 :
			raise ValueError('[*] Error : STRIDES must be greater than or equal to 1')
		else :
			self.STRIDES = STRIDES

		if input_shape is None :
			raise ValueError('[*] Error : Input Shape must be passed')
		else :
			self.input_shape = input_shape

		self.output_shape = (None,int((self.input_shape[1]-self.KERNEL_SIZE+1)/self.STRIDES),int((self.input_shape[2]-self.KERNEL_SIZE+1)/self.STRIDES),int(self.input_shape[3]))

	def maxpool(self,input_batch) :
		output_batch = []
		W,H,D = self.output_shape[1:]
		for X in input_batch :
			res = np.zeros((W,H,D))
			for i in range(W) :
				for j in range(H) :
					res[i,j] = np.max(np.max(X[i*self.KERNEL_SIZE:(i+1)*self.KERNEL_SIZE,j*self.KERNEL_SIZE:(j+1)*self.KERNEL_SIZE].T,axis=1),axis=1)

			output_batch.append(res)
		return output_batch

	def feed(self,input_batch) :
		return self.maxpool(input_batch)

	def plotImg(self,X_train) :
		for X in X_train :
			Filter_SIZE = int(X.shape[-1]**(1/2))
			_, axs = plt.subplots(Filter_SIZE,Filter_SIZE, figsize=(8,8))
			axs = axs.flatten()

			#print(X.shape)
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
	maxpool = MaxPool2D(KERNEL_SIZE=3,STRIDES=3,input_shape=conv.output_shape)
	print(maxpool.output_shape)
	output = maxpool.feed(conv.feed(X_train))
	#maxpool.plotImg(output)


if __name__ == '__main__':
	from Layer_Input import Input
	from Layer_Conv import Conv2D
	main()