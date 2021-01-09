import numpy as np
import os
import cv2

class Flatten() :

	def __init__(self,input_shape=None) :
		self.__Name__ = 'flat'
		self.input_shape = input_shape
		re = 1
		for i in input_shape[1:] :
			re *= i
		self.output_shape = (input_shape[0],re)

	def flatten(self,input_batch) :
		self.output_batch = []
		for i in input_batch :
			self.output_batch.append(i.ravel())
		return self.output_batch

	def feed(self,X) :
		return self.flatten(X)

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
	output = flat.flatten(output)
	print(flat.output_shape)
	print(output)

if __name__ == '__main__':
	from Layer_Input import Input
	from Layer_Conv import Conv2D
	from Layer_Pool import MaxPool2D
	main()