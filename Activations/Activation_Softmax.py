import numpy as np

class Softmax :

	def __init__(self) :
		self.__Name__ = 'Softmax' # __Name__ initialization
		self.__type__ = 'activation' # type of Layer

	def feed(self,X) :
		"""
		The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. 
		The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values 
		between 0 and 1, so that they can be interpreted as probabilities.
		"""

		e_x = np.exp(X - np.max(X))
		return e_x / e_x.sum(axis=0)