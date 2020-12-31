import numpy as np

class Sigmoid :

	def __init__(self) :
		self.__Name__ = "Sigmoid" # __Name__ initialization
		self.__type__ = 'activation' # type of Layer

	def feed(self,X) :
		"""
			Sigmoid is a mathematical function which has a characteristic S-Shaped Curve.There are a no of common Sigmoid functions such as 
			logistic , hyperbolic tangent , arc tangent .
			Common Sigmoid is Logistic function
			S(x) = 1 / (e^(-x) + 1)
			Input takes any real value and output ranges b/w 0 - 1
		"""
		self.output = 1 / (1+np.exp(-X))
		return self.output
