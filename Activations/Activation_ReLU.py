import numpy as np

class ReLU : # ReLU - Rectified Linear Unit

	def __init__(self) :
		self.__Name__ = 'ReLU' # __Name__ initialization
		self.__type__ = 'activation' # type of Layer

	def feed(self,X) :

		"""
			The function returns 0 if it receives any negative input, but for any positive value  x  it returns that value back. 
			So it can be written as  f(x)=max(0,x) .
		"""

		self.output = np.maximum(0,X) 
		return self.output # returning output after applying activation