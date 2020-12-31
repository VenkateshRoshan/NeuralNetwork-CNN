import numpy as np

class Input :

	"""
		This layer is used convert shape into (None,X,Y,Z) so that we can easily train data using our various batch_size at a time
	"""

	def __init__(self,shape=None) :

		if shape is None :
			raise ValueError('[*] Error : Shape must be passed') # Raises Error cause of shape required to train our model

		elif len(shape) <= 2 :
			raise ValueError('[*] Error : Shape must be atleast two dimensions')
			"""
				Raises error cause image shape would be 2D or 3D
			"""

		else :
			self.input_shape = tuple([None]+list(shape)) # Ex : if shape is (32,32,3) then it converts shape into (None,32,32,3)
			self.output_shape = self.input_shape

def main() :
	inp = Input(shape=(32,32,3))
	print(inp.input_shape)
	print(inp.output_shape)

if __name__ == '__main__':
	main()