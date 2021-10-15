import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Add, BatchNormalization,MaxPool2D

class CNNBlock(layers.Layer):
	"""
		This is a convolutional block.
		Here a convolutional layer is followed by a BatchNormalization layer

		Inputs:
		-> output_channels: Filers of the convolutional layer
		-> Kernals: Holds same meaning as the attributes of Conv2D layer
		-> stride: Holds same meaning as the attributes of Conv2D layer
		-> padding: Holds same meaning as the attributes of Conv2D layer
		-> activation: Holds same meaning as the attributes of Conv2D layer

		Output:
		-> output of the convolutional layer after passing through BatchNormalization and activation
	"""

	def __init__(self, filters, kernels, strides = 1, padding = 'valid', activation = 'relu'):
		super(CNNBlock, self).__init__()
		self.cnn = Conv2D(filters, kernels, strides = strides, padding = padding)
		self.bn = BatchNormalization()
		self.activation = tf.keras.activations.get(activation)

	def call(self, input_tensor, training = True):
		x = self.cnn(input_tensor)
		x = self.bn(x, training = training)
		return self.activation(x)


class ResBlock(layers.Layer):

	"""
		This is a Residual Block. 
		The input to the block is passed through 2 convolutional layers.
		The output of these convolutions is added to the input to the residual block through a skip connection.
		NOTE: An identity_mapping is a 1D convolution done in order to ensure that the dimensions match.

		Inputs:
		filters -> list of 2 elements. They are the filters in the Conv layers of the residual block
		kernels -> list of 2 elements. They are the kernel size of the Conv Layers of the residual block

		Outputs:
		Returns the output of the convolutions after adding it to the input of the block through a skip connection
	"""

	def __init__(self, filters, kernels = [3, 3]):
		super(ResBlock, self).__init__()
		self.cnn1 = CNNBlock(filters[0], kernels[0], padding = 'same')
		self.cnn2 = CNNBlock(filters[1], kernels[1], padding = 'same')
		self.pooling = MaxPool2D()
		self.identity_mapping = Conv2D(filters[1], 1, padding = 'same')

	def call(self, input_tensor, training = False):
		x = self.cnn1(input_tensor)
		x = self.cnn2(x)
		skip = self.identity_mapping(input_tensor)
		y = Add()([x, skip])
		return y
