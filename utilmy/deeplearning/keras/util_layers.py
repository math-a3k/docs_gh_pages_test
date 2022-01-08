# -*- coding: utf-8 -*-
HELP = """
 utils keras for layers
"""
import os,io, numpy as np, sys, glob, time, copy, json, pandas as pd, functools, sys
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Add, BatchNormalization, MaxPool2D, Layer, GlobalAveragePooling2D,
    Dropout,Input, Dense, DepthwiseConv2D, Flatten, Reshape)
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.utils.data_utils import Sequence

# import tensorflow_addons as tfa
# from box import Box
# from utilmy import pd_read_file


################################################################################################
from utilmy.utilmy import log, log2

def help():
    from utilmy import help_create
    ss = HELP + help_create("utilmy.deeplearning.keras.util_layers")
    print(ss)


################################################################################################
def test_all():
    test_resnetlayer()


def test_resnetlayer():
    """ basic implementation of the Residual block in a model architecture
    """
    model = Sequential(
        layers=[
            Input(shape=(224, 224, 3)),
            CNNBlock(32, 3, strides=1, padding='same', activation='relu'),
            ResBlock(filters=[32, 128], kernels=[3, 3]),
            Dropout(0.5),
            GlobalAveragePooling2D(),
            Dense(10, activation='softmax')
        ]
    )
    log(model.summary())





################################################################################################
def make_classifier_multihead(label_name_ncount:dict=None, 
                              layers_dim=[128, 1024], tag='1', cdim=3, n_filters=3):
    """ multi Label output head 
        Vector  --> Dense --> Multiple Softmax Classifier  ( 1 label per class)    
        label_name_ncount:   { 'gender' :  2       #  'male', 'female' 
                               'clothtype':  5     #  'top', 'shirt'
        }  
        
        
    """
    Input = tf.keras.layers.InputLayer
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu',
                              kernel_regularizer   = tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                              bias_regularizer     = regularizers.l2(1e-4),
                              activity_regularizer = regularizers.l2(1e-5))

    latent_dim = layers_dim[0]
    
    base_model = tf.keras.Sequential([
        Input(input_shape=(latent_dim,)),
        Dense(units= layers_dim[1] ),
    ])

    ### Internmediate output 
    x = base_model.output
    # x = layers.Flatten()(x) already flatten

    
    # Multi-heads Classifier
    outputs = []
    for class_name, num_classes in label_name_ncount.items():
       outputs.append( Dense(num_classes, activation='softmax', name=f'{class_name}_out')(x)  )
    
    clf = tf.keras.Model(name='clf_multihead_' + str(tag), 
                         inputs=base_model.input, outputs=outputs)
    return clf








################################################################################################
######## RESIDUAL BLOCK  #######################################################################
class DepthConvBlock(Layer):
    """ This is a Depthwise convolutional block.
        This performs the same function as a convolutional block with much fewer parameters.
        It saves a lot of computational power,
         by using Depthwise convolutional in addition to 1D convolution to replace Conv2D.
        Inputs:
        -> filers: Filter size of the 1d conv layer.
        Input shape:
        -> (n, h, w, c)
            here n is batch size.
            h and w are image dimensions 
            c refers to the number of channels
        Output shape:
        ->(n, h, w, filters)
    """

    def __init__(self, filters):
        super(DepthConvBlock, self).__init__()
        self.conv = DepthwiseConv2D(3, strides=1, padding='same', activation='relu', )
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.depth_ = Conv2D(filters, 1, strides=1, padding='same', activation='relu', )

    def call(self, inputs):
        return self.bn2(self.depth_(self.bn1(self.conv(inputs))))


class CNNBlock(Layer):
    """ This is a convolutional block.
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

    def __init__(self, filters, kernels, strides=1, padding='valid', activation=None):
        super(CNNBlock, self).__init__()
        self.cnn = Conv2D(filters, kernels, strides=strides, padding=padding)
        self.bn = BatchNormalization()
        self.activation = tf.keras.activations.get(activation)

    def call(self, input_tensor, training=True):
        x = self.cnn(input_tensor)
        x = self.bn(x, training=training)
        return self.activation(x)


class ResBlock(Layer):
    """ This is a Residual Block. 
        The input to the block is passed through 2 convolutional layers.
        The output of these convolutions is added to the input to the residual block through a skip connection.
        NOTE: An identity_mapping is a 1D convolution done in order to ensure that the dimensions match.
        Inputs:
        filters -> list of 2 elements. They are the filters in the Conv layers of the residual block
        kernels -> list of 2 elements. They are the kernel size of the Conv Layers of the residual block
        Outputs:
        Returns the output of the convolutions after adding it to the input of the block through a skip connection
    """

    def __init__(self, filters, kernels):
        try:
            assert (type(kernels) == list) and (type(filters) == list)
            assert (len(kernels) == 2) and (len(filters) == 2)
        except AssertionError:
            sys.exit('AssertionError: Please make sure that filters and kernels attribute in ResBlock are lists of 2\
                    elements each')

        super(ResBlock, self).__init__()
        self.cnn1 = CNNBlock(filters[0], kernels[0], padding='same')
        self.cnn2 = CNNBlock(filters[1], kernels[1], padding='same')
        self.pooling = MaxPool2D()
        self.identity_mapping = Conv2D(filters[1], 1, padding='same')

    def call(self, input_tensor, training=False):
        x = self.cnn1(input_tensor)
        x = self.cnn2(x)
        skip = self.identity_mapping(input_tensor)
        y = Add()([x, skip])
        return y












