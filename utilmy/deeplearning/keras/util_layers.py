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
from box import Box
# from utilmy import pd_read_file


################################################################################################
from utilmy.utilmy import log, log2
from tensorflow.python.keras.engine.functional import Functional
from typing import List, Optional

def help():
    """
    Summary: Extract code source from test code
    """
    from utilmy import help_create
    ss = HELP + help_create("utilmy.deeplearning.keras.util_layers")
    print(ss)


################################################################################################
def test_all() -> None:
    """
    Summary: test the Resnet network layer
    """
    test_resnetlayer()


def test_resnetlayer() -> None:
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
    
    
def test_classifier_multihead() -> None:
   """function test_classifier_multihead
   Args:
   Returns:
       
   """
   label_name_ncount = { 'color': 5, 'gender': 2, 'size': 7}
   clf = make_classifier_multihead(label_name_ncount=label_name_ncount,
                                   layers_dim=[128, 1024], tag='1')
   log(clf.summary())
 
 
def test_DFC_VAE():
   """function test_DFC_VAE
   Args:
   Returns:
       
   """
 
   log(m.summary())


def test_encoder():
   """function test_encoder
   Args:
   Returns:
       
   """
       
   log(m.summary())
       
       
def test_decoder():
   """function test_decoder
   Args:
   Returns:
       
   """
       
   log(m.summary())
       
       
       
def dataloader_get_mnist():
    """" train_laoder = dataloader_get_mnist()
    for i, (Xbatch, ybatch) in enumerate(train_loader):
        log(f'image shape : {Xbatch.shape}')
        log(f'label shape : {ybatch.shape}')
        break
    """        
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()
    
    from util.deeplearning.keras.util_dataloader_img import  CustomDataGenerator   
    train_loader = CustomDataGenerator(X_train, y_train)
    return train_loader   

       
       
################################################################################################
def make_classifier_multihead(label_name_ncount:dict=None, 
                              layers_dim: List[int]=[128, 1024], tag: str='1', latent_dim: int=512) -> Functional:
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
        """ DepthConvBlock:__init__
        Args:
            filters:     
        Returns:
           
        """
        super(DepthConvBlock, self).__init__()
        self.conv = DepthwiseConv2D(3, strides=1, padding='same', activation='relu', )
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.depth_ = Conv2D(filters, 1, strides=1, padding='same', activation='relu', )

    def call(self, inputs):
        """ DepthConvBlock:call
        Args:
            inputs:     
        Returns:
           
        """
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

    def __init__(self, filters: int, kernels: int, strides: int=1, padding: str='valid', activation: Optional[str]=None) -> None:
        """ CNNBlock:__init__
        Args:
            filters (function["arg_type"][i]) :     
            kernels (function["arg_type"][i]) :     
            strides (function["arg_type"][i]) :     
            padding (function["arg_type"][i]) :     
            activation (function["arg_type"][i]) :     
        Returns:
           
        """
        super(CNNBlock, self).__init__()
        self.cnn = Conv2D(filters, kernels, strides=strides, padding=padding)
        self.bn = BatchNormalization()
        self.activation = tf.keras.activations.get(activation)

    def call(self, input_tensor, training=True):
        """ CNNBlock:call
        Args:
            input_tensor:     
            training:     
        Returns:
           
        """
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

    def __init__(self, filters: List[int], kernels: List[int]) -> None:
        """ ResBlock:__init__
        Args:
            filters (function["arg_type"][i]) :     
            kernels (function["arg_type"][i]) :     
        Returns:
           
        """
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
        """ ResBlock:call
        Args:
            input_tensor:     
            training:     
        Returns:
           
        """
        x = self.cnn1(input_tensor)
        x = self.cnn2(x)
        skip = self.identity_mapping(input_tensor)
        y = Add()([x, skip])
        return y




################################################################################################
# The number given below are test cases, remove these before usage
cdim = 3
n_filters = 3


################################################################################################
class DFC_VAE(tf.keras.Model):
    """Deep Feature Consistent Variational Autoencoder Class"""

    def __init__(self, latent_dim, class_dict):
        """ DFC_VAE:__init__
        Args:
            latent_dim:     
            class_dict:     
        Returns:
           
        """
        super(DFC_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = make_encoder()
        self.decoder = make_decoder()

        self.classifier = make_classifier(class_dict)

    def encode(self, x):
        """ DFC_VAE:encode
        Args:
            x:     
        Returns:
           
        """
        z_mean, z_logsigma = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_logsigma

    def reparameterize(self, z_mean, z_logsigma):
        """ DFC_VAE:reparameterize
        Args:
            z_mean:     
            z_logsigma:     
        Returns:
           
        """
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return eps * tf.exp(z_logsigma * 0.5) + z_mean

    def decode(self, z, apply_sigmoid=False):
        """ DFC_VAE:decode
        Args:
            z:     
            apply_sigmoid:     
        Returns:
           
        """
        x_recon = self.decoder(z)
        if apply_sigmoid:
            new_x_recon = tf.sigmoid(x_recon)
            return new_x_recon
        return x_recon

    def call(self, x, training=True, mask=None, y_label_list=None):
        """ DFC_VAE:call
        Args:
            x:     
            training:     
            mask:     
            y_label_list:     
        Returns:
           
        """
        # out_classes = None        
        xcat_all = x[1]  # Category
        x = x[0]  # Image

        z_mean, z_logsigma = self.encode([x, xcat_all])
        z = self.reparameterize(z_mean, z_logsigma)
        x_recon = self.decode(z)

        # Classifier
        out_classes = self.classifier(z)

        return z_mean, z_logsigma, x_recon, out_classes


def make_encoder(xdim=256, ydim=256, latent_dim=10):
    """Creates encoder class"""
    # Functionally define the different layer types
    # Input = tf.keras.layers.InputLayer
    Conv2D = functools.partial(tf.keras.layers.Conv2D, padding='same', activation='relu',
                               kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                               activity_regularizer=regularizers.l2(1e-5))

    Dense = functools.partial(tf.keras.layers.Dense, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                              bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))

    # input0 = [Input( shape=(xdim, ydim, 3)), Input(shape=(cc.labels_onehotdim  , ))]
    input0 = [Input(shape=(xdim, ydim, 3)), Input(shape=(10,))]  # Replace this line with the previous line

    # Build the encoder network using the Sequential API
    encoder1 = tf.keras.Sequential([
        input0[0],
        Conv2D(filters=2 * n_filters, kernel_size=5, strides=2),
        BatchNormalization(),
        layers.Dropout(0.25),

        Conv2D(filters=4 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),
        layers.Dropout(0.25),

        Conv2D(filters=6 * n_filters, kernel_size=3, strides=2),
        BatchNormalization(),

        Flatten(),
        # Dense(512*2, activation='relu'),
    ])

    # Category Input
    encoder2 = tf.keras.Sequential([
        input0[1],
        Dense(64, activation='relu'),
    ])

    x = tf.keras.layers.concatenate([encoder1.output, encoder2.output])
    x = Dense(512 * 2, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = Dense(512 * 2, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    output0 = Dense(2 * latent_dim, activation="sigmoid")(x)

    encoder = tf.keras.Model(inputs=input0, outputs=output0)

    return encoder


def make_decoder(xdim, ydim, latent_dim):
    """
    ValueError: Dimensions must be equal, but are 3 and 4
    for '{{node sub}} = Sub[T=DT_FLOAT](x, sequential_1/conv2d_transpose_3/Relu)'
    with input shapes: [8,256,256,3], [8,256,256,4].
    """
    # Functionally define the different layer types
    # bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                              bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))

    Conv2DTranspose = functools.partial(tf.keras.layers.Conv2DTranspose, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                                        activity_regularizer=regularizers.l2(1e-5))

    # Build the decoder network using the Sequential API
    if xdim == 64:  # 64 x 64 img
        decoder = tf.keras.Sequential([
            Input(input_shape=(latent_dim,)),

            Dense(units=4 * 4 * 6 * n_filters),
            Dense(units=4 * 4 * 6 * n_filters),
            layers.Dropout(0.2),
            Dense(units=4 * 4 * 6 * n_filters),
            Reshape(target_shape=(4, 4, 6 * n_filters)),
            # ValueError: total size of new array must be unchanged, input_shape = [2304], output_shape = [7, 4, 144]

            # Conv. layer
            Conv2DTranspose(filters=4 * n_filters, kernel_size=3, strides=2),
            Conv2DTranspose(filters=2 * n_filters, kernel_size=3, strides=2),
            Conv2DTranspose(filters=1 * n_filters, kernel_size=5, strides=2),

            Conv2DTranspose(filters=3, kernel_size=5, strides=2),
            # Conv2DTranspose(filters=4, kernel_size=5,  strides=2),

        ])

    elif ydim == 256:  # 256 8 256 img
        decoder = tf.keras.Sequential([
            Input(input_shape=(latent_dim,)),

            Dense(units=16 * 16 * 6 * n_filters),
            Dense(units=16 * 16 * 6 * n_filters),
            layers.Dropout(0.2),
            Dense(units=16 * 16 * 6 * n_filters),
            Reshape(target_shape=(16, 16, 6 * n_filters)),

            # Conv. layer
            Conv2DTranspose(filters=4 * n_filters, kernel_size=3, strides=2),
            Conv2DTranspose(filters=2 * n_filters, kernel_size=3, strides=2),
            Conv2DTranspose(filters=1 * n_filters, kernel_size=5, strides=2),
            Conv2DTranspose(filters=3, kernel_size=5, strides=2),

        ])
    else:
        decoder = None

    return decoder


def make_classifier(class_dict, latent_dim=10):
    """ Supervised multi class
            self.gender         = nn.Linear(self.inter_features, self.num_classes['gender'])
            self.masterCategory = nn.Linear(self.inter_features, self.num_classes['masterCategory'])
    """
    Input = tf.keras.layers.InputLayer
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                              bias_regularizer=regularizers.l2(1e-4),
                              activity_regularizer=regularizers.l2(1e-5))
    # Reshape = tf.keras.layers.Reshape
    # BatchNormalization = tf.keras.layers.BatchNormalization

    # if xdim == 64 :   #### 64 x 64 img
    base_model = tf.keras.Sequential([
        Input(input_shape=(latent_dim,)),
        Dense(units=1024),
        # layers.Dropout(0.10),
        # Dense(units=512),
        # layers.Dropout(0.10),
        # Dense(units=512),
    ])

    x = base_model.output
    # x = layers.Flatten()(x) already flatten

    # Multi-heads
    outputs = [Dense(num_classes, activation='softmax', name=f'{class_name}_out')(x) for class_name, num_classes in
               class_dict.items()]
    clf = tf.keras.Model(name='clf', inputs=base_model.input, outputs=outputs)

    return clf


def make_classifier_2(latent_dim, class_dict):
    """ Supervised multi class
            self.gender         = nn.Linear(self.inter_features, self.num_classes['gender'])
            self.masterCategory = nn.Linear(self.inter_features, self.num_classes['masterCategory'])
            self.subCategory    = nn.Linear(self.inter_features, self.num_classes['subCategory'])
            self.articleType    = nn.Linear(self.inter_features, self.num_classes['articleType'])
    """
    Input = tf.keras.layers.InputLayer
    Dense = functools.partial(tf.keras.layers.Dense, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
                              bias_regularizer=regularizers.l2(1e-4),
                              activity_regularizer=regularizers.l2(1e-5))
    # Reshape = tf.keras.layers.Reshape
    # BatchNormalization = tf.keras.layers.BatchNormalization

    # if xdim == 64 :   #### 64 x 64 img
    base_model = tf.keras.Sequential([
        Input(input_shape=(latent_dim,)),
        Dense(units=512),
        layers.Dropout(0.10),
        Dense(units=512),
        layers.Dropout(0.10),
        Dense(units=512),
    ])

    x = base_model.output
    # x = layers.Flatten()(x) already flatten

    # Multi-heads
    outputs = [Dense(num_classes, activation='softmax', name=f'{class_name}_out')(x) for class_name, num_classes in
               class_dict.items()]
    clf = tf.keras.Model(name='clf', inputs=base_model.input, outputs=outputs)

    return clf

   







