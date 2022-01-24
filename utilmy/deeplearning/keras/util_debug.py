""""
Tools to debug Keras models, predictions.



"""

# -*- coding: utf-8 -*-
HELP = """
  Template models
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


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model
import tensorflow as tf, numpy as np, imutils, cv2



######################################################################################
def test_all():
    pass


def test_classactivation():
    pass


  
######################################################################################  
def keras_check_layer(mlayer,):
  """ Providing a layer,
      build and run the layer
  
      https://stackoverflow.com/questions/40143676/keras-output-simple-inputs-to-test-layers
  
  
  """
  pass
  
  

######################################################################################
def image_compare_modelpred(file_path, model_path, target_size):
    '''
        File Path (string): Path where the image is stores
        Model Path (string): Path where the model is stored
        Target Size(tuple, size 2): Dimension of input image (height, width)

        Output:
            Class Activation Map overlayed on the original image
    '''
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    preds = model.predict(image)
    i = np.argmax(preds[0])

    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    orig = cv2.imread(file_path)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
    output = np.hstack([orig, output])
    output = imutils.resize(output, height=400)
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    
class GradCAM:
    """ Summary: Gradient-weighted Class Activation Mapping  Model for mapping gradients onto images """

    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output.shape) == 4:
                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs= [self.model.get_layer(self.layerName).output, self.model.output]
        )

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
        colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)




