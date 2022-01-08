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
class GradCAM:
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


def get_final_image(file_path, model_path, target_size):
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







################################################################################################
def make_efficientet(xdim, ydim, cdim):
    # Input is 0-255, do not normalize input
    percep_model = tf.keras.applications.EfficientNetB2(
         include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(xdim, ydim, cdim), pooling=None, classes=1000,
         classifier_activation='softmax'
     )
    return percep_model






################################################################################################
def test_DFC_VAE():
    """
    model = DFC_VAE(5, 5)
    y_label_list = [ [ 1,2], [2,3 ]
    with tf.GradientTape() as tape:
        z_mean, z_logsigma, x_recon, out_classes = model(x, training=True, y_label_list= y_label_list)      #Forward pass through the VAE

    for epoch in range(epoch0, num_epochs):
        log2(f"Epoch {epoch+1}/{num_epochs}, in {kbatch} kbatches ")

        ###### Set learning rate
        cc.lr_actual            = learning_rate_schedule(mode= cc.lrate_mode, epoch=epoch, cc= cc)
        optimizer.learning_rate = cc.lr_actual

        for batch_idx, (x,  *y_label_list) in enumerate(train_data):
            if dostop: break
            # log("x", x)
            # log("y_label_list", y_label_list)
            # log('[Epoch {:03d} batch {:04d}/{:04d}]'.format(epoch + 1, batch_idx+1, kbatch))
            # print( str(y_label_list)[:100] )
            train_loss = train_step(x, model, y_label_list=y_label_list)
            dd.train_loss_hist.append( np.mean(train_loss.numpy()) )
            # log( dd.train_loss_hist[-1] )
            # image_check(name= f"{batch_idx}.png", img=x[0], renorm=False)
    """
    pass




class DFC_VAE(tf.keras.Model):
    """Deep Feature Consistent Variational Autoencoder Class
        classfier head

    with tf.GradientTape() as tape:
        z_mean, z_logsigma, x_recon, out_classes = model(x, training=True, y_label_list= y_label_list)      #Forward pass through the VAE


    """

    def __init__(self, latent_dim, class_dict):
        super(DFC_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder    = make_encoder()
        self.decoder    = make_decoder()
        self.classifier = make_classifier(class_dict)

    def encode(self, x):
        z_mean, z_logsigma = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return z_mean, z_logsigma

    def reparameterize(self, z_mean, z_logsigma):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return eps * tf.exp(z_logsigma * 0.5) + z_mean

    def decode(self, z, apply_sigmoid=False):
        x_recon = self.decoder(z)
        if apply_sigmoid:
            new_x_recon = tf.sigmoid(x_recon)
            return new_x_recon
        return x_recon

    def call(self, x, training=True, mask=None, y_label_list=None):
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
    # Functionally define the different layer types
    # Input = tf.keras.layers.InputLayer
    cdim = 3
    n_filters = 3
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
    cdim = 3
    n_filters = 3
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






