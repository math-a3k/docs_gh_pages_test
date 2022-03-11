from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model
import tensorflow as tf, numpy as np, imutils, cv2




def test_all():
    """
    Summary: test the network layer
    """
    pass



################################################################################################
def make_efficientet(xdim, ydim, cdim):
    """function make_efficientet
    Args:
        xdim:   
        ydim:   
        cdim:   
    Returns:
        
    """
    # Input is 0-255, do not normalize input
    percep_model = tf.keras.applications.EfficientNetB2(
         include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(xdim, ydim, cdim), pooling=None, classes=1000,
         classifier_activation='softmax'
     )
    return percep_model






################################################################################################
def test_DFC_VAE() -> None:
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
        """ DFC_VAE:__init__
        Args:
            latent_dim:     
            class_dict:     
        Returns:
           
        """
        super(DFC_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder    = make_encoder()
        self.decoder    = make_decoder()
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






