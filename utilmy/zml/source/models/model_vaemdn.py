# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Multi Density Variationnal Autoencoder
Only with TF1




"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn

####################################################################################################
from utilmy import global_verbosity, os_makedirs
verbosity = global_verbosity(__file__, "/../../config.json" ,default= 5)

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    if verbosity >= 3 : print(*s, flush=True)

####################################################################################################
global model, session
def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None

def reset():
    global model, session
    model, session = None, None


########Custom Model ################################################################################
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.parquet as pq

import tensorflow as tf
#assert "2.4"  in str(tf.version.VERSION), 'Compatible only with TF 2.4.1, keras 2.4.3, ' + str(tf.version.VERSION)

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Input, Dense, Reshape
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

####################################################################################################
##### Custom code  #################################################################################
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch, 1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


cols_ref_formodel = ['none']  ### No column group
def VAEMDN(model_pars):
    
    original_dim       = model_pars['original_dim']
    class_num          = model_pars['class_num']
    intermediate_dim   = model_pars['intermediate_dim']
    intermediate_dim_2 = model_pars['intermediate_dim_2']
    latent_dim         = model_pars['latent_dim']
    Lambda1            = model_pars['Lambda1']
    batch_size         = model_pars['batch_size']
    Lambda2            = model_pars['Lambda2']
    Alpha              = model_pars['Alpha']
    
    input_shape = (original_dim, )
    inputs = Input(shape=input_shape, name='encoder_input')
    inter_x1 = Dense(intermediate_dim, activation='tanh',
                     name='encoder_intermediate')(inputs)
    inter_x2 = Dense(intermediate_dim_2, activation='tanh',
                     name='encoder_intermediate_2')(inter_x1)
    inter_x3 = Dense(intermediate_dim_2, activation='tanh',
                     name='encoder_intermediate_3')(inter_x1)
    # add 3 means as additional parameters
    dummy = Input(shape=(1,), name='dummy')
    mu_vector = Dense(class_num*latent_dim, name='mu_vector',
                      use_bias=False)(dummy)
    mu = Reshape((class_num, latent_dim), name='mu')(mu_vector)

    # prior categorical distribution
    pi = Dense(class_num, activation='softmax', name='pi')(dummy)

    # posterior categorical distribution
    c = Dense(class_num, activation='softmax', name='c')(inter_x2)

    # outlier/non-outlier classification (Posterior Beta)
    # inter_outlier = Dense(128, activation='relu', name='inter_outlier')(x)
    c_outlier = Dense(2, activation='softmax', name='c_outlier')(inter_x3)

    # q(z|x)
    z_mean = Dense(latent_dim, name='z_mean')(inter_x2)
    z_log_var = Dense(latent_dim, name='z_log_var')(inter_x2)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,),
               name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = keras.models.Model([inputs, dummy], [z_mean, z_log_var, z,
                                      mu, c, c_outlier, pi], name='encoder')

    encoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    #log(encoder.summary())
    ####### build decoder model  ########################################################
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    inter_y1 = Dense(intermediate_dim_2, activation='tanh')(latent_inputs)
    inter_y2 = Dense(intermediate_dim, activation='tanh')(inter_y1)
    outputs = Dense(original_dim, activation='tanh')(inter_y2)

    ########### instantiate decoder model
    decoder = keras.models.Model(latent_inputs, outputs, name='decoder')
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    decoder.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    #log(decoder.summary())


    ################## instantiate VAE model  ####################################################
    outputs = decoder(encoder([inputs, dummy])[2])
    vae = keras.models.Model([inputs, dummy], outputs, name='vae_mlp')
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss = tf.compat.v1.multiply( reconstruction_loss, c_outlier[:, 0])
    reconstruction_loss *= original_dim

    kl_loss_all = tf.compat.v1.get_variable("kl_loss_all", [batch_size, 1],
                                    dtype=tf.compat.v1.float32, initializer=tf.compat.v1.zeros_initializer)
    kl_cat_all = tf.compat.v1.get_variable("kl_cat_all", [batch_size, 1],
                                   dtype=tf.compat.v1.float32, initializer=tf.compat.v1.zeros_initializer)

    dir_prior_all = tf.compat.v1.get_variable("dir_prior_all", [batch_size, 1],
                                      dtype=tf.compat.v1.float32, initializer=tf.compat.v1.zeros_initializer)
    for i in range(0, class_num):
        c_inlier = tf.compat.v1.multiply(c[:, i], c_outlier[:, 0])

        # kl-divergence between q(z|x) and p(z|c)
        kl_loss = 1 + z_log_var - \
            K.square(z_mean-mu[:, i, :]) - K.exp(z_log_var)
        kl_loss = tf.compat.v1.multiply(K.sum(kl_loss, axis=-1), c_inlier)
        kl_loss *= -0.5
        kl_loss_all = kl_loss_all + kl_loss

        # kl-divergence between q(c|x) and p(c) (not including outlier class)
        mc = K.mean(c[:, i])
        mpi = K.mean(pi[:, i])
        kl_cat = mc * K.log(mc) - mc * K.log(mpi)
        kl_cat_all = kl_cat_all + kl_cat

        # Dir prior: Dir(3, 3, ..., 3)
        dir_prior = -0.1*K.log(pi[:, i])
        dir_prior_all = dir_prior_all+dir_prior
    mco1 = K.mean(c_outlier[:, 0])
    mco2 = K.mean(c_outlier[:, 1])
    mpo1 = 1-Alpha
    mpo2 = Alpha
    kl_cat_outlier = (mco1 * K.log(mco1) - mco1 * np.log(mpo1) +
                      mco2 * K.log(mco2) - mco2 * np.log(mpo2))

    # total loss
    vae_loss = K.mean(reconstruction_loss +
                      kl_loss_all +
                      dir_prior_all +
                      Lambda1*kl_cat_all)+Lambda2*kl_cat_outlier

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder, decoder



def AUTOENCODER_BASIC(X_input_dim,loss_type="CosineSimilarity", lr=0.01, epsilon=1e-3, decay=1e-4,
                      optimizer='adam', encodingdim = 50, dim_list="50,25,10" ):
    import tensorflow as tf
    import keras.backend as K
    print(tf.__version__)

    def custom_loss_func_dice(y_true, y_pred, smooth=1):
      intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
      return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

    dim_list = dim_list.split(",")


    # encodingdim = 50
    # input =  tf.keras.layers.Input(X.shape[1],sparse=True) # use this if tensor is sparse
    inputs  = tf.keras.layers.Input(X_input_dim)
    encoded = tf.keras.layers.Dense( int(dim_list[0]), activation='relu')(inputs)
    encoded = tf.keras.layers.Dense( int(dim_list[1]), activation='relu')(encoded)
    encoded = tf.keras.layers.Dense( int(dim_list[2]), activation='relu')(encoded)


    decoded_input = tf.keras.layers.Input( int(dim_list[2]) )
    decoded = tf.keras.layers.Dense(int(dim_list[1]), activation='relu')(decoded_input)
    decoded = tf.keras.layers.Dense(int(dim_list[0]), activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(X_input_dim, activation='relu')(decoded)
    
    encoder = tf.keras.models.Model(inputs=inputs,outputs=encoded,name='encoder')
    encoder.compile(optimizer=optimizer)

    decoder = tf.keras.models.Model(inputs=decoded_input,outputs=decoded,name='decoder')
    decoder.compile(optimizer=optimizer)

    outputs     = decoder(encoder(inputs))
    autoencoder = tf.keras.models.Model(inputs,outputs)
    opt         = tf.keras.optimizers.Adagrad(lr=lr, epsilon=epsilon, decay=decay)

    if loss_type == "cosinesimilarity" :
       autoencoder.compile(optimizer=opt, loss=tf.keras.losses.CosineSimilarity(reduction="auto")) # cosine loss

    if loss_type == "dice" :
       autoencoder.compile(optimizer=opt, loss= custom_loss_func_dice) # dice loss (I used custom loss function and implemented it freehand to show how you can create your own custom function.)

    else :
       autoencoder.compile(optimizer=opt, loss= tf.keras.losses.categorical_crossentropy) # this is same algorith explained as "CustomLoss" topic in pytorch version.

    #log2(autoencoder.summary())
    return autoencoder,encoder,decoder


def AUTOENCODER_MULTIMODAL(input_shapes=[10],
                                  hidden_dims=[128, 64, 8],
                                  output_activations=['sigmoid', 'relu'],
                                  loss = ['bernoulli_divergence', 'poisson_divergence'],
                                  optimizer='adam'):
    """
    pip install mmae[keras]
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_validation, y_validation) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0
    x_validation = x_validation.astype('float32') / 255.0
    y_validation = y_validation.astype('float32') / 255.0

    data = [x_train, y_train]
    validation_data = [x_validation, y_validation]



    # Train model where input and output are the same
    autoencoder.fit(data, epochs=100, batch_size=256,
                    validation_data=validation_data)

    #To obtain a latent representation of the training data:
    latent_data = autoencoder.encode(data)

    #To decode the latent representation:
    reconstructed_data = autoencoder.decode(latent_data)

    #Encoding and decoding can also be merged into the following single statement:
    reconstructed_data = autoencoder.predict(data)

    :return:
    """
    # Remove 'tensorflow.' from the next line if you use just Keras

    from repo.keras_mmae.multimodal_autoencoder import MultimodalAutoencoder

    # Set network parameters
    #input_shapes = [x_train.shape[1:], (1,)]

    # Number of units of each layer of encoder network
    #hidden_dims = [128, 64, 8]

    # Output activation functions for each modality
    #output_activations = ['sigmoid', 'relu']

    #optimizer = 'adam'
    # Loss functions corresponding to a noise model for each modality
    #loss = ['bernoulli_divergence', 'poisson_divergence']

    # Construct autoencoder network
    autoencoder = MultimodalAutoencoder(input_shapes, hidden_dims,
                                        output_activations)
    autoencoder.compile(optimizer, loss)
    autoencoder.summary()
    return autoencoder



##################################################################################################
##################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
            self.encoder, self.decoder = None, None
            return

        model_class = model_pars['model_class']  
        ### get model params  #######################################################
        mdict_default = {
             'original_dim' : 15,
             'class_num':           5
            ,'intermediate_dim':    64
            ,'intermediate_dim_2':  16
            ,'latent_dim':          3
            ,'Lambda1':             1
            ,'batch_size':          256
            ,'Lambda2':             200
            ,'Alpha':               0.075
        }
        mdict = model_pars.get('model_pars', mdict_default)

        ### Dynamic Dimension : data_pars  ---> model_pars dimension  ###############
        # mdict['original_dim'] = np.uint32( data_pars['signal_dimension']*(data_pars['signal_dimension']-1)/2)
        dim = model_pars['model_pars']['original_dim']

        #### Model setup #############################################################
        self.model_pars['model_pars'] = mdict
        if 'VAEMDN' in model_class:
            self.model, self.encoder, self.decoder = VAEMDN( self.model_pars['model_pars'])
        else:
            self.model,self.encoder, self.decoder = AUTOENCODER_BASIC(dim)
        
        log2(self.model)
        # self.model.summary()


def fit(data_pars=None, compute_pars=None, out_pars=None,model_class='VAEMDN', **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train",)
    
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle
    early_stopping = EarlyStopping(monitor='loss', patience=3)
    path_check     = compute_pars.get('path_checkpoint', 'ztmp/model_dir/check_ckpt')
    #os.makedirs(os.path.abspath(path_check) , exist_ok= True)
    #model_ckpt     = ModelCheckpoint(filepath =  path_check,   save_best_only = True, monitor='loss')
    cpars['callbacks'] =  [early_stopping] # , model_ckpt]
    # cpars['callbacks'] = {}
    assert 'epochs' in cpars, 'epoch missing'


    ### Fake label
    Xtest_dummy  = np.ones((Xtest_tuple.shape[0], 1))
    Xtrain_dummy = np.ones((Xtrain_tuple.shape[0], 1))
    if 'VAEMDN' in model_class:
        hist = model.model.fit([Xtrain_tuple,ytrain],
                                validation_data=[Xtest_tuple,ytest],
                                **cpars)
    else:

        hist = model.model.fit(Xtrain_tuple,Xtrain_tuple,
                                validation_data=[Xtest_tuple],
                                **cpars)
    model.history = hist



def encode(Xpred=None, data_pars=None, compute_pars={}, out_pars={},model_class='VAEMDN', **kw):
    global model, session
    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")
    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    #log2(Xpred_tuple)
    Xdummy      = np.ones((Xpred_tuple.shape[0], 1))
    #log(Xpred_tuple)
    
    #### Saving on disk
    if model_class == 'VAEMDN':
        Xnew_encode = model.encoder.predict([Xpred_tuple,Xdummy])
        path_save = compute_pars.get('compute_extra', {}).get('path_encoding', None)
        if path_save is not None :
            os_makedirs(path_save)
            filename    = {0:'encoded_mean.parquet', 1:'encoded_logvar.parquet', 2:'encoded_mu.parquet'}
            for j,encodings in enumerate(Xnew_encode[:3]):
                parquetDic = {}
                for i in range(encodings.shape[1]):
                    name             = f'col_{i+1}'
                    parquetDic[name] = encodings[:,i]
                log2(f'Encoder Columns shape: {encodings.shape}')
                ndarray_table = pa.table(parquetDic)
                pq.write_table(ndarray_table,   path_save  + "/" + filename[j] )
                log(f'{path_save}/{filename[j]} created')
    
    else:
        Xnew_encode = model.encoder.predict(Xpred_tuple)
        log(Xnew_encode.shape)
        path_save = compute_pars.get('compute_extra', {}).get('path_encoding', None)
        if path_save is not None :
            os_makedirs(path_save)
            filename    = 'my_encode_basic_AE.parquet'
                
            parquetDic = {}
            for i in range(Xnew_encode.shape[1]):
                name             = f'col_{i+1}'
                parquetDic[name] = Xnew_encode[:,i]
            log2(f'Encoder Columns shape: {Xnew_encode.shape}')
            ndarray_table = pa.table(parquetDic)
            pq.write_table(ndarray_table,   path_save  + "/" + filename)
            log(f'{path_save}/{filename} created')
    return Xnew_encode


def decode(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, index = 0, **kw):
    global model, session
    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    #log2(Xpred_tuple)
    #Xdummy      = np.ones((Xpred_tuple.shape[0], 1))

    log(Xpred.shape)
    decoded_array = model.decoder.predict(Xpred )
        #### Saving on disk
    path_save = compute_pars.get('compute_extra', {}).get('path_encoding', None)
    if path_save is not None :
        parquetDic = {}
        for i in range(decoded_array.shape[1]):
            name = f'col_{i+1}'
            #### TODO : Actual Column names from data_pars, looks tricky ????
            parquetDic[name] = decoded_array[:,i]
        log2(f'Decoder Columns Shape {decoded_array.shape}')

        filename = {0:'decoded.parquet', 1:'decoded_logvar.parquet', 2:'decoded_mu.parquet'}
        log2(decoded_array)
        ndarray_table = pa.table(parquetDic)
        pq.write_table(ndarray_table,path_save  + "/" + filename[index])
        log(f'{filename[index]} created')

    return decoded_array


def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={},model_class='VAEMDN', **kw):
    global model, session
    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")

    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    #log2(Xpred_tuple)
    Xdummy = np.ones((Xpred_tuple.shape[0], 1))
    if 'VAEMDN' in model_class:
        Xnew   = model.model.predict([Xpred_tuple,Xdummy])
    else:
        Xnew   = model.model.predict(Xpred_tuple )
    return Xnew




####################################################################################################
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :
        return Xtrain

    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )

    return Xtuple_train



def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
    """
    # log(data_pars)
    if data_pars.get('dataset_name', '') == 'correlation' :
       x_train, ytrain = test_dataset_correlation(data_pars)
       return x_train, ytrain


    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

        if task_type == "predict":
            d = data_pars[task_type]
            Xtest     = d["X"]
            #Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtest

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            #Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            #Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref)
            #flog2("Xtuple_train", Xtuple_train)
            
            return Xtrain, ytrain, Xtest, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


def get_label(encoder, x_train, dummy_train, class_num=5, batch_size=256):
    [z_mean, z_log_var, z, mu, c, c_outlier, pi] = encoder.predict(
        [x_train, dummy_train], batch_size=batch_size)

    labels = np.zeros(x_train.shape[0])
    for i in range(0, x_train.shape[0]):
        max_prob = np.max(np.multiply(c[i, :], c_outlier[i, 0]))
        idx = np.argmax(np.multiply(c[i, :], c_outlier[i, 0]))
        if (max_prob > c_outlier[i, 1]):
            labels[i] = idx
        else:
            labels[i] = class_num
    return labels


######################################################################################
def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

    model.model.save(f"{path}/model_keras.h5")
    model.model.save_weights(f"{path}/model_keras_weights.h5")

    modelx = Model()  # Empty model  Issue with pickle
    modelx.model_pars   = model.model_pars
    modelx.data_pars    = model.data_pars
    modelx.compute_pars = model.compute_pars
    # log('model', modelx.model)
    pickle.dump(modelx, open(f"{path}/model.pkl", mode='wb'))  #

    pickle.dump(info, open(f"{path}/info.pkl", mode='wb'))  #


def load_model(path="",model_class='VAEMDN'):
    global model, session
    import dill as pickle

    model0      = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    if 'VAEMDN' in model_class:
        model.model, model.encoder, model.decoder        = VAEMDN( model0.model_pars['model_pars'])
    else:
        model.model, model.encoder, model.decoder        = AUTOENCODER_BASIC( model0.model_pars['model_pars']['original_dim'])
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars

    model.model.load_weights( f'{path}/model_keras_weights.h5')

    log(model.model.summary())
    #### Issue when loading model due to custom weights, losses, Keras erro
    #model_keras = get_model()
    #model_keras = keras.models.load_model(path + '/model_keras.h5' )
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd



#########################################################################################################
def test_dataset_correlation(n_rows=100):
    data_pars = {'dataset_name':  'correlation'}
    data_pars['state_num']           = 10
    data_pars['time_len']            = 500
    data_pars['signal_dimension']    = 15
    data_pars['CNR']                 = 1
    data_pars['window_len']          = 11
    data_pars['half_window_len']     = 5
    
    state_num = data_pars['state_num']
    time_len = data_pars['time_len']
    signal_dimension = data_pars['signal_dimension']
    CNR = data_pars['CNR']
    window_len = data_pars['window_len']
    half_window_len = data_pars['half_window_len']
    a = np.ones(shape=(state_num, state_num))
    alpha = np.ones(10)*10
    alpha[5:] = 1
    base_prob = np.random.dirichlet(alpha) * 0.1
    for t in range(state_num):
        a[t, :] = base_prob
        a[t, t] += 0.9

    # simulate states
    state = np.zeros(time_len, dtype=np.uint8)
    p = np.random.uniform()
    state[0] = np.floor(p*state_num)
    for t in range(0, time_len-1):
        p = np.random.uniform()
        for s in range(state_num):
            if (p <= np.sum(a[state[t], :s+1])):
                state[t+1] = s
                break

    freq = np.zeros(state_num)
    for t in range(state_num):
        freq[t] = np.sum(state == t)
    loading = np.random.randint(-1, 2, size=(state_num, signal_dimension))

    cov = np.zeros((state_num, signal_dimension, signal_dimension))
    for t in range(state_num):
        cov[t, :, :] = np.matmul(np.transpose(
            [loading[t, :]]), [loading[t, :]])

    # generate BOLD signal
    signal = np.zeros((time_len, signal_dimension))
    for t in range(0, time_len):
        signal[t, :] = np.random.multivariate_normal(
            np.zeros((signal_dimension)), cov[state[t], :, :])
    signal += np.random.normal(size=signal.shape)/CNR
    original_dim = np.uint32(signal_dimension*(signal_dimension-1)/2)

    x_train = np.zeros(
        shape=(time_len-window_len*2, np.uint32(original_dim)))
    sum_corr = np.zeros(shape=(state_num, original_dim))
    occupancy = np.zeros(state_num)

    for t in range(window_len, time_len-window_len):
        corr_matrix = np.corrcoef(np.transpose(
            signal[t-half_window_len:t+half_window_len+1, :]))
        upper = corr_matrix[np.triu_indices(signal_dimension, k=1)]
        x_train[t-window_len, :] = np.squeeze(upper)
        if (np.sum(state[t-half_window_len:t+half_window_len+1] == state[t]) == window_len):
            sum_corr[state[t], :] += x_train[t-window_len, :]
            occupancy[state[t]] += 1

    ### Dummy
    y_dummy = np.ones((x_train.shape[0], 1))   ### Not used, no label
    return x_train, y_dummy


def test():
    ######### Custom dataset
    m_signal_dim = 15
    X,y = test_dataset_correlation(n_rows=100)

    ######### size of NN (nb of correl)
    n_width = np.uint32( m_signal_dim * (m_signal_dim-1)/2)

    ######### Data
    d = {'task_type' : 'train', 'data_type': 'ram', }
    # d['signal_dimension'] = 15

    d["train"] ={
      "Xtrain":  X[:10,:],
      "ytrain":  y[:10,:],        ## Not used
      "Xtest":   X[10:1000,:],
      "ytest":   y[10:1000,:],    ## Nor Used
    }
    data_pars= d

    ########## Data
    m                       = {}
    m['original_dim']       = n_width
    m['class_num']          = 5
    m['intermediate_dim']   = 64
    m['intermediate_dim_2'] = 16
    m['latent_dim']         = 3
    m['Lambda1']            = 1
    m['batch_size']         = 256
    m['Lambda2']            = 200
    m['Alpha']              = 0.075
    model_pars = {'model_pars'  : m,
                  'model_class' : "class_VAEMDN"
                 }

    compute_pars = {}
    compute_pars['compute_pars'] = {'epochs': 1, }   ## direct feed


    ###  Tester #########################################################
    Xpred,_ = test_dataset_correlation()
    test_helper(model_pars, data_pars, compute_pars, Xpred)



def test2(n_sample          = 1000):
    from adatasets import test_dataset_classification_fake, pd_train_test_split2
    df, d=     test_dataset_classification_fake(n_sample)
    colnum, colcat, coly = d['colnum'], d['colcat'], d['coly']
    # df, colnum, colcat, coly = test_dataset_classi_fake(nrows= n_sample)
    X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes  = pd_train_test_split2(df, coly)

    #### Matching Big dict  ##################################################
    def post_process_fun(y): return int(y)
    def pre_process_fun(y):  return int(y)

    m = {'model_pars': {
        'model_class':  "model_vaem.py::VAEMDN"
        ,'model_pars' : {
            'original_dim':       len( colcat + colnum),
            'class_num':             2,
            'intermediate_dim':     64,
            'intermediate_dim_2':   16,
            'latent_dim' :           3,
            'Lambda1'    :           1,
            'batch_size' :         256,
            'Lambda2'    :         200,
            'Alpha'      :         0.075
        }
        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        ],
        }
        },

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score'],
                      'compute_pars' : {'epochs': 1 },
                    },

    'data_pars': { 'n_sample' : n_sample,
        'download_pars' : None,
        'cols_input_type' : {
            'colcat' : colcat,
            'colnum' : colnum,
            'coly'  :  coly,
        },
        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum_bin',   'colcat_bin',  ],

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' : colcat,
        }

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }


        ###################################################
        ,'train':   {'Xtrain': X_train,  'ytrain': y_train, 'Xtest':  X_valid,  'ytest':  y_valid}
        ,'eval':    {'X': X_valid,  'y': y_valid}
        ,'predict': {'X': X_valid}

        ,'task_type' : 'train', 'data_type': 'ram'

        }
    }

    ###  Tester #########################################################
    test_helper(m['model_pars'], m['data_pars'], m['compute_pars'])




def test3(n_sample = 1000):
    from adatasets import test_dataset_classification_petfinder, pd_train_test_split2
    df, d=     test_dataset_classification_petfinder(n_sample)
    colnum, colcat, coly = d['colnum'], d['colcat'], d['coly']
    # df, colnum, colcat, coly = test_dataset_classi_fake(nrows= n_sample)
    X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes  = pd_train_test_split2(df, coly)

    #df, colnum, colcat, coly,colyembed = test_dataset_petfinder(nrows= n_sample)
    #X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes  = train_test_split2(df, coly)

    #### Matching Big dict  ##################################################
    def post_process_fun(y): return int(y)
    def pre_process_fun(y):  return int(y)

    m = {'model_pars': {
        'model_class':  "model_vaem.py::Basic_AE"

        ,'model_pars' : {
            'original_dim':       len( colcat + colnum),
            'class_num':             2,
            'intermediate_dim':     64,
            'intermediate_dim_2':   16,
            'latent_dim' :           3,
            'Lambda1'    :           1,
            'batch_size' :         256,
            'Lambda2'    :         200,
            'Alpha'      :         0.075
        }
        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        ],
        }
        },

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score'],
                      'compute_pars' : {'epochs': 50 },

                      'compute_extra' : {'path_encoding': 'ztmp/'}
                    },

    'data_pars': { 'n_sample' : n_sample,
        'download_pars' : None,
        'cols_input_type' : {
            'colcat' : colcat,
            'colnum' : colnum,
            'coly'  :  coly,
        },

        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum_bin',   'colcat_bin',  ],

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' :      colcat,
        }
        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

        ###################################################
        ,'train':   {'Xtrain': X_train,  'ytrain': y_train, 'Xtest':  X_valid,  'ytest':  y_valid}
        ,'eval':    {'X': X_valid,  'y': y_valid}
        ,'predict': {'X': X_valid}

        ,'task_type' : 'train', 'data_type': 'ram'
        }
    }
    
    ###  Tester #########################################################
    test_helper(m['model_pars'], m['data_pars'], m['compute_pars'])
    



def test_helper(model_pars, data_pars, compute_pars):
    global model, session
    init()
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)
    model_class = model_pars['model_class']

    log('Training the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None,model_class=model_class)


    log('Predict data..')
    Xnew = predict(data_pars=data_pars,  compute_pars=compute_pars,model_class=model_class)

    
    if 'VAEMDN' in model_class:
        log('Encode data..')
        encoded = encode(data_pars=data_pars,  compute_pars=compute_pars,model_class='VAEMDN')
        encoded = encoded[:3]
        print('Encoded X (Batch 1): \n', encoded)

        #log(encoded)
        #There are different batches of Dataframe we have to perform on each batches

        log('Deccode data..')
        decoded_array = []
        for num,Xpred in enumerate(encoded):
            log(f'Shape of Decoded array: {Xpred.shape}')
            decoded = decode(Xpred = Xpred,data_pars=data_pars,index=num , compute_pars=compute_pars)
            decoded_array.append(decoded)
        print('Decoded X: \n')
        #log(decoded_array[0])

    else:
        log('Encode data..')
        encoded = encode(data_pars=data_pars,  compute_pars=compute_pars,model_class='BasicAE')
        print('Encoded X (Batch 1): \n', encoded)

        #log(encoded)
        #There are different batches of Dataframe we have to perform on each batches

        log('Deccode data..')
        decoded = decode(Xpred = encoded,data_pars=data_pars, compute_pars=compute_pars)
        print('Decoded X: \n')
        log(decoded)

    log('Saving model..')
    log( model.model.summary() )
    save(path= root + '/model_dir/')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/",model_class=model_class)

    log('Model architecture:')
    log(model.model.summary())



def benchmark(config='', dmin=5, dmax=6):
    from pmlb import fetch_data, classification_dataset_names
    from sdv.evaluation import evaluate

    for classification_dataset in classification_dataset_names[dmin:dmax]:
        X, y = fetch_data(classification_dataset, return_X_y=True)
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
        X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)
        def post_process_fun(y): return int(y)
        def pre_process_fun(y):  return int(y)
        #####
        # y = y.astype('uint8')
        num_classes  = len(np.unique(y))
        print(np.unique(y))
        model_pars = {
            'model_pars' : {
            'original_dim':       X.shape[1],
            'class_num':            num_classes,
            'intermediate_dim':     64,
            'intermediate_dim_2':   16,
            'latent_dim' :           3,
            'Lambda1'    :           1,
            'batch_size' :         256,
            'Lambda2'    :         200,
            'Alpha'      :         0.075
        }
        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################


        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        ],
        }
        }
        vae,vae_enc,vae_dec= VAEMDN(model_pars=model_pars['model_pars'])
        basic_ae,ae_enc,ae_dec = AUTOENCODER_BASIC(X.shape[1])

        vae.fit([X_train_full,y_train_full],epochs=50)
        basic_ae.fit(X_train_full,X_train_full,epochs=50)

        vae_data = vae.predict([X_test,y_test])
        basic_data = basic_ae.predict(X_test)

        print(f'{classification_dataset} Metrics: ------------')
        column = [f'col_{i}' for i in range(X.shape[1])]
        real_df = pd.DataFrame(X_test[:100],columns=column)
        vae_df = pd.DataFrame(vae_data[:100],columns=column)
        basic_df = pd.DataFrame(basic_data[:100],columns=column)
        #print(real_df,vae_df,basic_ae)
        print(evaluate(real_df,vae_df))




if __name__ == "__main__":
    # test()
    import fire
    fire.Fire()
















"""
def test_dataset_classi_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim=11
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(
              n_samples=10000, n_features=ndim, n_classes=1, n_redundant = 0, n_informative=ndim )
    df = pd.DataFrame(X,  columns= colnum)
    for ci in colcat :
      df[ci] = np.random.randint(0,1, len(df))
    df[coly]   = y.reshape(-1, 1)
    # log(df)
    return df, colnum, colcat, coly


def test_dataset_petfinder(nrows=1000):
    from sklearn.preprocessing import LabelEncoder
    # Dense features
    colnum = ['PhotoAmt', 'Fee','Age' ]

    # Sparse features
    colcat = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized',
              'Health', 'Breed1' ]

    colembed = ['Breed1']
    # Target column
    coly        = "y"

    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
    tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')

    log3('Data Frame Loaded')
    df      = pd.read_csv(csv_file)
    df      = df.iloc[:nrows, :]
    df['y'] = np.where(df['AdoptionSpeed']==4, 0, 1)
    df      = df.drop(columns=['AdoptionSpeed', 'Description'])
    df      = df.apply(LabelEncoder().fit_transform)
    log3(df.dtypes)
    return df, colnum, colcat, coly, colembed


def train_test_split2(df, coly):
    # log3(df.dtypes)
    y = df[coly] ### If clonassificati
    X = df.drop(coly,  axis=1)
    log3('y', np.sum(y[y==1]) , X.head(3))
    ######### Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)

    #####
    # y = y.astype('uint8')
    num_classes                                = len(set(y_train_full.values.ravel()))

    return X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes
"""





"""

    I had the same issue here using tf.data.Datasets and, for me, the problem was related with the inputs and outputs.

    I solved by naming each input layer and latter by creating a TF Dataset with the inputs as a dict and the output as a single value. Something like the following:

    # Example code

    x1 = tf.keras.layer.Input(..., name='input_1')
    x2 = tf.keras.layer.Input(..., name='input_2')
    ......
    concat_layer = tf.keras.layers.Concatenate([x1, x2])
    y = tf.keras.layer.Dense(1)(concat_layer)

    model = Model([x1, x2], y)

    dataset = tf.data.Dataset(....) # suppose each sample in dataset is a triple (2-features and 1 label)

    def input_solver(sample):
        return {'input_1': sample[0], 'input_2': sample[1]}, sample[2]

    dataset.map(input_solver) # this will map the first and the second feature in this triple-sample to the inputs.
    model.fit(dataset, epochs=5)

"""






'''from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sb

from pmlb import fetch_data, classification_dataset_names

logit_test_scores = []
gnb_test_scores = []

for classification_dataset in classification_dataset_names[:5]:
    X, y = fetch_data(classification_dataset, return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(X, y)

    logit = SVC()
    gnb = GaussianNB()

    logit.fit(train_X, train_y)
    gnb.fit(train_X, train_y)

    logit_test_scores.append(logit.score(test_X, test_y))
    gnb_test_scores.append(gnb.score(test_X, test_y))

print(logit_test_scores,gnb_test_scores)
sb.boxplot(data=[logit_test_scores, gnb_test_scores], notch=True)
plt.xticks([0, 1], ['LogisticRegression', 'GaussianNB'])
plt.ylabel('Test Accuracy')
plt.show()'''

'''import sdmetrics

# Load the demo data, which includes:
# - A dict containing the real tables as pandas.DataFrames.
# - A dict containing the synthetic clones of the real data.
# - A dict containing metadata about the tables.
real_data, synthetic_data, metadata = sdmetrics.load_demo()

# Obtain the list of multi table metrics, which is returned as a dict
# containing the metric names and the corresponding metric classes.
print(real_data)
'''
'''metrics = sdmetrics.multi_table.MultiTableMetric.get_subclasses()

# Run all the compatible metrics and get a report
print(sdmetrics.compute_metrics(metrics, real_data, synthetic_data, metadata=metadata))'''

def test4():
    from sdv.demo import load_tabular_demo

    from sdv.tabular import GaussianCopula

    real_data = load_tabular_demo('student_placements')

    model = GaussianCopula()

    model.fit(real_data)

    synthetic_data = model.sample()

    from sdv.evaluation import evaluate

    print(evaluate(synthetic_data, real_data))