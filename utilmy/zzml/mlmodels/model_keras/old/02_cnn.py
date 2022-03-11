# -*- coding: utf-8 -*-
"""
# json_api
https://scikit-learn.org/stable/modules/classes.html
https://stackoverflow.com/questions/38926078/meta-programming-to-parse-json-in-scala
Input : JSON API file
Ouput: Exeution of Script and save model storage
This is generic mapper between JSON and Python code script.
Execution is Asynchornous
"""
import os
import sys

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

# Less Keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
VERBOSE = False

from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri

MODEL_URI = get_model_uri(__file__)

####################################################################################################
# Helper functions
def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(filepath).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def log(*s, n=0, m=1):
    """function log
    Args:
        *s:   
        n:   
        m:   
    Returns:
        
    """
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)



####################################################################################################
class Model(object) :

    def __init__(self, model_pars=None, compute_pars=None, data_pars=None):
        """ Model:__init__
        Args:
            model_pars:     
            compute_pars:     
            data_pars:     
        Returns:
           
        """

        if model_pars is None and compute_pars is None :
            self.model = None
            return

        if not model_pars is None and not compute_pars is None:
            m = model_pars
            c = compute_pars

            rows, cols = data_pars["rows"], data_pars["cols"]

            if K.image_data_format() == 'channels_first':
                input_shape = (1, rows, cols)
            else:
                input_shape = (rows, cols, 1)

            nclasses = m["nclasses"]

            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(nclasses, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(), metrics=[keras.metrics.categorical_accuracy])

        self.model = model



def get_dataset( data_pars, **kw):
    """function get_dataset
    Args:
        data_pars:   
        **kw:   
    Returns:
        
    """
    log("#### Path params   ################################################")
    data_path = os_package_root_path(__file__, sublevel=0, path_add='')
    out_path = os.getcwd() + "/keras_deepAR/"
    os.makedirs(out_path, exist_ok=True)
    log(data_path, out_path)


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    rows, cols = data_pars["rows"], data_pars["cols"]


    # decide on input shape (depends on backend)
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, rows, cols)
        x_test = x_test.reshape(x_test.shape[0], 1, rows, cols)
        input_shape = (1, rows, cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
        x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
        input_shape = (rows, cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test
  

def fit(model, data_pars=None, model_pars=None, compute_pars=None, out_pars=None,session=None, **kwargs):
    """function fit
    Args:
        model:   
        data_pars:   
        model_pars:   
        compute_pars:   
        out_pars:   
        session:   
        **kwargs:   
    Returns:
        
    """
    # def fit(self,batch_size,epochs):
    data_pars['istrain'] = 1
    x_train, y_train, x_test, y_test = get_dataset(data_pars)
    y_train = keras.utils.to_categorical(y_train, model_pars["nclasses"])
    y_test = keras.utils.to_categorical(y_test, model_pars["nclasses"])
    mtmp =model.model.fit(x_train, y_train, batch_size=compute_pars["batch_size"], epochs=compute_pars["epochs"],
                          verbose=1, validation_data=(x_test, y_test))
    # model.model = mtmp
    return model


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    """function predict
    Args:
        model:   
        session:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    x_train, y_train, x_test, y_test = get_dataset(data_pars)

    ypred = model.model.predict(x_test, batch_size=compute_pars["batch_size"], verbose=1)
    return ypred


def metrics(ypred, model, session=None, model_pars=None, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    """function metrics
    Args:
        ypred:   
        model:   
        session:   
        model_pars:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    x_train, y_train, x_test, y_test = get_dataset(data_pars)
    y_test = keras.utils.to_categorical(y_test, model_pars["nclasses"])
    score = model.model.evaluate(x_test, y_test, verbose=0)
    return {  'loss_test:': score[0], 'accuracy_test:': score[1] }
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])


def save(model=None, session=None, save_pars={}):
    """function save
    Args:
        model:   
        session:   
        save_pars:   
    Returns:
        
    """
    from mlmodels.util import save_keras
    print(save_pars)
    save_keras(model, session, save_pars=save_pars)


def load(load_pars={}):
    """function load
    Args:
        load_pars:   
    Returns:
        
    """
    from mlmodels.util import load_keras
    model0 = load_keras(load_pars)

    model = Model()
    model.model = model0.model
    session = None
    return model, session

###################################################################################################
def get_params(choice=0, data_path="dataset/", **kw) :
    """function get_params
    Args:
        choice:   
        data_path:   
        **kw:   
    Returns:
        
    """
    if choice == 0 :
        log("#### Path params   ################################################")
        out_path = path_norm("ztest/model_keras/charcnn/")
        os.makedirs(out_path, exist_ok=True)
        log(out_path)

        train_data_path = data_path + "keras-keras-train.csv"
        test_data_path = data_path + "keras-test.csv"


        data_pars = {"train_data_path": train_data_path, "test_data_path": test_data_path, "train": False,
                     'prediction_length': 48, "rows": 28, "cols": 28,
                     "save_fig": "./series.png"}

        log("#### Model params   ################################################")
        model_pars = { "nclasses": 10, "nlayers": 2, "ncells": 40, "cell_type": 'lstm',
                       "dropout_rate": 0.1, "scaling": True, "nparallel_samples": 100}


        compute_pars = { "batch_size": 32, "clip_gradient": 100, "ctx": None, "epochs": 1, "init": "xavier",
                         "learning_rate": 1e-3,
                         "learning_rate_decay_factor": 0.5, "hybridize": False, "nbatches_per_epoch": 100,
                         'nsamples': 100, "minimum_learning_rate": 5e-05, "patience": 10, "weight_decay": 1e-08}


        outpath = out_path + "result"

        out_pars = {"path": outpath, "plot_prob": True, "quantiles": [0.1, 0.5, 0.9]}

    return model_pars, data_pars, compute_pars, out_pars



########################################################################################################################
def test2(data_path="dataset/", out_path="keras/keras.png", reset=True):
    """function test2
    Args:
        data_path:   
        out_path:   
        reset:   
    Returns:
        
    """
    ###loading the command line arguments
    # arg = load_arguments()

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=0, data_path=data_path)
    model_uri = "model_keras/0_cnn.py"

    log("#### Loading dataset   ############################################")
    kerast_ds = get_dataset(data_pars)


    log("#### Model init, fit   ############################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full(model_uri, model_pars)
    print(module, model)

    model= fit(model, None, data_pars, model_pars, compute_pars)


    log("#### Predict   ###################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)
    print(ypred)


    log("###Get  metrics   ################################################")
    metrics_val = metrics(model, data_pars, compute_pars, out_pars)


    log("#### Save   #######################################################")
    save(model, out_pars["save_path"])
    model2 = load(out_pars["save_path"])



def test(data_path="dataset/"):
    """function test
    Args:
        data_path:   
    Returns:
        
    """
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=0, data_path=data_path)


    log("#### Loading dataset   #############################################")
    kerast_ds = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    model = Model(model_pars, compute_pars, data_pars)
    model=fit(model, model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


    log("#### Predict   ####################################################")
    ypred = predict(model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(ypred)


    log("#### metrics   ####################################################")
    metrics_val = metrics(ypred, model, model_pars=model_pars, data_pars=data_pars,
                          compute_pars=compute_pars, out_pars=out_pars)
    print (metrics_val)


    log("#### Save   #######################################################")
    save_pars = {"path": out_pars['path']}
    save(model, save_pars=save_pars)
    model2 = load(save_pars)


if __name__ == '__main__':
    VERBOSE = True
    test()
