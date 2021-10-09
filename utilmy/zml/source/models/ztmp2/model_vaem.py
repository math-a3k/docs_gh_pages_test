# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""


https://github.com/microsoft/VAEM/blob/main/Main%20Notebook.ipynb


"""
import os, pandas as pd, numpy as np, sklearn, copy
from sklearn.model_selection import train_test_split
import tensorflow
import numpy as np
import tensorflow as tf
print(tf.__version__)
from scipy.stats import bernoulli
import os
import random
from random import sample
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
plt.switch_backend('agg')
tfd = tf.contrib.distributions
import utils.process as process
import json
import utils.params as params
import seaborn as sns; sns.set(style="ticks", color_codes=True)


####################################################################################################
verbosity =2

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 :
      print(*s, flush=True)


####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


# LOAD MODEL
from repo.vaem import model_main as Modelcustom

"""


cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input']

def WideDeep_dense(n_wide_cross, n_wide,n_deep, n_feat=8, m_EMBEDDING=10, loss='mse', metric = 'mean_squared_error'):
        #### Wide model with the functional API
        col_wide_cross          = layers.Input(shape=(n_wide_cross,))
        col_wide                = layers.Input(shape=(n_wide,))
        merged_layer            = layers.concatenate([col_wide_cross, col_wide])
        merged_layer            = layers.Dense(15, activation='relu')(merged_layer)
        predictions             = layers.Dense(1)(merged_layer)
        wide_model              = keras.Model(inputs=[col_wide_cross, col_wide], outputs=predictions)

        wide_model.compile(loss = 'mse', optimizer='adam', metrics=[ metric ])
        log2(wide_model.summary())

        #### Deep model with the Functional API
        deep_inputs             = layers.Input(shape=(n_deep,))
        embedding               = layers.Embedding(n_feat, m_EMBEDDING, input_length= n_deep)(deep_inputs)
        embedding               = layers.Flatten()(embedding)

        merged_layer            = layers.Dense(15, activation='relu')(embedding)

        embed_out               = layers.Dense(1)(merged_layer)
        deep_model              = keras.Model(inputs=deep_inputs, outputs=embed_out)
        deep_model.compile(loss='mse',   optimizer='adam',  metrics=[ metric ])
        log2(deep_model.summary())


        #### Combine wide and deep into one model
        merged_out = layers.concatenate([wide_model.output, deep_model.output])
        merged_out = layers.Dense(1)(merged_out)
        model      = keras.Model( wide_model.input + [deep_model.input], merged_out)
        model.compile(loss=loss,   optimizer='adam',  metrics=[ metric ])
        log2(model.summary())

        return model
"""


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars
        self.history = None
        if model_pars is None:
            self.model = None
        else:
            log2("data_pars", data_pars)

            model_class = model_pars['model_class']  #

            mdict = model_pars['model_pars']

            self.model  = Modelcustom(**mdict)
            log2(model_class, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle
    pass



def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")
    else :
        cols_type   = data_pars['cols_model_type2']  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)

    log2(Xpred_tuple)
    pass


    return ypred, ypred_proba





def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) < 1 :
        return Xtrain

    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )

    if len(cols_ref) == 1 :
        return Xtuple_train[0]  ### No tuple
    else :
        return Xtuple_train


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    cols_ref  = cols_ref_formodel

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
            Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref)


            log2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest


    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



########################################################################
def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

    model.model.save(f"{path}/model_keras.h5")

    modelx = Model()  # Empty model  Issue with pickle
    modelx.model_pars   = model.model_pars
    modelx.data_pars    = model.data_pars
    modelx.compute_pars = model.compute_pars
    # log('model', modelx.model)
    pickle.dump(modelx, open(f"{path}/model.pkl", mode='wb'))  #

    pickle.dump(info, open(f"{path}/info.pkl", mode='wb'))  #


def load_model(path=""):
    global model, session
    import dill as pickle

    model_keras = keras.models.load_model(path + '/model_keras.h5' )
    model0      = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model = model_keras
    model.model_pars = model0.model_pars
    model.compute_pars = model0.compute_pars
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


####################################################################################################
############ Do not change #########################################################################
def test(config=''):
    """
        Group of columns for the input model
           cols_input_group = [ ]
          for cols in cols_input_group,

    :param config:
    :return:
    """

    X = pd.DataFrame( np.random.rand(100,30), columns= [ 'col_' +str(i) for i in range(30)] )
    y = pd.DataFrame( np.random.binomial(n=1, p=0.5, size=[100]), columns = ['coly'] )
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)

    log(X_train.shape, )
    ##############################################################
    ##### Generate column actual names from
    colnum = [ 'col_0', 'col_11', 'col_8']
    colcat = [ 'col_13', 'col_17', 'col_13', 'col_9']

    cols_input_type_1 = {
        'colnum' : colnum,
        'colcat' : colcat
    }

    ###### Keras has 1 tuple input    ###########################
    colg_input = {
      'cols_cross_input':  ['colnum', 'colcat' ],
      'cols_deep_input':   ['colnum', 'colcat' ],
    }

    cols_model_type2= {}
    for colg, colist in colg_input.items() :
        cols_model_type2[colg] = []
        for colg_i in colist :
          cols_model_type2[colg].extend( cols_input_type_1[colg_i] )


    ##################################################################################
    model_pars = {'model_class': 'WideAndDeep',
                  'model_pars': {},
                }
    
    n_sample = 100
    data_pars = {'n_sample': n_sample,
                  'cols_input_type': cols_input_type_1,

                  'cols_model_group': ['colnum',
                                       'colcat',
                                       # 'colcross_pair'
                                       ],

                  'cols_model_type2' : cols_model_type2


        ### Filter data rows   #######################3############################
        , 'filter_pars': {'ymax': 2, 'ymin': -1}
                  }

    data_pars['train'] ={'Xtrain': X_train,  'ytrain': y_train,
                         'Xtest': X_test,  'ytest': y_test}
    data_pars['eval'] =  {'X': X_test,
                          'y': y_test}
    data_pars['predict'] = {'X': X_test}

    compute_pars = { 'compute_pars' : { 'epochs': 2,
                   } }

    ######## Run ###########################################
    test_helper(model_pars, data_pars, compute_pars)


def test_helper(model_pars, data_pars, compute_pars):
    global model, session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

    log('Evaluating the model..')
    log(eval(data_pars=data_pars, compute_pars=compute_pars))
    #
    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.summary())


#######################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire(test)






def load_dataset()
    seed = 3000
    bank_raw = pd.read_csv("./data/bank/bankmarketing_train.csv")
    print(bank_raw.info())
    label_column="y"
    matrix1 = bank_raw.copy()


    process.encode_catrtogrial_column(matrix1, ["job"])
    process.encode_catrtogrial_column(matrix1, ["marital"])
    process.encode_catrtogrial_column(matrix1, ["education"])
    process.encode_catrtogrial_column(matrix1, ["default"])
    process.encode_catrtogrial_column(matrix1, ["housing"])
    process.encode_catrtogrial_column(matrix1, ["loan"])
    process.encode_catrtogrial_column(matrix1, ["contact"])
    process.encode_catrtogrial_column(matrix1, ["month"])
    process.encode_catrtogrial_column(matrix1, ["day_of_week"])
    process.encode_catrtogrial_column(matrix1, ["poutcome"])
    process.encode_catrtogrial_column(matrix1, ["y"])

    Data = ((matrix1.values).astype(float))[0:,:]


    # the data will be mapped to interval [min_Data,max_Data]. Usually this will be [0,1] but you can also specify other values.
    max_Data = 0.7 
    min_Data = 0.3 
    # list of categorical variables
    list_cat = np.array([0,1,2,3,4,5,6,7])
    # list of numerical variables
    list_flt = np.array([8,9,10,11,12,13,14,15,16,17,18,19,20])
    # among numerical variables, which ones are discrete. This is referred as continuous-discrete variables in Appendix C.1.3 in our paper.
    # Examples include variables that take integer values, for example month, day of week, number of custumors etc. Other examples include numerical variables that are recorded on a discrete grid (for example salary). 
    list_discrete = np.array([8,9])




    # sort the variables in the data matrix, so that categorical variables appears first. The resulting data matrix is Data_sub
    list_discrete_in_flt = (np.in1d(list_flt, list_discrete).nonzero()[0])
    list_discrete_compressed = list_discrete_in_flt + len(list_cat)

    if len(list_flt)>0 and len(list_cat)>0:
        list_var = np.concatenate((list_cat,list_flt))
    elif len(list_flt)>0:
        list_var = list_flt
    else:
        list_var = list_cat
    Data_sub = Data[:,list_var]
    dic_var_type = np.zeros(Data_sub.shape[1])
    dic_var_type[0:len(list_cat)] = 1

    # In this notebook we assume the raw data matrix is fully observed
    Mask = np.ones(Data_sub.shape)
    # Normalize/squash the data matrix
    Data_std = (Data_sub - Data_sub.min(axis=0)) / (Data_sub.max(axis=0) - Data_sub.min(axis=0))
    scaling_factor = (Data_sub.max(axis=0) - Data_sub.min(axis=0))/(max_Data - min_Data)
    Data_sub = Data_std * (max_Data - min_Data) + min_Data

    # decompress categorical data into one hot representation
    Data_cat = Data[:,list_cat].copy()
    Data_flt = Data[:,list_flt].copy()
    Data_compressed = np.concatenate((Data_cat,Data_flt),axis = 1)
    Data_decompressed, Mask_decompressed, cat_dims, DIM_FLT = process.data_preprocess(Data_sub,Mask,dic_var_type)
    Data_train_decompressed, Data_test_decompressed, mask_train_decompressed, mask_test_decompressed,mask_train_compressed, mask_test_compressed,Data_train_compressed, Data_test_compressed = train_test_split(
            Data_decompressed, Mask_decompressed,Mask,Data_compressed,test_size=0.1, random_state=rs)

    list_discrete = list_discrete_in_flt + (cat_dims.sum()).astype(int)

    Data_decompressed = np.concatenate((Data_train_decompressed, Data_test_decompressed), axis=0)
    Data_train_orig = Data_train_decompressed.copy()
    Data_test_orig = Data_test_decompressed.copy()

    # Note that here we have added some noise to continuous-discrete variables to help training. Alternatively, you can also disable this by changing the noise ratio to 0.
    Data_noisy_decompressed,records_d, intervals_d = process.noisy_transform(Data_decompressed, list_discrete, noise_ratio = 0.99)
    noise_record = Data_noisy_decompressed - Data_decompressed
    Data_train_noisy_decompressed = Data_noisy_decompressed[0:Data_train_decompressed.shape[0],:]
    Data_test_noisy_decompressed = Data_noisy_decompressed[Data_train_decompressed.shape[0]:,:]


