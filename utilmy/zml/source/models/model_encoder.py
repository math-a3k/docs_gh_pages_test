# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Various Auto_Encoder :
  Clsutering, AE, Multi model Encoding

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
import os, sys, copy
sys.path.append( os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/")
### import util_feature

import pandas as pd, numpy as np,  sklearn
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.decomposition import TruncatedSVD, MiniBatchSparsePCA, FastICA

from umap import UMAP

### MMAE
try:
  from mmae.multimodal_autoencoder import MultimodalAutoencoder
except:
    os.system("pip install mmae[keras]")
####################################################################################################
# CONSTANTS
verbosity = 3

def log(*s):
    print(*s, flush=True)


def log2(*s):
    if verbosity >= 2 :
       print(*s, flush=True)


def log3(*s):
    if verbosity >= 3 :
       print(*s, flush=True)


####################################################################################################




####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model   = Model(*kw, **kwargs)
    session = None


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            model_class = globals()[model_pars['model_class']]
            self.model  = model_class(**model_pars['model_pars'])
            log2(model_class, self.model)



def fit(data_pars: dict=None, compute_pars: dict=None, out_pars: dict=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")

    cpars = copy.deepcopy(compute_pars.get("compute_pars", {}))

    ### Un-supervised
    model.model.fit(Xtrain_tuple, **cpars)



def transform(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    """ Geenrate Xtrain  ----> Xtrain_new  (ie transformed)
    :param Xpred:
              ==> dataframe       if you want to transorm by sklearn models like TruncatedSVD
    :param data_pars:
    :param compute_pars:
    :param out_pars:
    :param kw:
    :return:
    """

    global model, session
    name =  model.model_pars['model_pars']['model_pars']

    #######
    if Xpred is None:
        Xpred_tuple, y = get_dataset(data_pars, task_type="eval")
    else :
        cols_type         = data_pars['cols_model_type2']
        cols_ref_formodel = cols_type
        split             = kw.get("split", False)
        Xpred_tuple       = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel, split)


    cpars = compute_pars.get('compute_pars', {})
    if name in ['MultimodalAutoencoder'] :
       Xnew = model.model.encode( Xpred_tuple, **cpars )

    else :
       Xnew = model.model.transform( Xpred_tuple, **cpars )

    log3("generated data", Xnew)
    return Xnew



encode = transform   ### alias


def decode(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    """
       Embedding --> Actual values
    :param Xpred:
    :param data_pars:
    :param compute_pars:
    :param out_pars:
    :param kw:
    :return:
    """
    pass



def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    """
       Encode + Decode

    :param Xpred:
    :param data_pars:
    :param compute_pars:
    :param out_pars:
    :param kw:
    :return:
    """
    global model, session

    X1 = encode(Xpred, data_pars, compute_pars, out_pars, **kw)
    X2 = decode(X1, data_pars, compute_pars, out_pars, **kw)
    return X2


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model        = model0.model
    model.model_pars   = model0.model_pars
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
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split=False):
    """  Split into Tuples = (df1, df2, df3) to feed model, (ie Keras)
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :param split: 
        True :  split data to list of dataframe 
        False:  return same input of data
    :return:
    """
    if len(cols_ref) <= 1  or not split :
        return Xtrain
        
    Xtuple_train = []
    for cols_groupname in cols_ref :
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_i] )

    return Xtuple_train


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes OR single dataframe
    """
    #### log(data_pars)
    data_type = data_pars.get('type', 'ram')

    ### Sparse columns, Dense Columns
    cols_type_received     = data_pars.get('cols_model_type2', {} )
    cols_ref  = list( cols_type_received.keys())
    split = kw.get('split', False)

    if data_type == "ram":
        # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
        ### dict  colgroup ---> list of colname
        cols_type_received     = data_pars.get('cols_model_type2', {} )

        if task_type == "predict":
            d = data_pars[task_type]
            Xtrain       = d["X"]
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split)
            return Xtuple_train

        if task_type == "eval":
            d = data_pars[task_type]
            Xtrain, ytrain  = d["X"], d["y"]
            Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split)
            return Xtuple_train, ytrain

        if task_type == "train":
            d = data_pars[task_type]
            Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

            ### dict  colgroup ---> list of df
            Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref, split)
            Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref, split)

            log2("Xtuple_train", Xtuple_train)

            return Xtuple_train, ytrain, Xtuple_test, ytest

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')





def test_dataset_classi_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim=11
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(
        n_samples=1000,
        n_features=ndim,
        n_targets=1,
        n_informative=ndim )
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly




def test():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    df, colnum, colcat, coly= test_dataset_classi_fake(nrows=500)
    X = df[colnum + colcat ]
    y = df[coly]

    X['colid'] = np.arange(0, len(X))
    X_train, X_test, y_train, y_test    = train_test_split(X, y)
    X_train, X_valid, y_train, y_valid  = train_test_split(X_train, y_train, random_state=2021, stratify=y_train)


    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return int(y)

    def pre_process_fun(y):
        return int(y)


    m = {'model_pars': {
        ### LightGBM API model   #######################################
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  "torch_tabular.py::CategoryEmbeddingModelConfig"

        # Type of target prediction, evaluation metrics
        ,'model_pars' : {
                        }

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },

        {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },

        #### catcol INTO integer,   colcat into OneHot
        {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },

        ],
            }
        },

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                    },

    'data_pars': { 'n_sample' : n_sample,

        'download_pars' : None,

        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum_bin',
                                'colcat_bin',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly
                            }
        ###################################################
        ,'train': {'Xtrain': X_train,
                    'ytrain': y_train,
                        'Xtest': X_valid,
                        'ytest': y_valid},
                'eval': {'X': X_valid,
                        'y': y_valid},
                'predict': {'X': X_valid}

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },


        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' : colcat,
        },
        }
    }

    ##### Running loop
    ll = [
    ]


    #####################################################################
    log("test SVD")
    m['model_pars']['model_class'] =  'TruncatedSVD',
    m['model_pars']['model_pars']  = {  "n_components": 3,    'n_iter': 2,
                 }
    test_helper(m['model_pars'], m['compute_pars'], m['data_pars'])



    log("test autoencoder")
    m['model_pars']['model_class'] =  'keras_autoencoder_multimodal',
    m['model_pars']['model_pars']  = {  }
    test_helper(m['model_pars'], m['compute_pars'], m['data_pars'])





def test_helper(model_pars, data_pars, compute_pars):
    global model, session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)

    log('Predict data..')
    Xnew = transform(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Xnew', Xnew)

    # log('Evaluating the model..')
    # log(eval(data_pars=data_pars, compute_pars=compute_pars))

    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Load model..')
    model, session = load_model(path= root + "/model_dir/")
    log(model)




if __name__ == "__main__":
    import fire
    fire.Fire()
    
    
    




####################################################################################################
####################################################################################################
def pd_export(df, col, pars):
    """
       Export in train folder for next training
       colsall
    :param df:
    :param col:
    :param pars:
    :return:
    """
    colid, colsX, coly = pars['colid'], pars['colsX'], pars['coly']
    dfX   = df[colsX]
    dfX   = dfX.set_index(colid)
    dfX.to_parquet( pars['path_export'] + "/features.parquet")


    dfy = df[coly]
    dfy = dfy.set_index(colid)
    dfX.to_parquet( pars['path_export'] + "/target.parquet")



def pd_autoencoder(df, col, pars):
    """"
    (4) Autoencoder
    An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner.
    The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction,
    by training the network to ignore noise.
    (i) Feed Forward
    The simplest form of an autoencoder is a feedforward, non-recurrent
    neural network similar to single layer perceptrons that participate in multilayer perceptrons
    """
    from sklearn.preprocessing import minmax_scale
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    def encoder_dataset(df, drop=None, dimesions=20):
        # encode categorical columns
        cat_columns = df.select_dtypes(['category']).columns
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        print(cat_columns)

        # encode objects columns
        from sklearn.preprocessing import OrdinalEncoder

        def encode_objects(X_train):
            oe = OrdinalEncoder()
            oe.fit(X_train)
            X_train_enc = oe.transform(X_train)
            return X_train_enc

        selected_cols = df.select_dtypes(['object']).columns
        df[selected_cols] = encode_objects(df[selected_cols])

        # df = df[[c for c in df.columns if c not in df.select_dtypes(['object']).columns]]
        if drop:
            train_scaled = minmax_scale(df.drop(drop,axis=1).values, axis = 0)
        else:
           train_scaled = minmax_scale(df.values, axis = 0)
        return train_scaled
    # define the number of encoding dimensions
    encoding_dim = pars.get('dimesions', 2)
    # define the number of features
    train_scaled = encoder_dataset(df, pars.get('drop',None), encoding_dim)
    print("train scaled: ", train_scaled)
    ncol = train_scaled.shape[1]
    input_dim = tf.keras.Input(shape = (ncol, ))
    # Encoder Layers
    encoded1      = tf.keras.layers.Dense(3000, activation = 'relu')(input_dim)
    encoded2      = tf.keras.layers.Dense(2750, activation = 'relu')(encoded1)
    encoded3      = tf.keras.layers.Dense(2500, activation = 'relu')(encoded2)
    encoded4      = tf.keras.layers.Dense(750, activation = 'relu')(encoded3)
    encoded5      = tf.keras.layers.Dense(500, activation = 'relu')(encoded4)
    encoded6      = tf.keras.layers.Dense(250, activation = 'relu')(encoded5)
    encoded7      = tf.keras.layers.Dense(encoding_dim, activation = 'relu')(encoded6)
    encoder       = tf.keras.Model(inputs = input_dim, outputs = encoded7)
    encoded_input = tf.keras.Input(shape = (encoding_dim, ))
    encoded_train = pd.DataFrame(encoder.predict(train_scaled),index=df.index)
    encoded_train = encoded_train.add_prefix('encoded_')
    if 'drop' in pars :
        drop = pars['drop']
        encoded_train = pd.concat((df[drop],encoded_train),axis=1)

    return encoded_train
    # df_out = mapper.encoder_dataset(df.copy(), ["Close_1"], 15); df_out.head()





def pd_covariate_shift_adjustment():
    """
    https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766
     Covariate shift has been extensively studied in the literature, and a number of proposals to work under it have been published. Some of the most important ones include:
        Weighting the log-likelihood function (Shimodaira, 2000)
        Importance weighted cross-validation (Sugiyama et al, 2007 JMLR)
        Integrated optimization problem. Discriminative learning. (Bickel et al, 2009 JMRL)
        Kernel mean matching (Gretton et al., 2009)
        Adversarial search (Globerson et al, 2009)
        Frank-Wolfe algorithm (Wen et al., 2015)
    """
    import numpy as np
    from scipy import sparse
    import pylab as plt

    # .. to generate a synthetic dataset ..
    from sklearn import datasets
    n_samples, n_features = 1000, 10000
    A, b = datasets.make_regression(n_samples, n_features)
    def FW(alpha, max_iter=200, tol=1e-8):
        # .. initial estimate, could be any feasible point ..
        x_t = sparse.dok_matrix((n_features, 1))
        trace = []  # to keep track of the gap
        # .. some quantities can be precomputed ..
        Atb = A.T.dot(b)
        for it in range(max_iter):
            # .. compute gradient. Slightly more involved than usual because ..
            # .. of the use of sparse matrices ..
            Ax = x_t.T.dot(A.T).ravel()
            grad = (A.T.dot(Ax) - Atb)
            # .. the LMO results in a vector that is zero everywhere except for ..
            # .. a single index. Of this vector we only store its index and magnitude ..
            idx_oracle = np.argmax(np.abs(grad))
            mag_oracle = alpha * np.sign(-grad[idx_oracle])
            g_t = x_t.T.dot(grad).ravel() - grad[idx_oracle] * mag_oracle
            trace.append(g_t)
            if g_t <= tol:
                break
            q_t = A[:, idx_oracle] * mag_oracle - Ax
            step_size = min(q_t.dot(b - Ax) / q_t.dot(q_t), 1.)
            x_t = (1. - step_size) * x_t
            x_t[idx_oracle] = x_t[idx_oracle] + step_size * mag_oracle
        return x_t, np.array(trace)

    # .. plot evolution of FW gap ..
    sol, trace = FW(.5 * n_features)
    plt.plot(trace)
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('FW gap')
    plt.title('FW on a Lasso problem')
    plt.grid()
    plt.show()
    sparsity = np.mean(sol.toarray().ravel() != 0)
    print('Sparsity of solution: %s%%' % (sparsity * 100))




    
    
"""
def pd_generic_transform(df, col=None, pars={}, model=None)  :
 
     Transform or Samples using  model.fit()   model.sample()  or model.transform()
    params:
            df    : (pandas dataframe) original dataframe
            col   : column name for data enancement
            pars  : (dict - optional) contains:                                          
                path_model_save: saving location if save_model is set to True
                path_model_load: saved model location to skip training
                path_data_new  : new data where saved 
    returns:
            model, df_new, col, pars
   
    path_model_save = pars.get('path_model_save', 'data/output/ztmp/')
    pars_model      = pars.get('pars_model', {} )
    model_method    = pars.get('method', 'transform')
    
    # model fitting 
    if 'path_model_load' in pars:
            model = load(pars['path_model_load'])
    else:
            log('##### Training Started #####')
            model = model( **pars_model)
            model.fit(df)
            log('##### Training Finshed #####')
            try:
                 save(model, path_model_save )
                 log('model saved at: ' + path_model_save  )
            except:
                 log('saving model failed: ', path_model_save)
    log('##### Generating Samples/transform #############')    
    if model_method == 'sample' :
        n_samples =pars.get('n_samples', max(1, 0.10 * len(df) ) )
        new_data  = model.sample(n_samples)
        
    elif model_method == 'transform' :
        new_data = model.transform(df.values)
    else :
        raise Exception("Unknown", model_method)
        
    log_pd( new_data, n=7)    
    if 'path_newdata' in pars :
        new_data.to_parquet( pars['path_newdata'] + '/features.parquet' ) 
        log('###### df transform save on disk', pars['path_newdata'] )    
    
    return model, df_new, col, pars
"""