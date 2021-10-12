""""

 python source/models/keras_deepctr.py test


# DeepCTR
https://github.com/shenweichen/DeepCTR
https://deepctr-doc.readthedocs.io/en/latest/Examples.html#classification-criteo


DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models
along with lots of core components layers which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with `model.fit()`and `model.predict()` .


## Models List
|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|                AutoInt                 | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                  NFFM                  | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|                 FGCNN                  | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |
|     Deep Session Interest Network      | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |


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
import warnings
warnings.filterwarnings("ignore")

from jsoncomment import JsonComment ; json = JsonComment()
from pathlib import Path
import importlib
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import deepctr
from deepctr.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat, get_feature_names
from deepctr.models import DeepFM

# from preprocess import _preprocess_criteo, _preprocess_movielens

# Note: keep that to disable eager mode with tf 2.x
import tensorflow as tf
if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()

####################################################################################################
# Helper functions
#from mlmodels.preprocess.tabular_keras  import get_test_data, get_xy_fd_dien, get_xy_fd_din, get_xy_fd_dsin



####################################################################################################
DATA_PARAMS = {
    "AFM"     : {"sparse_feat_num": 3, "dense_feat_num": 0},
    "AutoInt" : {"sparse_feat_num": 1, "dense_feat_num": 1},
    "CCPM"    : {"sparse_feat_num": 3, "dense_feat_num":0},
    "DCN"     : {"sparse_feat_num": 3, "dense_feat_num": 3},
    "DCNMix"  : {"sparse_feat_num": 3, "dense_feat_num": 3},
    "DeepFM"  : {"sparse_feat_num": 1, "dense_feat_num": 1},
    "DIEN"    : {},
    "DIN"     : {},
    "DSIN"    : {},
    "FGCNN"   : {"embedding_size": 8, "sparse_feat_num": 1, "dense_feat_num": 1},
    "FiBiNET" : {"sparse_feat_num": 2, "dense_feat_num": 2},
    "FLEN"    : {"embedding_size": 2, "sparse_feat_num": 6, "dense_feat_num": 6, "use_group": True},
    "FNN"     : {"sparse_feat_num": 1, "dense_feat_num": 1},
    "MLR"     : {"sparse_feat_num": 0, "dense_feat_num": 2, "prefix": "region"},
    "NFM"     : {"sparse_feat_num": 1, "dense_feat_num": 1},
    "ONN"     : {"sparse_feat_num": 2, "dense_feat_num": 2, "sequence_feature":('sum', 'mean', 'max',), "hash_flag":True},
    "PNN"     : {"sparse_feat_num": 1, "dense_feat_num": 1},
    "WDL"     : {"sparse_feat_num": 2, "dense_feat_num": 0},
    "xDeepFM" : {"sparse_feat_num": 1, "dense_feat_num": 1}
}

MODEL_PARAMS = {
    "AFM"     : {"use_attention": True, "afm_dropout": 0.5},
    "AutoInt" : {"att_layer_num": 1, "dnn_hidden_units": (), "dnn_dropout": 0.5},
    "CCPM"    : {"conv_kernel_width": (3, 2), "conv_filters": (2, 1), "dnn_hidden_units": [32,], "dnn_dropout": 0.5},
    "DCN"     : {"cross_num": 0, "dnn_hidden_units": (8,), "dnn_dropout": 0.5},
    "DCNMix"  : {"cross_num": 0, "dnn_hidden_units": (8,), "dnn_dropout": 0.5},
    "DeepFM"  : {"dnn_hidden_units": (2,), "dnn_dropout": 0.5},
    "DIEN"    : {"dnn_hidden_units": [4, 4, 4], "dnn_dropout": 0.5, "gru_type": "GRU"},
    "DIN"     : {"dnn_hidden_units":[4, 4, 4], "dnn_dropout":0.5},
    "DSIN"    : {"sess_max_count":2, "dnn_hidden_units":[4, 4, 4], "dnn_dropout":0.5},
    "FGCNN"   : {"conv_kernel_width":(3,2), "conv_filters":(2, 1), "new_maps":(2, 2), "pooling_width":(2, 2), "dnn_hidden_units": (32, ), "dnn_dropout":0.5},
    "FiBiNET" : {"bilinear_type": "all", "dnn_hidden_units":[4,], "dnn_dropout":0.5},
    "FLEN"    : {"dnn_hidden_units": (3,), "dnn_dropout":0.5},
    "FNN"     : {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "MLR"     : {},
    "NFM"     : {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "ONN"     : {"dnn_hidden_units": [32, 32], "embedding_size":4, "dnn_dropout":0.5},
    "PNN"     : {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5, "use_inner": True, "use_outter": True},
    "WDL"     : {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "xDeepFM" : {"dnn_dropout": 0.5, "dnn_hidden_units": (8,), "cin_layer_size": (), "cin_split_half": True, "cin_activation": 'linear'}
}


class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        if model_pars is None :
          return self

        model_name = model_pars.get("model_name", "DeepFM")
        model_list = list(MODEL_PARAMS.keys())

        if not model_name in model_list :
          raise ValueError('Not existing model', model_name)
          return self

        modeli = getattr(importlib.import_module("deepctr.models"), model_name)

        if model_name == "MLR":
            region_feat_col = model_pars.get('region_feat_col', None)
            base_feat_col   = model_pars.get('base_feat_col', None)
        else:
            linear_feat_col = model_pars.get('linear_feat_col', None)
            dnn_feat_col    = model_pars.get('dnn_feat_col', None)
            behavior_feat_list  = model_pars.get('behavior_feat_list', None)


        task = model_pars.get('task', 'binary')
        # 4.Define Model
        if model_name in ["DIEN", "DIN", "DSIN"]:
            self.model = modeli(dnn_feat_col, behavior_feat_list, task=task, **MODEL_PARAMS[model_name])

        elif model_name == "MLR":
            self.model = modeli(region_feat_col, base_feat_col, task=task, **MODEL_PARAMS[model_name])

        elif model_name == "PNN":
            self.model = modeli(dnn_feat_col, task=task, **MODEL_PARAMS[model_name])

        else:
            self.model = modeli(linear_feat_col, dnn_feat_col, task=task, **MODEL_PARAMS[model_name])

        model_pars = model_pars.get("model_pars", {})
        self.model.compile(**model_pars)
        self.model.summary()



def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    #Xtrain, ytrain, Xval, yval = get_dataset(data_pars, task_type="train")
    
    Xtrain, ytrain, Xval, yval, col_dict = get_dataset(data_pars, task_type="train")


    # if VERBOSE: log(Xtrain.shape, model.model)

    cpars = compute_pars.get("compute_pars", {})
    assert 'epochs' in cpars, 'epoch'

    hist = model.model.fit(Xtrain, ytrain,
                           validation_data=(Xval, yval), **cpars)
    model.history = hist


def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    # data_pars['train'] = True
    Xtest, ytest = get_dataset(data_pars, task_type="eval")
    results      = model.model.evaluate(Xtest, ytest)
    ddict        = [{"metric_val": results, 'metric_name': model.model.metrics_names}]
    return ddict


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session

    Xpred = get_dataset(data_pars, task_type="predict")
    ypred = model.model.predict(Xpred)

    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba


def save(path=None, save_weight=False):
    global model, session
    os.makedirs(path, exist_ok=True)

    if save_weight:  # only saving the weight
        filename = "weight.h5"
        model.model.save_weights(path+filename)
    else:  # save all model params
        filename = "model.h5"
        filepath = path + filename
        keras.models.save_model(model.model, filepath)
    # model.model.save(filepath)


def load_model(path="", load_weight=False):
    global model, session

    if load_weight:
        filename = "weight.h5"
        model = model.model.load_weights(path+filename)
    else:
        filepath = path + 'model.h5'
        model = keras.models.load_model(filepath, deepctr.layers.custom_objects)
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


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values,
                                                        stratify=dfy.values,test_size=0.1)
        return Xtrain, ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]

        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest


####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  :
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]




        if task_type == "train":
            d = data_pars[task_type]

            #### run_preprocessor --->   Model feeding  in run_train.py ? Line 253
            name = data_pars['target_data_type']   #######  WDL,MLR, ...
            cols_family = {}
            cols_family['coldense']  = data_pars[ "cols_model_type" ]['coldense']
            cols_family['colsparse'] = data_pars[ "cols_model_type" ]['colsparse']


            if name == 'MLR':
                X_train,  y_train, region_feat_col, base_feat_col = get_xy_random2(d['Xtrain'], d['ytrain'], cols_family)
                X_test,  y_test, region_feat_col, base_feat_col   = get_xy_random2(d['Xval'], d['yval'], cols_family )
                col_dict = {  'linear_feat_col' : region_feat_col,
                              'dnn_feat_col'  :   base_feat_col 
                           }

            elif name in ['WDL', 'FNN', 'DCN', 'DCNMix', 'FLEN', 'DeepFM', 'xDeepFM', 'AutoInt', 'FNN', 'ONN',
                          'NFM', 'FiBiNET', 'FGCNN', 'AFM', 'CCPM', 'PNN' ]:
                X_train, y_train, linear_feat_col, dnn_feat_col = get_xy_random2(d['Xtrain'], d['ytrain'], cols_family)
                X_test, y_test,   linear_feat_col, dnn_feat_col = get_xy_random2(d['Xval'], d['yval'], cols_family )
                col_dict = {  'linear_feat_col' : linear_feat_col,
                              'dnn_feat_col'    : dnn_feat_col
                           }

            elif name in ['DIN', 'DIEN', 'DSIN']:
                ##### Complicated to after
                if name=="DIN" : x, y, dnn_feat_col, behavior_feat_list = get_xy_fd2()
                if name=="DIEN": x, y, dnn_feat_col, behavior_feat_list = get_xy_fd2(use_neg=True)
                if name=="DSIN": x, y, dnn_feat_col, behavior_feat_list = get_xy_fd2(hash_flag=True, use_session=True)


            return X_train, y_train, X_test, y_test, col_dict










    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


########################################################################################################################


########################################################################################################################
def get_xy_random2(X, y, cols_family={}):
    # X = np.random.rand(100,30)
    # y = np.random.binomial(n=1, p=0.5, size=[100])

    ## PREPROCESSING STEPS
    # change into dataframe
    target = 'y'
    cols      = [str(i) for i in range(X.shape[1])]  # define column pd dataframe, need to be string type
    data      = pd.DataFrame(X, columns=cols)  # need to convert into df, following the step from documentation
    #data['y'] = y

    # define which feature columns sparse or dense type
    # since our data categorize as Dense Features, we define the sparse features as empty list
    #cols_sparse_features = []
    #cols_dense_features  = [str(i) for i in range(X.shape[1])]

    cols_sparse_features = cols_family['colsparse']
    cols_dense_features  = cols_family['coldense']


    # convert feature type into SparseFeat or DenseFeat type, adjusting from DeepCTR library
    sparse_feat_l = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                     for i,feat in enumerate(cols_sparse_features)]
                    
    dense_feat_l       = [DenseFeat(feat, dimension=1) for feat in cols_dense_features]
    feature_col        = sparse_feat_l + dense_feat_l

    linear_feat_col = feature_col  # containing all the features used by linear part of the model
    dnn_feat_col    = feature_col  # containing all the features used by deep part of the model
    feature_names    = get_feature_names(linear_feat_col + dnn_feat_col)


    train_model_input  = {name: data[name] for name in feature_names}
    X_train, y_train   = train_model_input, y.values

    return X_train, y_train, linear_feat_col, dnn_feat_col




def get_xy_random():
    X = np.random.rand(100,30)
    y = np.random.binomial(n=1, p=0.5, size=[100])

    ## PREPROCESSING STEPS
    # change into dataframe
    cols      = [str(i) for i in range(X.shape[1])]  # define column pd dataframe, need to be string type
    data      = pd.DataFrame(X, columns=cols)  # need to convert into df, following the step from documentation
    data['y'] = y

    # define which feature columns sparse or dense type
    # since our data categorize as Dense Features, we define the sparse features as empty list
    cols_sparse_features = []
    cols_dense_features = [str(i) for i in range(X.shape[1])]

    # convert feature type into SparseFeat or DenseFeat type, adjusting from DeepCTR library
    sparse_feat_l = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                     for i,feat in enumerate(cols_sparse_features)]
                    
    dense_feat_l       = [DenseFeat(feat, dimension=1) for feat in cols_dense_features]
    feature_col        = sparse_feat_l + dense_feat_l

    linear_feat_col = feature_col  # containing all the features used by linear part of the model
    dnn_feat_col    = feature_col  # containing all the features used by deep part of the model
    feature_names      = get_feature_names(linear_feat_col + dnn_feat_col)




    train_full, test   = train_test_split(data, random_state=2021, stratify=data['y'])
    train, val         = train_test_split(train_full, random_state=2021, stratify=train_full['y'])


    train_model_input  = {name:train[name] for name in feature_names}
    val_model_input    = {name:val[name] for name in feature_names}
    test_model_input   = {name:test[name] for name in feature_names}
    target             = 'y'
    ## END OF PREPROCESSING STEPS

    X_train, y_train   = train_model_input, train[target].values
    X_val, y_val       = val_model_input, val[target].values
    X_test, y_test     = test_model_input, test[target].values
    return X_train, X_val, X_test, y_train, y_val, y_test, linear_feat_col, dnn_feat_col






def get_xy_fd(use_neg=False, hash_flag=False, use_session=False):
    feature_col = [SparseFeat('user', 3, embedding_dim=10, use_hash=hash_flag),
                       SparseFeat('gender'   , 2     , embedding_dim=4 , use_hash=hash_flag) ,
                       SparseFeat('item_id'  , 3 + 1 , embedding_dim=4 , use_hash=hash_flag) ,
                       SparseFeat('cate_id'  , 2 + 1 , embedding_dim=4 , use_hash=hash_flag) ,
                       DenseFeat('pay_score' , 1)]

    behavior_feat_list = ["item_id", "cate_id"]
    uid                = np.array([0, 1, 2])
    ugender            = np.array([0, 1, 0])
    iid                = np.array([1, 2, 3])  # 0 is mask value
    cate_id            = np.array([1, 2, 2])  # 0 is mask value
    score              = np.array([0.1, 0.2, 0.3])

    if use_session:
        feature_col += [
            VarLenSparseFeat(SparseFeat('sess_0_item_id', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item_id'),
                             maxlen=4), VarLenSparseFeat(
                SparseFeat('sess_0_cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='cate_id'),
                maxlen=4)]

        feature_col += [
            VarLenSparseFeat(SparseFeat('sess_1_item_id', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item_id'),
                             maxlen=4), VarLenSparseFeat(
                SparseFeat('sess_1_cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='cate_id'),
                maxlen=4)]
        sess1_iid     = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [0, 0, 0, 0]])
        sess1_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [0, 0, 0, 0]])

        sess2_iid     = np.array([[1, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        sess2_cate_id = np.array([[1, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        sess_number   = np.array([2, 1, 0])

        feature_dict  = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                        'sess_0_item_id': sess1_iid, 'sess_0_cate_id': sess1_cate_id, 'pay_score': score,
                        'sess_1_item_id': sess2_iid, 'sess_1_cate_id': sess2_cate_id, }
    else:
        feature_col += [
                VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                                 maxlen=4, length_name="seq_length"),
                VarLenSparseFeat(SparseFeat('hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'), maxlen=4,
                                 length_name="seq_length")]
        hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        hist_cate_id = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])

        behavior_length = np.array([3, 3, 2])

        feature_dict = {'user': uid, 'gender': ugender, 'item_id': iid, 'cate_id': cate_id,
                        'hist_item_id': hist_iid, 'hist_cate_id': hist_cate_id,
                        'pay_score': score, "seq_length": behavior_length}

    if use_neg:
        feature_dict['neg_hist_item_id'] = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
        feature_dict['neg_hist_cate_id'] = np.array([[1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]])
        feature_col += [
            VarLenSparseFeat(SparseFeat('neg_hist_item_id', vocabulary_size=3 + 1, embedding_dim=8, embedding_name='item_id'),
                             maxlen=4, length_name="seq_length"),
            VarLenSparseFeat(SparseFeat('neg_hist_cate_id', 2 + 1, embedding_dim=4, embedding_name='cate_id'),
                             maxlen=4, length_name="seq_length")]

    x = {name: feature_dict[name] for name in get_feature_names(feature_col)}
    if use_session: x["sess_length"] = sess_number
    y = np.array([1, 0, 1])
    return x, y, feature_col, behavior_feat_list


def get_xy_dataset(data_sample=None):
    if data_sample == "avazu":
        df         = pd.read_csv('https://raw.githubusercontent.com/shenweichen/DeepCTR/master/examples/avazu_sample.txt')
        df['day']  = df['hour'].apply(lambda x: str(x)[4:6])
        df['hour'] = df['hour'].apply(lambda x: str(x)[6:])

        sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                           'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                           'device_model', 'device_type', 'device_conn_type',  # 'device_ip',
                           'C14',
                           'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]

        df[sparse_features] = df[sparse_features].fillna('-1', )
        target = ['click']

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe      = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

        # 2.count #unique features for each sparse field,and record dense feature field name
        field_info = dict(C14              = 'user',    C15='user', C16='user', C17='user',
                          C18              = 'user',    C19='user', C20='user', C21='user', C1='user',
                          banner_pos       = 'context', site_id='context',
                          site_domain      = 'context', site_category='context',
                          app_id           = 'item',    app_domain='item', app_category='item',
                          device_model     = 'user',    device_type='user',
                          device_conn_type = 'context', hour='context',
                          device_id        = 'user'
                          )

        fixlen_feat_col = [
            SparseFeat(name, vocabulary_size=df[name].nunique(), embedding_dim=16, use_hash=False, dtype='int32',
                       group_name=field_info[name]) for name in sparse_features]

        dnn_feat_col    = fixlen_feat_col
        linear_feat_col = fixlen_feat_col
        feature_names          = get_feature_names(linear_feat_col + dnn_feat_col)

    elif data_sample == "criteo":
        df = pd.read_csv('https://raw.githubusercontent.com/shenweichen/DeepCTR/master/examples/criteo_sample.txt')
        sparse_features       = ['C' + str(i) for i in range(1, 27)]
        dense_features        = ['I' + str(i) for i in range(1, 14)]

        df[sparse_features] = df[sparse_features].fillna('-1', )
        df[dense_features]  = df[dense_features].fillna(0, )
        target                = ['label']

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe        = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_features] = mms.fit_transform(df[dense_features])

        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feat_col = [SparseFeat(feat, vocabulary_size=df[feat].nunique(),embedding_dim=4)
                                for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                for feat in dense_features]

        dnn_feat_col    = fixlen_feat_col
        linear_feat_col = fixlen_feat_col
        feature_names   = get_feature_names(linear_feat_col + dnn_feat_col)

    elif data_sample == "movielens":
        df = pd.read_csv("https://raw.githubusercontent.com/shenweichen/DeepCTR/master/examples/movielens_sample.txt")
        sparse_features = ["movie_id", "user_id",   "gender", "age", "occupation", "zip"]
        target = ['rating']

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])

        # 2.count #unique features for each sparse field
        fixlen_feat_col = [SparseFeat(feat, df[feat].nunique(),embedding_dim=4)  for feat in sparse_features]
        linear_feat_col = fixlen_feat_col
        dnn_feat_col    = fixlen_feat_col
        feature_names   = get_feature_names(linear_feat_col + dnn_feat_col)

    # 3.generate input data for model
    train_full, test  = train_test_split(df, random_state=2021, stratify=df[target])
    train, val        = train_test_split(train_full, random_state=2021, stratify=train_full[target])

    train_model_input = {name:train[name] for name in feature_names}
    val_model_input   = {name:val[name]   for name in feature_names}
    test_model_input  = {name:test[name]  for name in feature_names}

    X_train, y_train  = train_model_input, train[target].values
    X_val, y_val      = val_model_input,   val[target].values
    X_test, y_test    = test_model_input,  test[target].values
    return X_train, X_val, X_test, y_train, y_val, y_test, linear_feat_col, dnn_feat_col


def test(config=''):
    global model, session

    # model list succeed on running
    model_l = ['WDL', 'FNN', 'MLR', 'DCN', 'DCNMix', 'DIEN', 'DIN', 'DSIN', 'FLEN', 'DeepFM', 'xDeepFM', 'AutoInt', 
               'FNN', 'ONN', 'NFM', 'AFM', 'FiBiNET', 'PNN', 'FGCNN']

    # iterate to test each model on the list model
    for name in model_l:

        # get dataset for testing
        linear_feat_col, dnn_feat_col  = None, None
        behavior_feat_list             = None
        region_feat_col, base_feat_col = None, None  # only for MLR model


        # setting up model_pars for dataset task
        task = 'binary'
        loss = 'binary_crossentropy'
        metrics = ['binary_crossentropy']


        if name in ['WDL', 'FNN', 'DCN', 'DCNMix', 'MLR']:
            if name=='MLR':
                X_train, X_val, X_test, y_train, y_val, y_test, region_feat_col, base_feat_col = get_xy_random()
            else:
                X_train, X_val, X_test, y_train, y_val, y_test, linear_feat_col, dnn_feat_col = get_xy_random()
            # setting up model_pars for dataset task
            metrics = ['accuracy']

        elif name in ['DIN', 'DIEN', 'DSIN']:
            if name=="DIN" : x, y, dnn_feat_col, behavior_feat_list = get_xy_fd()
            if name=="DIEN": x, y, dnn_feat_col, behavior_feat_list = get_xy_fd(use_neg=True)
            if name=="DSIN": x, y, dnn_feat_col, behavior_feat_list = get_xy_fd(hash_flag=True, use_session=True)
            # since the example data very small, we don't split the data
            X_train, X_val, X_test, y_train, y_val, y_test = x, x, x, y, y, y

        elif name == 'FLEN':
            # classification dataset
            X_train, X_val, X_test, y_train, y_val, y_test, linear_feat_col, dnn_feat_col = get_xy_dataset("avazu")

        elif name in ['DeepFM', 'xDeepFM', 'AutoInt', 'FNN', 'ONN', 'NFM', 'FiBiNET', 'FGCNN']:
            # classification dataset
            X_train, X_val, X_test, y_train, y_val, y_test, linear_feat_col, dnn_feat_col = get_xy_dataset("criteo")

        elif name in ['AFM', 'CCPM', 'PNN']:
            # regression dataset
            X_train, X_val, X_test, y_train, y_val, y_test, linear_feat_col, dnn_feat_col = get_xy_dataset("movielens")
            # setting up model_pars for dataset task
            task = 'regression'
            loss = 'mse'
            metrics = ['mae']

        # initalize model_pars
        opt = keras.optimizers.Adam()

        # initialize compute_pars
        early_stopping = EarlyStopping(monitor='loss', patience=1)
        # Note: ModelCheckpoint error when used
        # model_ckpt = ModelCheckpoint(filepath='', save_best_only=True, monitor='loss')
        callbacks = [early_stopping]

        m = {
          'model_pars' : {'model_name': name,
                      'linear_feat_col'    : linear_feat_col,
                      'dnn_feat_col'       : dnn_feat_col,
                      'behavior_feat_list' : behavior_feat_list,
                      'region_feat_col'    : region_feat_col,
                      'base_feat_col'      : base_feat_col,
                      'task'                  : task,
                      'model_pars': {'optimizer': opt,
                                     'loss': loss,
                                     'metrics': metrics}
                     },

        'data_pars' : {'train': {'Xtrain': X_train,
                               'ytrain' : y_train,
                               'Xval'   : X_val,
                               'yval'   : y_val},
                     'eval': {'X': X_test,
                              'y': y_test},
                     'predict': {'X': X_test},
                    },

        # compute_pars = {}
        'compute_pars' : {'compute_pars': {'epochs': 1,
                        'callbacks': callbacks} }
        }

        test_helper(name, m['model_pars'], m['data_pars'], m['compute_pars'])
        # log('Model architecture:')
        # log(model.summary())



def test_helper(model_name, model_pars, data_pars, compute_pars):
    global model, session
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log(f'===> Running process for model {model_name}')
    log('> Training the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)
    log('Training completed!')

    log('> Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')
    log('Data successfully predicted!')
    #

    log('> Evaluating the model..')
    log(eval(data_pars=data_pars, compute_pars=compute_pars))
    log('Evaluating completed!')
    #

    log('> Saving model..')
    if model_name == 'FGCNN':
        save(path='model_dir/', save_weight=True)
    else:
        save(path='model_dir/')
    log('Model successfully saved!')

    log('> Load model..')
    if model_name == 'FGCNN':
        model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)
        model, session = load_model(path="model_dir/", load_weight=True)
    else:
        model, session = load_model(path="model_dir/")
    log('Model successfully loaded!')
    log(f'===> Running process for model {model_name} completed!\n\n')








if __name__ == '__main__':
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    import fire
    fire.Fire()








########################################################################################################################
########################################################################################################################

