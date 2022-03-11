""""

Most difficult part is pre-processing.

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


Names"

model_list = ["AFM",
"AUTOINT",
"CCPM",
"DCN",
"DeepFM",
"DIEN",
"DIN",
"DSIN",
"FGCNN",
"FIBINET",
"FLEN",
"FNN",
"MLR",
"NFM",
"ONN",
"PNN",
"WDL",
"XDEEPFM", ]


"""
from jsoncomment import JsonComment ; json = JsonComment()
import os
from pathlib import Path
import importlib


import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                            get_feature_names)
from deepctr.models import DeepFM

# from preprocess import _preprocess_criteo, _preprocess_movielens

# Note: keep that to disable eager mode with tf 2.x
import tensorflow as tf
if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()


####################################################################################################
# Helper functions
from mlmodels.util import os_package_root_path, log, path_norm
from mlmodels.util import save_keras, load_keras
from mlmodels.preprocess.tabular_keras  import get_test_data, get_xy_fd_dien, get_xy_fd_din, get_xy_fd_dsin



####################################################################################################
DATA_PARAMS = {
    "AFM": {"sparse_feature_num": 3, "dense_feature_num": 0},
    "AutoInt":{"sparse_feature_num": 1, "dense_feature_num": 1},
    "CCPM": {"sparse_feature_num": 3, "dense_feature_num":0},
    "DCN": {"sparse_feature_num": 3, "dense_feature_num": 3},
    "DeepFM": {"sparse_feature_num": 1, "dense_feature_num": 1},
    "DIEN": {},
    "DIN": {},
    "DSIN": {},
    "FGCNN": {"embedding_size": 8, "sparse_feature_num": 1, "dense_feature_num": 1},
    "FiBiNET": {"sparse_feature_num": 2, "dense_feature_num": 2},
    "FLEN": {"embedding_size": 2, "sparse_feature_num": 6, "dense_feature_num": 6, "use_group": True},
    "FNN": {"sparse_feature_num": 1, "dense_feature_num": 1},
    "MLR": {"sparse_feature_num": 0, "dense_feature_num": 2, "prefix": "region"},
    "NFM": {"sparse_feature_num": 1, "dense_feature_num": 1},
    "ONN": {"sparse_feature_num": 2, "dense_feature_num": 2, "sequence_feature":('sum', 'mean', 'max',), "hash_flag":True},
    "PNN": {"sparse_feature_num": 1, "dense_feature_num": 1},
    "WDL": {"sparse_feature_num": 2, "dense_feature_num": 0},
    "xDeepFM": {"sparse_feature_num": 1, "dense_feature_num": 1}
}

MODEL_PARAMS = {
    "AFM": {"use_attention": True, "afm_dropout": 0.5},
    "AutoInt":{"att_layer_num": 1, "dnn_hidden_units": (), "dnn_dropout": 0.5},
    "CCPM": {"conv_kernel_width": (3, 2), "conv_filters": (2, 1), "dnn_hidden_units": [32,], "dnn_dropout": 0.5},
    "DCN": {"cross_num": 0, "dnn_hidden_units": (8,), "dnn_dropout": 0.5},
    "DeepFM": {"dnn_hidden_units": (2,), "dnn_dropout": 0.5},
    "DIEN": {"dnn_hidden_units": [4, 4, 4], "dnn_dropout": 0.5, "gru_type": "GRU"},
    "DIN": {"dnn_hidden_units":[4, 4, 4], "dnn_dropout":0.5},
    "DSIN": {"sess_max_count":2, "dnn_hidden_units":[4, 4, 4], "dnn_dropout":0.5},
    "FGCNN": {"conv_kernel_width":(3,2), "conv_filters":(2, 1), "new_maps":(2, 2), "pooling_width":(2, 2), "dnn_hidden_units": (32, ), "dnn_dropout":0.5},
    "FiBiNET":{"bilinear_type": "all", "dnn_hidden_units":[4,], "dnn_dropout":0.5},
    "FLEN": {"dnn_hidden_units": (3,), "dnn_dropout":0.5},
    "FNN": {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "MLR": {},
    "NFM": {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "ONN": {"dnn_hidden_units": [32, 32], "embedding_size":4, "dnn_dropout":0.5},
    "PNN": {"embedding_size":4, "dnn_hidden_units":[4, 4], "dnn_dropout":0.5, "use_inner": True, "use_outter": True},
    "WDL": {"dnn_hidden_units":[32, 32], "dnn_dropout":0.5},
    "xDeepFM": {"dnn_dropout": 0.5, "dnn_hidden_units": (8,), "cin_layer_size": (), "cin_split_half": True, "cin_activation": 'linear'}
}

class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        """ Model:__init__
        Args:
            model_pars:     
            data_pars:     
            compute_pars:     
            **kwargs:     
        Returns:
           
        """
        if model_pars is None :
          return self
       
        model_name = model_pars.get("model_name", "DeepFM")   
        model_list = list(MODEL_PARAMS.keys())
        
        if not model_name in model_list :
          raise ValueError('Not existing model', model_name)
          return self

        modeli = getattr(importlib.import_module("deepctr.models"), model_name)
        # 4.Define Model
        x, y, feature_columns, behavior_feature_list = kwargs["dataset"]
        if model_name in ["DIEN", "DIN", "DSIN"]:
            self.model = modeli(feature_columns, behavior_feature_list, **MODEL_PARAMS[model_name])
        elif model_name == "MLR":
            self.model = modeli(feature_columns)
        elif model_name == "PNN":
            self.model = modeli(feature_columns, **MODEL_PARAMS[model_name])
        else:
            self.model = modeli(feature_columns, feature_columns, **MODEL_PARAMS[model_name])

        self.model.compile(model_pars['optimization'], model_pars['cost'],
                           metrics=compute_pars.get("metrics", ['binary_crossentropy']), )
        self.model.summary()


        
##################################################################################################
def _preprocess_criteo(df, **kw):
    """function _preprocess_criteo
    Args:
        df:   
        **kw:   
    Returns:
        
    """
    hash_feature = kw.get('hash_feature')
    sparse_col = ['C' + str(i) for i in range(1, 27)]
    dense_col = ['I' + str(i) for i in range(1, 14)]
    df[sparse_col] = df[sparse_col].fillna('-1', )
    df[dense_col] = df[dense_col].fillna(0, )
    target = ["label"]

    # set hashing space for each sparse field,and record dense feature field name
    if hash_feature:
        # Transformation for dense features
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_col] = mms.fit_transform(df[dense_col])
        sparse_col = ['C' + str(i) for i in range(1, 27)]
        dense_col = ['I' + str(i) for i in range(1, 14)]

        fixlen_cols = [SparseFeat(feat, vocabulary_size=1000, embedding_dim=4, use_hash=True, dtype='string')
                       # since the input is string
                       for feat in sparse_col] + [DenseFeat(feat, 1, ) for feat in dense_col]

    else:
        for feat in sparse_col:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_col] = mms.fit_transform(df[dense_col])
        fixlen_cols = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                       for i, feat in enumerate(sparse_col)] + [DenseFeat(feat, 1, ) for feat in dense_col]

    linear_cols = fixlen_cols
    dnn_cols = fixlen_cols
    train, test = train_test_split(df, test_size=kw['test_size'])

    return df, linear_cols, dnn_cols, train, test, target, test[target].values


def _preprocess_movielens(df, **kw):
    """function _preprocess_movielens
    Args:
        df:   
        **kw:   
    Returns:
        
    """
    multiple_value = kw.get('multiple_value')
    sparse_col = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_col:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    if not multiple_value:
        # 2.count #unique features for each sparse field
        fixlen_cols = [SparseFeat(feat, df[feat].nunique(), embedding_dim=4) for feat in sparse_col]
        linear_cols = fixlen_cols
        dnn_cols = fixlen_cols
        train, test = train_test_split(df, test_size=0.2)
        ytrue = test[target].values
    else:
        ytrue = df[target].values
        hash_feature = kw.get('hash_feature', False)
        if not hash_feature:
            def split(x):
                key_ans = x.split('|')
                for key in key_ans:
                    if key not in key2index:
                        # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                        key2index[key] = len(key2index) + 1
                return list(map(lambda x: key2index[x], key_ans))

            # preprocess the sequence feature
            key2index = {}
            genres_list = list(map(split, df['genres'].values))
            genres_length = np.array(list(map(len, genres_list)))
            max_len = max(genres_length)
            # Notice : padding=`post`
            genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
            fixlen_cols = [SparseFeat(feat, df[feat].nunique(), embedding_dim=4) for feat in sparse_col]

            use_weighted_sequence = False
            if use_weighted_sequence:
                varlen_cols = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
                    key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                weight_name='genres_weight')]  # Notice : value 0 is for padding for sequence input feature
            else:
                varlen_cols = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
                    key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

            linear_cols = fixlen_cols + varlen_cols
            dnn_cols = fixlen_cols + varlen_cols

            # generate input data for model
            model_input = {name: df[name] for name in sparse_col}  #
            model_input["genres"] = genres_list
            model_input["genres_weight"] = np.random.randn(df.shape[0], max_len, 1)


        else:
            df[sparse_col] = df[sparse_col].astype(str)

            # 1.Use hashing encoding on the fly for sparse features,and process sequence features
            genres_list = list(map(lambda x: x.split('|'), df['genres'].values))
            genres_length = np.array(list(map(len, genres_list)))
            max_len = max(genres_length)

            # Notice : padding=`post`
            genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str, value=0)

            # 2.set hashing space for each sparse field and generate feature config for sequence feature
            fixlen_cols = [
                SparseFeat(feat, df[feat].nunique() * 5, embedding_dim=4, use_hash=True, dtype='string')
                for feat in sparse_col]
            varlen_cols = [
                VarLenSparseFeat(
                    SparseFeat('genres', vocabulary_size=100, embedding_dim=4, use_hash=True, dtype="string"),
                    maxlen=max_len, combiner='mean',
                )]  # Notice : value 0 is for padding for sequence input feature

            linear_cols = fixlen_cols + varlen_cols
            dnn_cols = fixlen_cols + varlen_cols
            feature_names = get_feature_names(linear_cols + dnn_cols)

            # 3.generate input data for model
            model_input = {name: df[name] for name in feature_names}
            model_input['genres'] = genres_list

        train, test = model_input, model_input

    return df, linear_cols, dnn_cols, train, test, target, ytrue


def get_dataset(data_pars=None, **kw):
    """function get_dataset
    Args:
        data_pars:   
        **kw:   
    Returns:
        
    """
    ##check whether dataset is of kind train or test
    data_path = data_pars.get("train_data_path", "")
    data_type = data_pars['dataset_type']
    test_size = data_pars['test_size']


    #### To test all models
    if data_type == "synthesis":
        if data_pars["dataset_name"] == "DIEN":
            x, y, feature_columns, behavior_feature_list = get_xy_fd_dien(hash_flag=True)
        elif data_pars["dataset_name"] == "DIN":
            x, y, feature_columns, behavior_feature_list = get_xy_fd_din(hash_flag=True)
        elif data_pars["dataset_name"] == "DSIN":
            x, y, feature_columns, behavior_feature_list = get_xy_fd_dsin(hash_flag=True)
        else:
            x, y, feature_columns = get_test_data(**DATA_PARAMS[data_pars["dataset_name"]])
            behavior_feature_list = None

        return x, y, feature_columns, behavior_feature_list
  
    #### read from csv file
    if data_pars.get("uri_type") == "pickle":
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)

    if data_type == "criteo":
        df, linear_cols, dnn_cols, train, test, target, ytrue = _preprocess_criteo(df, **data_pars)
    
    elif data_type == "movie_len":
        df, linear_cols, dnn_cols, train, test, target, ytrue = _preprocess_movielens(df, **data_pars)

    else:  ## Already define
        linear_cols = data_pars['linear_cols']
        dnn_cols    = data_pars['dnn_cols']
        train, test = train_test_split(df, test_size=data_pars['test_size'])
        target      = data_pars['target_col']
        ytrue       = data_pars['target_col']

    return df, linear_cols, dnn_cols, train, test, target, ytrue



def fit(model, session=None, compute_pars=None, data_pars=None, out_pars=None,
        **kwargs):
    ##loading dataset
    """
          Classe Model --> model,   model.model contains thte sub-model
    """
    x, y, feature_columns, behavior_feature_list = kwargs["dataset"]

    model.model.fit(x, y,
                    batch_size=compute_pars["batch_size"],
                    epochs=compute_pars["epochs"],
                    validation_split=compute_pars["validation_split"])

    return model


# Model p redict
def predict(model, session=None, compute_pars=None, data_pars=None, out_pars=None, **kwargs):
    """function predict
    Args:
        model:   
        session:   
        compute_pars:   
        data_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    x, y, feature_columns, behavior_feature_list = kwargs["dataset"]
    pred_ans = model.model.predict(x, batch_size=compute_pars['batch_size'])

    return pred_ans


def metrics(ypred, ytrue=None, session=None, compute_pars=None, data_pars=None, out_pars=None, **kwargs):
    """function metrics
    Args:
        ypred:   
        ytrue:   
        session:   
        compute_pars:   
        data_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    metrics_dict = {"MSE": mean_squared_error(ytrue, ypred)}
    return metrics_dict


def reset_model():
    """function reset_model
    Args:
    Returns:
        
    """
    pass


########################################################################################################################


########################################################################################################################
def path_setup(out_folder="", sublevel=0, data_path="dataset/"):
    """function path_setup
    Args:
        out_folder:   
        sublevel:   
        data_path:   
    Returns:
        
    """
    #### Relative path
    data_path = os_package_root_path(__file__, sublevel=sublevel, path_add=data_path)
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)
    log(data_path, out_path)
    return data_path, out_path


def _config_process(config):
    """function _config_process
    Args:
        config:   
    Returns:
        
    """
    data_pars = config["data_pars"]
    model_pars = config["model_pars"]
    compute_pars = config["compute_pars"]
    out_pars = config["out_pars"]
    return model_pars, data_pars, compute_pars, out_pars


def config_load(data_path, file_default, config_mode):
    """function config_load
    Args:
        data_path:   
        file_default:   
        config_mode:   
    Returns:
        
    """
    data_path = Path(os.path.realpath(
        __file__)).parent.parent / file_default if data_path == "dataset/" else data_path

    config = json.load(open(data_path, encoding='utf-8'))
    config = config[config_mode]

    model_pars, data_pars, compute_pars, out_pars = _config_process(config)
    return model_pars, data_pars, compute_pars, out_pars


def get_params(choice="", data_path="dataset/", config_mode="test", **kwargs):
    """function get_params
    Args:
        choice:   
        data_path:   
        config_mode:   
        **kwargs:   
    Returns:
        
    """
    if choice == "json":
        model_pars, data_pars, compute_pars, out_pars = config_load(data_path,
                                                                    file_default="model_keras/01_deepctr.json",
                                                                    config_mode=config_mode)
        return model_pars, data_pars, compute_pars, out_pars

    if choice == 0:
        log("#### Path params   ###################################################")
        data_path, _ = path_setup(out_folder="/deepctr_test/", data_path=data_path)
        out_path = path_norm("ztest/model_keras/deepctr/model.h5")

        train_data_path = data_path + "recommender/criteo_sample.txt"
        data_pars = {"train_data_path": train_data_path, "dataset_type": "criteo", "test_size": 0.2}

        log("#### Model params   #################################################")
        model_pars = {"task": "binary", "model_name": "DeepFM", "optimization": "adam", "cost": "binary_crossentropy"}
        compute_pars = {"batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"path": out_path}


    elif choice == 1:
        log("#### Path params   ##################################################")
        data_path, _ = path_setup(out_folder="/deepctr_test/", data_path=data_path)
        out_path = path_norm("ztest/model_keras/deepctr/model.h5")

        train_data_path = data_path + "recommender/criteo_sample.txt"
        data_pars = {"train_data_path": train_data_path, "hash_feature": True,
                     "dataset_type": "criteo", "test_size": 0.2}

        log("#### Model params   #################################################")
        model_pars = {"task": "binary", "model_name": "DeepFM", "optimization": "adam", "cost": "binary_crossentropy"}
        compute_pars = {"batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"path": out_path}


    elif choice == 2:
        log("#### Path params   ################################################")
        data_path, _ = path_setup(out_folder="/ here_test/", data_path=data_path)
        out_path = path_norm("ztest/model_keras/deepctr/model.h5")

        train_data_path = data_path + "/recommender/movielens_sample.txt"
        data_pars = {"train_data_path": train_data_path, "dataset_type": "movie_len",
                     "test_size": 0.2}

        log("#### Model params   ################################################")
        model_pars = {"task": "regression", "model_name": "DeepFM", "optimization": "adam", "cost": "mse"}
        compute_pars = {"batch_size": 256, "epochs": 10,
                        "validation_split": 0.2}
        out_pars = {"path": out_path}


    elif choice == 3:
        log("#### Path params   ##################################################")
        data_path, _ = path_setup(out_folder="/deepctr_test/", data_path=data_path)
        out_path = path_norm("ztest/model_keras/deepctr/model.h5")

        train_data_path = data_path + "/recommender/movielens_sample.txt"
        data_pars = {"train_data_path": train_data_path, "multiple_value": True,
                     "dataset_type": "movie_len", "test_size": 0.2}

        log("#### Model params   ################################################")
        model_pars = {"task": "regression", "model_name": "DeepFM", "optimization": "adam", "cost": "mse"}
        compute_pars = {"batch_size": 256, "epochs": 10,
                        "validation_split": 0.2}
        out_pars = {"path": out_path}

    elif choice == 4:
        log("#### Path params   #################################################")
        data_path, _ = path_setup(out_folder="/deepctr_test/", data_path=data_path)
        out_path = path_norm("ztest/model_keras/deepctr/model.h5")

        train_data_path = data_path + "/recommender/movielens_sample.txt"
        data_pars = {"train_data_path": train_data_path, "multiple_value": True,
                     "hash_feature": True, "dataset_type": "movie_len", "test_size": 0.2}

        log("#### Model params   ################################################")
        model_pars = {"task": "regression", "model_name": "DeepFM", "optimization": "adam", "cost": "mse"}
        compute_pars = {"batch_size": 256, "epochs": 10,
                        "validation_split": 0.2}
        out_pars = {"path": out_path}

    elif choice == 5:
        model_name = kwargs["model_name"]

        log("#### Path params   #################################################")
        model_name = kwargs["model_name"]
        out_path = path_norm(f"ztest/model_keras/deepctr/model_{model_name}.h5")

        data_pars = {"dataset_type": "synthesis", "sample_size": 8, "test_size": 0.2, "dataset_name": model_name, **DATA_PARAMS[model_name]}

        log("#### Model params   ################################################")
        model_pars = {"model_name": model_name, "optimization": "adam", "cost": "mse"}
        compute_pars = {"batch_size": 100, "epochs": 1,
                        "validation_split": 0.5}
        out_pars = {"path": out_path}

    return model_pars, data_pars, compute_pars, out_pars


########################################################################################################################
########################################################################################################################
def test(data_path="dataset/", pars_choice=0, **kwargs):
    """function test
    Args:
        data_path:   
        pars_choice:   
        **kwargs:   
    Returns:
        
    """
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice,
                                                               data_path=data_path, **kwargs)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    dataset = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    from mlmodels.models import module_load_full, fit, predict
    module, model = module_load_full("model_keras.01_deepctr", model_pars, data_pars, compute_pars, dataset=dataset)
    model = fit(module, model, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars, dataset=dataset)

    # log("#### Predict   ####################################################")
    ypred = predict(module, model, compute_pars=compute_pars, data_pars=data_pars, out_pars=out_pars, dataset=dataset)

    log("#### metrics   ####################################################")
    metrics_val = metrics(ypred, dataset[1], compute_pars=compute_pars, data_pars=data_pars, out_pars=out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")

    log("#### Save/Load   ##################################################")
    save_keras(model, save_pars=out_pars)
    from deepctr.layers import custom_objects
    model2 = load_keras(out_pars, custom_pars={"custom_objects": custom_objects})
    model2.model.summary()


if __name__ == '__main__':
    VERBOSE = True
    for model_name in MODEL_PARAMS.keys():
        if model_name == "FGCNN": # TODO: check save io
            continue
        test(pars_choice=5, **{"model_name": model_name})

    # test(pars_choice=1)
    # test(pars_choice=2)
    # test(pars_choice=3)
    # test(pars_choice=4)
