# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
python torch_ease.py test
"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn
from sklearn.utils.validation import _num_samples
import wget, zipfile
import scipy.sparse as scipy_sparse
####################################################################################################
try   : verbosity = int(json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../../config.json", mode='r'))['verbosity'])
except Exception as e : verbosity = 2
#raise Exception(f"{e}")

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    if verbosity >= 3 : print(*s, flush=True)

def os_makedirs(dir_or_file):
    if os.path.isfile(dir_or_file) :os.makedirs(os.path.dirname(os.path.abspath(dir_or_file)), exist_ok=True)
    else : os.makedirs(os.path.abspath(dir_or_file), exist_ok=True)

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
from pathlib import Path
from collections import namedtuple

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
##### Add custom repo to Python Path ################################################################
thisfile_dirpath = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
import_path      = thisfile_dirpath + "/repo/TorchEASE/"
sys.path.append(import_path)

##### Import from src/core_models/
from main.EASE import TorchEASE

##### pkg
path_pkg =  thisfile_dirpath + "/repo/TorchEASE/"


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):

        if model_pars is None:
            self.model = None
            return

        # n_samples = data_pars["n_sample"]
        # train_df = data_pars["df"].iloc[:n_samples]
        train_df = get_dataset(data_pars=data_pars, task_type="train")
        # self.args = model_pars2
        self.model = TorchEASE(
            train_df, user_col="user_id", item_col="book_id", score_col="rating", reg=250.0
        )
        log2(self.model)






def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    model.model.fit()



def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        Xpred = get_dataset(data_pars=data_pars, task_type="predict")

    ypred = model.model.predict_all(Xpred)

    return ypred


def save(path=None, info=None):
    """ Custom saving
    """
    global model, session
    import cloudpickle as pickle
    os.makedirs(path + "/model/", exist_ok=True)

    #### Torch part
    model.model.save_model(path + "/model/torch_ease_checkpoint")

    #### Wrapper
    model.model = None   ## prevent issues
    pickle.dump(model,  open(path + "/model/model.pkl", mode='wb')) # , protocol=pickle.HIGHEST_PROTOCOL )
    pickle.dump(info, open(path   + "/model/info.pkl", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )



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
############ Test  #################################################################################

def test_dataset_goodbooks(nrows=1000):
    from sklearn.preprocessing import LabelEncoder
    data_path = "./goodbooks_dataset"
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)

        wget.download(
            "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip",
            out=data_path
        )

        with zipfile.ZipFile(f"{data_path}/goodbooks-10k.zip") as zip_ref:
            zip_ref.extractall(data_path)
    df = pd.read_csv(data_path + "/ratings.csv")
    # Dense features
    coly = ['rating',  ]

    # Sparse features
    colcat = ['user_id', 'book_id' ]
    colnum = []
    return df, colnum, colcat, coly


def train_test_split2(df, coly):
    log3(df.dtypes)
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



def get_dataset(data_pars=None, task_type="train", **kwargs):

    if task_type == "train":
        n_samples = data_pars["n_sample"]
        return pd.DataFrame(
            data=np.hstack(
                (data_pars["data_flow"][task_type]["Xtrain"],
                data_pars["data_flow"][task_type]["ytrain"])
            ),
            columns=["user_id", "book_id", "rating"]
        ).iloc[:n_samples]

        # return data_pars["df"].iloc[:n_samples]

    elif task_type == "predict":
        df_pred =  data_pars["data_flow"][task_type]["X"]
        return df_pred

    else:
        raise Exception(f"Task [{task_type}] not supported, TorchEASE supports 'train', 'predict' tasks")

def test(n_sample          = 1000):
    df, colnum, colcat, coly = test_dataset_goodbooks(nrows= n_sample)
    X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes  = train_test_split2(df, coly)

    #### Matching Big dict  ##################################################
    def post_process_fun(y): return int(y)
    def pre_process_fun(y):  return int(y)
    m = {
    'model_pars': {
         'model_class' :  "torch_ease.py::EASE"
         ,'model_pars' : {  }
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
                      'compute_pars': { 'epochs' : 1}
    },

    'data_pars': { 'n_sample' : n_sample,
        'data_pars' :{
        },

        'download_pars'   : None,
        # 'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  ##################
         'cols_model_group': [ 'colnum_bin',   'colcat_bin', ]

        ### Filter data rows   ###########################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        ### Added continuous & sparse features groups ###
        'cols_input_type' : {
            'colcat' : colcat,
            'colnum' : colnum,
            'coly'  :  coly,
        }


        ####### ACTUAL data Values #############################################################
        ,'data_flow' : {
            'train':   {'Xtrain': X_train,  'ytrain': y_train}#, 'Xtest':  X_valid,  'ytest':  y_valid}  #{'X_train': train_df,'Y_train':train_label, 'X_test':  val_df,'Y_test':val_label }
            ,'val':     {'X': X_valid,  'y': y_valid}  #{  'X':  val_df ,'Y':val_label }
            ,'predict': {'X': X_valid}

        },

        }
    }
    # m = {'model_pars': {
    #     'model_class':  "torch_ease.py::EASE"
    #     ,'model_pars' : {
    #         'original_dim':       len( colcat + colnum),
    #         'class_num':             2,

    #     }
    #     , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
    #     , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

    #     ### Pipeline for data processing ##############################
    #     'pipe_list': [  #### coly target prorcessing
    #         {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
    #         {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
    #         {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
    #     ],
    #     }
    #     },

    # 'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score'],
    #                   'compute_pars' : {'epochs': 1 },
    #                 },

    # 'data_pars': {
    #     'n_sample' : n_sample,
    #     'download_pars' : None,
    #     'cols_input_type' : {
    #         'colcat' : colcat,
    #         'colnum' : colnum,
    #         'coly'  :  coly,
    #     },
    #     ### family of columns for MODEL  #########################################################
    #     'cols_model_group': [ 'colnum_bin',   'colcat_bin',  ],

    #     ### Added continuous & sparse features groups ###
    #     'cols_model_type2': {
    #         'colcontinuous':   colnum ,
    #         'colsparse' : colcat,
    #     }

    #     ### Filter data rows   ##################################################################
    #     ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },


    #     ###################################################
    #     'df' : df
    #     ,'train':   {'Xtrain': X_train,  'ytrain': y_train, 'Xtest':  X_valid,  'ytest':  y_valid}
    #     ,'eval':    {'X': X_valid,  'y': y_valid}
    #     ,'predict': {'X': X_valid}

    #     ,'task_type' : 'train', 'data_type': 'ram'

    #     }
    # }

    ###  Tester #########################################################
    test_helper(m['model_pars'], m['data_pars'], m['compute_pars'])



def test_helper(model_pars, data_pars, compute_pars):
    global model,session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars)

    log('Predict data..')
    ypred = predict(data_pars=data_pars,compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')


    # log('Saving model..')
    # save(path= root + '/model_dir/')

    # log('Model architecture:')
    # log(model.model.summary())

    # log('Model Snapshot')
    # model_summary()





if __name__ == "__main__":
    import fire
    fire.Fire()








# def get_dataset(data_pars, task_type="train"):
#     """
#     :param data_pars:
#     :param task_type:
#     :return:
#     """
#     clean       = data_pars["data_pars"].get('clean', True)
#     data_path   = data_pars["data_pars"]["data_path"]
#     batch_size  = data_pars["data_pars"]["batch_size"]

#     if task_type == 'pred_encode':
#             train_loader, X_train, target_errors_train, dataset_obj,  attributes = utils.load_data(data_path, batch_size,
#                                             is_train=True,
#                                             get_data_idxs=False)

#             return X_train

#     elif task_type == 'pred_decode':
#         train_loader, X_train, target_errors_train, dataset_obj,  attributes = utils.load_data(data_path, batch_size,
#                                         is_train=True,
#                                         get_data_idxs=False)

#         return target_errors_train


#     if not clean:
#         if task_type == 'train':
#             train_loader, X_train, target_errors_train, dataset_obj,  attributes = utils.load_data(data_path, batch_size,
#                                             is_train=True,
#                                             get_data_idxs=False)

#             return train_loader, X_train, target_errors_train, dataset_obj, attributes

#         elif task_type == 'test':

#             test_loader, X_test, target_errors_test, _, _ = utils.load_data(
#                 data_path, batch_size, is_train=False
#             )

#             return test_loader, X_test, target_errors_test
#         elif task_type == 'predict':
#             train_loader, _, _, _,  _ = utils.load_data(data_path, batch_size,
#                                             is_train=True,
#                                             get_data_idxs=False)

#             return train_loader

#     # -- clean versions for evaluation
#     else:

#         if task_type == 'train':
#             _, X_train_clean, _, _, _ = utils.load_data(
#                 data_path, batch_size, is_train=True, is_clean=True, stdize_dirty=True
#             )

#             return X_train_clean

#         elif task_type == 'test':
#             _, X_test_clean, _, _, _ = utils.load_data(
#                 data_path, batch_size, is_train=False, is_clean=True, stdize_dirty=True
#             )

#             return X_test_clean


# def make_rand_sparse_dataset(
#         n_rows=1000,
#     ):
#     # we need a single source of all user_ids and item_ids
#     # to avoid ids apearing in test that wasn't available in train
#     all_train_data = np.random.randint(0, 10000000, (n_rows, 2)).astype(np.int32)

#     # Split data
#     train_data, val_data = train_test_split(all_train_data, test_size=0.1, shuffle=True)
#     val_data, test_data = train_test_split(val_data, test_size=0.5)

#     # add val to train_data and create df
#     val = np.ones((train_data.shape[0],1)).astype(np.int32)
#     train_data = np.hstack((train_data, val))
#     train_set_df = pd.DataFrame(
#         data=train_data
#     )

#     # add val to val_data and create df
#     val = np.ones((val_data.shape[0],1))
#     val_data = np.hstack((val_data, val))
#     val_set_df = pd.DataFrame(
#         data=val_data
#     )

#     # add val to test_data and create df
#     val = np.ones((test_data.shape[0],1))
#     test_data = np.hstack((test_data, val))
#     test_set_df = pd.DataFrame(
#         data=test_data
#     )

#     ds = Dataset(
#         uids=np.unique(all_train_data[:,0]).astype(np.int32),
#         iids=np.unique(all_train_data[:,1]).astype(np.int32),
#         train_set=train_set_df,
#         valid_set=val_set_df,
#         test_set=test_set_df
#     )

#     train_sampler = SparseDummySampler(
#         data=ds,
#         mode='train',
#         batch_size=128,
#         shuffle=True
#     )

#     return train_sampler


# def test_sparse(n_sample=1000):

#     train_sampler = make_rand_sparse_dataset(
#         n_rows=1000,
#     )

#     model = EASE()

#     log("Training...")

#     model.train(train_sampler)
#     test_te = train_sampler.data_te

#     test_uids = np.random.choice(train_sampler.data.unique_uid, 500)
#     uid_to_internal_rectorch_id = lambda uid: train_sampler.data.u2id[uid]
#     id_mapper = np.vectorize(uid_to_internal_rectorch_id)
#     test_mapped_ids = id_mapper(test_uids)

#     log("Predicting...")

#     res = model.predict(test_mapped_ids, test_te, False)
#     log(res)


# cols_ref_formodel = ['cols_single_group']
# cols_ref_formodel = ['colcontinuous', 'colsparse']
# def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
#     """  Split into Tuples to feed  Xyuple = (df1, df2, df3) OR single dataframe
#     :param Xtrain:
#     :param cols_type_received:
#     :param cols_ref:
#     :return:
#     """
#     if len(cols_ref) <= 1 :
#         return Xtrain

#     Xtuple_train = []
#     # cols_ref is the reference for types of cols groups (sparse/continuous)
#     # This will result in deviding the dataset into many groups of features
#     for cols_groupname in cols_ref :
#         # Assert the group name is in the cols reference
#         assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
#         cols_i = cols_type_received[cols_groupname]
#         # Add the columns of this group to the list
#         Xtuple_train.append( Xtrain[cols_i] )

#     if len(cols_ref) == 1 :
#         return Xtuple_train[0]  ### No tuple
#     else :
#         return Xtuple_train


# def get_dataset2(data_pars=None, task_type="train", **kw):
#     """  Return tuple of dataframes
#     """
#     # log(data_pars)
#     data_type = data_pars.get('type', 'ram')
#     cols_ref  = cols_ref_formodel

#     if data_type == "ram":
#         # cols_ref_formodel = ['cols_cross_input', 'cols_deep_input', 'cols_deep_input' ]
#         ### dict  colgroup ---> list of colname

#         cols_type_received     = data_pars.get('cols_model_type2', {} )  ##3 Sparse, Continuous

#         if task_type == "predict":
#             d = data_pars[task_type]
#             Xtrain       = d["X"]
#             Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
#             return Xtuple_train

#         if task_type == "eval":
#             d = data_pars[task_type]
#             Xtrain, ytrain  = d["X"], d["y"]
#             Xtuple_train    = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
#             return Xtuple_train, ytrain

#         if task_type == "train":
#             d = data_pars[task_type]
#             Xtrain, ytrain, Xtest, ytest  = d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

#             ### dict  colgroup ---> list of df
#             Xtuple_train = get_dataset_tuple(Xtrain, cols_type_received, cols_ref)
#             Xtuple_test  = get_dataset_tuple(Xtest, cols_type_received, cols_ref)
#             log2("Xtuple_train", Xtuple_train)

#             return Xtuple_train, ytrain, Xtuple_test, ytest


#     elif data_type == "file":
#         raise Exception(f' {data_type} data_type Not implemented ')

#     raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


# def eval(Xpred=None, data_pars: dict={}, compute_pars: dict={}, out_pars: dict={}, **kw):
#     global model, session
#     """
#          Encode + Decode
#     """
#     Xencoded = encode(Xpred=Xpred, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
#     log("\nEncoded : ", Xencoded)

#     log('\nDecoding : ')
#     Xnew_original = decode(Xpred=Xencoded, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
#     log('\nDecoded : ', Xnew_original)
