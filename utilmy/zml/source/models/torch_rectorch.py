# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
python torch_rectorch.py test



"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn
import scipy.sparse as scipy_sparse
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
# from main.EASE import TorchEASE

from rectorch.data import Dataset
from rectorch.samplers import SparseDummySampler
from rectorch.models.mf import EASE
##### pkg
path_pkg =  thisfile_dirpath + "/repo/TorchEASE/"


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):

        if model_pars is None:
            self.model = None
            return

        lam = model_pars.get("lam", 100.)
        self.model = EASE(
            lam=lam
        )
        log2(self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    train_sampler = get_dataset(data_pars, task_type="train")
    model.model.train(train_sampler)



def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session

    test_mapped_ids, test_te = get_dataset(data_pars, task_type="predict")
    log("Predicting...")
    print("model shape : ", model.model.model.shape)
    ypred = model.model.predict(test_mapped_ids, test_te, False)

    return ypred

def get_dataset(data_pars=None, task_type="train"):


    if task_type == "train":
        # df = data_pars.get('df')
        train_set_df = data_pars.get('train').get('df')
        val_set_df = data_pars.get('val').get('df')
        test_set_df = data_pars.get('predict').get('df')
        batch_size = data_pars.get('data_pars').get('batch_size')

        df = pd.concat((
            train_set_df,
            val_set_df,
            test_set_df
        ))

        ds = Dataset(
            uids=np.unique(df.iloc[:,0]).astype(np.int32),
            iids=np.unique(df.iloc[:,1]).astype(np.int32),
            train_set=train_set_df,
            valid_set=val_set_df,
            test_set=test_set_df
        )

        train_sampler = SparseDummySampler(
            data=ds,
            mode='train',
            batch_size=batch_size,
            shuffle=True
        )

        data_pars['train']['train_sampler'] = train_sampler

        return train_sampler
    elif task_type == "predict":

        train_sampler = data_pars['train']['train_sampler']
        test_te = train_sampler.data_te

        test_uids = np.random.choice(train_sampler.data.unique_uid, 500)
        uid_to_internal_rectorch_id = lambda uid: train_sampler.data.u2id[uid]
        id_mapper = np.vectorize(uid_to_internal_rectorch_id)
        test_mapped_ids = id_mapper(test_uids)

        return test_mapped_ids, test_te
    else:
        raise Exception(f"Task {task_type} is not supported")

def save(path=None, info=None):
    """ Custom saving
    """
    global model, session
    import cloudpickle as pickle
    os.makedirs(path + "/model/", exist_ok=True)

    #### Torch part
    model.model.save_model(path + "/model/torch_rectorch")

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

def make_rand_sparse_dataset(
        n_rows=1000,
    ):
    # we need a single source of all user_ids and item_ids
    # to avoid ids apearing in test that wasn't available in train
    all_train_data = np.random.randint(0, 10000000, (n_rows, 2)).astype(np.int32)
    val = np.ones((all_train_data.shape[0], 1)).astype(np.int32)
    all_train_data = np.hstack(
        (all_train_data, val)
    )
    df = pd.DataFrame(data=all_train_data)

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.1, shuffle=True)
    val_df, test_df = train_test_split(val_df, test_size=0.5)


    return train_df, val_df, test_df



def test(n_sample          = 1000):
    train_df, val_df, test_df = make_rand_sparse_dataset(n_rows= n_sample)

    #### Matching Big dict  ##################################################
    def post_process_fun(y): return int(y)
    def pre_process_fun(y):  return int(y)

    # m = {'model_pars': {
    #     'model_class':  "torch_rectorch.py::EASE"
    #     ,'model_pars' : {
    #             'lam' : 100.
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





    #     ### Added continuous & sparse features groups ###
    #     'cols_model_type': {
    #         'uid':   0 ,
    #         'iid' : 1,
    #     }

    #     ### Filter data rows   ##################################################################
    #     ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }


    #     ###################################################
    #     # ,'df' : df
    #     ,'train':   {'df': train_df, 'train_sampler' : None}
    #     ,'eval':    {'df': val_df}
    #     ,'predict': {'df': test_df}


    #     }
    # }
    m = {
    'model_pars': {
        'model_class' :  "torch_rectorch.py::EASE"

        ,'model_pars' : {

            'lam' : 100.
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

    'compute_pars': { 
        'compute_extra' :{
             
            },

        'compute_pars' :{
            'metric_list': ['accuracy_score','average_precision_score'],
            'compute_pars' : {'epochs': 1 },
        },

    },

    'data_pars': { 'n_sample' : n_sample,
  
        'download_pars'   : None,
        # 'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  ##################
         'cols_model_group': [ 'colnum_bin',   'colcat_bin', ]

        ### Filter data rows   ###########################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
        },


        'data_pars' :{
                'cols_model_type': {
                    'uid':   0 ,
                    'iid' : 1,
                },
                # Raw dataset, pre preprocessing
                "dataset_path" : "",
                "batch_size":128,   ### Mini Batch from data
                # Needed by getdataset
                "clean" : False,
                "data_path": "",
        }
        ####### ACTUAL data Values #############################################################
        ,'train':   {'df': train_df, 'train_sampler' : None}
        ,'val':     {'df': val_df}
        ,'predict': {'df': test_df}

    },

    'global_pars' :{
    }
    }
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
    log(f'y_pred : ', ypred)


    log('Saving model..')
    save(path= root + '/model_dir/')

# def test_dataset_goodbooks(nrows=1000):
#     from sklearn.preprocessing import LabelEncoder
#     data_path = "./goodbooks_dataset"
#     if not os.path.isdir(data_path):
#         os.makedirs(data_path, exist_ok=True)
    
#         wget.download(
#             "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip",
#             out=data_path
#         )

#         with zipfile.ZipFile(f"{data_path}/goodbooks-10k.zip") as zip_ref:
#             zip_ref.extractall(data_path)
#     df = pd.read_csv(data_path + "/ratings.csv")
#     # Dense features
#     coly = ['rating',  ]

#     # Sparse features
#     colcat = ['user_id', 'book_id' ]
#     colnum = []
#     return df, colnum, colcat, coly


# def train_test_split2(df, coly):
#     log3(df.dtypes)
#     y = df[coly] ### If clonassificati
#     X = df.drop(coly,  axis=1)
#     log3('y', np.sum(y[y==1]) , X.head(3))
#     ######### Split the df into train/test subsets
#     X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
#     X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)

#     #####
#     # y = y.astype('uint8')
#     num_classes                                = len(set(y_train_full.values.ravel()))

#     return X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes



if __name__ == "__main__":
    import fire
    fire.Fire()
















# def test_dataset_goodbooks(nrows=1000):
#     from sklearn.preprocessing import LabelEncoder
#     data_path = "./goodbooks_dataset"
#     if not os.path.isdir(data_path):
#         os.makedirs(data_path, exist_ok=True)

#         wget.download(
#             "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip",
#             out=data_path
#         )

#         with zipfile.ZipFile(f"{data_path}/goodbooks-10k.zip") as zip_ref:
#             zip_ref.extractall(data_path)
#     df = pd.read_csv(data_path + "/ratings.csv")
#     # Dense features
#     coly = ['rating',  ]

#     # Sparse features
#     colcat = ['user_id', 'book_id' ]
#     colnum = []
#     return df, colnum, colcat, coly


# def train_test_split2(df, coly):
#     log3(df.dtypes)
#     y = df[coly] ### If clonassificati
#     X = df.drop(coly,  axis=1)
#     log3('y', np.sum(y[y==1]) , X.head(3))
#     ######### Split the df into train/test subsets
#     X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021)
#     X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021)

#     #####
#     # y = y.astype('uint8')
#     num_classes                                = len(set(y_train_full.values.ravel()))

#     return X,y, X_train, X_valid, y_train, y_valid, X_test,  y_test, num_classes

