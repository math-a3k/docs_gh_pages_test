# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
python torch_ease.py test



"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn
from sklearn.model_selection import train_test_split

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
import wget
import zipfile
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)
# from torch.utils import data
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from rectorch.evaluation import evaluate
from rectorch.utils import collect_results

##### Add custom repo to Python Path ################################################################
#thisfile_dirpath = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
#import_path      = thisfile_dirpath + "/repo/TorchEASE/"
#sys.path.append(import_path)

##### Import from src/core_models/
import rectorch

##### pkg
# path_pkg =  thisfile_dirpath + "/repo/TorchEASE/"
from rectorch.models.mf import EASE
from rectorch.data import DataProcessing
from rectorch.samplers import ArrayDummySampler

####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, global_pars=None):
        self.model_pars, self.compute_pars, self.data_pars, self.global_pars = model_pars, compute_pars, data_pars, global_pars
        if model_pars is None:
            self.model = None
            return 

        dataset = data_pars['dataset']
        self.model = EASE(**model_pars["model_cfg"])
                # enc_dims=None,
                # dropout=0.5,
                # beta=.2,
                # anneal_steps=100000,
                # opt_conf=None,
                # device="cpu",
                # trainer=None)



def get_dataset(data_pars, task_type="train"):
    """
    :param data_pars:
    :param task_type:
    :return:
    """
    clean       = data_pars["data_pars"].get('clean', True)
    data_path   = data_pars["data_pars"]["data_path"]
    batch_size  = data_pars["data_pars"]["batch_size"]

    if task_type == 'pred_encode':
            train_loader, X_train, target_errors_train, dataset_obj,  attributes = utils.load_data(data_path, batch_size,
                                            is_train=True,
                                            get_data_idxs=False)

            return X_train
        
    elif task_type == 'pred_decode':
        train_loader, X_train, target_errors_train, dataset_obj,  attributes = utils.load_data(data_path, batch_size,
                                        is_train=True,
                                        get_data_idxs=False)

        return target_errors_train


    if not clean:
        if task_type == 'train':
            train_loader, X_train, target_errors_train, dataset_obj,  attributes = utils.load_data(data_path, batch_size,
                                            is_train=True,
                                            get_data_idxs=False)

            return train_loader, X_train, target_errors_train, dataset_obj, attributes
        
        elif task_type == 'test':

            test_loader, X_test, target_errors_test, _, _ = utils.load_data(
                data_path, batch_size, is_train=False
            )

            return test_loader, X_test, target_errors_test
        elif task_type == 'predict':
            train_loader, _, _, _,  _ = utils.load_data(data_path, batch_size,
                                            is_train=True,
                                            get_data_idxs=False)

            return train_loader
        
    # -- clean versions for evaluation
    else:

        if task_type == 'train':
            _, X_train_clean, _, _, _ = utils.load_data(
                data_path, batch_size, is_train=True, is_clean=True, stdize_dirty=True
            )

            return X_train_clean

        elif task_type == 'test':
            _, X_test_clean, _, _, _ = utils.load_data(
                data_path, batch_size, is_train=False, is_clean=True, stdize_dirty=True
            )

            return X_test_clean
        

def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    sampler = data_pars["sampler"]
    valid_metric = compute_pars["train"]["valid_metric"]
    model.model.train(sampler)



def predict(Xpred=None, data_pars=None, compute_pars={}, out_pars={}, **kw):
    global model, session
    data_sampler = Xpred
    if data_sampler is None:
        data_sampler = data_pars['sampler']

        one_batch = next(iter(data_sampler))
        X_one_batch = one_batch[0]

    ypred = model.model.predict(X_one_batch[0], X_one_batch[1])

    
    return ypred[0], None

    

def eval(Xpred=None, data_pars: dict={}, compute_pars: dict={}, out_pars: dict={}, **kw):
    global model, session
  
    data_sampler = data_pars["sampler"]
    results = evaluate(model.model, data_sampler, ["ndcg@100", "recall@100", "ndcg@20", "recall@20"])
    return results

def save(path=None, info=None):
    """ Custom saving
    """
    global model, session
    import cloudpickle as pickle
    os.makedirs(os.path.normpath(path + "/model/model_rec_ease_checkpoint/"), exist_ok=True)

    #### Torch part
    model.model.save_model(os.path.normpath(path + "/model/model_rec_ease_checkpoint"))


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


# cols_ref_formodel = ['cols_single_group']
cols_ref_formodel = ['colcontinuous', 'colsparse']
def get_dataset_tuple(Xtrain, cols_type_received, cols_ref):
    """  Split into Tuples to feed  Xyuple = (df1, df2, df3) OR single dataframe
    :param Xtrain:
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    if len(cols_ref) <= 1 :
        return Xtrain

    Xtuple_train = []
    # cols_ref is the reference for types of cols groups (sparse/continuous)
    # This will result in deviding the dataset into many groups of features
    for cols_groupname in cols_ref :
        # Assert the group name is in the cols reference
        assert cols_groupname in cols_type_received, "Error missing colgroup in config data_pars[cols_model_type] "
        cols_i = cols_type_received[cols_groupname]
        # Add the columns of this group to the list
        Xtuple_train.append( Xtrain[cols_i] )

    if len(cols_ref) == 1 :
        return Xtuple_train[0]  ### No tuple
    else :
        return Xtuple_train


def get_dataset2(data_pars=None, task_type="train", **kw):
    """  Return tuple of dataframes
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



####################################################################################################
############ Test  #################################################################################

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

def get_dataset_sampler(data_pars):

    try:
        cfg_data = data_pars['data_cfg']
        dataset = DataProcessing(cfg_data).process_and_split()

        sampler = ArrayDummySampler(dataset, mode="train", batch_size=500)
    except:
        data_path = data_pars["data_path"]
        log("Dataset not downloaded, downloading now ....")
        
        # if not os.path.isdir(f"{data_path}ml-20m"):
        #     log("\n\nDownloading ml-20m ....")

        #     wget.download(
        #         "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        #         out=data_path
        #     )
        #     with zipfile.ZipFile(f"{data_path}ml-20m.zip") as zip_ref:
        #         zip_ref.extractall(data_path)

        if not os.path.isdir(f"{data_path}ml-1m"):
            log("\n\nDownloading ml-1m ....\n")

            wget.download(
                "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                out=data_path
            )
            with zipfile.ZipFile(f"{data_path}ml-1m.zip") as zip_ref:
                zip_ref.extractall(data_path)

        dataset, sampler = get_dataset_sampler(data_pars)


    return dataset, sampler

def init_dataset(data_pars):
    data_path = data_pars['data_pars']['data_cfg']["processing"]["data_path"]
    
def test(n_sample          = 1000):

    # #### Matching Big dict  ##################################################
    def post_process_fun(y): return int(y)
    def pre_process_fun(y):  return int(y)
    data_path = './rec_data/Movies/'
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)
        
    m = {'model_pars': {
            'model_class':  "model_rec.py::REC"
            ,'model_pars' : {
                'original_dim':       None,
                'class_num':             2,

            }
            , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
            , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

            ### Pipeline for data processing ##############################
        
            },
            'model_cfg' : {
                "lam" : 200.,
            }
        },

    'compute_pars': { 'metric_list': ['ndcg@100'],
                      "train": {
                            "valid_metric": "ndcg@100"
                        },
                        "test":{
                            "metrics": ["ndcg@100", "ndcg@10", "recall@20", "recall@50"]
                        },
                    },

    'data_pars': {
        'data_path' : data_path,
        'data_cfg' : {
                "processing": {
                "data_path": data_path + "ml-1m/ratings.dat",
                "threshold": 3.5,
                "separator": "::",
                #"header": 0,
                "u_min": 5,
                "i_min": 0
            },
            "splitting": {
                "split_type": "vertical",
                "sort_by": None,
                "seed": 98765,
                "shuffle": True,
                "valid_size": 200,
                "test_size": 200,
                "test_prop": 0.2
            }
        }

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },


        ###################################################
        'sampler' : None,
        'dataset' : None

        }
    }
    dataset, sampler = get_dataset_sampler(m["data_pars"])

    m["data_pars"]["sampler"] = sampler
    m["data_pars"]["dataset"] = dataset

    model = Model(
        model_pars=m["model_pars"],
        data_pars=m["data_pars"],
        compute_pars=m["compute_pars"]
    )
            
    # model.model.train(sampler, valid_metric="ndcg@100")
    # ###  Tester #########################################################
    test_helper(m['model_pars'], m['data_pars'], m['compute_pars'])



def test_helper(model_pars, data_pars, compute_pars):
    global model,session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars)

    log('Predict data..')
    ypred, ypred_proba = predict(data_pars=data_pars,compute_pars=compute_pars)
    log(f'y_pred: {ypred}')
    log(ypred.shape)

    log('Eval data..')
    eval_results = eval(data_pars=data_pars, compute_pars=compute_pars)
    log(collect_results(eval_results))
    # log('Saving model..')
    # save(path= root + '/model_rec/')

    log('Model architecture:')
    log(model.model)

    log('Model Snapshot')
    # model_summary()


if __name__ == "__main__":
    import fire
    fire.Fire()


