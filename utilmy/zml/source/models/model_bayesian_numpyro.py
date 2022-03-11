# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn

####################################################################################################
from utilmy import global_verbosity, os_makedirs
verbosity = global_verbosity(__file__, "/../../config.json" ,default= 5)

def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    """function log
    Args:
        *s:   
    Returns:
        
    """
    print(*s, flush=True)

def log2(*s):
    """function log2
    Args:
        *s:   
    Returns:
        
    """
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    """function log3
    Args:
        *s:   
    Returns:
        
    """
    if verbosity >= 3 : print(*s, flush=True)

####################################################################################################
global model, session
def init(*kw, **kwargs):
    """function init
    Args:
        *kw:   
        **kwargs:   
    Returns:
        
    """
    """function init
    Args:
        *kw:   
        **kwargs:   
    Returns:
        
    """
    global model, session
    model = Model(*kw, **kwargs)
    session = None

def reset():
    """function reset
    Args:
    Returns:
        
    """
    """function reset
    Args:
    Returns:
        
    """
    global model, session
    model, session = None, None


########Custom Model ################################################################################
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import sklearn

from torch import nn
from pyro.nn import PyroModule
import logging

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroSample

####################################################################################################
VERBOSE = False


# MODEL_URI = get_model_uri(__file__)


# from mlmodels.util import log, path_norm, get_model_uri
def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session


def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        """ Model:__init__
        Args:
            model_pars:     
            data_pars:     
            compute_pars:     
        Returns:
           
        """
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            ###############################################################
            from model_numpyro import Normal
            class BayesianRegression(Normal):
                def __init__(self, in_features, out_features):
                    dv = "y"
                    features = dict(
                        const=dict(transformer=1, prior=dist.Normal(0, 1)),
                        x=dict(transformer=lambda df: df.x, prior=dist.Normal(0, 1)),
                    )




            input_width = model_pars['model_pars']['input_width']
            y_width = model_pars['model_pars'].get('y_width', 1)
            self.model = BayesianRegression(input_width, y_width)
            self.guide = None
            self.pred_summary = None  ### All MC summary

            if VERBOSE: log(self.guide, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")

    Xtrain = torch.tensor(Xtrain.values, dtype=torch.float)
    Xtest = torch.tensor(Xtest.values, dtype=torch.float)
    ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    ytest = torch.tensor(ytest.values, dtype=torch.float)

    if VERBOSE: log(Xtrain, model.model)

    ###############################################################
    compute_pars2 = compute_pars.get('compute_pars', {})
    n_iter = compute_pars2.get('n_iter', 1000)
    lr = compute_pars2.get('learning_rate', 0.01)
    method = compute_pars2.get('method', 'svi_elbo')


    RNG_KEY = np.array([0, 0])
    model = model.model.fit(Xtrain, rng_key=RNG_KEY)



    df_loss = pd.DataFrame(losses)
    df_loss['loss'].plot()
    return df_loss


def predict(Xpred=None, data_pars={}, compute_pars=None, out_pars={}, **kw):
    """function predict
    Args:
        Xpred:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kw:   
    Returns:
        
    """
    global model, session
    # data_pars['train'] = False

    compute_pars2 = model.compute_pars if compute_pars is None else compute_pars
    num_samples = compute_pars2.get('num_samples', 300)

    ###### Data load
    if Xpred is None:
        Xpred = get_dataset(data_pars, task_type="predict")
    cols_Xpred = list(Xpred.columns)

    max_size = compute_pars2.get('max_size', len(Xpred))

    Xpred = Xpred.iloc[:max_size, :]
    Xpred_ = torch.tensor(Xpred.values, dtype=torch.float)

    ###### Post processing normalization
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y

    from pyro.infer import Predictive
    def summary(samples):
        site_stats = {}
        for k, v in samples.items():
            site_stats[k] = {
                "mean": torch.mean(v, 0),
                "std": torch.std(v, 0),
                # "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
                # "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
            }
        return site_stats

    predictive = Predictive(model.model, guide=model.guide, num_samples=num_samples,
                            return_sites=("linear.weight", "obs", "_RETURN"))
    pred_samples = predictive(Xpred_)
    pred_summary = summary(pred_samples)

    mu = pred_summary["_RETURN"]
    y = pred_summary["obs"]
    dd = {
        "mu_mean": post_process_fun(mu["mean"].detach().numpy()),
        # "mu_perc_5"    : post_process_fun( mu["5%"].detach().numpy() ),
        # "mu_perc_95"   : post_process_fun( mu["95%"].detach().numpy() ),
        "y_mean": post_process_fun(y["mean"].detach().numpy()),
        # "y_perc_5"     : post_process_fun( y["5%"].detach().numpy() ),
        # "y_perc_95"    : post_process_fun( y["95%"].detach().numpy() ),
        # "true_salary" : y_data,
    }
    for i, col in enumerate(cols_Xpred):
        dd[col] = Xpred[col].values  # "major_PHYSICS": x_data[:, -8],
    # print(dd)
    ypred_mean = pd.DataFrame(dd)
    model.pred_summary = {'pred_mean': ypred_mean, 'pred_summary': pred_summary, 'pred_samples': pred_samples}
    print('stored in model.pred_summary')
    # print(  dd['y_mean'], dd['y_mean'].shape )
    # import pdb; pdb.set_trace()
    return dd['y_mean']


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    """function save
    Args:
        path:   
        info:   
    Returns:
        
    """
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    """function load_model
    Args:
        path:   
    Returns:
        
    """
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model = model0.model
    model.model_pars = model0.model_pars
    model.compute_pars = model0.compute_pars
    session = None
    return model, session


def load_info(path=""):
    """function load_info
    Args:
        path:   
    Returns:
        
    """
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


def preprocess(prepro_pars):
    """function preprocess
    Args:
        prepro_pars:   
    Returns:
        
    """
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
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values)
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
            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


def get_params(param_pars={}, **kw):
    """function get_params
    Args:
        param_pars:   
        **kw:   
    Returns:
        
    """
    import json
    # from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")
