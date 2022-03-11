# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

python model_bayesian_pyro.py      test


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
    global model, session
    model = Model(*kw, **kwargs)
    session = None

def reset():
    """function reset
    Args:
    Returns:
        
    """
    global model, session
    model, session = None, None


########Custom Model ################################################################################
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
import torch
from torch import nn


####################################################################################################
class BayesianRegression(PyroModule):
    def __init__(self, X_dim:int=17, y_dim:int=1):
        """ BayesianRegression:__init__
        Args:
            X_dim (function["arg_type"][i]) :     
            y_dim (function["arg_type"][i]) :     
        Returns:
           
        """
        super().__init__()
        self.linear = PyroModule[nn.Linear](X_dim, y_dim)

        # W of shape (y_width, input_width)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([y_dim, X_dim]).to_event(2))
        # bias (y_width, 1)
        self.linear.bias   = PyroSample(dist.Normal(0., 10.).expand([y_dim]).to_event(1))

    def forward(self, x, y=None):
        """ BayesianRegression:forward
        Args:
            x:     
            y:     
        Returns:
           
        """
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

# Supported Models
MODEL_LIST = [ BayesianRegression ]


# Map model_name to class : [model_bayesian_pyro.py::BayesianRegression] -> Model Class
def model_class_loader(m_name='BayesianRegression', class_list:list=None):
  """function model_class_loader
  Args:
      m_name:   
      class_list ( list ) :   
  Returns:
      
  """
  class_list_dict = { myclass.__name__ : myclass for myclass in class_list }
  class_name = m_name.split("::")[-1]
  return class_list_dict.get(class_name)


####################################################################################################
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
            # Load Model Class
            model_class = model_class_loader(model_pars['model_class'], MODEL_LIST ) 
            
            mpars    = model_pars.get('model_pars', {})  ## default already in model

            # change from dict keys (to __init__ params (X_dim, y_dim)
            self.model  = model_class( **mpars )
            self.guide  = None
            self.pred_summary = None  ### All MC summary
            self.history      = None

            log(self.guide, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")

    Xtrain = torch.tensor(Xtrain.values, dtype=torch.float)
    Xtest  = torch.tensor(Xtest.values, dtype=torch.float)
    ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    ytest  = torch.tensor(ytest.values, dtype=torch.float)

    log(Xtrain, model.model)

    ###############################################################
    compute_pars2 = compute_pars.get('compute_pars', {})
    n_iter        = compute_pars2.get('n_iter', 1000)
    lr            = compute_pars2.get('learning_rate', 0.01)
    method        = compute_pars2.get('method', 'svi_elbo')

    ### SVI + Elbo is faster than HMC
    guide = AutoDiagonalNormal(model.model)
    adam  = pyro.optim.Adam({"lr": lr})
    svi = SVI(model.model, guide, adam, loss=Trace_ELBO())

    pyro.clear_param_store()
    losses = []
    for j in range(n_iter):
        # calculate the loss and take a gradient step
        loss = svi.step(Xtrain, ytrain)
        losses.append({'loss': loss, 'iteration': j})
        if j % 100 == 0:
            log("[iteration %04d] loss: %.4f" % (j + 1, loss / len(Xtrain)))

    model.guide = guide

    df_loss = pd.DataFrame(losses)
    df_loss['loss'].plot()
    model.history =df_loss



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

    compute_pars2 = model.compute_pars if compute_pars is None else compute_pars
    num_samples = compute_pars2.get('num_samples', 300)

    ###### Data load
    if Xpred is None:
        Xpred = get_dataset(data_pars, task_type="predict")
    cols_Xpred = list(Xpred.columns)

    max_size = compute_pars2.get('max_size', len(Xpred))

    Xpred    = Xpred.iloc[:max_size, :]
    Xpred_   = torch.tensor(Xpred.values, dtype=torch.float)

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
            }
        return site_stats

    # If the model is loaded, it drops the guide param if it's None
    guide      = getattr(model, "guide", None)
    predictive = Predictive(model.model, guide=guide, num_samples=num_samples,
                            return_sites=("linear.weight", "obs", "_RETURN"))
    pred_samples = predictive(Xpred_)
    pred_summary = summary(pred_samples)

    mu = pred_summary["_RETURN"]
    y  = pred_summary["obs"]
    dd = {
        "mu_mean": post_process_fun(mu["mean"].detach().numpy()),
        "y_mean": post_process_fun(y["mean"].detach().numpy()),
    }
    for i, col in enumerate(cols_Xpred):
        dd[col] = Xpred[col].values  # "major_PHYSICS": x_data[:, -8],

    ypred_mean = pd.DataFrame(dd)
    model.pred_summary = {'pred_mean': ypred_mean, 'pred_summary': pred_summary, 'pred_samples': pred_samples}
    print('stored in model.pred_summary')

    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return dd['y_mean'], ypred_proba


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
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))


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


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  : 
      "file" :
    """
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


########################################################################################################################
########################################################################################################################
def y_norm(y, inverse=True, mode='boxcox'):
    """function y_norm
    Args:
        y:   
        inverse:   
        mode:   
    Returns:
        
    """
    ## Normalize the input/output
    if mode == 'boxcox':
        width0 = 53.0  # 0,1 factor
        k1 = 0.6145279599674994  # Optimal boxCox lambda for y
        if inverse:
                y2 = y * width0
                # Numpy Warns of raising power to a neg nbr to in case they result in complex nbrs  
                tmp = ((y2 * k1) + 1)
                y2 = np.sign(tmp) * np.abs(tmp) ** (1 / k1)
                return y2
        else:
                y1 = (y ** k1 - 1) / k1
                y1 = y1 / width0
                return y1

    if mode == 'norm':
        m0, width0 = 0.0, 0.0  ## Min, Max
        if inverse:
                y1 = (y * width0 + m0)
                return y1

        else:
                y2 = (y - m0) / width0
                return y2
    else:
            return y


def test_dataset_regress_fake(nrows=500):
    """function test_dataset_regress_fake
    Args:
        nrows:   
    Returns:
        
    """
    from sklearn import datasets as sklearn_datasets
    coly   = ['y']
    # 16 num features
    colnum = ["colnum_" +str(i) for i in range(0, 16) ]
    # 1 cat features
    colcat = ['colcat_1']

    # Generate a regression dataset
    X, y    = sklearn_datasets.make_regression( n_samples=nrows, n_features=17, n_targets=1, n_informative=15,noise=0.1)
    df      = pd.DataFrame(X,  columns= colnum+colcat)
    df[coly]= y.reshape(-1, 1)
    df[coly] = (df[coly] -df[coly].min() ) / (df[coly].max() -df[coly].min() )

    # Assign categ values the cat columns 
    for ci in colcat :
      df[ci] = np.random.randint(    low=0, high=1, size=df[ci].shape )

    return df, colnum, colcat, coly


def test(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    global model, session

    #### Regression PLEASE RANDOM VALUES AS TEST
    ### Fake Regression dataset
    df, colcat, colnum, coly = test_dataset_regress_fake(nrows=nrows)
    X = df[colcat + colnum]
    y = df[coly]


    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, )#stratify=y) Regression no classes to stratify to
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021,)# stratify=y_train_full)
    log("X_train", X_train)
    log("y_train", y_train)

    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')


    m = {'model_pars': {

        # Input features, output features
        'model_pars' : {},

        'post_process_fun' : post_process_fun   ### After prediction  ##########################################
     
        },

        'compute_pars': { 'metric_list': ['accuracy_score', 'median_absolute_error']
                        },

        'data_pars': { 
            'n_sample' : n_sample,
        ###################################################  
        'train': {  'Xtrain': X_train,
                    'ytrain': y_train,
                    'Xtest': X_valid,
                    'ytest': y_valid
        },
        'eval': {   'X': X_valid,
                    'y': y_valid
        },
        'predict': {'X': X_valid}

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        }
    }

    ##### Running loop
    ll = [
        ('model_bayesian_pyro.py::BayesianRegression', {'X_dim': 17,  'y_dim': 1 } )
    ]
    for cfg in ll:
        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg[0]
        m['model_pars']['model_pars']  = cfg[1]

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')


        log('Saving model..')
        save(path= "ztmp/data/output/torch_tabular")

        log('Load model..')
        model, session = load_model(path="ztmp/data/output/torch_tabular")
        
        log('Model architecture:')
        log(model.model)

        log('Predict data..check')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])

        reset()



if __name__ == "__main__":
    import fire
    fire.Fire()



