# coding: utf-8
"""
Generic template for new model.
Check parameters template in models_config.json

"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 2 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }



"""
import inspect
import os
import sys
from jsoncomment import JsonComment ; json = JsonComment()
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd



from mlmodels.util import (os_package_root_path, log, path_norm, get_model_uri,
                           config_path_pretrained, config_path_dataset, os_path_split  )


from mlmodels.data import (download_data, import_data  )



VERBOSE = False
MODEL_URI = get_model_uri(__file___)
####################################################################################################



####################################################################################################
class Model:
  def __init__(self, model_pars=None, data_pars=None, compute_pars=None  ):
    """ Model:__init__
    Args:
        model_pars:     
        data_pars:     
        compute_pars:     
    Returns:
       
    """
    ### Model Structure        ################################
    if model_pars is None :
        self.model = None

    else :
        self.model = None






def fit(model, data_pars=None, compute_pars=None, out_pars=None,   **kw):
  """
  """

  sess = None # Session type for compute
  Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
  

  return model, sess




def evaluate(model, data_pars=None, compute_pars=None, out_pars=None,  **kw):
    """
       Return metrics of the model when fitted.
    """
    ddict = {}
    
    return ddict

    
        

  

def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw):
  """function predict
  Args:
      model:   
      sess:   
      data_pars:   
      compute_pars:   
      out_pars:   
      **kw:   
  Returns:
      
  """
  ##### Get Data ###############################################
  Xpred, ypred = None, None

  #### Do prediction
  ypred = model.model.fit(Xpred)

  ### Save Results
  
  
  ### Return val
  if compute_pars.get("return_pred_not") is not None :
    return ypred


  
  
def reset_model():
  """function reset_model
  Args:
  Returns:
      
  """
  pass





def save(model=None, session=None, save_pars={}):
    """function save
    Args:
        model:   
        session:   
        save_pars:   
    Returns:
        
    """
    from mlmodels.util import save_tf
    print(save_pars)
    save_tf(model, session, save_pars)
     


def load(load_pars={}):
    """function load
    Args:
        load_pars:   
    Returns:
        
    """
    from mlmodels.util import load_tf
    print(load_pars)
    input_tensors, output_tensors =  load_tf(load_pars)

    model = Model()
    model.model = None
    session = None
    return model, session




####################################################################################################
def get_dataset(data_pars=None, **kw):
  """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
  """

  if data_pars['train'] :
    Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
    return Xtrain, Xtest, ytrain, ytest 


  else :
    Xtest, ytest = None, None  # data for training.
    return Xtest, ytest 



def get_params(param_pars={}, **kw):
    """function get_params
    Args:
        param_pars:   
        **kw:   
    Returns:
        
    """
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']


    if choice == "json":
       data_path = path_norm(data_path)
       cf = json.load(open(data_path, mode='r'))
       cf = cf[config_mode]
       return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path  = path_norm( "dataset/text/imdb.csv"  )   
        out_path   = path_norm( "ztest/model_keras/crf_bilstm/" )   
        model_path = os.path.join(out_path , "model")


        data_pars ={
            "path"            :  "",
            "path_type"   :  "local/absolute/web",

            "data_type"   :   "text" / "recommender"  / "timeseries" /"image",
            "data_split"  : {"istrain" :  1   , "split_size" : 0.5, "randomseed" : 1   },

            "data_loader"      :   "mlmodels.data.pd_reader",
            "data_loader_pars" :   { "ok"  },

            "data_preprocessor" : "mlmodels.model_keras.prepocess:process",
            "data_preprocessor_pars" :  {  },

            "size" : [0,1,2],
            "output_size": [0, 6]            
          }

        model_pars   = {}

        
        compute_pars = {}
        

        out_pars     = {}

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")






################################################################################################
########## Tests are normalized Do not Change ##################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    """function test
    Args:
        data_path:   
        pars_choice:   
        config_mode:   
    Returns:
        
    """
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)


    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    metrics_val = evaluate(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save   ########################################################")
    save(model, session, out_pars)

    log("#### Load   ########################################################")
    model2 = load( out_pars )
    ypred = predict(model2, data_pars, compute_pars, out_pars)
    print(model2)




if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"

    ### Local fixed params
    test(pars_choice="test01")

    ### Local json file
    # test(pars_choice="json")

    ####    test_module(model_uri="model_xxxx/yyyy.py", param_pars=None)
    from mlmodels.models import test_module

    param_pars = {'choice': "test01", 'config_mode': 'test', 'data_path': '/dataset/'}
    test_module(module_uri=MODEL_URI, param_pars=param_pars)

    ##### get of get_params
    # choice      = pp['choice']
    # config_mode = pp['config_mode']
    # data_path   = pp['data_path']

    ####    test_api(model_uri="model_xxxx/yyyy.py", param_pars=None)
    from mlmodels.models import test_api

    param_pars = {'choice': "test01", 'config_mode': 'test', 'data_path': 'dataset/'}
    test_api(module_uri=MODEL_URI, param_pars=param_pars)



