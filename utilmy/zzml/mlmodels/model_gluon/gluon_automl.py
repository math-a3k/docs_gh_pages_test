# -*- coding: utf-8 -*-
"""
AutoGluon : Automatic ML using gluon platform.
# First install package from terminal:  pip install mxnet autogluon
https://autogluon.mxnet.io/tutorials/tabular_prediction/tabular-quickstart.html



"""
from jsoncomment import JsonComment ; json = JsonComment()
import os
from pathlib import Path

import autogluon as ag
from autogluon import TabularPrediction as tabular_task
from mlmodels.model_gluon.util_autogluon import (
    fit, get_dataset, load, predict, save)


from mlmodels.util import path_norm, os_package_root_path, log



########################################################################################################################
#### Model defintion
class Model(object):
    def __init__(self, model_pars=None, compute_pars=None):
        """ Model:__init__
        Args:
            model_pars:     
            compute_pars:     
        Returns:
           
        """
        ## Empty model for Seaialization
        if model_pars is None and compute_pars is None:
            self.model = None

        else:
            if model_pars['model_type'] == 'tabular':
                self.model = tabular_task


########################################################################################################################
def path_setup(out_folder="", sublevel=0, data_path="dataset/"):
    """function path_setup
    Args:
        out_folder:   
        sublevel:   
        data_path:   
    Returns:
        
    """
    data_path = os_package_root_path(path_add=data_path)
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)
    model_path = out_path + "/model_gluon_automl/"
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path


def _config_process(config):
    """function _config_process
    Args:
        config:   
    Returns:
        
    """
    data_pars = config["data_pars"]

    log("#### Model params   ################################################")
    m = config["model_pars"]
    model_pars = {"model_type": m["model_type"],
                  "learning_rate": ag.space.Real(m["learning_rate_min"],
                                                 m["learning_rate_max"],
                                                 default=m["learning_rate_default"],
                                                 log=True),
                  
                  "activation": ag.space.Categorical(*tuple(m["activation"])),
                  "layers": ag.space.Categorical(*tuple(m["layers"])),
                  "dropout_prob": ag.space.Real(m["dropout_prob_min"],
                                                m["dropout_prob_max"],
                                                default=m["dropout_prob_default"]),
                  
                  "num_boost_round": m["num_boost_round"],
                  "num_leaves": ag.space.Int(lower=m["num_leaves_lower"],
                                             upper=m["num_leaves_upper"],
                                             default=m["num_leaves_default"])
                  }

    compute_pars = config["compute_pars"]
    out_pars = config["out_pars"]
    return model_pars, data_pars, compute_pars, out_pars


def get_params(choice="", data_path="dataset/", config_mode="test", **kw):
    """function get_params
    Args:
        choice:   
        data_path:   
        config_mode:   
        **kw:   
    Returns:
        
    """
    if choice == "json":
        data_path = Path(os.path.realpath(
            __file__)).parent.parent / "model_gluon/gluon_automl.json" if data_path == "dataset/" else data_path

        with open(data_path, encoding='utf-8') as config_f:
            config = json.load(config_f)
            config = config[config_mode]

        model_pars, data_pars, compute_pars, out_pars = _config_process(config)
        return model_pars, data_pars, compute_pars, out_pars

    if choice == "test01":
        log("#### Path params   #################################################")
        data_path, out_path, model_path = path_setup(out_folder="", sublevel=0,
                                                     data_path="dataset/")

        data_pars = {"train": True, "uri_type": "amazon_aws", "dt_name": "Inc"}

        model_pars = {"model_type": "tabular",
                      "learning_rate": ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),
                      "activation": ag.space.Categorical(*tuple(["relu", "softrelu", "tanh"])),
                      "layers": ag.space.Categorical(
                          *tuple([[100], [1000], [200, 100], [300, 200, 100]])),
                      'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),
                      'num_boost_round': 10,
                      'num_leaves': ag.space.Int(lower=26, upper=30, default=36)}

        compute_pars = {"hp_tune": True, "num_epochs": 1, "time_limits": 100, "num_trials": 2,
                        "search_strategy": "skopt"}
        out_pars = {"out_path": out_path}

    return model_pars, data_pars, compute_pars, out_pars


########################################################################################################################
def test(data_path="dataset/", pars_choice="json"):
    """function test
    Args:
        data_path:   
        pars_choice:   
    Returns:
        
    """
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice,
                                                               data_path=data_path)

    log("#### Loading dataset   #############################################")
    gluon_ds = get_dataset(**data_pars)

    log("#### Model init, fit   #############################################")
    model = Model(model_pars, compute_pars)
    model = fit(model, data_pars, model_pars, compute_pars, out_pars)

    log("#### save the trained model  #######################################")
    # save(model, data_pars["modelpath"])


    log("#### Predict   ####################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)

    #log("#### metrics   ####################################################")
    #metrics_val = metrics(model, ypred, data_pars, compute_pars, out_pars)
    #print(metrics_val)

    log("#### Plot   #######################################################")

    log("#### Save/Load   ##################################################")
    save(model, out_pars)
    model2 = load(out_pars['out_path'])
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)


if __name__ == '__main__':
    VERBOSE = True
    test(pars_choice="json")
    #test(pars_choice="test01")
