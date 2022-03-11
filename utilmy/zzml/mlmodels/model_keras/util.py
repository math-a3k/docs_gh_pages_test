from jsoncomment import JsonComment ; json = JsonComment()
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor

VERBOSE = False




####################################################################################################
# Helper functions
def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(filepath).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def log(*s, n=0, m=1):
    """function log
    Args:
        *s:   
        n:   
        m:   
    Returns:
        
    """
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


####################################################################################################
def _config_process(data_path, config_mode="test"):
    """function _config_process
    Args:
        data_path:   
        config_mode:   
    Returns:
        
    """
    data_path = Path(os.path.realpath(
        __file__)).parent.parent / "model_gluon/gluon_deepar.json" if data_path == "dataset/" else data_path

    with open(data_path, encoding='utf-8') as config_f:
        config = json.load(config_f)
        config = config[config_mode]

    return config["model_pars"], config["data_pars"], config["compute_pars"], config["out_pars"]



# Dataaset
def get_dataset(**kw):
    """function get_dataset
    Args:
        **kw:   
    Returns:
        
    """
    ##check whether dataset is of kind train or test
    data_path = kw['train_data_path'] if  kw['train'] else kw['test_data_path']

    #### read from csv file
    if  kw.get("uri_type") == "pickle" :
        data_set = pd.read_pickle(data_path)
    else :
        data_set = pd.read_csv(data_path)

    ### convert to gluont format
    gluonts_ds = ListDataset([{FieldName.TARGET: data_set.iloc[i].values, FieldName.START: kw['start'] }
                             for i in range(kw['num_series'])],  freq=kw['freq'])

    if VERBOSE:
        entry = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig = kw['save_fig']
        plt.savefig(save_fig)

    return gluonts_ds


# Model fit
def fit(model, data_pars=None, model_pars=None, compute_pars=None, out_pars=None,session=None, **kwargs):
        ##loading dataset
        """
          Classe Model --> model,   model.model contains thte sub-model

        """
        pass



# Model p redict
def predict(model, data_pars, compute_pars=None, out_pars=None, **kwargs):
    """function predict
    Args:
        model:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    ##  Model is class
    ## load test dataset
    pass


def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    """function metrics
    Args:
        ypred:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    ## load test dataset
    pass


###############################################################################################################




###############################################################################################################
# save and load model helper function
class Model_empty(object):
    def __init__(self, model_pars=None, compute_pars=None):
        """ Model_empty:__init__
        Args:
            model_pars:     
            compute_pars:     
        Returns:
           
        """
        ## Empty model for Seaialization
        self.model = None


def save(model, path):
    """function save
    Args:
        model:   
        path:   
    Returns:
        
    """
    if not os.path.exists(os.path.dirname(path)):
        print("model file path do not exist!")
    else:
        save_model(model.model, path)


def load(path):
    """function load
    Args:
        path:   
    Returns:
        
    """
    if not os.path.exists(path):
        print("model file do not exist!")
        return None
    else:
        model = Model_empty()
        model_keras = load_model(path, custom_objects)
        model.model = model_keras

        #### Add back the model parameters...
        return model
