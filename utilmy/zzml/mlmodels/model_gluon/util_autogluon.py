import os

from autogluon import TabularPrediction as tabular_task

#### Requieres to install mlmodels
#### print(data)
VERBOSE = False


###################################################################################################
######## Helper functions
def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(os.path.realpath(filepath)).parent
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
################ Dataset   #########################################################################
def _get_dataset_from_aws(**kw):
    """function _get_dataset_from_aws
    Args:
        **kw:   
    Returns:
        
    """
    URL_INC_TRAIN = 'https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv'
    URL_INC_TEST = 'https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv'

    dt_name = kw['dt_name']
    if dt_name == 'Inc':
        if kw['train']:
            data = tabular_task.Dataset(file_path=URL_INC_TRAIN)
        else:
            data = tabular_task.Dataset(file_path=URL_INC_TEST)
        label = 'occupation'
        if kw.get('label'):
            label = kw.get('label')

        return data, label
    else:
        print(f"Not support {dt_name} yet!")


def import_data_fromfile(**kw):
   """
       data_pars["data_path"]
   
   """ 
   import pandas as pd
   import numpy as np

   m = kw
   if m.get("uri_type") in ["pickle", "pandas_pickle" ]:
       df = pd.read_pickle(m["data_path"])
       return df


   if m.get("uri_type") in ["csv", "pandas_csv" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if m.get("uri_type") in [ "dask" ]:
       df = pd.read_csv(m["data_path"])
       return df


   if ".npz" in m['data_path'] :
       arr = import_to_numpy(data_pars, mode=mode, **kw)
       return arr


   if ".csv" in m['data_path'] or ".txt" in m['data_path']  :
       df = import_to_pandas(data_pars, mode=mode, **kw)
       return df


   if ".pkl" in m['data_path']   :
       df = pd.read_pickle(m["data_path"], **kw)
       return df








def get_dataset(**kw):
    """function get_dataset
    Args:
        **kw:   
    Returns:
        
    """

    if kw['uri_type'] == 'amazon_aws':
        data, label = _get_dataset_from_aws(**kw)
        return data, label

    ##check whether dataset is of kind train or test
    # data_path = kw['train_data_path'] if kw['train'] else kw['test_data_path']
    df = import_data_fromfile(**kw )

    col_target = kw.get('col_target') if kw.get('col_target') else 'y'
    
    return df, col_target
  

    if VERBOSE:
        pass




####################################################################################################
# Model fit
def fit(model, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,
        **kwargs):
    ##loading dataset
    """
      Classe Model --> model,   model.model contains thte sub-model

    """
    data  = get_dataset(**data_pars)
    if data is None or not isinstance(data, (list, tuple)):
        raise Exception("Missing data or invalid data format for fitting!")

    train_ds, label = data
    nn_options = {
        'num_epochs'    : compute_pars['num_epochs'],
        'learning_rate' : model_pars['learning_rate'],
        'activation'    : model_pars['activation'],
        'layers'        : model_pars['layers'],
        'dropout_prob'  : model_pars['dropout_prob'],
    }
    
    gbm_options = {
        'num_boost_round': model_pars['num_boost_round'],
        'num_leaves':      model_pars['num_leaves'],
    }
  
    ## Attribut model has the model
    predictor = model.model.fit(train_data=train_ds, label=label,
                                output_directory    = out_pars['out_path'],
                                time_limits         = compute_pars['time_limits'],
                                num_trials          = compute_pars['num_trials'],
                                hyperparameter_tune = compute_pars['hp_tune'],
                                hyperparameters     = {'NN': nn_options, 'GBM': gbm_options},
                                search_strategy     = compute_pars['search_strategy'],
                                verbosity=3)
    model.model = predictor
    return model


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
    data_pars['train'] = False
    test_ds, label = get_dataset(**data_pars)
    # remove label in test data if have
    if label in test_ds.columns:
        test_ds = test_ds.drop(labels=[label], axis=1)

    y_pred = model.model.predict(test_ds)

    ### output stats for prediction
    if VERBOSE:
        pass
    return y_pred


def metrics(model, ypred, ytrue, data_pars, compute_pars=None, out_pars=None, **kwargs):
    """function metrics
    Args:
        model:   
        ypred:   
        ytrue:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    ## load test dataset
    #data_pars['train'] = False
    #test_ds, label = get_dataset(**data_pars)
    #y_test = test_ds[label]

    ## evaluate
    acc = model.model.evaluate_predictions(y_true=ytrue, y_pred=ypred, auxiliary_metrics=False)
    metrics_dict = {"ACC": acc}
    return metrics_dict


###############################################################################################################
### different plots and output metric




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
        self.model = tabular_task


def save(model, out_pars):
    """function save
    Args:
        model:   
        out_pars:   
    Returns:
        
    """
    if not model:
        print("model do not exist!")
    else:
        model.model.save()


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
        model.model = tabular_task.load(path)

        #### Add back the model parameters...
        return model




if __name__ == '__main__':
   VERBOSE = True
   df = get_dataset(data_path="../dataset/milk.csv", uri_type="csv")
   print(df)
