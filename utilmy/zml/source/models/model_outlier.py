# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
python model_fraud.py
"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn

####################################################################################################
from utilmy import global_verbosity, os_makedirs, pd_read_file
verbosity = global_verbosity(__file__,"/../../config.json", 3 )

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


########Custom Model #########################################################################################
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.tree import *
from lightgbm import LGBMModel, LGBMRegressor, LGBMClassifier

try :
    #### All are Un-supervised Model
    from pyod.models.abod  import *
    from pyod.models.auto_encoder import *
    from pyod.models.cblof import *
    from pyod.models.cof import *
    from pyod.models.combination import *
    from pyod.models.copod import *
    from pyod.models.feature_bagging import *
    from pyod.models.hbos import *
    from pyod.models.iforest import *
    from pyod.models.knn import *
    from pyod.models.lmdd import *
    from pyod.models.loda import *
    from pyod.models.lof import *
    from pyod.models.loci import *
    from pyod.models.lscp import *
    from pyod.models.mad import *
    from pyod.models.mcd import *
    from pyod.models.mo_gaal import *
    from pyod.models.ocsvm import *
    from pyod.models.pca import *
    from pyod.models.sod import *
    from pyod.models.so_gaal import *
    from pyod.models.sos import *
    from pyod.models.vae import *
    # from pyod.models.xgbod import *
except Exception as e :
    print("Error import  pyod", e)


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            ##
            self.model_pars['model_path']  = self.model_pars['model_class'].split(":")[0]
            self.model_pars['model_class'] = self.model_pars['model_class'].split(":")[-1]

            model_class = globals()[self.model_pars['model_class']]  ## globals() Buggy when doing loop
            self.model  = model_class( **self.model_pars['model_pars'])
            log2(model_class, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset2(data_pars, task_type="train")
    log2(Xtrain.shape, model.model)

    if  model.model_pars['model_class'] in [ 'HBOS', 'ABOD'  ]:  ## Numba issues
        Xtrain = Xtrain.astype('float32')
        ytrain = ytrain.astype('float32')


    if "LGBM" in model.model_pars['model_class']:
        model.model.fit(Xtrain, ytrain, eval_set=[(Xtest, ytest)], **compute_pars.get("compute_pars", {}))
    else:
        model.model.fit(Xtrain, ytrain, **compute_pars.get("compute_pars", {}))


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session
    if Xpred is None:
        data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")


    if  model.model_pars['model_class'] in [ 'HBOS', 'ABOD'  ]:
        Xpred = Xpred.astype('float32')


    ypred = model.model.predict(Xpred)
    #ypred = post_process_fun(ypred)

    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
          if  model.model_pars['model_class'] in [  ]:
             ypred_proba = model.model.decision_scores  (Xpred)
          else :
             """
               bug in site-packages\pyod\models/base.py
               Correct code :
                     self._classes = max(2, len(np.unique(y)) )
             """
             ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba



def save(path=None, info=None):
    global model, session
    # import cloudpickle as pickle
    import dill as pickle
    os.makedirs(path, exist_ok=True)

    try :
       pickle.dump(model, open(f"{path}/model.pkl", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    except Exception as e :
       mkeras = model.model.combine_model   ## Keras version
       mkeras.save( f"{path}/model_keras.h5" )
       model.model = None
       pickle.dump(model, open(f"{path}/model.pkl", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )
       model.model = mkeras
       log( f"{path}/model_keras.h5"  )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))    # ,protocol=pickle.HIGHEST_PROTOCOL )



def load_model(path=""):
    global model, session
    # import cloudpickle as pickle
    import dill as pickle
    try  :
        model0      = pickle.load(open(f"{path}/model.pkl", mode='rb'))
        model       = Model()  # Empty model
        model.model_pars = model0.model_pars
        model.compute_pars = model0.compute_pars

        if model.model_pars['model_class'] in  [ 'SO_GAAL', 'VAE'] :
           import keras
           model.model =  keras.models.load_model(f"{path}/model_keras.h5")
        else :
           model.model = model0.model

        session = None
    except Exception as e :
       log(e)

    return model, session


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
THISMODEL_COLGROUPS = []
def get_dataset_split_for_model_pandastuple(Xtrain, ytrain=None, data_pars=None, ):
    """  Split data for moel input/
    Xtrain  ---> Split INTO  tuple of data  Xtuple= (df1, df2, df3) to fit model input.
    :param Xtrain:
    :param coldataloader_received:
    :param colmodel_ref:
    :return:
    """
    from utilmy import pd_read_file
    coldataloader_received  = data_pars.get('cols_model_type2', {})   ### column defined in Data
    colmodel_ref            = THISMODEL_COLGROUPS   ### Column defined here

    ### Into RAM
    if isinstance(Xtrain, str) : Xtrain = pd_read_file(Xtrain + "*", verbose=False)
    if isinstance(ytrain, str) : ytrain = pd_read_file(ytrain + "*", verbose=False)


    ##########################################################################
    if len(colmodel_ref) <= 1 :   ## No split
        return Xtrain, ytrain

    ### Split the pandas columns into different pieces  ######################
    Xtuple_train = []
    for cols_groupname in colmodel_ref :
        assert cols_groupname in coldataloader_received, "Error missing colgroup in  data_pars[cols_model_type] "
        cols_name_list = coldataloader_received[cols_groupname]
        Xtuple_train.append( Xtrain[cols_name_list] )

    return Xtuple_train, ytrain


def get_dataset2(data_pars=None, task_type="train", **kw):
    """
       Raw Data (Path)   --->  Input Object (ie Pandas, ...) for Model training
    """
    log3('data_pars', data_pars)
    data_type            = data_pars.get('type', 'pandas')
    d                    = data_pars[task_type]
    data_pars[task_type] = None    ### Save memory

    if data_type in [ 'pandas', 'ram'] :
        if task_type == "predict":
            Xtrain, _ = get_dataset_split_for_model_pandastuple(d['X'], None,  data_pars)
            return Xtrain

        if task_type == "eval":
            Xtrain, ytrain = get_dataset_split_for_model_pandastuple(d['X'], d['y'],  data_pars)
            return Xtrain, ytrain

        if task_type == "train":
            Xtrain, ytrain = get_dataset_split_for_model_pandastuple(d['Xtrain'], d['ytrain'],  data_pars)
            Xtest, ytest   = get_dataset_split_for_model_pandastuple(d['Xtest'],  d['ytest'],   data_pars)
            return Xtrain, ytrain, Xtest, ytest



def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  :
      "file" :
    """
    # log(data_pars)
    data_type  = data_pars.get('type', 'ram')
    cols_type  = data_pars.get('cols_model_type2', {})   #### Split input by Sparse, Continous

    log3("Cols Type:", cols_type)

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




####################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()





"""
class EnsembleDetector:
    def save(self, folder):
        #Saves the EnsembleDetector (as multiple files) in a given folder.'''
        # Save TF-based AutoEncoders in separate sub-directories (they don't pickle)
        tf_models = {}   # {index for self.models: model} 
        for i, model in enumerate(self.models):
            if 'AutoEncoder' in str(type(model)):
                model.model_.save(Path(folder)/str(i))
                tf_models[i] = model.model_
                model.model_ = None  # Remove non-pickleable TF models from self so we can pickle self
            if 'VAE' in str(type(model)):
                raise Exception('VAE is not supported when saving the ensemble yet, since it uses a Lambda layer.')
        # Pickle the entire EnsembleDetector after the TF models are removed
        Path(folder).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(folder)/'ensemble_detector.joblib')
        # Add the TF model objects back into self
        for i in tf_models: self.models[i].model_ = tf_models[i]
    @staticmethod
    def load(folder):
        '''Loads the EnsembleDetector (from multiple files) in a given folder.'''
        # Unpickle the EnsembleDetector object
        ed = joblib.load(Path(folder)/'ensemble_detector.joblib')
        # Load TF-based AutoEncoders from separate sub-directories (they don't pickle)
        for i, model in enumerate(ed.models):
            if 'AutoEncoder' in str(type(model)):
                model.model_ = keras.models.load_model(Path(folder)/str(i))
        return ed
"""