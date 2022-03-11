# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.train.html
https://github.com/optuna/optuna/blob/master/examples/lightgbm_tuner_simple.py
### https://github.com/optuna/optuna/blob/master/examples/pruning/lightgbm_integration.py
"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn
# from repo.model_gefs.experiments.run_missing import X_train
from sklearn.model_selection import train_test_split
####################################################################################################
try   : verbosity = int(json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/../../config.json", mode='r'))['verbosity'])
except Exception as e : verbosity = 2
#raise Exception(f"{e}")

def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    if verbosity >= 1 : print(*s, flush=True)

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

def os_makedirs(dir_or_file):
    """function os_makedirs
    Args:
        dir_or_file:   
    Returns:
        
    """
    if os.path.isfile(dir_or_file) :os.makedirs(os.path.dirname(os.path.abspath(dir_or_file)), exist_ok=True)
    else : os.makedirs(os.path.abspath(dir_or_file), exist_ok=True)

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
from lightgbm import LGBMModel, LGBMRegressor, LGBMClassifier
import optuna.integration.lightgbm as LGBMModel_optuna



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
            model_object_name = 'LGBMModel_optuna'
            if 'LGBMClassifier' in model_pars['model_class']  :
               self.model_pars['model_pars']['objective'] =  'binary'

            elif  'LGBMRegressor' in model_pars['model_class'] :
               self.model_pars['model_pars']['objective'] =  'huber'
            
            # Suppress warnings
            self.model_pars['model_pars']['verbose'] = -1
            # To avoid warings that occure due to large num_leaves 
            self.model_pars['model_pars']['num_leaves'] = 5

             
            
            model_class     = globals()[model_object_name]
            self.model_meta = model_class  ### Hyper param seerch Model
            self.model      = None         ### Best model saved after train
            #self.model = model_class()
            log2(model_class, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xval, yval = get_dataset(data_pars, task_type="train")
    log2(Xtrain.shape, model.model)

    dtrain = model.model_meta.Dataset(Xtrain,label = ytrain)
    dval   = model.model_meta.Dataset(Xval,  label = yval)
    # dtrain = LGBMModel_optuna.Dataset(Xtrain, label=ytrain)
    # dval = LGBMModel_optuna.Dataset(Xtest, label=ytest)

    pars_lightgbm = model.model_pars['model_pars']   #### from model_pars
    # Suppress logs for each leaf, removing this will cause 1000s of ouputs
    pars_optuna   = compute_pars.get("optuna_params", {"verbose_eval":-1})    ### Specific to Optuna
    optuna_engine = compute_pars.get('optuna_engine', 'simple')
    print("pars_lightgbm", pars_lightgbm)
    print("pars_optuna", pars_optuna)

    if optuna_engine == 'LightGBMTuner':
        model_fit = model.model_meta.LightGBMTuner(pars_lightgbm, dtrain, 
                    valid_sets=[dtrain, dval], **pars_optuna).run()

    elif optuna_engine == 'LightGBMTunerCV':
        model_fit = model.model_meta.LightGBMTunerCV(pars_lightgbm, dtrain, 
                    valid_sets=[dtrain, dval], **pars_optuna).run()

    else :
        model_fit = model.model_meta.train( pars_lightgbm, dtrain, 
                    valid_sets=[dtrain, dval], **pars_optuna)
    """
       print("Best score:", tuner.best_score)
       best_params = tuner.best_params         
    """

    ### Best model store as
    print(model_fit.params)
    model.model                    = model_fit
    model.model_pars['model_pars'] = model_fit.params
    return model_fit


def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
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
    ## optuna_model = model.model_pars.get('optuna_model', None)   #### NO model is saved in model.model

    if Xpred is None:
        Xpred = get_dataset(data_pars, task_type="predict")

    ypred = model.model.predict(Xpred,) # num_iteration=model.model.best_iteration)

    ypred_proba = None  ### No proba    
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred) 
    return ypred, ypred_proba


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

    filename = "model_pars.pkl"
    pickle.dump(model.model_pars, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )


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
    model.model      = model0.model
    model.model_meta = model0.model_meta

    model.model_pars   = model0.model_pars
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

####################################################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
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


####################################################################################################################
def test_dataset_classi_fake(nrows=500):
    """function test_dataset_classi_fake
    Args:
        nrows:   
    Returns:
        
    """
    from sklearn import datasets as sklearn_datasets
    ndim=11
    coly   = 'y'
    colnum = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(
        # nbr_informative + nbr_redundent < n_features
        n_samples=1000, n_features=ndim, n_classes=2, n_informative=ndim-2)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[colcat] = np.random.randint(0,1, len(df))

    return df, colnum, colcat, coly


def test(config=''):
    """function test
    Args:
        config:   
    Returns:
        
    """
    global model, session
    df, colnum, colcat, coly = test_dataset_classi_fake(nrows=500)

    #### Matching Big dict  ##################################################
    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return int(y)

    def pre_process_fun(y):
        return int(y)

    m = {
    'model_pars': {
        
        'objective' : 'binary',
         'model_class' :  "optuna_lightgbm.py::LGBMClassifier"
         ,'model_pars' : {}
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

        'download_pars' : None,
        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
         'cols_model_group': [ 'colnum_bin',   'colcat_bin', ]
        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly  }
        ####### ACTUAL data pipeline #############################################################
        ,'train':   {} #{'X_train': train_df,'Y_train':train_label, 'X_test':  val_df,'Y_test':val_label }
        ,'val':     {}  #{  'X':  val_df ,'Y':val_label }
        ,'predict': {}


        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' :     colcat,
        },
        }
    }


    log("##### Sparse Tests  ############################################### ")
    ##### Dict update
    m['model_pars']['model_pars'] = {  }
    train_df = df[colnum + colcat]
    train_df, test_df = train_test_split(df, test_size=0.2)
    # test_df, val_df = train_test_split(val_df, test_size=0.5)

    m['data_pars']['train']     = {
        'Xtrain': train_df[colcat+colnum], 
        'ytrain': train_df[coly],
        'Xtest':  test_df[colnum+colcat],
        'ytest': test_df[coly]
    }
    m['data_pars']['predict']   = {'X':       test_df }
    m['data_pars']['data_pars'] = {
        # 'colcat_unique' : colcat_unique,
        'colcat'        : colcat,
        'colnum'        : colnum,
        # 'colembed_dict' : colembed_dict
    }

    test_helper( m['model_pars'], m['data_pars'], m['compute_pars'])



def test_helper(model_pars, data_pars, compute_pars):
    """function test_helper
    Args:
        model_pars:   
        data_pars:   
        compute_pars:   
    Returns:
        
    """
    global model,session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars)

    log('Predict data..')
    ypred, ypred_proba = predict(data_pars=data_pars,compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

    log('Saving model..')
    save(path= root + '/model_dir/')

    log('Model architecture:')
    log(model.model)

    # log('Model Snapshot')
    # model_summary()

def benchmark():
    """function benchmark
    Args:
    Returns:
        
    """
    global model
    try:
        from pmlb import fetch_data, classification_dataset_names
    except:
        log("Installing pmlb...")
        os.system("pip install pmlb")
        from pmlb import fetch_data, classification_dataset_names


    for classification_dataset in classification_dataset_names:
        df = fetch_data(classification_dataset, return_X_y=False)
        train_df, test_df = train_test_split(df)
        log("\n\n")
        log(f"\t\t######################## {classification_dataset} ########################\n")
        benchmark_helper(train_df, test_df)
        log(f"\t\t######################## !! END !! ########################\n")

        

def benchmark_helper(train_df, test_df):
    """function benchmark_helper
    Args:
        train_df:   
        test_df:   
    Returns:
        
    """
    global model, session
    # plmb has no meta data available with the datasets
    # to get dynamicaly, but it keeps seperate datatypes for cat/num (float64/int64)
    colcat = train_df.iloc[:,:-1].select_dtypes(["int64"]).head(1).columns.to_list()
    colnum = train_df.iloc[:,:-1].select_dtypes(["float64"]).head(1).columns.to_list()
    coly = train_df.columns.to_list()[-1]
    #### Matching Big dict  ##################################################
    cols_input_type_1 = []
    n_sample = 100
    def post_process_fun(y):
        return int(y)

    def pre_process_fun(y):
        return int(y)

    m = {
    'model_pars': {
        
        'objective' : 'binary',
         'model_class' :  "optuna_lightgbm.py::LGBMClassifier"
         ,'model_pars' : {}
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

        'download_pars' : None,
        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
         'cols_model_group': [ 'colnum_bin',   'colcat_bin', ]
        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly  }
        ####### ACTUAL data pipeline #############################################################
        ,'train':   {} #{'X_train': train_df,'Y_train':train_label, 'X_test':  val_df,'Y_test':val_label }
        ,'val':     {}  #{  'X':  val_df ,'Y':val_label }
        ,'predict': {}


        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' :     colcat,
        },
        }
    }


    log("##### Sparse Tests  ############################################### ")
    ##### Dict update
    m['model_pars']['model_pars'] = {  }
 
    # test_df, val_df = train_test_split(val_df, test_size=0.5)

    m['data_pars']['train']     = {
        'Xtrain': train_df[colcat+colnum], 
        'ytrain': train_df[coly],
        'Xtest':  test_df[colnum+colcat],
        'ytest': test_df[coly]
    }
    m['data_pars']['predict']   = {'X':       test_df }
    m['data_pars']['data_pars'] = {
        # 'colcat_unique' : colcat_unique,
        'colcat'        : colcat,
        'colnum'        : colnum,
        # 'colembed_dict' : colembed_dict
    }

    test_helper( m['model_pars'], m['data_pars'], m['compute_pars'])

# def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
#     """
#        Return metrics of the model when fitted.
#     """
#     pass

####################################################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()

