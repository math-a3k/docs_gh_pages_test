# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""

python torch_rvae.py test --nrows 1000


"""
import os, sys,  numpy as np,  pandas as pd, wget, copy
from pathlib import Path
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)
# from torch.utils import data
from sklearn.model_selection import train_test_split
import torch


######################################################################################################
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/repo/RVAE_MixedTypes/core_models/" )
import main, train_eval_models

root     = os.path.dirname(os.path.abspath(__file__)) 
path_pkg =  root + "/repo/RVAE_MixedTypes/"





####################################################################################################
VERBOSE = False

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if VERBOSE :
      print(*s, flush=True)

####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model, session = Model(*kw, **kwargs), None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None

        else:
            ###############################################################
            dm          = data_pars['cols_model_group_custom']


            self.model = None
            self.guide = None
            self.pred_summary = None  ### All MC summary

            if VERBOSE: log(self.guide, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    cpars          = copy.deepcopy( compute_pars.get("compute_pars", {}))   ## issue with pickle

    # if data_pars is not None :
    Xtrain_tuple, ytrain, Xtest_tuple, ytest = get_dataset(data_pars, task_type="train")

    train = pd.concat((Xtrain_tuple[0], Xtrain_tuple[1]), axis=1)
    train = pd.concat((train, ytrain), axis=1)

    val   = pd.concat((Xtest_tuple[0], Xtest_tuple[1]), axis=1)
    val   = pd.concat((val, ytest), axis=1)

    ###############################################################
    model.model.fit(train=train, validation=val, **cpars)



def predict(Xpred=None, data_pars: dict={}, compute_pars: dict={}, out_pars: dict={}, **kw):
    global model, session

    if Xpred is None:
        Xpred_tuple = get_dataset(data_pars, task_type="predict")
    else :
        cols_type   = data_pars.get('cols_model_type2', {})  ##
        Xpred_tuple = get_dataset_tuple(Xpred, cols_type, cols_ref_formodel)   

    Xpred_tuple_concat = pd.concat((Xpred_tuple[0], Xpred_tuple[1]), axis=1)
    ypred = model.model.predict(Xpred_tuple_concat)
    
    #####################################################################
    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba



def reset():
    global model, session
    model, session = None, None

def save(path=None, info=None):
    """ Custom saving
    """
    global model, session
    import cloudpickle as pickle
    os.makedirs(path + "/model/", exist_ok=True)

    #### Torch part
    model.model.save_model(path + "/model/torch_checkpoint")

    #### Wrapper
    model.model = None   ## prevent issues
    pickle.dump(model,  open(path + "/model/model.pkl", mode='wb')) # , protocol=pickle.HIGHEST_PROTOCOL )
    pickle.dump(info, open(path   + "/model/info.pkl", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(path + '/model/model.pkl', mode='rb'))
         
    model = Model()  # Empty model
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars
    model.data_pars    = model0.data_pars

    ### Custom part
    # model.model        = TabularModel.load_from_checkpoint( "ztmp/data/output/torch_tabular/torch_checkpoint")
    model.model        = TabularModel.load_from_checkpoint(  path +"/model/torch_checkpoint")
 
    session = None
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


def get_dataset(data_pars=None, task_type="train", **kw):
    """
      return tuple of dataframes
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
def test_dataset_1(nrows=1000):
    # Dense features
    colnum = []

    # Sparse features
    colcat = [  ]



def test(nrows=1000):
    """
        nrows : take first nrows from dataset
    """
    global model, session
    df, colnum, colcat, coly = test_dataset_covtype()

    #### Matching Big dict  ##################################################    
    X = df
    y = df[coly].astype('uint8')
    log('y', np.sum(y[y==1]) )

    # Split the df into train/test subsets
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)
    num_classes = len(set(y_train_full[coly].values.ravel()))
    log(X_train)





    m = {'model_pars': {
        ### LightGBM API model   #######################################
        # Specify the ModelConfig for pytorch_tabular
        'model_class':  "torch_tabular.py::CategoryEmbeddingModelConfig"
        
        # Type of target prediction, evaluation metrics
        ,'model_pars' : { 
                        # 'task': "classification",
                        # 'metrics' : ["f1","accuracy"],
                        # 'metrics_params' : [{"num_classes":num_classes},{}]
                        }  

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################

        ### Pipeline for data processing ##############################
        'pipe_list': [  #### coly target prorcessing
        {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },

        {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },

        #### catcol INTO integer,   colcat into OneHot
        {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
        {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },

        ],
            }
        },

    'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                    },

    'data_pars': { 'n_sample' : n_sample,
        'download_pars' : None,
        'cols_input_type' : cols_input_type_1,
        ### family of columns for MODEL  #########################################################
        'cols_model_group': [ 'colnum_bin',   'colcat_bin',
                            ]

        ,'cols_model_group_custom' :  { 'colnum' : colnum,
                                        'colcat' : colcat,
                                        'coly' : coly
                            }
        ###################################################  
        ,'train': {'Xtrain': X_train, 'ytrain': y_train,
                   'Xtest': X_valid,  'ytest':  y_valid},
                'eval': {'X': X_valid,  'y': y_valid},
                'predict': {'X': X_valid}

        ### Filter data rows   ##################################################################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        
        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
            'colcontinuous':   colnum ,
            'colsparse' : colcat, 
        },
        }
    }


    ll = [
        ('torch_tabular.py::CategoryEmbeddingModelConfig', 
            {   'task': "classification",
                'metrics' : ["f1","accuracy"],
                'metrics_params' : [{"num_classes":num_classes},{}]
            }  
        ),

    
    ]
    for cfg in ll:
        log("******************************************** New Model ********************************************")
        log(f"******************************************** {cfg[0]} ********************************************")
        reset()
        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg[0]
        m['model_pars']['model_pars']  = {**m['model_pars']['model_pars'] , **cfg[1] }

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

        if cfg != "torch_tabular.py::NodeConfig":
            log('Saving model..')
            save(path= "ztmp/data/output/torch_tabular")
            #  os.path.join(root, 'data\\output\\torch_tabular\\model'))

            log('Load model..')
            model, session = load_model(path="ztmp/data/output/torch_tabular")
            #os.path.join(root, 'data\\output\\torch_tabular\\model'))
        else:
            log('\n*** !!! Saving Bug in pytorch_tabular for NodeConfig !!! ***\n')
            
        log('Model architecture:')
        log(model.model)

        log('Model config:')
        log(model.model.config._config_name)
        reset()


def test2(nrow=10000):
    """
       python source/models/torch_tabular.py test

    """
    global model, session


if __name__ == "__main__":
    import fire
    fire.Fire()
    # test()








