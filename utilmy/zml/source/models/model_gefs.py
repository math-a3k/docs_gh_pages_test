# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
python model_gef.py test
"""
import os, sys,copy, pathlib, pprint, json, pandas as pd, numpy as np, scipy as sci, sklearn

####################################################################################################
from utilmy import global_verbosity, os_makedirs, pd_read_file
verbosity = global_verbosity(__file__,"/../../config.json", 3 )

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


#######Custom model #################################################################################
from sklearn.model_selection import train_test_split
# from gefs import prep
# from prep import train_test_split

thisfile_dirpath = os.path.dirname(os.path.abspath(__file__) ).replace("\\", "/")
try :
  sys.path.append( thisfile_dirpath + "/repo/model_gefs/" )
  from gefs import RandomForest
except :
  #   os.system( " python -m pip install git+https://github.com/arita37/GeFs/GeFs.git@aa32d657013b7cacf62aaad912a9b88110cee5d1  -y ")
  # Updated GeFs
  os.system( "pip install git+git://github.com/arita37/GeFs.git@f5725d7787149eea3886f52437cec77513e30666")
  sys.path.append( thisfile_dirpath + "/repo/model_gefs/" )
  from gefs import RandomForest


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
            self.n_estimators = model_pars.get('n_estimators', 100)
            self.ncat         = model_pars.get('ncat', None)  # Number of categories of each variable This is an ndarray
            if self.ncat is None:
                self.model = None  # In order to create an instance of the model we need to calculate the ncat mentioned above on our dataset
                log('ncat is not define')
            else:
                """
                    def __init__(self, n_estimators=100, imp_measure='gini', min_samples_split=2,
                 min_samples_leaf=1, max_features=None, bootstrap=True,
                 ncat=None, max_depth=1e6, surrogate=False):
                """
                self.model = RandomForest(n_estimators=self.n_estimators, ncat=self.ncat)
            log(None, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")
    log(Xtrain.shape, model.model)

    if model.ncat is None:
        log("#!IMPORTANT This indicates that the preprocessing pipeline was not adapted to GEFS! and we need to calculate ncat")
        cont_cols  = data_pars['data_pars'].get("colnum")  #  continous, float column is this correct?
        temp_train = pd.concat([Xtrain, ytrain], axis=1)
        temp_test  = pd.concat([Xtest, ytest],   axis=1)
        df         = pd.concat([temp_train, temp_test], ignore_index=True, sort=False)
        model.ncat = pd_colcat_get_catcount(
            df,
            # categ cols
            colcat=data_pars["data_pars"]["colcat"],
            # target col index
            coly=-1,
            # num cols indices
            continuous_ids=[df.columns.get_loc(c) for c in cont_cols]
        )
        ncat = np.array(list(model.ncat.values()))

        # In case of warnings make sure ncat is consistent
        # check this issue : https://github.com/AlCorreia/GeFs/issues/6
        """
         def __init__(self, n_estimators=100, imp_measure='gini', min_samples_split=2,
         min_samples_leaf=1, max_features=None, bootstrap=True,
         ncat=None, max_depth=1e6, surrogate=False):
        """
        model.model = RandomForest(n_estimators=model.n_estimators, ncat=ncat, )

    # Remove the target col
    X = Xtrain.iloc[:,:-1]
    # y should be 1-dim
    model.model.fit(X.values, ytrain.values.reshape(-1))

    # Make sure ncat is consistent, otherwise model.topc()
    # will throw all kind of numba errors
    # check this issue : https://github.com/AlCorreia/GeFs/issues/5
    model.model = model.model.topc()  # Convert to a GeF


def eval(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    data_pars['train'] = True
    Xval, yval        = get_dataset(data_pars, task_type="val")
    print("Xval : ",Xval, "\nyval : ", yval)
    ypred, ypred_prob = predict(Xval, data_pars, compute_pars, out_pars)
    mpars = compute_pars.get("metrics_pars", {'metric_name': 'auc'})

    scorer = { "auc": sklearn.metrics.roc_auc_score, }[mpars['metric_name']]

    mpars2 = mpars.get("metrics_pars", {})  ##Specific to score
    score_val = scorer(yval, ypred_prob, **mpars2)
    ddict = [{"metric_val": score_val, 'metric_name': mpars['metric_name']}]

    return ddict


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
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y.astype(np.int)

    if Xpred is None:
        data_pars['train'] = False
        Xpred              = get_dataset(data_pars, task_type="predict")

    # target column index
    coly_index = Xpred.columns.get_loc(data_pars["data_pars"]["coly"][0])
    # Models expect no target
    X = Xpred.iloc[:,:-1].values
    ypred, y_prob = model.model.classify(X, classcol=coly_index, return_prob=True)

    ypred         = post_process_fun(ypred)
    y_prob        = np.max(y_prob, axis=1)
    ypred_proba = y_prob  if compute_pars.get("compute_pars").get("probability", False) else None
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

####################################################################################################
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

        if task_type == "val":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]

            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



####################################################################################################
############ Test ##################################################################################
def test(n_sample = 100):
    """function test
    Args:
        n_sample :   
    Returns:
        
    """
    from adatasets import test_data_classifier_fake
    df, d = test_data_classifier_fake(nrows=500)
    colnum, colcat, coly = d['colnum'], d['colcat'], d['coly']
    df[coly].iloc[:50] = 1  ## Force 2 class

    ### Unique values
    colcat_unique = {  col: list(df[col].unique())  for col in colcat }

    X = df[colcat + colnum + [coly]]
    y = df[ [coly]]
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.05, random_state=2021, )#stratify=y) Regression no classes to stratify to
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021,)# stratify=y_train_full)
    log("X_train", X_train)


    def post_process_fun(y):   ### After prediction is done
        return  y.astype(np.int)

    def pre_process_fun(y):    ### Before the prediction is done
        return  int(y)


    ####################################################
    m = {
    'model_pars': {
        'model_class' :  "model_gefs.py::RandomForest"
        ,'model_pars' : {
            'cat': 10, 'n_estimators': 5
        }
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

    'compute_pars': {
        'compute_extra' :{
        },

        'compute_pars' :{
            'metric_list': ['accuracy_score','average_precision_score'],
            # Eval returns a probability
            "probability" : True,
            'epochs': 1,
        },

    },

    'data_pars': {
        "n_sample" : n_sample,
        "download_pars" : None,
        ### Raw data:  column input #####################
        "cols_input_type" : {
            "colnum" : colnum,
            "colcat" : colcat,
            "coly" : coly
        },
        ### family of columns for MODEL  ##################
         'cols_model_group': [ 'colnum_bin',   'colcat_bin', ]


        ### Filter data rows   ###########################
        ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 },

        ### Added continuous & sparse features groups ###
        'cols_model_type2': {
        },

        'data_pars' :{
                'cols_model_type': {
                },
                # Raw dataset, pre preprocessing
                "dataset_path" : "",
                "batch_size":128,   ### Mini Batch from data
                # Needed by getdataset
                "clean" : False,
                "data_path": "",

                'colcat_unique' : colcat_unique,
                'colcat'        : colcat,
                'colnum'        : colnum,
                'coly'          : coly,
                'colembed_dict' : None
        }
        ####### ACTUAL data Values #############################################################

        ,'train':   {'Xtrain': X_train, 'ytrain': y_train,
                    'Xtest':  X_valid, 'ytest': y_valid}
        ,'val':     {'X': X_valid, 'y': y_valid}
        ,'predict': {'X': X_valid}

    },

    'global_pars' :{
    }
    }
    ######## Run ###########################################
    test_helper(m['model_pars'], m['data_pars'], m['compute_pars'])


def test_helper(model_pars, data_pars, compute_pars):
    """function test_helper
    Args:
        model_pars:   
        data_pars:   
        compute_pars:   
    Returns:
        
    """
    global model, session
    root  = "ztmp/"
    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    log('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

    log('Evaluating the model..')
    # log(eval(data_pars=data_pars, compute_pars=compute_pars))


    # Open Issue with GeFs, not pickle-able and with no native saving mechanism
    # https://github.com/AlCorreia/GeFs/issues/7
    log('Saving model..')
    print("Can't save, open issue with GeFs : https://github.com/AlCorreia/GeFs/issues/7")
    # save(path= root + '/model_dir/')

    log('Load model..')
    # model, session = load_model(path= root + "/model_dir/")
    # log('Model successfully loaded!\n\n')

    log('Model architecture:')
    log(model.model)

    log('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')





####################################################################################################
def pd_colcat_get_catcount(df, colcat, coly, continuous_ids=None):
    """  Learns the number of categories in each variable and standardizes the df.
        ncat: numpy m The number of categories of each variable. One if the variable is continuous.
    """
    data = data.copy()
    ncat = np.ones(data.shape[1])
    if not classcol:
        classcol = data.shape[1]-1
    for i in range(data.shape[1]):
        if i != classcol and (i in continuous_ids or gef_is_continuous(data[:, i])):
            continue
        else:
            data[:, i] = data[:, i].astype(int)
            ncat[i] = max(data[:, i]) + 1
    return ncat



def is_continuous(v_array):
    """ Returns true if df was sampled from a continuous variables, and false
    """
    observed = v_array[~np.isnan(v_array)]  # not consider missing values for this.
    rules    = [np.min(observed) < -1,
                np.sum((observed) != np.round(observed)) > 0,
                len(np.unique(observed)) > min(30, len(observed) / 3)]
    if any(rules):
        return True
    else:
        return False


def test2():
    """function test2
    Args:
    Returns:
        
    """
    # Load toy dataset
    df   = pd.read_csv('https://raw.githubusercontent.com/arita37/GeFs/master/data/winequality_white.csv', sep=',')
    print(df.head(3).T, df.dtypes, df.shape)

    df = df.iloc[:500,:]
    colcat = "fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol".split(",")
    coly   = "quality"
    print(colcat, coly)

    cols = list(df.columns)
    icoly = cols.index(coly)

    ncat_white = learncats(df.values, classcol= icoly,   continuous_ids=[])
    ncat_white[-1] = 2   ### coly Force to be binary
    ncat_white = [  int(t) for t in ncat_white]
    print('ncat_white', ncat_white)


    X_train, y_train, data_train, data_test, mean, std = train_test_split2(df.values, ncat_white, 0.7)
    y_train = np.where(y_train <= 6, 0, 1)


    model_pars = {
        'n_estimators':10,
        'ncat': ncat_white
    }

    model_white = Model(model_pars=model_pars)
    model_white.model.fit(X_train, y_train)
    gef_white = model_white.model.topc(learnspn=np.Inf)

    log('gefs model test ok')




import numpy as np
import pandas as pd


# Auxiliary functions
def get_dummies(data):
    """function get_dummies
    Args:
        data:   
    Returns:
        
    """
    data = data.copy()
    if isinstance(data, pd.Series):
        data = pd.factorize(data)[0]
        return data
    for col in data.columns:
        data.loc[:, col] = pd.factorize(data[col])[0]
    return data


def learncats(data, classcol=None, continuous_ids=[]):
    """
        Learns the number of categories in each variable and standardizes the data.
        ----------
        data: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        classcol: int  The column index of the class variables (if any).
        continuous_ids: list of ints
            List containing the indices of known continuous variables. Useful for
            discrete data like age, which is better modeled as continuous.
        Returns
        -------
        ncat: numpy m  The number of categories of each variable. One if the variable is  continuous.
    """
    data = data.copy()
    ncat = np.ones(data.shape[1])
    if not classcol:
        classcol = data.shape[1]-1
    for i in range(data.shape[1]):
        if i != classcol and (i in continuous_ids or gef_is_continuous(data[:, i])):
            continue
        else:
            data[:, i] = data[:, i].astype(int)
            ncat[i] = max(data[:, i]) + 1
    return ncat

def gef_is_continuous(data):
    """
        Returns true if data was sampled from a continuous variables, and false
    """
    observed = data[~np.isnan(data)]  # not consider missing values for this.
    rules = [np.min(observed) < 0,
             np.sum((observed) != np.round(observed)) > 0,
             len(np.unique(observed)) > min(30, len(observed)/3)]
    if any(rules):
        return True
    else:
        return False



def gef_get_stats(data, ncat=None):
    """
        Compute univariate statistics for continuous variables.
    """
    data = data.copy()
    maxv = np.ones(data.shape[1])
    minv = np.zeros(data.shape[1])
    mean = np.zeros(data.shape[1])
    std = np.zeros(data.shape[1])
    if ncat is not None:
        for i in range(data.shape[1]):
            if ncat[i] == 1:
                maxv[i] = np.max(data[:, i])
                minv[i] = np.min(data[:, i])
                mean[i] = np.mean(data[:, i])
                std[i] = np.std(data[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                data[:, i] = (data[:, i] - minv[i])/(maxv[i] - minv[i])
    else:
        for i in range(data.shape[1]):
            if gef_is_continuous(data[:, i]):
                maxv[i] = np.max(data[:, i])
                minv[i] = np.min(data[:, i])
                mean[i] = np.mean(data[:, i])
                std[i] = np.std(data[:, i])
                assert maxv[i] != minv[i], 'Cannot have constant continuous variable in the data'
                data[:, i] = (data[:, i] - minv[i])/(maxv[i] - minv[i])
    return data, maxv, minv, mean, std


def gef_normalize_data(data, maxv, minv):
    """
        Normalizes the data given the maximum and minimum values of each variable.
    """
    data = data.copy()
    for v in range(data.shape[1]):
        if maxv[v] != minv[v]:
            data[:, v] = (data[:, v] - minv[v])/(maxv[v] - minv[v])
    return data


def gef_standardize_data(data, mean, std):
    """
        Standardizes the data given the mean and standard deviations values of
    """
    data = data.copy()
    for v in range(data.shape[1]):
        if std[v] > 0:
            data[:, v] = (data[:, v] - mean[v])/(std[v])
            #  Clip values more than 6 standard deviations from the mean
            data[:, v] = np.clip(data[:, v], -6, 6)
    return data





def train_test_split2(data, ncat, train_ratio=0.7, prep='std'):
    """function train_test_split2
    Args:
        data:   
        ncat:   
        train_ratio:   
        prep:   
    Returns:
        
    """
    assert train_ratio >= 0
    assert train_ratio <= 1
    shuffle = np.random.choice(range(data.shape[0]), data.shape[0], replace=False)
    data_train = data[shuffle[:int(train_ratio*data.shape[0])], :]
    data_test = data[shuffle[int(train_ratio*data.shape[0]):], :]
    if prep=='norm':
        data_train, maxv, minv, _, _, = gef_get_stats(data_train, ncat)
        data_test = gef_normalize_data(data_test, maxv, minv)
    elif prep=='std':
        _, maxv, minv, mean, std = gef_get_stats(data_train, ncat)
        data_train = gef_standardize_data(data_train, mean, std)
        data_test = gef_standardize_data(data_test, mean, std)

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    return X_train, X_test, y_train, y_test, data_train, data_test



if __name__ == "__main__":
    import fire
    fire.Fire()
    # test()








def test_converion():
    """
    General comments on the API¶
    There are four different functions to do classification with GeFs.

    classify
    classify_avg
    classify_lspn
    classify_avg_lspn
    The first two, classify and classify_avg, exploit class factorised leaves to run inference faster
    (propagate the probabilities of all classes at once). That, of course, only works if the leaves
    are class factorised (e.g. learnsp=np.Inf). Otherwise, one should use classify_lspn and classify_avg_lspn which work
     with any PC (in particular those with a LearnSPN network at the leaves, hence the name).

    The other important distinction is that avg methods assume a model learned as an ensemble and performs inference by 'averaging' the distribution of each of the base models. These are the methods that match the original Random Forest in terms of classification (with complete data, and class factorised leaves). In contrast, the other methods run inference as if the model is a single PC. One can interpret that as giving different weights to each of the base models according to the likelihood of the instance to be classified (base models under which the instance is more likely are given higher weights).
    This inference method is referred to as GeF+ in the paper, as it defines a mixture over the base models.


    :return:
    """
    ### RF ---> GeFs model
    from sklearn.ensemble import RandomForestClassifier
    from gefs.sklearn_utils import tree2pc, rf2pc


    # Define a synthetic dataset
    n_samples = 100
    n_features = 20
    n_classes = 2

    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=2, n_redundant=2, n_repeated=0,
                               n_classes=n_classes, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0,
                               hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
    # We need to specify the number of categories of each feature (with 1 for continuous features).
    ncat = np.ones(n_features+1)  # Here all features are continuous
    ncat[-1] = n_classes  # The class variable is naturally categorical

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    GeF = rf2pc(rf, X_train, y_train, ncat, learnspn=np.Inf, minstd=1., smoothing=1e-6)
    pred, prob = GeF.classify_avg(X_test, return_prob=True)







def train_test_split(data, ncat, train_ratio=0.7, prep='std'):
    """function train_test_split
    Args:
        data:   
        ncat:   
        train_ratio:   
        prep:   
    Returns:
        
    """
    assert train_ratio >= 0
    assert train_ratio <= 1
    shuffle = np.random.choice(range(data.shape[0]), data.shape[0], replace=False)
    data_train = data[shuffle[:int(train_ratio*data.shape[0])], :]
    data_test = data[shuffle[int(train_ratio*data.shape[0]):], :]
    if prep=='norm':
        data_train, maxv, minv, _, _, = gef_get_stats(data_train, ncat)
        data_test = gef_normalize_data(data_test, maxv, minv)
    elif prep=='std':
        _, maxv, minv, mean, std = gef_get_stats(data_train, ncat)
        data_train = gef_standardize_data(data_train, mean, std)
        data_test = gef_standardize_data(data_test, mean, std)

    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]

    return X_train, X_test, y_train, y_test, data_train, data_test


# Preprocessing functions
def adult(data):
    """function adult
    Args:
        data:   
    Returns:
        
    """
    cat_cols = ['workclass', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country', 'y']
    cont_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def australia(data):
    """function australia
    Args:
        data:   
    Returns:
        
    """
    cat_cols = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13', 'class']
    cont_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    data = data.replace('?', np.nan)
    ncat = learncats(data.values.astype(float), classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def bank(data):
    """function bank
    Args:
        data:   
    Returns:
        
    """
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome', 'y']
    cont_cols = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate',
                'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    data.loc[:, 'pdays'] = np.where(data['pdays']==999, 0, 1)
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def credit(data):
    """function credit
    Args:
        data:   
    Returns:
        
    """
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'default payment next month']
    cont_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def electricity(data):
    """function electricity
    Args:
        data:   
    Returns:
        
    """
    cat_cols = ['day', 'class']
    cont_cols = ['date', 'period', 'nswprice', 'nswdemand', 'vicprice',
       'vicdemand', 'transfer']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def segment(data):
    """function segment
    Args:
        data:   
    Returns:
        
    """
    data = data.drop(columns=['region.centroid.col', 'region.pixel.count'])
    cat_cols = ['short.line.density.5', 'short.line.density.2', 'class']
    cont_cols = ['region.centroid.row', 'vedge.mean', 'vegde.sd', 'hedge.mean', 'hedge.sd',
                 'intensity.mean', 'rawred.mean', 'rawblue.mean', 'rawgreen.mean', 'exred.mean', 'exblue.mean' ,
                 'exgreen.mean', 'value.mean', 'saturation.mean', 'hue.mean']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def german(data):
    """function german
    Args:
        data:   
    Returns:
        
    """
    cat_cols = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20]
    cont_cols = [1, 4, 7, 10, 12, 15, 17]
    data.iloc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=cont_cols)
    return data.values.astype(float), ncat


def vowel(data):
    """function vowel
    Args:
        data:   
    Returns:
        
    """
    cat_cols = ['Speaker_Number', 'Sex', 'Class']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=data.shape[1]-1)
    return data.values.astype(float), ncat


def cmc(data):
    """function cmc
    Args:
        data:   
    Returns:
        
    """
    cat_cols = ['Wifes_education', 'Husbands_education', 'Wifes_religion', 'Wifes_now_working%3F',
            'Husbands_occupation', 'Standard-of-living_index', 'Media_exposure', 'Contraceptive_method_used']
    cont_cols = ['Wifes_age', 'Number_of_children_ever_born']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=data.shape[1]-1)
    return data.values.astype(float), ncat


def get_data(name):
    """function get_data
    Args:
        name:   
    Returns:
        
    """
    if 'wine' in name:
        data_red = pd.read_csv('../data/winequality_red.csv')
        data_white = pd.read_csv('../data/winequality_white.csv')
        data = pd.concat([data_red, data_white]).values
        data[:, -1] = np.where(data[:, -1] <= 6, 0, 1)
        ncat = learncats(data, classcol=data.shape[1]-1)
    elif 'bank' in name:
        data = pd.read_csv('../data/bank-additional-full.csv', sep=';')
        data, ncat = bank(data)
    elif 'segment' in name:
        data = pd.read_csv('../data/segment.csv')
        data, ncat = segment(data)
    elif 'german' in name:
        data = pd.read_csv('../data/german.csv', sep=' ', header=None)
        data, ncat = german(data)
    elif 'vehicle' in name:
        data = pd.read_csv('../data/vehicle.csv')
        data['Class'] = get_dummies(data['Class'])
        ncat = np.ones(data.shape[1])
        ncat[-1] = len(np.unique(data['Class']))
        data = data.values.astype(float)
    elif 'vowel' in name:
        data = pd.read_csv('../data/vowel.csv')
        data, ncat = vowel(data)
    elif 'authent' in name:
        data = pd.read_csv('../data/authent.csv')
        data['Class'] = get_dummies(data['Class'])
        ncat = learncats(data.values).astype(int)
        data = data.values.astype(float)
    elif 'diabetes' in name:
        data = pd.read_csv('../data/diabetes.csv')
        data['class'] = get_dummies(data['class'])
        ncat = learncats(data.values,
                         continuous_ids=[0] # Force first variable to be continuous
                         ).astype(int)
        data = data.values.astype(float)
    elif 'cmc' in name:
        data = pd.read_csv('../data/cmc.csv')
        data, ncat = cmc(data)
    elif 'electricity' in name:
        data = pd.read_csv('../data/electricity.csv')
        data, ncat = electricity(data)
    elif 'gesture' in name:
        data = pd.read_csv('../data/gesture.csv')
        data['Phase'] = get_dummies(data['Phase'])
        data = data.values.astype(float)
        ncat = np.ones(data.shape[1])
        ncat[-1] = 5
    elif 'breast' in name:
        data = pd.read_csv('../data/wdbc.csv')
        data['Class'] = get_dummies(data['Class'])
        data = data.values.astype(float)
        ncat = np.ones(data.shape[1])
        ncat[-1] = 2
    elif 'krvskp' in name:
        data = pd.read_csv('../data/kr-vs-kp.csv')
        data = get_dummies(data)
        ncat = learncats(data.values)
        data = data.values.astype(float)
    elif 'dna' in name:
        data = pd.read_csv('../data/dna.csv')
        data = get_dummies(data).values.astype(float)
        ncat = learncats(data)
    elif 'robot' in name:
        data = pd.read_csv('../data/robot.csv')
        data['Class'] = get_dummies(data['Class'])
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'mice' in name:
        data = pd.read_csv('../data/miceprotein.csv')
        data['class'] = get_dummies(data['class'])
        data = data.replace('?', np.nan)
        data = data.drop(['MouseID', 'Genotype', 'Treatment', 'Behavior'], axis=1)
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'dresses' in name:
        data = pd.read_csv('../data/dresses.csv')
        data = data.replace('?', np.nan)
        data = get_dummies(data)
        data = data.values.astype(float)
        data[data < 0] = np.nan
        ncat = learncats(data)
    elif 'texture' in name:
        data = pd.read_csv('../data/texture.csv')
        data['Class'] = get_dummies(data['Class'])
        data = data.values.astype(float)
        ncat = np.ones(data.shape[1])
        ncat[-1] = 11
    elif 'splice' in name:
        data = pd.read_csv('../data/splice.csv')
        data = data.drop('Instance_name', axis=1)
        data = get_dummies(data).values.astype(float)
        ncat = learncats(data)
    elif 'jungle' in name:
        data = pd.read_csv('../data/jungle.csv')
        data = get_dummies(data)
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'phishing' in name:
        data = pd.read_csv('../data/phishing.csv')
        data = get_dummies(data)
        data = data.values.astype(float)
        ncat = learncats(data)
    elif 'fashion' in name:
        data = pd.read_csv('../data/fashion.csv')
        data = data.values.astype(np.float64)
        ncat = np.ones(data.shape[1]).astype(np.int64)
        ncat[-1] = 10
    elif 'mnist' in name:
        data = pd.read_csv('../data/mnist.csv')
        data = data.values.astype(np.float64)
        ncat = np.ones(data.shape[1]).astype(np.int64)
        ncat[-1] = 10
    else:
        print("Sorry, dataset {} is not available.".format(name))
        print("You have to provide the data and run the appropriate pre-processing steps yourself.")
        raise ValueError

    return data, ncat








"""
def test_dataset_classi_fake(nrows=500):
    from sklearn import datasets as sklearn_datasets
    ndim=11
    coly   = ['y']
    colnum = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(
        n_samples=1000,
        n_features=ndim,
        # No n_targets param for make_classification
        # n_targets=1,
        # Fake dataset, classification on 2 classes
        n_classes=2,
        # In classification, n_informative should be less than n_features
        n_informative=ndim - 2
    )
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)
    for ci in colcat :
      df[colcat] = np.random.randint(2, len(df))
    return df, colnum, colcat, coly
"""






"""
python model_gef.py test_model
    def learncats(data, classcol=None, continuous_ids=[]):
            Learns the number of categories in each variable and standardizes the data.
            Parameters
            ----------
            data: numpy n x m
                Numpy array comprising n realisations (instances) of m variables.
            classcol: int
                The column index of the class variables (if any).
            continuous_ids: list of ints
                List containing the indices of known continuous variables. Useful for
                discrete data like age, which is better modeled as continuous.
            Returns
            -------
            ncat: numpy m
                The number of categories of each variable. One if the variable is
                continuous.
        data = data.copy()
        ncat = np.ones(data.shape[1])
        if not classcol:
            classcol = data.shape[1] - 1
        for i in range(data.shape[1]):
            if i != classcol and (i in continuous_ids or gef_is_continuous(data[:, i])):
                continue
            else:
                data[:, i] = data[:, i].astype(int)
                ncat[i] = max(data[:, i]) + 1
        return ncat
"""