# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from jsoncomment import JsonComment ; json = JsonComment()
import pandas as pd

from lightgbm import LGBMModel


from mlmodels.util import  os_package_root_path, log, path_norm, get_model_uri


####################################################################################################
VERBOSE = False
MODEL_URI = get_model_uri(__file__)



####################################################################################################
global model, session

def init(*kw, **kwargs) :
    global model, session
    model   = Model(*kw, **kwargs)
    session = None


class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        """
         lightgbm.LGBMModel(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000,
          objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0,
          colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, importance_type='split', **kwargs)
        """
        self.model_pars, self.data_pars, self.compute_pars = copy.deepcopy(model_pars), copy.deepcopy(data_pars), copy.deepcopy(compute_pars)
        if model_pars is None :
            self.model = None
        else :        
          #### Specific to SKLEARN model
          mpars      = model_pars.get('model_pars', {})  
          self.model = LGBMModel(**mpars)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    X (array-like or sparse matrix of shape = [n_samples, n_features]) – Input feature matrix.
y (array-like of shape = [n_samples]) – The target values (class labels in classification, real numbers in regression).
sample_weight (array-like of shape = [n_samples] or None, optional (default=None)) – Weights of training data.
init_score (array-like of shape = [n_samples] or None, optional (default=None)) – Init score of training data.
group (array-like or None, optional (default=None)) – Group data of training data.
eval_set (list or None, optional (default=None)) – A list of (X, y) tuple pairs to use as validation sets.
eval_names (list of strings or None, optional (default=None)) – Names of eval_set.
eval_sample_weight (list of arrays or None, optional (default=None)) – Weights of eval data.
eval_class_weight (list or None, optional (default=None)) – Class weights of eval data.
eval_init_score (list of arrays or None, optional (default=None)) – Init score of eval data.
eval_group (list of arrays or None, optional (default=None)) – Group data of eval data.
eval_metric (string, list of strings, callable or None, optional (default=None)) – If string, it should be a built-in evaluation metric to use. If callable, it should be a custom evaluation metric, see note below for more details. In either case, the metric from the model parameters will be evaluated and used as well. Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier, ‘ndcg’ for LGBMRanker.
early_stopping_rounds (int or None, optional (default=None)) – Activates early stopping. The model will train until the validation score stops improving. Validation score needs to improve at least every early_stopping_rounds round(s) to continue training. Requires at least one validation data and one metric. If there’s more than one, will check all of them. But the training data is ignored anyway. To check only the first metric, set the first_metric_only parameter to True in additional parameters **kwargs of the model constructor.
verbose (bool or int, optional (default=True)) –
Requires at least one evaluation data. If True, the eval metric on the eval set is printed at each boosting stage. If int, the eval metric on the eval set is printed at every verbose boosting stage. The last boosting stage or the boosting stage found by using early_stopping_rounds is also printed.
    """
    global model, session
    Xtrain, ytrain, Xtest,  ytest = get_dataset(data_pars, task_type="train")
    cpars = compute_pars.get("compute_pars", {})
    model.model = model.model.fit(Xtrain, ytrain, **cpars)



def evaluate(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    Xval, yval = get_dataset(data_pars, task_type="eval")

    #### Do prediction
    ypred = model.model.predict(Xval)    
    if verbose : print(data_pars)


    ddict = {}
    metric_score_name = compute_pars.get('metric_eval') 
    if metric_score_name is None :
        raise Exception("  compute_pars.get('metric_eval')    requires a sklearn metrics name")
    
    
    from sklearn.metrics import roc_auc_score,mean_squared_error,accuracy_score
    if metric_score_name == "roc_auc_score" :
        score = roc_auc_score(yval, ypred)
        ddict[metric_score_name] = score

    if metric_score_name == "mean_squared_error" :
        score = mean_squared_error(yval, ypred)
        ddict[metric_score_name] = score
        
    if metric_score_name == "accuracy_score":
        ypred = ypred.argmax(axis=1)
        score = accuracy_score(yval, ypred)
        ddict[metric_score_name] = score

    else :
        from mlmodels.metrics import metrics_eval
        ddict = metrics_eval( metric_list=[ metric_score_name ], ytrue= yval, ypred= ypred, 
                              ypred_proba=None, return_dict=1   )
        return ddict

    return ddict



def predict(data_pars=None, compute_pars=None, out_pars=None, **kw):
    global model, session
    Xpred = get_dataset(data_pars, task_type="pred")
    ppars = compute_pars.get("pred_pars", {})
    ypred = model.model.predict(Xpred, **ppars)


    ### Return val
    if compute_pars.get("return_pred_not") is not None:
        return ypred



def reset():
    global model, session
    model, session = None, None


def save(path=None, info={}):
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)
    
    filename = "model.pkl"
    pickle.dump(model, open( f"{path}/{filename}" , mode='wb')) #, protocol=pickle.HIGHEST_PROTOCOL )
    
    filename = "info.pkl"
    pickle.dump(info, open( f"{path}/{filename}" , mode='wb')) #,protocol=pickle.HIGHEST_PROTOCOL )   
    
    

def load(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open( f"{path}/model.pkl", mode='rb') )
 
    model = Model() # Empty model    
    model.model        = model0.model
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars    
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl") :   
      if not "model.pkl" in fp :  
        obj = pickle.load(open( fp, mode='rb') )
        key = fp.split("/")[-1]
        dd[key] = obj
    return dd


####################################################################################################
def get_dataset(data_pars=None, **kw):
    """
            "data_pars": {
            "data_info": {
                "data_path": "dataset/tabular/",
                "dataset":   "csv titanic",
                "data_type": "csv",
                // "batch_size": 10,
                "task_type": "train"    // "pred"  "eval"
            },
            "preprocessors": [
            {
                "uri"  : "mlmodels.preprocess.generic.pandas_reader",
                "args"  : {
                           "task_type": "train", 
                           "path" : "dataset/tabular/titanic_train_preprocessed.csv",
                           "colX": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S", "Title"],
                           "coly": "Survived",
                           "path_eval" : "",
                           "train_split_ratio" : 0.8,
                          },
                "out"   :  [  "Xtrain", "ytrain", "Xtest", "ytest" ]          
            }]    
            },
    """ 
    from mlmodels.dataloader import DataLoader

    if "dataset" not in data_pars['data_info'] :
      raise Exception("need dataset in data_info")        

    task   = data_pars['task_type']
    loader = DataLoader(data_pars)
    loader.compute()

    if task == 'train' :
       Xtrain, ytrain, Xtest, ytest  = loader.get_data()       
       return Xtrain, ytrain, Xtest, ytest

    if task == 'eval' :
       Xtest, ytest  = loader.get_data()       
       return Xtest, ytest

    if task == 'pred' :
       X  = loader.get_data()       
       return X

    else :
       raise Exception(f"Unknown task {task}") 



def get_dataset2(data_pars=None, **kw):
    """
      JSON data_pars to get dataset
      "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
      "size": [0, 0, 6], "output_size": [0, 6] },

      mode= train, eval, predict
           test


    """
    d = data_pars

    if d['mode'] == 'test_data':
        from sklearn.datasets import  make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain,  ytrain, Xtest, ytest


    if d['mode'] == 'train'  :
        from sklearn.model_selection import train_test_split

        path  =  path_norm( d.get('train_path', d.get('path', None)) )
        path2 =  path_norm( d.get('test_path', None) )
        
        if path2 is None :
           df  = pd.read_csv(path)
           X = df[d['colX']].values
           y = df[d['coly']].values

           Xtrain, Xtest, ytrain, ytest =  train_test_split(X, y)
           return Xtrain,  ytrain, Xtest, ytest


        df     = pd.read_csv(path)
        Xtrain = df[d['colX']].values
        ytrain = df[d['coly']].values

        df     = pd.read_csv(path2 )
        Xtest  = df[d['colX']].values
        ytest  = df[d['coly']].values

        return Xtrain,  ytrain, Xtest, ytest


    if d['mode'] == 'eval'  :
        path  = path_norm( d.get('test_path', d.get('path', None)) )
        df    = pd.read_csv(path)
        Xtest = df[d['colX']].values
        ytest = df[d['coly']].values

        return None,  None, Xtest, ytest


    if d['mode'] == 'predict'  :
        path  = path_norm( d.get('test_path', d.get('path', None)) )
        df    = pd.read_csv(path)
        Xtest = df[d['colX']].values
        ytest = df[d['coly']].values

        return None,  None, Xtest, None


    else:
        df  = pd.read_csv(data_pars['path'])
        Xtest = df[data_pars['colX']]
        return None, None, Xtest, None



def get_params(param_pars={}, **kw):
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    """
    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path  = path_norm( "dataset/tabular/titanic_train_preprocessed.csv"  )   
        out_path   = path_norm( "ztest/model_sklearn/model_lightgbm/" )
        model_path = os.path.join(out_path , "model.pkl")

        data_pars = {'mode': 'test', 'path': data_path, 'data_type' : 'pandas' }
        model_pars = {"objective":  "regression", "max_depth" : 4 , "random_state":0}
        compute_pars = { "meta" : 1,  "compute_pars":{}}
        out_pars = {'path' : out_path, "model_path": model_path}

        return model_pars, data_pars, compute_pars, out_pars
    """
    else:
        raise Exception(f"Not support choice {choice} yet")


################################################################################################
########## Tests are normalized Do not Change ##################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    log("### Local test ")

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)


    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)


    log("#### Model init   ##################################################")
    init(model_pars, data_pars, compute_pars)


    log("#### Model  fit   ##################################################")
    fit(data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    ypred = predict(data_pars, compute_pars, out_pars)


    log("#### Metrics   #####################################################")
    metrics_val = evaluate(data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")



    log("#### Save   ########################################################")
    save_pars = {"path" : out_pars['model_path'] }
    save(save_pars)


    log("#### Load   ########################################################")
    global model, session
    load_pars = {"path" : out_pars['model_path'] }
    model, session = load( load_pars)


    log("#### Save/Load   ###################################################")    
    ypred = predict(data_pars, compute_pars, out_pars)
    
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    log(model.model)



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
    test_module(model_uri=MODEL_URI, param_pars=param_pars)

    ##### get of get_params
    # choice      = pp['choice']
    # config_mode = pp['config_mode']
    # data_path   = pp['data_path']

    ####    test_api(model_uri="model_xxxx/yyyy.py", param_pars=None)
    from mlmodels.models import test_api

    param_pars = {'choice': "test01", 'config_mode': 'test', 'data_path': '/dataset/'}
    test_api(model_uri=MODEL_URI, param_pars=param_pars)


"""
Generic template for new model.
Check parameters template in models_config.json
boosting_type (string, optional (default='gbdt')) – ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
num_leaves (int, optional (default=31)) – Maximum tree leaves for base learners.
max_depth (int, optional (default=-1)) – Maximum tree depth for base learners, <=0 means no limit.
learning_rate (float, optional (default=0.1)) – Boosting learning rate. You can use callbacks parameter of fit method to shrink/adapt learning rate in training using reset_parameter callback. Note, that this will ignore the learning_rate argument in training.
n_estimators (int, optional (default=100)) – Number of boosted trees to fit.
subsample_for_bin (int, optional (default=200000)) – Number of samples for constructing bins.
objective (string, callable or None, optional (default=None)) – Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below). Default: ‘regression’ for LGBMRegressor, ‘binary’ or ‘multiclass’ for LGBMClassifier, ‘lambdarank’ for LGBMRanker.
class_weight (dict, 'balanced' or None, optional (default=None)) – Weights associated with classes in the form {class_label: weight}. Use this parameter only for multi-class classification task; for binary classification task you may use is_unbalance or scale_pos_weight parameters. Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities. You may want to consider performing probability calibration (https://scikit-learn.org/stable/modules/calibration.html) of your model. The ‘balanced’ mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)). If None, all classes are supposed to have weight one. Note, that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
min_split_gain (float, optional (default=0.)) – Minimum loss reduction required to make a further partition on a leaf node of the tree.
min_child_weight (float, optional (default=1e-3)) – Minimum sum of instance weight (hessian) needed in a child (leaf).
min_child_samples (int, optional (default=20)) – Minimum number of data needed in a child (leaf).
subsample (float, optional (default=1.)) – Subsample ratio of the training instance.
subsample_freq (int, optional (default=0)) – Frequence of subsample, <=0 means no enable.
colsample_bytree (float, optional (default=1.)) – Subsample ratio of columns when constructing each tree.
reg_alpha (float, optional (default=0.)) – L1 regularization term on weights.
reg_lambda (float, optional (default=0.)) – L2 regularization term on weights.
random_state (int, RandomState object or None, optional (default=None)) – Random number seed. If int, this number is used to seed the C++ code. If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code. If None, default seeds in C++ code are used.
n_jobs (int, optional (default=-1)) – Number of parallel threads.
silent (bool, optional (default=True)) – Whether to print messages while running boosting.
importance_type (string, optional (default='split')) – The type of feature importance to be filled into feature_importances_. If ‘split’, result contains numbers of times the feature is used in a model. If ‘gain’, result contains total gains of splits which use the feature.
**kwargs –
Other parameters for the model. Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.
https://github.com/optuna/optuna/blob/master/examples/lightgbm_tuner_simple.py#L22
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM tuner.
In this example, we optimize the validation log loss of cancer detection.
You can execute this code directly.
    $ python lightgbm_tuner_simple.py
if __name__ == '__main__':
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
    }
    best_params, tuning_history = dict(), list()
    model = lgb.train(params,
                      dtrain,
                      valid_sets=[dtrain, dval],
                      best_params=best_params,
                      tuning_history=tuning_history,
                      verbose_eval=100,
                      early_stopping_rounds=100,
                      )
    prediction = np.rint(model.predict(val_x, num_iteration=model.best_iteration))
    accuracy = accuracy_score(val_y, prediction)
    print('Number of finished trials: {}'.format(len(tuning_history)))
    print('Best params:', best_params)
    print('  Accuracy = {}'.format(accuracy))
    print('  Params: ')
    for key, value in best_params.items():
        print('    {}: {}'.format(key, value))
"""
