# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for automatic ML
  tpot
  mlbox


"""
import copy
import os
from collections import OrderedDict
from importlib import import_module

import numpy as np
import pandas as pd
import scipy as sci
from dateutil.parser import parse

import sklearn as sk
from matplotlib import pyplot as plt

from sklearn.preprocessing import train_test_split
####################################################################################################


from util import save_all


def import_(abs_module_path, class_name=None):
    try:
        module_object = import_module(abs_module_path)
        print("imported", module_object)
        if class_name is None:
            return module_object
        target_class = getattr(module_object, class_name)
        return target_class
    except Exception as e:
        print(abs_module_path, class_name, e)


# from attrdict import AttrDict as dict2
# from kmodes.kmodes import KModes
# from tabulate import tabulate


########### Dynamic Import   #######################################################################
# EvolutionaryAlgorithmSearchCV = import_("evolutionary_search", "EvolutionaryAlgorithmSearchCV")
esearch = import_("evolutionary_search")



####################################################################################################
DIRCWD = os.getcwd()
print("os.getcwd", os.getcwd())

class dict2(object):
    def __init__(self, d):
        self.__dict__ = d





############# TPOT  ################################################################################
def model_auto_tpot(
    df,
    colX, coly,
    outfolder="aaserialize/",
    model_type="regressor/classifier",
    train_size=0.5,
    generation=1,
    population_size=5,
    verbosity=2,
):
    """ Automatic training of Xmat--->Y, Generate SKlearn code in outfile
      Very Slow Process, use lower number of Sample
  :param Xmat:
  :param y:
  :param outfolder:
  :param model_type:
  :param train_size:
  :param generation:
  :param population_size:
  :param verbosity:
  :return:
    """
    tpot = import_("tpot")

    X = df[colX].values
    y = df[coly].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.5)

    if model_type == "regressor":
        clf = tpot.TPOTRegressor(
            generations=generation, population_size=population_size, verbosity=verbosity
        )
    elif model_type == "classifier":
        clf = tpot.TPOTClassifier(
            generations=generation, population_size=population_size, verbosity=verbosity
        )


    print("Start")
    clf.fit(X_train, y_train)
    
    score = tpot.score(X_test, y_test)
    print("score", score)

    file1 =  outfolder + "/tpot_regression_pipeline_" + str(np.random.randint(1000, 9999)) + ".py"
    tpot.export(file1)
    return file1




######################MLBOX  ######################################################################
def model_auto_mlbox( filepath= [ "train.csv", "test.csv" ],
    colX=None, coly=None,
    do="predict",
    outfolder="aaserialize/",
    model_type="regressor/classifier",
    params={ "csv_seprator" : ",", "train_size" : 0.5, "score_metric" : "accuracy",
             "n_folds": 3, "n_step": 10},
    param_space =  {
        'est__strategy':{"search":"choice",                         "space":["LightGBM"]},
        'est__n_estimators':{"search":"choice",                     "space":[150]},
        'est__colsample_bytree':{"search":"uniform",                "space":[0.8,0.95]},
        'est__subsample':{"search":"uniform",                       "space":[0.8,0.95]},
        'est__max_depth':{"search":"choice",                        "space":[5,6,7,8,9]},
        'est__learning_rate':{"search":"choice",                    "space":[0.07]}
    },
    generation=1,
    population_size=5,
    verbosity=2,
):
    """
      Using mlbox
      https://www.analyticsvidhya.com/blog/2017/07/mlbox-library-automated-machine-learning/


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    colX : TYPE
        DESCRIPTION.
    coly : TYPE
        DESCRIPTION.
    outfolder : TYPE, optional
        DESCRIPTION. The default is "aaserialize/".
    model_type : TYPE, optional
        DESCRIPTION. The default is "regressor/classifier".
    params : TYPE, optional
        DESCRIPTION. The default is {"train_size" : 0.5}.
    generation : TYPE, optional
        DESCRIPTION. The default is 1.
    population_size : TYPE, optional
        DESCRIPTION. The default is 5.
    verbosity : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    """
    from mlbox.preprocessing import Reader,Drift_thresholder
    from mlbox.optimisation import Optimiser
    from mlbox.prediction import Predictor

    p = dict2(params)


    ## Pre-process
    """
    df (dict, default = None) –
    Dataset dictionary. Must contain keys and values:

    ”train”: pandas DataFrame for the train set.
    ”test” : pandas DataFrame for the test set.
    ”target” : encoded pandas Serie for the target on train set (with dtype=’float’ for a regression or dtype=’int’ for a classification). Indexes should match the train set.

    """
    rd = Reader(sep = p.csv_separator)
    df = rd.train_test_split( filepath, coly)   # Reading and preprocessing (dates, ...)
    dft = Drift_thresholder()
    df = dft.fit_transform(df)      # Removing non-stable features (like ID,...)


    ### Optimal parameter
    # score_rmse = make_scorer(lambda y_true, y_pred: np.sqrt(np.sum((y_true - y_pred)**2)/len(y_true)), greater_is_better=False, needs_proba=False)
    #                    opt = Optimiser(scoring = rmse, n_folds = 3)

    opt = Optimiser(scoring = p.score_metric, n_folds = p.n_folds)
    param_optim = opt.optimise(param_space, df, p.n_step)


    if do == "prediction" :
      clf = Predictor(to_path= outfolder, verbose=True)

      #Fit and predict and save on disk
      clf.fit_predict(param_optim, df)

      # Load the predictions
      preds = pd.read_csv("save/"+coly+"_predictions.csv")
      print(preds.shape, preds.head(5))




      """
      submit = pd.read_csv("../input/gendermodel.csv", sep=',')
      submit[coly] =  preds[coly+"_predicted"].values

      submit.to_csv( outfolder + "/mlbox.csv", index=False)
      """



######################  ATOML GS  ##################################################################
def model_auto_automlgs( filepath= [ "train.csv", "test.csv" ],
    colX=None, coly=None,
    do="predict",
    outfolder="aaserialize/",
    model_type="regressor/classifier",
    params={ "csv_seprator" : ",", "train_size" : 0.5, "score_metric" : "accuracy",
             "n_folds": 3, "n_step": 10},
    param_space =  {
        'est__strategy':{"search":"choice",                         "space":["LightGBM"]},
        'est__n_estimators':{"search":"choice",                     "space":[150]},
        'est__colsample_bytree':{"search":"uniform",                "space":[0.8,0.95]},
        'est__subsample':{"search":"uniform",                       "space":[0.8,0.95]},
        'est__max_depth':{"search":"choice",                        "space":[5,6,7,8,9]},
        'est__learning_rate':{"search":"choice",                    "space":[0.07]}
    },
    generation=1,
    population_size=5,
    verbosity=2,
):
    """
     https://github.com/minimaxir/automl-gs

     https://github.com/minimaxir/automl-gs

     automl_gs titanic.csv Survived --framework xgboost --num_trials 1000


csv_path: Path to the CSV file (must be in the current directory) [Required]
target_field: Target field to predict [Required]
target_metric: Target metric to optimize [Default: Automatically determined depending on problem type]
framework: Machine learning framework to use [Default: 'tensorflow']
config_name: Name of the model (if you want to train models with different names) [Default: 'automl']
num_trials: Number of trials / different hyperparameter combos to test. [Default: 100]
split: Train-validation split when training the models [Default: 0.7]
num_epochs: Number of epochs / passes through the data when training the models. [Default: 20]
col_types: Dictionary of fields:data types to use to override automl-gs's guesses. (only when using in Python) [Default: {}]
gpu: For non-Tensorflow frameworks and Pascal-or-later GPUs, boolean to determine whether to use GPU-optimized training methods (TensorFlow can detect it automatically) [Default: False]
tpu_address: For TensorFlow, hardware address of the TPU on the system. [Default: None]   

   The output of the automl-gs training is:

A timestamped folder (e.g. automl_tensorflow_20190317_020434) with contains:
model.py: The generated model file.
pipeline.py: The generated pipeline file.
requirements.txt: The generated requirements file.
/encoders: A folder containing JSON-serialized encoder files
/metadata: A folder containing training statistics + other cool stuff not yet implemented!
The model itself (format depends on framework)
automl_results.csv: A CSV containing the training results after each epoch and the hyperparameters used to train at that time.
Once the training is done, you can run the generated files from the command line within the generated folder above.


    Parameters
    ----------
    filepath : TYPE, optional
        DESCRIPTION. The default is [ "train.csv", "test.csv" ].
    colX : TYPE, optional
        DESCRIPTION. The default is None.
    coly : TYPE, optional
        DESCRIPTION. The default is None.
    do : TYPE, optional
        DESCRIPTION. The default is "predict".
    outfolder : TYPE, optional
        DESCRIPTION. The default is "aaserialize/".
    model_type : TYPE, optional
        DESCRIPTION. The default is "regressor/classifier".
    params : TYPE, optional
        DESCRIPTION. The default is { "csv_seprator" : ",", "train_size" : 0.5, "score_metric" : "accuracy",             "n_folds": 3, "n_step": 10}.
    param_space : TYPE, optional
        DESCRIPTION. The default is {        'est__strategy':{"search":"choice",                         "space":["LightGBM"]},        'est__n_estimators':{"search":"choice",                     "space":[150]},        'est__colsample_bytree':{"search":"uniform",                "space":[0.8,0.95]},        'est__subsample':{"search":"uniform",                       "space":[0.8,0.95]},        'est__max_depth':{"search":"choice",                        "space":[5,6,7,8,9]},        'est__learning_rate':{"search":"choice",                    "space":[0.07]}    }.
    generation : TYPE, optional
        DESCRIPTION. The default is 1.
    population_size : TYPE, optional
        DESCRIPTION. The default is 5.
    verbosity : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    """

    p = dict2(params)

    from automl_gs import automl_grid_search
    automl_grid_search( filepath, coly, framework = p.framework,
                        model_name=p.model_name,
                        col_type=p.col_types,
                        num_epochs= p.num_epochs
                        gpu=False)


    """
      csv_path: Path to the CSV file (must be in the current directory) [Required]
target_field: Target field to predict [Required]
target_metric: Target metric to optimize [Default: Automatically determined depending on problem type]
framework: Machine learning framework to use [Default: 'tensorflow']
config_name: Name of the model (if you want to train models with different names) [Default: 'automl']
num_trials: Number of trials / different hyperparameter combos to test. [Default: 100]
split: Train-validation split when training the models [Default: 0.7]
num_epochs: Number of epochs / passes through the data when training the models. [Default: 20]
col_types: Dictionary of fields:data types to use to override automl-gs's guesses. (only when using in Python) [Default: {}]
gpu: For non-Tensorflow frameworks and Pascal-or-later GPUs, boolean to determine whether to use GPU-optimized training methods (TensorFlow can detect it automatically) [Default: False]
tpu_address: For TensorFlow, hardware address of the TPU on the system. [Default: None]   


    """












"""


from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

paths = ["../input/train.csv","../input/test.csv"]
target_name = "Survived"


rd = Reader(sep = ",")
df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)


dft = Drift_thresholder()
df = dft.fit_transform(df)   #removing non-stable features (like ID,...)


opt = Optimiser(scoring = "accuracy", n_folds = 5)



space = {
    
        'est__strategy':{"search":"choice",
                                  "space":["LightGBM"]},    
        'est__n_estimators':{"search":"choice",
                                  "space":[150]},    
        'est__colsample_bytree':{"search":"uniform",
                                  "space":[0.8,0.95]},
        'est__subsample':{"search":"uniform",
                                  "space":[0.8,0.95]},
        'est__max_depth':{"search":"choice",
                                  "space":[5,6,7,8,9]},
        'est__learning_rate':{"search":"choice",
                                  "space":[0.07]} 
    
        }

params = opt.optimise(space, df,15)



prd = Predictor()
prd.fit_predict(params, df)


submit = pd.read_csv("../input/gendermodel.csv",sep=',')
preds = pd.read_csv("save/"+target_name+"_predictions.csv")

submit[target_name] =  preds[target_name+"_predicted"].values

submit.to_csv("mlbox.csv", index=False)



Below is the table of the four broad optimisations that are done in the MLBox library with terms to the right of hyphen that can be optimised for different values.

Missing Values Encoder(ne) – numerical_strategy (when the column to be imputed is a continuous column eg- mean, median etc), categorical_strategy(when the column to be imputed is a categorical column e.g.- NaN values etc)

Categorical Values Encoder(ce)– strategy (method of encoding categorical variables e.g.- label_encoding, dummification, random_projection, entity_embedding)

Feature Selector(fs)– strategy (different methods for feature selection e.g. l1, variance, rf_feature_importance), threshold (the percentage of features to be discarded)

Estimator(est)–strategy (different algorithms that can be used as estimators eg- LightGBM, xgboost etc.), **params(parameters specific to the algorithm being used eg- max_depth, n_estimators etc.)

Let us take an example and create a hyperparameter space to be optimised. Let us state all the parameters that I want to optimise:

Algorithm to be used- LightGBM
LightGBM max_depth-[3,5,7,9]
LightGBM n_estimators-[250,500,700,1000]
Feature selection-[variance, l1, random forest feature importance]
Missing values imputation – numerical(mean,median),categorical(NAN values)
categorical values encoder- label encoding, entity embedding and random projection

Let us now create our hyper-parameter space. Before that, remember, hyper-parameter is a dictionary of key and value pairs where value is also a dictionary given by the syntax
{“search”:strategy,”space”:list}, where strategy can be either “choice” or “uniform” and list is the list of values.

space={'ne__numerical_strategy':{"search":"choice","space":['mean','median']},
'ne__categorical_strategy':{"search":"choice","space":[np.NaN]},
'ce__strategy':{"search":"choice","space":['label_encoding','entity_embedding','random_projection']},
'fs__strategy':{"search":"choice","space":['l1','variance','rf_feature_importance']},
'fs__threshold':{"search":"uniform","space":[0.01, 0.3]},
'est__max_depth':{"search":"choice","space":[3,5,7,9]},
'est__n_estimators':{"search":"choice","space":[250,500,700,1000]}}

Now we will see the steps to choose the best combination from the above space using the following steps:

Step1: Create an object of class Optimiser which has the parameters as ‘scoring’ and ‘n_folds’. Scoring is the metric against which we want to optimise our hyper-parameter space and n_folds is the number of folds of cross-validation
Scoring values for Classification- "accuracy", "roc_auc", "f1", "log_loss", "precision", "recall"
Scoring values for Regression- "mean_absolute_error", "mean_squarred_error", "median_absolute_error", "r2"
opt=Optimiser(scoring="accuracy",n_folds=5)

Step2: Use the optimise function of the object created above which takes the hyper-parameter space, dictionary created by the train_test_split and number of iterations as the parameters. This function returns the best hyper-paramters from the hyper-parameter space.
best=opt.optimise(space,data,40)


"""