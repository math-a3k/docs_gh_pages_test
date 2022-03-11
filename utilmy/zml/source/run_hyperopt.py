# -*- coding: utf-8 -*-
"""


Input a dict of variables.

pars_dict

pars_dict_range :


Optimize all

obj_fun




"""
DEBUG = True

def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    print(*s)

import copy, optuna

def run_hyper_optuna(obj_fun, pars_dict_init,  pars_dict_range,  engine_pars, ntrials=3):
    """
      pars_dict_init =  {  'boosting_type':'gbdt',
						'importance_type':'split', 'learning_rate':0.001, 'max_depth':10,
						'n_estimators': 50, 'n_jobs':-1, 'num_leaves':31 }
	  pars_dict_range =   {  'boosting_type':  ( 'categorical',  ['gbdt', 'gbdt']      ) ,
						 'importance_type':'split',
						 'learning_rate':  ('log_uniform' , 0.001, 0.1,  ),
						 'max_depth':      ('int',  1, 10, 'uniform')
						 'n_estimators':   ('int', 0, 10,  'uniform' )
						 'n_jobs':-1,
						 'num_leaves':31 }
      obj_fun(pars_dict) :  Objective function
      engine_pars :    {   }  optuna parameters
      
      
      API interface integration :
           https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html
      
    """
    import os
    print(pars_dict_init)
    def parse_dict(mdict, trial):
        #### Recursive parse the dict
        mnew = copy.deepcopy((mdict))
        print(mnew)
        for t, p in mdict.items():
            if t=="objective":
                continue
            if isinstance(p, dict) :
                pres = parse_dict(p, trial)

            else :
                # 'learning_rate':  ('log_uniform' , 0.001, 0.1,  ),
                pres  = None
                x     = p[0]
                if   x == 'log_uniform':      pres = trial.suggest_loguniform(t, p[1], p[2])
                elif x == 'int':              pres = trial.suggest_int(t,        p[1], p[2])
                elif x == 'uniform':          pres = trial.suggest_uniform(t,    p[1], p[2])
                elif x == 'categorical':      pres = trial.suggest_categorical(t, p[1])
                elif x == 'discrete_uniform': pres = trial.suggest_discrete_uniform(t, p[1], p[2], p[3])
                else:
                    print(f'Not supported type {t}, {p}')

            mnew[t] = pres
            if DEBUG : log(t, pres)
        return mnew

    def merge_dict(src, dst):
        for key, value in src.items():
            if isinstance(value, dict):
                node = dst.setdefault(key, {})
                merge_dict(value, node)
            else:
                dst[key] = value

        return dst

    def obj_custom(trial) :
        model_pars = copy.deepcopy( pars_dict_init )
        model_add  = parse_dict(pars_dict_range, trial)
        #model_pars = {**model_pars, **model_add}  ### Merge overwrite
        model_pars = merge_dict(model_pars,model_add)
        score      = obj_fun(model_pars)
        return score


    log("###### Hyper-optimization through study   ##################################")
    pruner = optuna.pruners.MedianPruner() if engine_pars.get("method", '') == 'prune' else None

    if "study_name" in engine_pars:
        study_name = engine_pars['study_name']
        storage    = engine_pars.get('storage', 'sqlite:///optunadb.db')  # {}
        # os.makedirs( os.path.dirname(storage), exist_ok= True)
        # study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
        try:
            study = optuna.load_study(study_name, storage)
        except:
            study = optuna.create_study(study_name=study_name, storage=storage, pruner=pruner)
    else:
        study = optuna.create_study(pruner=pruner)
        
    print(pars_dict_init)
    study.optimize(obj_custom, n_trials=ntrials)  # Invoke optimization of the objective function.
    log("Optim, finished", ntrials)
    pars_best  = study.best_params
    score_best = study.best_value

    log("####  Save on disk ##########################################################")





    return pars_best, score_best




def test_hyper():
    """function test_hyper
    Args:
    Returns:
        
    """
    model_pars ={



    }


    pars_range ={



    }

    from source import run_train
    def objective1(ddict):
         res = run_train.run_train(model_pars = model_pars,
                             n_sample          =  nsample,
                        )
         return res

    engine_pars = {}
    result_p = run_hyper_optuna(objective1, model_pars, pars_range, engine_pars, ntrials= 3)
    log(result_p)





def test_hyper3():
    """function test_hyper3
    Args:
    Returns:
        
    """
    import pandas as pd
    import numpy as np

    pars = {'x':2,
            'y':{'z':3, 't':2}}

    pars_range = {'x': ('uniform',  -10,10),
                  'y': {'z':('int',-10, 10)}
                  }

    def objective1(ddict):
         x   = ddict['x']
         z   = ddict['y']['z']
         obj = ((x - 2)**3 + z**2 )
         return obj

    engine_pars = {'metric_target':'loss'}
    result_p = run_hyper_optuna(objective1,pars, pars_range, engine_pars, ntrials= 3)
    log(result_p)




def test_hyper2():
    """function test_hyper2
    Args:
    Returns:
        
    """
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score
    param_dict={'n_estimators':100,
                'max_depth':3,
                'min_samples_split':2,
                'min_samples_leaf':2,
                'min_weight_fraction_leaf':0.0,
                'criterion':'gini',
                'max_features':'sqrt'}

    def objective(param_dict):
      model=RandomForestClassifier(**param_dict)
      model.fit(X_train,y_train)
      y_pred=model.predict(X_test)
      score=model.score(X_test, y_test)
      return -score

    param_dict_range={'max_depth':('int',  1, 10, 'uniform'),
                                  'n_estimators':('int', 0, 1000,  'uniform'),
                      'min_samples_split':('uniform',0,1,'uniform'),
                      'min_samples_leaf':('int',2,10,'uniform'),
                      'min_weight_fraction_leaf':('uniform',0.001,0.5,'uniform'),
                      'criterion':('categorical',['gini','entropy']),
                      'max_features':('categorical',['auto', 'sqrt', 'log2'])}
    engine_pars={'metric_target':'roc_auc_score'}

    def objective1(param_dict1):
         x=param_dict1.get('x')
         z=param_dict1['y']['z']
         obj=-((x - 2)**3 + z**2)
         return obj

    engine_pars1={'metric_target':'loss'}
    result_p=run_hyper_optuna(objective1, param_dict, param_dict_range, engine_pars,100)




if __name__ == "__main__":
    import fire
    fire.Fire()





def eval_dict(src, dst={}):
    """function eval_dict
    Args:
        src:   
        dst:   
    Returns:
        
    """
    import pandas as pd
    for key, value in src.items():
        if isinstance(value, dict):
            node = dst.setdefault(key, {})
            eval_dict(value, node)
        else:
            if "@lazy" not in key :
               dst[key] = value
            else :
                key2 = key.split(":")[-1]
                if 'pandas.read_csv' in key :
                    dst[key2] = pd.read_csv(value)
                elif 'pandas.read_parquet' in key :
                    dst[key2] = pd.read_parquet(value)
    return dst





"""
### Distributed
https://optuna.readthedocs.io/en/latest/tutorial/distributed.html
  { 'distributed' : 1,
   'study_name' : 'ok' ,
  'storage' : 'sqlite'
 }

 ###### 1st engine is optuna
 https://optuna.readthedocs.io/en/stable/installation.html
https://github.com/pfnet/optuna/blob/master/examples/tensorflow_estimator_simple.py
https://github.com/pfnet/optuna/tree/master/examples
Interface layer to Optuna  for hyperparameter optimization
optuna create-study --study-name "distributed-example" --storage "sqlite:///example.db"
https://optuna.readthedocs.io/en/latest/tutorial/distributed.html
study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
study.optimize(objective, n_trials=100)

weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam']) # Categorical parameter
num_layers = trial.suggest_int('num_layers', 1, 3)      # Int parameter
dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)      # Uniform parameter
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)      # Loguniform parameter
drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1) # Discrete-uniform parameter

"""
"""



############################################################################################################
def optim_optuna_old(model_uri="model_tf.1_lstm.py",
                 hypermodel_pars = {"engine_pars": {}},
                 model_pars      = {},
                 data_pars       = {},
                 compute_pars    = {},  # only Model pars
                 out_pars        = {}):

    import optuna
    # these parameters should be inside the key "engine_pars" which should be inside "hypermodel_pars" of the config json
    engine_pars   = hypermodel_pars['engine_pars']
    ntrials       = engine_pars['ntrials']
    # metric target should be a value that is available in the model.stats of the model we are optimizing it can simply be called "loss" or
    # specific like "roc_auc_score"
    metric_target = engine_pars["metric_target"]

    save_path     = out_pars['path']
    # log_path      = out_pars['log_path']
    os.makedirs(save_path, exist_ok=True)

    model_name    = model_pars.get("model_name")  #### Only for sklearn model
    # model_type    = model_pars['model_type']
    log(model_pars, data_pars, compute_pars, hypermodel_pars)

    module = module_load(model_uri)
    log(module)

    def objective(trial):
        # log("check", module, data_pars)
        for t, p in hypermodel_pars.items():

            if t == 'engine_pars': continue  ##Skip
            # type, init, range[0,1]
            x = p['type']
            if x == 'log_uniform':
                pres = trial.suggest_loguniform(t, p['range'][0], p['range'][1])
            elif x == 'int':
                pres = trial.suggest_int(t, p['range'][0], p['range'][1])
            elif x == 'categorical':
                pres = trial.suggest_categorical(t, p['value'])
            elif x == 'discrete_uniform':
                pres = trial.suggest_discrete_uniform(t, p['init'], p['range'][0], p['range'][1])
            elif x == 'uniform':
                pres = trial.suggest_uniform(t, p['range'][0], p['range'][1])
            else:
                raise Exception(f'Not supported type {x}')
                pres = None

            model_pars[t] = pres

        module = model_create(module, model_pars, data_pars, compute_pars)  # module.Model(**param_dict)
        if VERBOSE: log(model)

        module.fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
        metrics = module.evaluate(data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
        mtarget = metrics[metric_target]
        try:
            module.reset_model()  # Reset Graph for TF
        except Exception as e:
            log(e)

        return mtarget


    log("###### Hyper-optimization through study   ##################################")
    pruner = optuna.pruners.MedianPruner() if engine_pars["method"] == 'prune' else None

    if engine_pars.get("distributed") is not None:
        # study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
        try:
            study = optuna.load_study(study_name=engine_pars['study_name'], storage=engine_pars['storage'])
        except:
            study = optuna.create_study(study_name=engine_pars['study_name'], storage=engine_pars['storage'],
                                        pruner=pruner)
    else:
        study = optuna.create_study(pruner=pruner)

    study.optimize(objective, n_trials=ntrials)  # Invoke optimization of the objective function.
    log("Optim, finished", n=35)
    param_dict_best = study.best_params
    # param_dict.update(module.config_get_pars(choice="test", )


    log("### Save Stats   ##########################################################")
    study_trials = study.trials_dataframe()
    study_trials.to_csv(f"{save_path}/{model_uri}_study.csv")
    param_dict_best["best_value"] = study.best_value
    json.dump(param_dict_best, open(f"{save_path}/{model_uri}_best-params.json", mode="w"))


    log("### Run Model with best   #################################################")
    model_pars_update = copy.deepcopy(model_pars)
    model_pars_update.update(param_dict_best)
    model_pars_update["model_name"] = model_name  ###SKLearn model

    module = model_create(module, model_pars_update, data_pars, compute_pars)
    module.fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)


    log("#### Saving     ###########################################################")
    model_uri = model_uri.replace(".", "-")
    save_pars = {'path': save_path, 'model_type': model_uri.split("-")[0], 'model_uri': model_uri}
    module.save(save_pars=save_pars)

    #log( os.stats(save_path))
    ## model_pars_update["model_name"] = model_name
    return model_pars_update






import pandas as pd
import numpy as np
#import xgboost as xgb
import lightgbm as lgb
import gc

from skopt.space import Real, Integer
from skopt.utils import use_named_args
import itertools
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, json
from pandas.io.json import json_normalize
import lightgbm as lgb
from sklearn.feature_selection import RFE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output


https://rbfopt.readthedocs.io/en/latest/rbfopt_aux_problems.html





“RBFOpt: an open-source library for black-box optimization with costly function ...

https://github.com/coin-or/rbfopt

After installation, the easiest way to optimize a function is to use the RbfoptUserBlackBox class to define a black-box, and execute RbfoptAlgorithm on it. This is a minimal example to optimize the 3-dimensional function defined below:

import rbfopt
import numpy as np
def obj_funct(x):
  return x[0]*x[1] - x[2]

bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                               np.array(['R', 'I', 'R']), obj_funct)
settings = rbfopt.RbfoptSettings(max_evaluations=50)
alg = rbfopt.RbfoptAlgorithm(settings, bb)
val, x, itercount, evalcount, fast_evalcount = alg.optimize()
Another possibility is to define your own class derived from RbfoptBlackBox in a separate file, and execute the command-line interface on the file. An example is provided under src/rbfopt/examples, in the file rbfopt_black_box_example.py. This can be executed with:

rbfopt_cl_interface.py src/rbfopt/examples/rbfopt_black_box_example.py







# Feature importance

#lightGBM model fit
gbm = lgb.LGBMRegressor()
gbm.fit(train, target)
gbm.booster_.feature_importance()

# importance of each attribute
fea_imp_ = pd.DataFrame({'cols':train.columns, 'fea_imp':gbm.feature_importances_})
fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)







TRAINING_SIZE = 300000


#Recursive Feature Elimination(RFE)

# create the RFE model and select 10 attributes
rfe = RFE(gbm, 10)
rfe = rfe.fit(train, target)

# summarize the selection of the attributes
print(rfe.support_)

# summarize the ranking of the attributes
fea_rank_ = pd.DataFrame({'cols':train.columns, 'fea_rank':rfe.ranking_})
fea_rank_.loc[fea_rank_.fea_rank > 0].sort_values(by=['fea_rank'], ascending = True)















TEST_SIZE = 50000

# Load data
train = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TRAINING_SIZE-TEST_SIZE), 
    nrows=TRAINING_SIZE,
    parse_dates=['click_time']
)

val = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TEST_SIZE), 
    nrows=TEST_SIZE,
    parse_dates=['click_time']
)

# Split into X and y
y_train = train['is_attributed']
y_val = val['is_attributed']


# from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(3, 10, name='max_depth'),
          Integer(6, 30, name='num_leaves'),
          Integer(50, 200, name='min_child_samples'),
          Real(1, 400,  name='scale_pos_weight'),
          Real(0.6, 0.9, name='subsample'),
          Real(0.6, 0.9, name='colsample_bytree')
         ]





res_gp = gp_minimize(objective, space, n_calls=20,
                     random_state=0,n_random_starts=10)

"Best score=%.4f" % res_gp.fun






TRAINING_SIZE = 300000
TEST_SIZE = 50000

# Load data
train = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TRAINING_SIZE-TEST_SIZE), 
    nrows=TRAINING_SIZE,
    parse_dates=['click_time']
)

val = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TEST_SIZE), 
    nrows=TEST_SIZE,
    parse_dates=['click_time']
)

# Split into X and y
y_train = train['is_attributed']
y_val = val['is_attributed']

Specify the parameter space we want to explore.

# from that dimension (`'log-uniform'` for the learning rate)
space  = [Integer(3, 10, name='max_depth'),
          Integer(6, 30, name='num_leaves'),
          Integer(50, 200, name='min_child_samples'),
          Real(1, 400,  name='scale_pos_weight'),
          Real(0.6, 0.9, name='subsample'),
          Real(0.6, 0.9, name='colsample_bytree')
         ]

Below is the fun part. The function gp_minimize requires an objective function and what the function all needs is basically a metric we want to minimize. Of course, we can just use whatever training setup we have been using but just tweak it to return a AUC to minimize..(negative AUC)

def objective(values):
    

    params = {'max_depth': values[0], 
          'num_leaves': values[1], 
          'min_child_samples': values[2], 
          'scale_pos_weight': values[3],
            'subsample': values[4],
            'colsample_bytree': values[5],
             'metric':'auc',
             'nthread': 8,
             'boosting_type': 'gbdt',
             'objective': 'binary',
             'learning_rate':0.15,
             'max_bin': 100,
             'min_child_weight': 0,
             'min_split_gain': 0,
             'subsample_freq': 1}
    

    print('\nNext set of params.....',params)
    
    feature_set = ['app','device','os','channel']
    categorical = ['app','device','os','channel']
    
    
    early_stopping_rounds = 50
    num_boost_round       = 1000
    
        # Fit model on feature_set and calculate validation AUROC
    xgtrain = lgb.Dataset(train[feature_set].values, label=y_train,feature_name=feature_set,
                           categorical_feature=categorical)
    xgvalid = lgb.Dataset(val[feature_set].values, label=y_val,feature_name=feature_set,
                          categorical_feature=categorical)
    
    evals_results = {}
    model_lgb     = lgb.train(params,xgtrain,valid_sets=[xgtrain, xgvalid], 
                              valid_names=['train','valid'], 
                               evals_result=evals_results, 
                               num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=None, feval=None)
    
    auc = -roc_auc_score(y_val, model_lgb.predict(val[model_lgb.feature_name()]))
    
    print('\nAUROC.....',-auc,".....iter.....", model_lgb.current_iteration())
    
    gc.collect()
    
    return  auc


"""


