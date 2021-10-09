# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
python  test_hyperopt.py  train   --nsample 200

python  test_hyperopt.py  hyperparam  --ntrials 2


"""
import warnings, copy, os, sys, numpy as np
warnings.filterwarnings('ignore')

####################################################################################
###### Path ########################################################################
root_repo      =  os.path.abspath(os.getcwd()).replace("\\", "/") + "/"     ; print(root_repo)
THIS_FILEPATH  =  os.path.abspath(__file__) 

sys.path.append(root_repo)
from source.util_feature import save, os_get_function_name, log


def global_pars_update(model_dict,  data_name, config_name):
    print("config_name", config_name)
    dir_data  = root_repo + "/data/"  ; print("dir_data", dir_data)

    m                      = {}
    m['config_path']       = THIS_FILEPATH  
    m['config_name']       = config_name

    #### peoprocess input path
    m['path_data_preprocess'] = dir_data + f'/input/{data_name}/train/'

    #### train input path
    m['path_data_train']      = dir_data + f'/input/{data_name}/train/'
    m['path_data_test']       = dir_data + f'/input/{data_name}/test/'
    #m['path_data_val']       = dir_data + f'/input/{data_name}/test/'

    #### train output path
    m['path_train_output']    = dir_data + f'/output/{data_name}/{config_name}/'
    m['path_train_model']     = dir_data + f'/output/{data_name}/{config_name}/model/'
    m['path_features_store']  = dir_data + f'/output/{data_name}/{config_name}/features_store/'
    m['path_pipeline']        = dir_data + f'/output/{data_name}/{config_name}/pipeline/'


    #### predict  input path
    m['path_pred_data']       = dir_data + f'/input/{data_name}/test/'
    m['path_pred_pipeline']   = dir_data + f'/output/{data_name}/{config_name}/pipeline/'
    m['path_pred_model']      = dir_data + f'/output/{data_name}/{config_name}/model/'

    #### predict  output path
    m['path_pred_output']     = dir_data + f'/output/{data_name}/pred_{config_name}/'

    #####  Generic
    m['n_sample']             = model_dict['data_pars'].get('n_sample', 5000)

    model_dict[ 'global_pars'] = m
    return model_dict


####################################################################################
##### Params########################################################################
config_default   = 'titanic1'          ### name of function which contains data configuration


cols_input_type_2 = {
     "coly"     :  "Survived"
    ,"colid"    :  "PassengerId"
    ,"colcat"   :  ["Sex", "Embarked" ]
    ,"colnum"   :  ["Pclass", "Age","SibSp", "Parch","Fare"]
    ,"coltext"  :  ["Name", "Ticket"]
    ,"coldate"  :  []
    ,"colcross" :  [ "Name", "Sex", "Ticket","Embarked","Pclass", "Age","SibSp", "Parch","Fare" ]

    ,'colgen'  :   [ "Pclass", "Age","SibSp", "Parch","Fare" ]
}


#################################################################################
def post_process_fun(y): return  int(y)
def pre_process_fun(y):  return  int(y)

def titanic1(path_model_out="") :
    """ One big dict
    """
    config_name  = os_get_function_name()
    data_name    = "titanic"         ### in data/input/
    model_class  = 'LGBMClassifier'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 500

    model_dict = {
    'model_pars': {
         'model_class': model_class
        ,'model_pars' : {'objective': 'binary', 'n_estimators':10,
                        }

        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },

            # {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },

            # {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            # {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'},
        ],
        }
  },

  'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                  },

  'data_pars': { 'n_sample' : n_sample,
      'cols_input_type' : cols_input_type_2,
      'cols_model_group': [ 'colnum',  ### should be optional 'colcat'
                            'colcat_bin',
                          ],
      'cols_model_type' : {
         'continuous'   : [ 'colnum',   ],
         'sparse'       : [ 'colcat_bin', 'colnum_bin',  ],
      }

      ### Filter data rows   ##################################################################
     ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
  }

    ##### Filling Global parameters    ########################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name )
    return model_dict




#####################################################################################
########## Hyper-parans #############################################################
def hyperparam(config_full="",
               ntrials=2, n_sample=5000, debug=1,
               path_output         = "data/output/titanic1/",
               path_optuna_storage = 'data/output/optuna_hyper/optunadb.db'):
    """
        python test_hyperopt.py  hyperparam  --ntrials 2
    """
    # from core_run import  hyperparam_wrapper

    #### Base nodel_dict params ##########################################################
    config_name = 'titanic1'
    config_full = THIS_FILEPATH + "::" + config_name

    ###### model_dict  adding range for parameters sampling  #############################
    mdict_range =   {'model_pars': {
        ### LightGBM API model
        'model_pars' : { 'objective':      'binary',
                         'n_estimators':   ('int', 20, 100),
                         'learning_rate' : ('log_uniform', 0.0001, 0.1),
                         'reg_alpha ' :    ('uniform', 0.0001, 0.01),  ### l1 reg
                       },
        },

        'data_pars' :  {
            #### Categorical sampling
            'cols_model_group' : ('categorical',  [  [ 'colnum'  ] ,
                                                     [ 'colnum', 'colcat_bin']
                                                  ]
            ),
        }

    }

    ###################################################################################
    metric_name = "accuracy_score"

    hyperparam_wrapper(config_full,
                       ntrials, n_sample, debug,
                       path_output, path_optuna_storage,
                       metric_name,
                       mdict_range
                       )


#####################################################################################
########## Hyper-parans Wrapper #####################################################
def hyperparam_wrapper(config_full="",
                       ntrials=2, n_sample=5000, debug=1,
                       path_output         = "data/output/titanic1/",
                       path_optuna_storage = 'data/output/optuna_hyper/optunadb.db',
                       metric_name='accuracy_score', mdict_range=None):

    from source.util_feature import load_function_uri
    from source.run_train import  run_train
    from source.run_hyperopt import run_hyper_optuna
    import json

    ##############################################################################
    ####### model_dict initial dict of params  ###################################
    config_name = config_full.split("::")[-1]
    mdict       = load_function_uri(config_full) #titanic1() dictionnary
    mdict       = mdict()

    ####### Objective   ##########################################################
    def objective_fun(mdict):
        if debug : log(mdict)#
        ddict       = run_train(config_name="", config_path="", n_sample= n_sample,
                                mode="run_preprocess", model_dict=mdict,
                                return_mode='dict')

        # print(ddict['stats']['metrics_test'].to_dict('records')[0])
        #res = ddict['stats']['metrics_test'].to_dict('records')[0]['metric_val']
        df  =  ddict['stats']['metrics_test']

        #### Beware of the sign
        res = -np.mean(df[ df['metric_name'] == metric_name ]['metric_val'].values)
        return res

    ##### Optuna Params   ####################################################
    engine_pars = {'metric_target' :'loss',
                   'study_name'    : config_name  ,
                   'storage'       : "sqlite:///:memory:" }
                    # f"sqlite:///" + os.path.abspath(path_optuna_storage).replace("\\", "/") }

    ##### Running the optim
    best_dict   = run_hyper_optuna(objective_fun, mdict, mdict_range, engine_pars, ntrials= ntrials)


    ##### Export
    os.makedirs(path_output, exist_ok=True)
    json.dump(best_dict, open(path_output + "/hyper_params_best.json", mode='a'))

    log(engine_pars['storage'])
    log(best_dict)
    log(path_output)


###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess



##################################################################################
########## Train #################################################################
from core_run import train



####################################################################################
####### Inference ##################################################################
# predict(config='', nsample=10000)
from core_run import predict




###########################################################################################################
###########################################################################################################
if __name__ == "__main__":
    from pyinstrument import Profiler;  profiler = Profiler() ; profiler.start()
    import fire
    fire.Fire()
    profiler.stop() ; print(profiler.output_text(unicode=True, color=True))







