# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Run template

python core_run.py data_profile --config outlier_predict.py::titanic_lightgbm

python core_run.py preprocess --config outlier_predict.py::titanic_lightgbm

python core_run.py train --config outlier_predict.py::titanic_lightgbm

python core_run.py predict --config outlier_predict.py::titanic_lightgbm


"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

####################################################################################
def log(*s):
  print(*s)


root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
log(root, dir_data)


####################################################################################
def get_global_pars(config_uri=""):
  log("#### Model Params Dynamic loading  ##########################################")
  from source.util_feature import load_function_uri
  log("config_uri",config_uri)
  model_dict_fun = load_function_uri(uri_name=config_uri )

  #### Get dict + Update Global variables
  try :
     model_dict     = model_dict_fun()   ### params
  except :
     model_dict  = model_dict_fun

  return model_dict


def get_config_path(config=''):
    #### Get params where the file is imported  #####################
    path0 =  os.path.abspath( sys.modules['__main__'].__file__)
    print("file where imported", path0)

    config_default = get_global_pars( path0 + "::config_default")


    if len(config)  == 0 :
        config_uri  = path0  + "::" + config_default
        config_name = config_default

    elif "::" not in config :
        config_uri  = path0  + "::" + config
        config_name = config

    else :
        config_uri  = config
        config_name = config.split("::")[1]

    ##################################################################
    log("default: ", config_uri)
    return config_uri, config_name


#####################################################################################
########## Profile data #############################################################
def data_profile2(config=''):
    """
    :param config:
    :return:
    """
    config_uri, config_name = get_config_path(config)
    from source.run_feature_profile import run_profile
    mdict = get_global_pars( config_uri)
    m     = mdict['global_pars']
    log(mdict)

    run_profile(path_data   = m['path_data_train'],
               path_output  = m['path_model'] + "/profile/",  
               n_sample     = 5000,
              ) 


def data_profile(path_data="NO PATH", path_output="NO PATH@", n_sample= 5000):
   from source.run_feature_profile import run_profile
   run_profile(path_data   = path_data,
               path_output = path_output + "/profile/",
               n_sample    = n_sample,
              )

###################################################################################
########## Preprocess #############################################################
def preprocess(config='', nsample=None):
    """

    :param config:
    :param nsample:
    :return:
    """
    config_uri, config_name = get_config_path(config)
    mdict = get_global_pars( config_uri)
    m     = mdict['global_pars']
    log(mdict)

    from source import run_preprocess
    run_preprocess.run_preprocess(config_name   =  config_name,
                                  config_path   =  m['config_path'],
                                  n_sample      =  nsample if nsample is not None else m['n_sample'],

                                  ### Optonal
                                  mode          =  'run_preprocess')


####################################################################################
########## Train ###################################################################
def train(config='', nsample=None):
    """  train a model with  confi_name  and nsample
    :param config:
    :param nsample:
    :return:
    """

    config_uri, config_name = get_config_path(config)

    mdict = get_global_pars(  config_uri)
    m     = mdict['global_pars']
    log(mdict)
    from source import run_train
    run_train.run_train(config_name       =  config_name,
                        config_path       =  m['config_path'],
                        n_sample          =  nsample if nsample is not None else m['n_sample'],
                        # use_mlmflow       =  False
                        )


####################################################################################
######### Check model ##############################################################
def check(config='outlier_predict.py::titanic_lightgbm'):
    mdict = get_global_pars(config)
    m     = mdict['global_pars']
    log(mdict)
    pass




########################################################################################
####### Inference ######################################################################
def predict(config='', nsample=None):
    """
    :param config:
    :param nsample:
    :return:
    """
    config_uri, config_name = get_config_path(config)

    mdict = get_global_pars( config_uri)
    m     = mdict['global_pars']
    log(mdict)

    from source import run_inference
    run_inference.run_predict(config_name = config_name,
                              config_path = m['config_path'],
                              n_sample    = nsample if nsample is not None else m['n_sample'],

                              #### Optional
                              path_data   = m['path_pred_data'],
                              path_output = m['path_pred_output'],
                              model_dict  = None
                              )





#########################################################################################
#########################################################################################
########## source/ru_sampler Train Sampler ###############################################
def train_sampler(config='', nsample=None):
    """  train a model with  confi_name  and nsample
    :param config:
    :param nsample:
    :return:
    """
    config_uri, config_name = get_config_path(config)

    mdict = get_global_pars(  config_uri)
    m     = mdict['global_pars']
    log(mdict)
    from source import run_sampler
    run_sampler.run_train(config_name     =  config_name,
                        config_path       =  m['config_path'],
                        n_sample          =  nsample if nsample is not None else m['n_sample'],
                        # use_mlmflow       =  False
                        )

####### source/run_sampler tranform ####################################################
def transform(config='', nsample=None):
    """
    :param config:
    :param nsample:
    :return:
    """
    config_uri, config_name = get_config_path(config)

    mdict = get_global_pars( config_uri)
    m     = mdict['global_pars']
    log(mdict)


    from source import run_sampler
    run_sampler.run_transform(config_name = config_name,
                              config_path = m['config_path'],
                              n_sample    = nsample if nsample is not None else m['n_sample'],

                              #### Optional
                              path_data   = m['path_pred_data'],
                              path_output = m['path_pred_output'],
                              model_dict  = None
                              )






########################################################################################
#######  HYPER  ########################################################################
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
    mdict       = load_function_uri(config_full) #titanic1()
    mdict       = mdict()

    ####### Objective   ##########################################################
    def objective_fun(mdict):
        if debug : log(mdict)#
        ddict       = run_train(config_name="", config_path="", n_sample= n_sample,
                                mode="run_preprocess", model_dict=mdict,
                                return_mode='dict')

        # print(ddict['stats']['metrics_test'].to_dict('records')[0])
        ddict['stats'][metric_name] = ddict['stats']['metrics_test'].to_dict('records')[0]['metric_val']

        if debug : print(ddict)
        res = ddict['stats'][metric_name]
        return res

    ##### Optuna Params   ####################################################
    engine_pars = {'metric_target' :'loss',
                   'study_name'    : config_name  ,
                   'storage'       : f"sqlite:///" + os.path.abspath(path_optuna_storage).replace("\\", "/") }

    ##### Running the optim
    best_dict   = run_hyper_optuna(objective_fun, mdict, mdict_range, engine_pars, ntrials= ntrials)


    ##### Export
    os.makedirs(path_output, exist_ok=True)
    json.dump(best_dict, open(path_output + "/hyper_params_best.json", mode='a'))

    log(engine_pars['storage'])
    log(best_dict)
    log(path_output)






########################################################################################
####### Inference / Deploy #############################################################
def deploy():
    """
    Simple deploy using uvicorn on runtime. U can use gunicorn instead.

    #### Important info: First request will always take time

    return: Service
    """
    import uvicorn
    uvicorn.run("core_deploy:app", host="127.0.0.1", port=8000, log_level="info")





##########################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
    








