# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titanic dataset (along with optuna)
All in one file config
  python example/classifier/classifier_optuna.py  train

  python classifier_optuna.py  predict  > zlog


"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

####################################################################################
###### Path ########################################################################
root_repo      =  os.path.abspath(os.getcwd()).replace("\\", "/") + "/"     ; print(root_repo)
THIS_FILEPATH  =  os.path.abspath(__file__)

sys.path.append(root_repo)
from source.util_feature import save,os_get_function_name

def global_pars_update(model_dict,  data_name, config_name):
    print("config_name", config_name)
    dir_data  = root_repo + "/data/"  ; print("dir_data", dir_data)

    m                      = {}
    m['config_path']       = THIS_FILEPATH
    m['config_name']       = config_name

    #### peoprocess input path
    m['path_data_preprocess'] = dir_data + f'/input/{data_name}/train/'

    #### train input path
    dir_data_url              = "https://github.com/arita37/dsa2_data/tree/master/"  #### Remote Data directory
    m["path_data_train"]      = dir_data_url + f"/input/{data_name}/train/"
    m["path_data_test"]       = dir_data_url + f"/input/{data_name}/test/"
    #m["path_data_val"]       = dir_data + f"/input/{data_name}/test/"

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
config_default = 'titanic_lightoptuna'  ### name of function which contains data configuration

# data_name    = "titanic"     ### in data/input/
cols_input_type_1 = {
    "coly": "Survived"
    , "colid": "PassengerId"
    , "colcat": ["Sex", "Embarked"]
    , "colnum": ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    , "coltext": []
    , "coldate": []
    , "colcross": ["Name", "Sex", "Ticket", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"]
}

cols_input_type_2 = {
    "coly": "Survived"
    , "colid": "PassengerId"
    , "colcat": ["Sex", "Embarked"]
    , "colnum": ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    , "coltext": ["Name", "Ticket"]
    , "coldate": []
    , "colcross": ["Name", "Sex", "Ticket", "Embarked", "Pclass", "Age", "SibSp", "Parch", "Fare"]
}


####################################################################################
def titanic_lightoptuna():
    """
       Contains all needed informations for Light GBM Classifier model,
       used for titanic classification task
    """
    config_name = os_get_function_name()
    data_name   = "titanic"  ### in data/input/
    model_class = 'LGBMModel_optuna'  ### ACTUAL Class name for model_sklearn.py
    n_sample    = 1000

    def post_process_fun(y):
        ### After prediction is done
        return int(y)

    def pre_process_fun(y):
        ### Before the prediction is done
        return int(y)

    model_dict = {'model_pars': {
    ### LightGBM API model   #######################################
    'model_file'  : 'optuna_lightgbm.py',  ###Optional one
    'model_class' :  model_class
    ,'model_pars':  {'objective': 'binary',
                     'n_estimators': 50,
                     'learning_rate': 0.001,
                     'boosting_type': 'gbdt',  ### Model hyperparameters
                     'early_stopping_rounds': 5
                     }


    ### After prediction  ##########################################
    , 'post_process_fun': post_process_fun

    ### Before training  ##########################################
    , 'pre_process_pars': {'y_norm_fun': pre_process_fun,

       ### Pipeline for data processing ##############################
       'pipe_list': [
           {'uri': 'source/prepro.py::pd_coly', 'pars': {}, 'cols_family': 'coly', 'cols_out': 'coly', 'type': 'coly'},
           {'uri': 'source/prepro.py::pd_colnum_bin', 'pars': {},  'cols_family': 'colnum', 'cols_out': 'colnum_bin', 'type': ''},
           # {'uri': 'source/prepro.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot', 'type': ''},

           {'uri': 'source/prepro.py::pd_colcat_bin', 'pars': {}, 'cols_family': 'colcat', 'cols_out': 'colcat_bin', 'type': ''},
           # {'uri': 'source/prepro.py::pd_colcat_to_onehot', 'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot', 'type': ''},
           #{'uri': 'source/prepro.py::pd_colcross', 'pars': {},'cols_family': 'colcross', 'cols_out': 'colcross_pair_onehot', 'type': 'cross'}
       ],
       }
    },

    #classoptuna.integration.lightgbm.LightGBMTuner(params: Dict[str, Any], train_set: lgb.Dataset,
    # num_boost_round: int = 1000, valid_sets: Optional[VALID_SET_TYPE] = None,
    # valid_names: Optional[Any] = None, fobj: Optional[Callable[[…], Any]] = None,
    # feval: Optional[Callable[[…], Any]] = None, feature_name: str = 'auto', categorical_feature: str = 'auto', early_stopping_rounds: Optional[int] = None, evals_result: Optional[Dict[Any, Any]] = None, verbose_eval: Union[bool, int, None] = True, learning_rates: Optional[List[float]] = None, keep_training_booster: bool = False, callbacks: Optional[List[Callable[[…], Any]]] = None, time_budget: Optional[int] = None, sample_size: Optional[int] = None, study: Optional[optuna.study.Study] = None, optuna_callbacks: Optional[List[Callable[[optuna.study.Study, optuna.trial._frozen.FrozenTrial], None]]] = None, model_dir: Optional[str] = None, verbosity: Optional[int] = None, show_progress_bar: bool = True)[source]
    'compute_pars': {'metric_list': ['accuracy_score', 'average_precision_score'],
                     'optuna_params': {
                         "early_stopping_rounds": 5,
                          'verbose_eval' :        100,
                           #  folds=KFold(n_splits=3)
                     },

                     'optuna_engine' : 'LightGBMTuner'   ###  LightGBMTuner', LightGBMTunerCV

                     },


    'data_pars': {'n_sample': n_sample,
                  'cols_input_type': cols_input_type_1,
                  ### family of columns for MODEL  #########################################################
                  #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
                  #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
                  #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
                  #  'coldate',
                  #  'coltext',
                  'cols_model_group': ['colnum_bin',
                                       'colcat_bin',
                                       # 'coltext',
                                       # 'coldate',
                                       # 'colcross_pair'
                                       ]

                  ### Filter data rows   ##################################################################
        , 'filter_pars': {'ymax': 2, 'ymin': -1}

                  }
    }

    ##### Filling Global parameters    ############################################################
    model_dict = global_pars_update(model_dict, data_name, config_name)
    return model_dict




#####################################################################################
########## Profile data #############################################################
#### def data_profile(path_data="", path_output="", n_sample= 5000):
from core_run import data_profile




 ###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess


##################################################################################
########## Train #################################################################
## def train(config_uri='titanic_classifier.py::titanic_lightgbm'):
from core_run import train



###################################################################################
######### Check data ##############################################################
def check():
   pass



####################################################################################
####### Inference ##################################################################
# def  predict(config='', nsample=10000)
from core_run import predict



###########################################################################################################
###########################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()



