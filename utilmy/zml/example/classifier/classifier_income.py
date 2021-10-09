# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
    python  example/classifier/classifier_income.py  train       --nsample 32560

	python  classifier_income.py  predict


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
config_default  = 'income_status_lightgbm'        ### name of function which contains data configuration


cols_input_type_1 = {
     "coly"   :   "status"
    ,"colid"  :   "id"
    ,"colcat" :   ["occupation","workclass","native-country","education-num","marital-status","relationship","race","sex"]
    ,"colnum" :   ["age", "final_weight", "capital-gain", "capital-loss", "hours-per-week"]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : ["occupation","workclass","native-country","education-num","marital-status","relationship","race","sex","age", "final_weight", "capital-gain", "capital-loss", "hours-per-week"]
}




####################################################################################
def income_status_lightgbm(path_model_out="") :
    """


    """
    data_name    = "income_status"       ### in data/input/
    model_class  = 'LGBMClassifier' #  'LGBMClassifier_optuna' ACTUAL Class name for model_sklearn.py
    n_sample     = 32500 # 32560

    def post_process_fun(y):  ### After prediction is done
        return  int(y)

    def pre_process_fun(y):   ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        ### LightGBM API model   #######################################
        'model_class': model_class
        ,'model_pars' : {'boosting_type':'gbdt', 'class_weight':None, 'colsample_bytree':1.0,
						'importance_type':'split', 'learning_rate':0.001, 'max_depth':-1,
						'min_child_samples':20, 'min_child_weight':0.001, 'min_split_gain':0,
						'n_estimators': 5,
                         'n_jobs':-1, 'num_leaves':31, 'objective':None,
						'random_state':None, 'reg_alpha':0, 'reg_lambda':0.0, 'silent':True,
						'subsample':1.0, 'subsample_for_bin':200000, 'subsample_freq':0
        }


        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


        ### Pipeline for data processing ##############################
        'pipe_list': [
            #{'uri': 'data/input/income/manual_preprocessing.py::pd_income_processor',      'pars': {}, 'cols_family': 'colall',   'cols_out': 'colall',
            #        'type': 'filter'         },

            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },

            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },

            ### Cross Features
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair',  'type': 'cross'},

            ### Quantile normalization
            {'uri': 'source/prepro.py::pd_colnum_quantile_norm',       'pars': {'colsparse' :  [] },
             'cols_family': 'colnum',     'cols_out': 'colnum_quantile_norm',     'type': ''             },

        ],
               }
        },

      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score'],
                      'optuna_params': {
                          "early_stopping_rounds": 5,
                          'verbose_eval' :        100,
                           #  folds=KFold(n_splits=3)
                      },
                      'optuna_engine' : 'LightGBMTuner'   ###  LightGBMTuner', LightGBMTunerCV
                    },

      'data_pars': { 'n_sample' : n_sample,
          'cols_input_type' : cols_input_type_1,

          ### family of columns for MODEL  ########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',  'coltext',
          'cols_model_group': [ # 'colnum_bin',
                                'colcat_bin',
                                'colnum_quantile_norm',

                                # 'coltext',
                                # 'coldate',
                                'colcross_pair',

                              ]

          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
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

