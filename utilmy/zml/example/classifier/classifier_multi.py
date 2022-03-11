# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
https://www.kaggle.com/tapioca/multiclass-lightgbm

https://medium.com/@nitin9809/lightgbm-binary-classification-multi-class-classification-regression-using-python-4f22032b36a2


  python example/classifier/classifier_multi.py  train

  python example/classifier/classifier_multi.py  predict


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
    """function global_pars_update
    Args:
        model_dict:   
        data_name:   
        config_name:   
    Returns:
        
    """
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
config_default  = 'multi_lightgbm'

coly     = 'pet_category'
coldate  = ['issue_date','listing_date']
colcat   = ['color_type','condition']
colnum   = ['length(m)','height(cm)','X1','X2'] # ,'breed_category'
colcross = ['condition', 'color_type','length(m)', 'height(cm)', 'X1', 'X2']  # , 'breed_category'


cols_input_type_1 = {  "coly"  :   coly
                    ,"colid"   :   'pet_id'
                    ,"colcat" :    colcat
                    ,"colnum" :    colnum
                    ,"coltext" :   []
                    ,"coldate" :   []
                    ,"colcross" :  colcross
                   }

#####################################################################################
##### Params ########################################################################
def multi_lightgbm() :
    """
       multiclass
    """
    data_name         = f"multiclass"     ### in data/input/
    model_name        = 'LGBMClassifier'
    n_sample          = 6000

    def post_process_fun(y):          ### After prediction is done
        return  int(y)

    def pre_process_fun_multi(y):     ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        #'model_path'       : path_model_out

        ### LightGBM API model  ########################################
        'model_class': model_name    ## ACTUAL Class name for model_sklearn.py
        ,'model_pars'       : {'objective': 'multiclass','num_class':4,'metric':'multi_logloss',
                                'learning_rate':0.03,'boosting_type':'gbdt',
                                "n_estimators": 3,
                              }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun_multi ,
                                
        ### Pipeline for data processing.
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],

        },
        },

      'compute_pars': { 'metric_list': ['roc_auc_score','accuracy_score'],
                        'probability': True,  ### output probability for classifier
                      },

      'data_pars': {
          'n_sample' : n_sample,

          ### columns from raw file, based on data type, #############
          'cols_input_type' : cols_input_type_1,

          ### family of columns for MODEL  ########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',
          #  'coltext',
          'cols_model_group': [ 'colnum_bin','colcat_bin']

          ,'cols_model_type' : {}

          ### Filter data rows   #####################################
         ,'filter_pars': { 'ymax' : 5 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    #############################################################
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
   """function check
   Args:
   Returns:
       
   """
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





