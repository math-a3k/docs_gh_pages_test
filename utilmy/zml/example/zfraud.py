# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

  python  zfraud.py  data_profile  --path_data_train data/input/adfraud/train/  --path_out zlog/

  python  zfraud.py    train

  python  zfraud.py    predict



https://github.com/arita37/dsa2/tree/main/data/input/adfraud

ip,app,device,os,channel,click_time,attributed_time,is_attributed
83230,3,1,13,379,11/6/2017 14:32,,0
17357,3,1,19,379,11/6/2017 14:33,,0
35810,3,1,13,379,11/6/2017 14:34,,0
45745,14,1,13,478,11/6/2017 14:34,,0


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
    m['path_data_train']      = dir_data + f'/input/{data_name}/train_100k/'
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
config_default  = 'adfraud_lightgbm'   ### name of function which contains data configuration



####################################################################################
##### Params########################################################################
"""

Index(['ip', 'app', 'os', 'device', 'channel', 'ip_app_count_channel', 
       'ip_app_os_count_channel', 'ip_max_hour', 'ip_min_hour',        
       'ip_max_dayweek', 'ip_min_dayweek', 'ip_AvgViewPerDistinct_hour'
,                                                                      
       'ip_AvgViewPerDistinct_dayweek', 'ip_channel_max_hour',         
       'ip_channel_min_hour', 'ip_channel_max_dayweek',                
       'ip_channel_min_dayweek', 'ip_nunique_channel', 'ip_nunique_app'
,                                                                      
       'ip_nunique_device', 'ip_AvgViewPerDistinct_app',               
       'ip_AvgViewPerDistinct_channel', 'ip_app_nunique_os',           
       'ip_device_os_nunique_app', 'app_AvgViewPerDistinct_ip',        
       'app_count_channel', 'app_nunique_channel', 'channel_count_app']



"""
cols_input_type_1 = {
     "coly"   :   "is_attributed"     ### 
    ,"colid"  :   "ip"
    ,"colcat" :   [  "app", "device", "os", "channel",   ]
    ,"colnum" :   [ 'ip_app_count_channel', 
       'ip_app_os_count_channel', 'ip_max_hour', 'ip_min_hour',        
       'ip_max_dayweek', 'ip_min_dayweek', 'ip_AvgViewPerDistinct_hour'
,                                                                      
       'ip_AvgViewPerDistinct_dayweek', 'ip_channel_max_hour',         
       'ip_channel_min_hour', 'ip_channel_max_dayweek',                
       'ip_channel_min_dayweek', 'ip_nunique_channel', 'ip_nunique_app'
,                                                                      
       'ip_nunique_device', 'ip_AvgViewPerDistinct_app',               
       'ip_AvgViewPerDistinct_channel', 'ip_app_nunique_os',           
       'ip_device_os_nunique_app', 'app_AvgViewPerDistinct_ip',        
       'app_count_channel', 'app_nunique_channel', 'channel_count_app' 



        ]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : []
}




####################################################################################
def  adfraud_lightgbm(path_model_out="") :
    """

    """
    config_name  = os_get_function_name()
    data_name    = "adfraud"         ### in data/input/
    model_class  = 'LGBMClassifier'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 5000000

    def post_process_fun(y):   ### After prediction is done
        return  int(y)

    def pre_process_fun(y):    ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        ### LightGBM API model   #######################################
         'info' : """
              Use large max_bin (may be slower)
              Use small learning_rate with large num_iterations
              Use large num_leaves (may cause over-fitting)
              Use bigger training data

            Try dart
              Deal with Over-fitting
              Use small max_bin
              Use small num_leaves
              Use min_data_in_leaf and min_sum_hessian_in_leaf
              Use bagging by set bagging_fraction and bagging_freq
              Use feature sub-sampling by set feature_fraction
              Use bigger training data
              Try lambda_l1, lambda_l2 and min_gain_to_split for regularization
              Try max_depth to avoid growing deep tree
              Try extra_trees
              Try increasing path_smooth

         """,
         'model_class': model_class
        ,'model_pars' : {'objective': 'binary',
            'boosting_type'     : 'gbdt',  #  "seed": 1, 'boosting_type': 'dart',
            'metric'            : 'auc,average_precision',
            #'scale_pos_weight'  : 99,
            'is_unbalance' : True,
            'learning_rate'     : 0.001,
            'num_leaves'        : 31,      # we should let it be smaller than 2^(max_depth)
            'max_depth'         : -1,      # -1 means no limit
            'min_child_samples' : 20,      # Minimum number of df need in a child(min_data_in_leaf)
            'max_bin'           : 255,     # Number of bucketed bin for feature values
            'subsample'         : 0.6,     # Subsample ratio of the training instance.
            'subsample_freq'    : 0,       # frequence of subsample, <=0 means no enable
            'colsample_bytree'  : 0.3,     # Subsample ratio of columns when constructing each tree.
            'min_child_weight'  : 5,       # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin' : 2000,  # Number of samples for constructing bin
            'min_split_gain'    : 0,       # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha'         : 0,       # L1 regularization term on weights
            'reg_lambda'        : 0,       # L2 regularization term on weights
            # 'nthread'           : -1,
            'verbose'           : 0,
          }

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################


        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            # {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            # {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            # {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            # {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair',  'type': 'cross'},

            #### Example of Custom processor
            # {'uri': 'titanic_classifier.py::pd_colnum_quantile_norm',   'pars': {}, 'cols_family': 'colnum',   'cols_out': 'colnum_quantile_norm',  'type': '' },          
          
        ],
               }
        },

       #### Sklearn 
      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score', 'f1_score',
                                         'recall_score' ]
                        },

      'data_pars': { 'n_sample' : n_sample,
          'cols_input_type' : cols_input_type_1,
          ### family of columns for MODEL  ######################################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',
          'cols_model_group': [ # 'colnum_bin',
                                'colnum',
                                'colcat_bin',
                                # 'coltext',
                                # 'coldate',
                                # 'colcross_pair',
                               
                               ### example of custom
                               # 'colnum_quantile_norm'
                              ]

          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name )
    return model_dict

# from adfraud import adfraud_lightgbm
# print( adfraud_lightgbm )



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



####################################################################################
####### Inference ##################################################################
# def  predict(config='', nsample=10000)
from core_run import predict





###########################################################################################################
###########################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
    


