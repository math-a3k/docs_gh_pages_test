# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
### Usage:
  python tseries.py  train      --config  config1
  python tseries.py  predict    --config  config1

"""
import warnings, copy, os, sys, pandas as pd
warnings.filterwarnings("ignore")

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
    m["config_path"]       = THIS_FILEPATH  
    m["config_name"]       = config_name

    #### peoprocess input path
    m["path_data_preprocess"] = dir_data + f"/input/{data_name}/train/"

    #### train input path
    #dir_data_url              = "https://github.com/arita37/dsa2_data/tree/master/"  #### Remote Data directory
    m["path_data_train"]      = dir_data + f"/input/{data_name}/train/"
    m["path_data_test"]       = dir_data + f"/input/{data_name}/test/"
    #m["path_data_val"]       = dir_data + f"/input/{data_name}/test/"

    #### train output path
    m["path_train_output"]    = dir_data + f"/output/{data_name}/{config_name}/"
    m["path_train_model"]     = dir_data + f"/output/{data_name}/{config_name}/model/"
    m["path_features_store"]  = dir_data + f"/output/{data_name}/{config_name}/features_store/"
    m["path_pipeline"]        = dir_data + f"/output/{data_name}/{config_name}/pipeline/"


    #### predict  input path
    m["path_pred_data"]       = dir_data + f"/input/{data_name}/test/"
    m["path_pred_pipeline"]   = dir_data + f"/output/{data_name}/{config_name}/pipeline/"
    m["path_pred_model"]      = dir_data + f"/output/{data_name}/{config_name}/model/"

    #### predict  output path
    m["path_pred_output"]     = dir_data + f"/output/{data_name}/pred_{config_name}/"

    #####  Generic
    m["n_sample"]             = model_dict["data_pars"].get("n_sample", 5000)

    model_dict[ "global_pars"] = m
    return model_dict


####################################################################################
##### Params########################################################################
config_default   = "config1"    ### name of function which contains configuration

cols_input_type_1 = {
     "coly"   :   "sales"
    ,"colid"  :   "id_date"   ### used for JOIN tables, duplicate date
    ,"colcat" :   ["store", "item" ]
    ,"colnum" :   []
    ,"coltext" :  []
    ,"coldate" :  []

    ### Specific for time sereis
    ,"col_tseries" :  ['date', 'store', 'item', 'sales']

    ,"colcross" : [ ]
}


####################################################################################
def config1() :
    """  ONE SINGLE DICT Contains all needed informations forused for tseries_demand
    """
    data_name    = "tseries_demand"         ### in data/input/
    model_class  = "source/models/model_tseries.py:LGBMRegressor"  ### ACTUAL Class name
    n_sample     = 500

    def post_process_fun(y):   ### After prediction is done
        # ynew = np.exp(y) - 1.0
        ynew = float(y)
        return  ynew

    def pre_process_fun(y):    ### Before the prediction is done
        # ynew = np.log(y+1)
        ynew = float(y)
        return  ynew


    model_dict = {"model_pars": {
         "model_class": model_class
        ,"model_pars" : {"objective": "huber",    ### Regression Type Loss
                           "n_estimators": 100,
                           "learning_rate":0.001,
                           "boosting_type":"gbdt",     ### Model hyperparameters
                           "early_stopping_rounds": 5

        }

        , "post_process_fun" : post_process_fun
        , "pre_process_pars" : {"y_norm_fun" :  pre_process_fun ,
            ### Pipeline for data processing ##############################
            "pipe_list": [
            #### Example of Custom processor
            {"uri":  THIS_FILEPATH + "::pd_dsa2_custom",
                "pars"        : {'coldate': 'date'},
                "cols_family" : "col_tseries",
                "cols_out"    : "tseries_feat",  "type": "" },

            ],
        }
    },

    "compute_pars": { "metric_list": ['root_mean_squared_error', 'mean_absolute_error',
                                       'explained_variance_score', 'r2_score', 'median_absolute_error']
    },

    "data_pars": { "n_sample" : n_sample,
          "download_pars" : None,
          ### Raw data:  column input ##################
          "cols_input_type" : cols_input_type_1,

          ### Model Input :  Merge family of columns
          "cols_model_group": [ ### cols_out of  pd_dsa2_custom
                               "tseries_feat"
                              ]

          #### Model Input : Separate Category Sparse from Continuous : Aribitrary name is OK (!)
         ,'cols_model_type': {
             'My123_continuous' : [ 'tseries_feat',   ],
             'my_sparse'        : [ 'colcat',  ],
          }

          ### Filter data rows   #################
         ,"filter_pars": { "ymax" : 999999999 ,"ymin" : -1 }

         }
    }

    ##### Filling Global parameters
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict




def pd_dsa2_custom(df: pd.DataFrame, col: list=None, pars: dict=None):
    """
    Example of custom Processor Combining
    Usage :
    ,{"uri":  THIS_FILEPATH + "::pd_dsa2_custom",   "pars": {'coldate': 'date'}, "cols_family": "coldate",   "cols_out": "coldate_features1",  "type": "" },

    """
    prefix = "tseries_feat"  ### Used acolumn index
    #### Inference time LOAD previous pars  ###########################################
    from prepro import prepro_load, prepro_save
    prepro, pars_saved, cols_saved = prepro_load(prefix, pars)


    #### Do something #################################################################
    from source.prepro_tseries import pd_ts_date, pd_ts_rolling
    if prepro is None :   ###  Training time
        dfy, coly = pars['dfy'], pars['coly']
        coldate   = pars['coldate']
        #df        = df.set_index(coldate)

        #### time features
        dfi, coli = pd_ts_date(df, cols=[coldate], pars={'col_add': ['day', 'month', 'year', 'weekday']})
        df_new    = dfi

        #### Rolling features
        dfi, coli = pd_ts_rolling(df,  cols= ['date', 'item', 'store', 'sales'],
                                  pars= {'col_groupby' : ['store','item'],
                                         'col_stat'    : 'sales',
                                         'lag_list'    : [7, 30]})
        df_new    = pd.concat([df_new , dfi], axis=1)

    else :  ### predict time
        """
          y is NOT provided, need to calculate y based on past values.
          Auto-regressive feature engineering.

        """


        pars = pars_saved  ##merge
    ###################################################################################



    ###################################################################################
    ### Clean up the df ###############################################################
    df_new.index   = df.index  ### Impt for JOIN
    df_new.columns = [col + f"_{prefix}"  for col in df_new.columns ]
    cols_new       = list(df_new.columns)

    ###### Training time save all #####################################################
    df_new, col_pars = prepro_save(prefix, pars, df_new, cols_new, prepro)
    return df_new, col_pars






########## Preprocess #############################################################
### def preprocess(config="", nsample=1000):
from core_run import preprocess


########## Train #################################################################
# def train(config=None, nsample=None):
from core_run import train


####### Inference ##################################################################
# predict(config="", nsample=10000)
from core_run import predict


###########################################################################################################
if __name__ == "__main__":
    from pyinstrument import Profiler;  profiler = Profiler() ; profiler.start()
    import fire
    fire.Fire()
    profiler.stop() ; print(profiler.output_text(unicode=True, color=True))





