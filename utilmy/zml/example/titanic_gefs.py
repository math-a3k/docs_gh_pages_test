# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
  ipython titanic_gefs.py  train      --config  config1  --pdb
  ipython titanic_gefs.py  predict    --config  config1


"""
import warnings, copy, os, sys
warnings.filterwarnings("ignore")

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
    dir_data_url              = "https://github.com/arita37/dsa2_data/tree/master/"  #### Remote Data directory
    m["path_data_train"]      = dir_data_url + f"/input/{data_name}/train/"
    m["path_data_test"]       = dir_data_url + f"/input/{data_name}/test/"
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
config_default   = "config1"    ### name of function which contains data configuration


cols_input_type_1 = {
     "coly"   :   "Survived"
    ,"colid"  :   "PassengerId"
    ,"colcat" :   []  # ["Sex", "Embarked", "Pclass", ]
    ,"colnum" :   [ "Age","SibSp", "Parch","Fare"]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : [  ]
}


colcat = cols_input_type_1['colcat']
colnum = cols_input_type_1['colnum']
coly   = cols_input_type_1['coly']

# colcat_unique = {  col: list(df[col].unique())  for col in colcat }
colcat_unique = {  'Sex': 2, 'Embarked': 2 }   ### nbf unique values

colcat_bin  = []  ### Compute the bins for category


####################################################################################
def config1() :
    """
       ONE SINGLE DICT Contains all needed informations for
       used for titanic classification task
    """
    data_name    = "titanic"         ### in data/input/
    model_class  = "source/models/model_gefs.py::RandomForest"  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 1000

    def post_process_fun(y):   ### After prediction is done
        return  int(y)

    def pre_process_fun(y):    ### Before the prediction is done
        return  int(y)


    model_dict = { "model_pars": {
         "model_class": model_class
        ,"model_pars" : {'cat': 10, 'n_estimators': 5
                        }

        , "post_process_fun" : post_process_fun   ### After prediction  ##########################################
        , "pre_process_pars" : {"y_norm_fun" :  pre_process_fun ,  ### Before training  ##########################
            ### Pipeline for data processing ##############################
            "pipe_list": [
            {"uri": "source/prepro.py::pd_coly",                 "pars": {}, "cols_family": "coly",       "cols_out": "coly",           "type": "coly"         },

            # {"uri": "source/prepro.py::pd_colnum_bin",           "pars": {}, "cols_family": "colnum",     "cols_out": "colnum_bin",     "type": ""             },
            # {"uri": "source/prepro.py::pd_colnum_binto_onehot",  "pars": {}, "cols_family": "colnum_bin", "cols_out": "colnum_onehot",  "type": ""             },

            {"uri": "source/prepro.py::pd_colcat_bin",           "pars": {}, "cols_family": "colcat",     "cols_out": "colcat_bin",     "type": ""             },
            #{"uri": "source/prepro.py::pd_colcat_to_onehot",     "pars": {}, "cols_family": "colcat_bin", "cols_out": "colcat_onehot",  "type": ""             },

            ],
        }
    },

    "compute_pars": { "metric_list": ["accuracy_score","average_precision_score"]
                        # ,"mlflow_pars" : {}   ### Not empty --> use mlflow
    },

    "data_pars": { "n_sample" : n_sample, "download_pars" : None,
        ### Raw data:  column input
        "cols_input_type" : cols_input_type_1,

          ### Model Input :  Merge family of columns
          "cols_model_group": [ "colnum",
                                "colcat_bin",  ]

          #### Model Input : Separate Category Sparse from Continuous : Aribitrary name is OK (!)
          ,'cols_model_type': {
             #'continuous'   : [ 'colnum',   ],
             #'sparse'       : [ 'colcat_bin', 'colnum_bin',  ],
             #'my_split_23'  : [ 'colnum_bin',   ],
          },

        'data_pars' :{ 'cols_model_type': {},
            # Raw dataset, pre preprocessing
            "dataset_path" : "",
            "batch_size":128,   ### Mini Batch from data
            # Needed by getdataset
            "clean" : False,
            "data_path": "",

            'colcat_unique' : colcat_unique,
            'colcat_bin'    : colcat_bin,
            'colcat'        : colcat,
            'colnum'        : colnum,
            'coly'          : coly,
            'colembed_dict' : None
        }

        ### Filter data rows   #################################
        ,"filter_pars": { "ymax" : 2 ,"ymin" : -1 }
    }
    }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict





###################################################################################
########## Preprocess #############################################################
### def preprocess(config="", nsample=1000):
from core_run import preprocess


##################################################################################
########## Train #################################################################
# def train(config=None, nsample=None):
from core_run import train



####################################################################################
####### Inference ##################################################################
# predict(config="", nsample=10000)
from core_run import predict




###########################################################################################################
###########################################################################################################
if __name__ == "__main__":
    from pyinstrument import Profiler;  profiler = Profiler() ; profiler.start()
    import fire
    fire.Fire()
    profiler.stop() ; print(profiler.output_text(unicode=True, color=True))
    



