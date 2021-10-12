# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
  python regress_house.py  train    > zlog/log-house.txt 2>&1
  python regress_house.py  check
  python regress_house.py  predict



"""
import warnings
warnings.filterwarnings('ignore')
import os, sys, copy
############################################################################
from source import util_feature



####################################################################################################
###### Path ########################################################################################
root_repo      =  os.path.abspath(os.getcwd()).replace("\\", "/") + "/"     ; print(root_repo)
THIS_FILEPATH  =  os.path.abspath(__file__)

sys.path.append(root_repo)
from source.util_feature import save,os_get_function_name


def global_pars_update(model_dict,  data_name, config_name):
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
config_default  = 'house_price_lightgbm'



cols_input_type_2 = {
     "coly"   : "SalePrice"
    ,"colid"  : "Id"

    ,"colcat" : [ "MSSubClass", "MSZoning", "Street" ]

    ,"colnum" : [ "LotArea", "OverallQual", "OverallCond", 	]

    ,"coltext"  : []
    ,"coldate" : []   # ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
    ,"colcross" : []

}



cols_input_type_1 = {
     "coly"   : "SalePrice"
    ,"colid"  : "Id"

    ,"colcat" : [  "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
          "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

    ,"colnum" : [ "LotArea", "OverallQual", "OverallCond", "MasVnrArea",
          "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]

    ,"coltext"  : []
    ,"coldate" : []   # ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
    ,"colcross" : []

}




#####################################################################################
####### y normalization #############################################################
def y_norm(y, inverse=True, mode='boxcox'):
    ## Normalize the input/output
    if mode == 'boxcox':
        width0 = 53.0  # 0,1 factor
        k1 = 1.0  # Optimal boxCox lambda for y
        if inverse:
                y2 = y * width0
                y2 = ((y2 * k1) + 1) ** (1 / k1)
                return y2
        else:
                y1 = (y ** k1 - 1) / k1
                y1 = y1 / width0
                return y1

    if mode == 'norm':
        m0, width0 = 0.0, 100000.0  ## Min, Max
        if inverse:
                y1 = (y * width0 + m0)
                return y1

        else:
                y2 = (y - m0) / width0
                return y2
    else:
            return y



####################################################################################
##### Params########################################################################
def house_price_lightgbm(path_model_out="") :
    """
        Huber Loss includes L1  regurarlization
        We test different features combinaison, default params is optimal
    """
    data_name         = 'house_price'
    model_name        = 'LGBMRegressor'
    n_sample          = 20000


    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')


    model_dict = {'model_pars': {  'model_path'       : path_model_out

        , 'model_class': model_name   ### Actual Class Name
        , 'model_pars'       : {}  # default ones of the model name

        , 'post_process_fun' : post_process_fun
        , 'pre_process_pars' : {'y_norm_fun' : copy.deepcopy(pre_process_fun),

            ### Pipeline for data processing.
            # 'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
           'pipe_list'  : [ 'filter', 'label',   'dfcat_bin'  ]

           }
                                                         },
    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                    },
    'data_pars': {
        'cols_input_type' : cols_input_type_1,

        # 'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]
        'cols_model_group': [ 'colnum', 'colcat_bin' ]


       ,'filter_pars': { 'ymax' : 1000000.0 ,'ymin' : 0.0 }   ### Filter data

    }}


    ################################################################################################
    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict







def house_price_elasticnetcv(path_model_out=""):
    model_name   = 'ElasticNetCV'
    config_name  = 'house_price_elasticnetcv'
    n_sample     = 1000


    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')


    model_dict = {'model_pars': {'model_class': 'ElasticNetCV'
        , 'model_path': path_model_out



        , 'model_pars': {}  # default ones
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

                        ### Pipeline for data processing.
                       # 'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
                       'pipe_list' : [ 'filter', 'label',   'dfcat_hot' ]
                                                     }
                                                         },
    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                    },

    'data_pars': {
        'cols_input_type' : cols_input_type_1,

        # 'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]
        'cols_model_group': [ 'colnum', 'colcat_onehot' ]

         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
    }}


    ################################################################################################
    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict




####################################################################################################
########## Init variable ###########################################################################
globals()[config_name]()



###################################################################################
########## Profile data #############################################################
def data_profile():
   from source.run_feature_profile import run_profile
   run_profile(path_data   = path_data_train,
               path_output = path_model + "/profile/",
               n_sample    = 5000,
              )



###################################################################################
########## Preprocess #############################################################
def preprocess():
    from source import run_preprocess_old
    run_preprocess_old.run_preprocess(model_name =  config_name,
                                      path_data         =  path_data_train,
                                      path_output       =  path_model,
                                      path_config_model =  path_config_model,
                                      n_sample          =  n_sample,
                                      mode              =  'run_preprocess')


############################################################################
########## Train ###########################################################
def train():
    from source import run_train
    run_train.run_train(config_name=  config_name,
                        path_data_train=  path_data_train,
                        path_output       =  path_model,
                        config_path=  path_config_model, n_sample = n_sample)


###################################################################################
######### Check model #############################################################
def check():
   pass


########################################################################################
####### Inference ######################################################################
def predict():
    from source import run_inference
    run_inference.run_predict(model_name,
                            path_model  = path_model,
                            path_data   = path_data_test,
                            path_output = path_output_pred,
                            n_sample    = n_sample)


def run_all():
    data_profile()
    preprocess()
    train()
    check()
    predict()



###########################################################################################################
###########################################################################################################
"""
python  regress_house.py  preprocess
python  regress_house.py  train
python  regress_house.py  check
python  regress_house.py  predict
python  regress_house.py  run_all
"""
if __name__ == "__main__":
        import fire
        fire.Fire()
