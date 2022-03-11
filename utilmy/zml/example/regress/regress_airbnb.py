# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

  python regress_airbnb.py  preprocess
  python regress_airbnb.py  train
  python regress_airbnb.py  predict


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



######################################################################################
config_default  = 'airbnb_lightgbm'



#####################################################################################
####### y normalization #############################################################   
def y_norm(y, inverse=True, mode='boxcox'):
    """function y_norm
    Args:
        y:   
        inverse:   
        mode:   
    Returns:
        
    """
    ## Normalize the input/output
    if mode == 'boxcox':
        width0 = 53.0  # 0,1 factor
        k1 = 0.6145279599674994  # Optimal boxCox lambda for y
        if inverse:
                y2 = y * width0
                y2 = ((y2 * k1) + 1) ** (1 / k1)
                return y2
        else:
                y1 = (y ** k1 - 1) / k1
                y1 = y1 / width0
                return y1

    if mode == 'norm':
        m0, width0 = 0.0, 0.01  ## Min, Max
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
cols_input_type_1 = {
	 "coly"     : "price"
	,"colid"    : "id"
	,"colcat"   :  [ "cancellation_policy", "host_response_rate", "host_response_time" ]
	,"colnum"   :  [ "review_scores_communication", "review_scores_location", "review_scores_rating"         ]
	,"coltext"  :  [ "house_rules", "neighborhood_overview", "notes", "street"  ]
	,"coldate"  :  [ "calendar_last_scraped", "first_review", "host_since" ]
	,"colcross" :  [ "review_scores_communication", "review_scores_location", "cancellation_policy", "host_response_rate"]
}


cols_input_type_2 = {
	 "coly"     : "price"
	,"colid"    : "id"
	,"colcat"   : ["host_id", "host_location", "host_response_time","host_response_rate","host_is_superhost","host_neighbourhood","host_verifications","host_has_profile_pic","host_identity_verified","street","neighbourhood","neighbourhood_cleansed", "neighbourhood_group_cleansed","city","zipcode", "smart_location","is_location_exact","property_type","room_type", "accommodates","bathrooms","bedrooms", "beds","bed_type","guests_included","calendar_updated", "license","instant_bookable","cancellation_policy","require_guest_profile_picture","require_guest_phone_verification","scrape_id"]
	,"colnum"   : ["host_listings_count","latitude", "longitude","square_feet","weekly_price","monthly_price", "security_deposit","cleaning_fee","extra_people", "minimum_nights","maximum_nights","availability_30","availability_60","availability_90","availability_365","number_of_reviews","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication", "review_scores_location","review_scores_value","calculated_host_listings_count","reviews_per_month"]
	,"coltext"  : ["name","summary", "space","description", "neighborhood_overview","notes","transit", "access","interaction", "house_rules","host_name","host_about","amenities"]
	, "coldate" : ["last_scraped","host_since","first_review","last_review"]
	,"colcross" : ["name","host_is_superhost","is_location_exact","monthly_price","review_scores_value","review_scores_rating","reviews_per_month"]
}


####################################################################################
def airbnb_lightgbm(path_model_out="") :
    """

    """
    data_name    = "airbnb"   ###in data/
    model_name   = 'LGBMRegressor'


    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')


    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')


    #############################################################################
    model_dict = {'model_pars': {'model_class': model_name

        ,'model_path': path_model_out
        ,'model_pars': {'objective': 'huber',

                       }  # lightgbm one
        ,'post_process_fun': post_process_fun
        ,'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

        ### Pipeline for data processing ########################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_coltext',              'pars': {}, 'cols_family': 'coltext',    'cols_out': 'coltext_svd',    'type': ''             },
            {'uri': 'source/prepro.py::pd_coldate',              'pars': {}, 'cols_family': 'coldate',    'cols_out': 'coldate',        'type': ''             },
            {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],

        }
    },

    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',   #### sklearm names
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                    },

    'data_pars': {
          'cols_input_type' : cols_input_type_1

         # "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
         # "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
         # 'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
         # 'coldate', #'coltext', 'coltext_svd'

         ,'cols_model_group': [  'colnum'
                                ,'colcat_bin'
                                ,'coltext_svd'
                              ]

         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data

         }}


    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, os_get_function_name() )

    return model_dict
 



#####################################################################################
########## Profile data #############################################################
# def data_profile(path_data="", path_output="", n_sample= 5000):
from core_run import  data_profile



###################################################################################
########## Preprocess #############################################################
### def preprocess(config="", nsample=1000):
from core_run import preprocess



##################################################################################
########## Train #################################################################
## def train(config=None, nsample=None):
from core_run import train



####################################################################################
####### Inference ##################################################################
# predict(config="", nsample=10000)
from core_run import predict



#######################################################################################
#######################################################################################
"""
python  regress_airbnb.py  preprocess
python  regress_airbnb.py  train
python  regress_airbnb.py  predict
python  regress_airbnb.py  run_all
"""
if __name__ == "__main__":
        import fire
        fire.Fire()






def airbnb_elasticnetcv(path_model_out=""):
    """function airbnb_elasticnetcv
    Args:
        path_model_out:   
    Returns:
        
    """
    global model_name
    model_name        = 'ElasticNetCV'

    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'ElasticNetCV'
        , 'model_path': path_model_out
        , 'model_pars': {}  # default ones
        , 'post_process_fun': post_process_fun
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

                        ### Pipeline for data processing.
                       'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot',
                                        'dfdate', 'dftext'

                        ]
                                                     }
                                                         },
    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                                    },
    'data_pars': {
            'cols_input_type' : cols_input_type_1,

            'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]


         ,'cols_model': []  # cols['colcat_model'],
         ,'coly': []        # cols['coly']
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                            }}
    return model_dict



