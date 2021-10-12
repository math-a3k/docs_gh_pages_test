# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

  python example/regress/regress_salary.py  train
  python example/regress/regress_salary.py  predict


"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

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
config_default  = 'salary_lightgbm'
n_sample     = 1000




####################################################################################
cols_input_type_1 = {
     "coly"   :   "salary"
    ,"colid"  :   "jobId"
    ,"colcat" :   [ "companyId", "jobType", "degree", "major", "industry" ]
    ,"colnum" :   ["yearsExperience", "milesFromMetropolis"]
    ,"coltext" :  []
    ,"coldate" :  []
    ,"colcross" : [ "jobType", "degree", "major", "industry", "yearsExperience", "milesFromMetropolis" ]
}





####### y normalization #############################################################   
def y_norm(y, inverse=True, mode='boxcox'):
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
        m0, width0 = 0.0, 350.0  ## Min, Max
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
def salary_lightgbm(path_model_out="") :
    """
        Huber Loss includes L1  regurarlization
        We test different features combinaison, default params is optimal
    """
    data_name     = "salary"
    model_class   = 'LGBMRegressor'
    n_sample      = 10**5

    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')


    model_dict = {'model_pars':
        {'model_class': model_class
        ,'model_path':  path_model_out
        ,'model_pars':  {'objective': 'huber',


        }  # default
        ,'post_process_fun': copy.deepcopy( post_process_fun)
        ,'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },


    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                                    },

    'data_pars': {
            'cols_input_type' : cols_input_type_1

            # cols['cols_model'] = cols["colnum"] + cols["colcat_bin"]  # + cols[ "colcross_onehot"]
            ,'cols_model_group': [ 'colnum', 'colcat_bin']

           ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data

    }}

    ################################################################################################
    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, data_name, os_get_function_name() )
    return model_dict


 
def salary_elasticnetcv(path_model_out=""):
    global model_name
    model_name        = 'ElasticNetCV'

    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'ElasticNetCV'
        , 'model_path': path_model_out
        , 'model_pars': {}  # default ones
        , 'post_process_fun': copy.deepcopy(post_process_fun)
        , 'pre_process_pars': {'y_norm_fun' : copy.deepcopy(pre_process_fun),

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },



    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                                    },
    'data_pars': {
            'cols_input_type' : cols_input_type_1

         ,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot', 'colcross_onehot' ]

         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
    }}


    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, model_class=os_get_function_name() )
    return model_dict





def salary_bayesian_pyro(path_model_out="") :
    global model_name
    model_name        = 'model_bayesian_pyro'
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')

    model_dict = {'model_pars': {'model_class': 'model_bayesian_pyro'
        , 'model_path': path_model_out
        , 'model_pars': {'input_width': 112, }  # default
        , 'post_process_fun': post_process_fun

        , 'pre_process_pars': {'y_norm_fun' :  copy.deepcopy(pre_process_fun) ,

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },

    'compute_pars': {'compute_pars': {'n_iter': 1200, 'learning_rate': 0.01}
                                 , 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                                    'explained_variance_score', 'r2_score', 'median_absolute_error']
                                 , 'max_size': 1000000
                                 , 'num_samples': 300
     },
    'data_pars':  {
            'cols_input_type' : cols_input_type_1


            ,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]


           ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                            }}

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, os_get_function_name() )
    return model_dict





def salary_glm( path_model_out="") :
    global model_name
    model_name        = 'TweedieRegressor'
    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='norm')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='norm')



    model_dict = {'model_pars': {'model_class': 'TweedieRegressor'  # Ridge
        , 'model_path': path_model_out
        , 'model_pars': {'power': 0, 'link': 'identity'}  # default ones
        , 'pre_process_pars': {'y_norm_fun' : pre_process_fun,

        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
               }
        },



        'compute_pars': {'metric_list': ['root_mean_squared_error', 'mean_absolute_error',
                                         'explained_variance_score',  'r2_score', 'median_absolute_error']
                                        },

    'data_pars': {
            'cols_input_type' : cols_input_type_1
            ,'cols_model_group': [ 'colnum_onehot', 'colcat_onehot' ]
         ,'filter_pars': { 'ymax' : 100000.0 ,'ymin' : 0.0 }   ### Filter data
                                }
    }

    ##### Filling Global parameters    #############################################################
    model_dict        = global_pars_update(model_dict, model_name, data_name)
    return model_dict





###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess



##################################################################################
########## Train #################################################################
from core_run import train




###################################################################################
######### Check data ##############################################################
def check():
   pass



####################################################################################
####### Inference ##################################################################
# predict(config='', nsample=10000)
from core_run import predict





###########################################################################################################
###########################################################################################################
if __name__ == "__main__":
        import fire
        fire.Fire()

