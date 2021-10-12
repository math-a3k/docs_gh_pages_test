# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
  python outlier_predict.py  train    > zlog/log_outlier_train.txt 2>&1
  python outlier_predict.py  predict  > zlog/log_outlier_predict.txt 2>&1


"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')


####################################################################################
###### Path ########################################################################
from source import util_feature
config_file  = os.path.basename(__file__)

print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(root, dir_data)


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name


def global_pars_update(model_dict,  data_name, config_name):
    m                      = {}
    m['config_path']       = root + f"/{config_file}"
    m['config_name']       = config_name

    ##### run_Preoprocess ONLY
    m['path_data_preprocess'] = root + f'/data/input/{data_name}/train/'

    ##### run_Train  ONLY
    m['path_data_train']   = root + f'/data/input/{data_name}/train/'
    m['path_data_test']    = root + f'/data/input/{data_name}/test/'
    #m['path_data_val']    = root + f'/data/input/{data_name}/test/'
    m['path_train_output']    = root + f'/data/output/{data_name}/{config_name}/'
    m['path_train_model']     = root + f'/data/output/{data_name}/{config_name}/model/'
    m['path_features_store']  = root + f'/data/output/{data_name}/{config_name}/features_store/'
    m['path_pipeline']        = root + f'/data/output/{data_name}/{config_name}/pipeline/'


    ##### Prediction
    m['path_pred_data']    = root + f'/data/input/{data_name}/test/'
    m['path_pred_pipeline']= root + f'/data/output/{data_name}/{config_name}/pipeline/'
    m['path_pred_model']   = root + f'/data/output/{data_name}/{config_name}/model/'
    m['path_pred_output']  = root + f'/data/output/{data_name}/pred_{config_name}/'


    #####  Generic
    m['n_sample']             = model_dict['data_pars'].get('n_sample', 5000)

    model_dict[ 'global_pars'] = m
    return model_dict


####################################################################################
##### Params########################################################################
config_default   = 'titanic_pyod'          ### name of function which contains data configuration


# data_name    = "titanic"     ### in data/input/
cols_input_type_2 = {
     "coly"   :   "Survived"
    ,"colid"  :   "PassengerId"
    ,"colcat" :   ["Sex", "Embarked" ]
    ,"colnum" :   ["Pclass", "Age","SibSp", "Parch","Fare"]
    ,"coltext" :  ["Name", "Ticket"]
    ,"coldate" :  []
    ,"colcross" : [ "Name", "Sex", "Ticket","Embarked","Pclass", "Age","SibSp", "Parch","Fare" ]
}


####################################################################################
def titanic_pyod(path_model_out="") :
    """ All Models  :    https://pyod.readthedocs.io/en/latest/pyod.html
        pyod.models.abod 
        pyod.models.auto_encoder 
        pyod.models.cblof 
        pyod.models.cof 
        pyod.models.combination 
        pyod.models.copod 
        pyod.models.feature_bagging 
        pyod.models.hbos 
        pyod.models.iforest 
        pyod.models.knn 
        pyod.models.lmdd 
        pyod.models.loda 
        pyod.models.lof 
        pyod.models.loci 
        pyod.models.lscp 
        pyod.models.mad 
        pyod.models.mcd 
        pyod.models.mo_gaal 
        pyod.models.ocsvm 
        pyod.models.pca 
        pyod.models.sod 
        pyod.models.so_gaal 
        pyod.models.sos 
        pyod.models.vae 
        pyod.models.xgbod 
         contents
        Utility Functions
        pyod.utils.data 
        pyod.utils.example 
        pyod.utils.stat_models 
        pyod.utils.utility 

        clf.fit(X)
        scores_pred = clf.decision_function(X) * -1
        y_pred = clf.predict(X)

    """
    config_name  = os_get_function_name()
    data_name    = "titanic"         ### in data/input/
    model_class  = 'COPOD'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 1000

    def post_process_fun(y):   ### After prediction is done
        return  float(y)

    def pre_process_fun(y):    ### Before the prediction is done
        return   1 if y > 0.5 else 0    ### proba

    ##### Model pars for each
    mpars = {
      'IForest' : {},
      'VAE' :   { 'encoder_neurons' : [13, 8, 4 ],
                 'decoder_neurons' : [4, 8, 13]
        },

       'COPOD' : {'contamination' : 0.05}
    }


    model_dict = {'model_pars': {
        ### LightGBM API model   #######################################
         'model_class':  model_class
        ,'model_pars' :  mpars[model_class]

        , 'post_process_fun' : post_process_fun   ### After prediction  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,  ### Before training  ##########################


        ### Pipeline for data processing ##############################
        'pipe_list': [
            {'uri': 'source/prepro.py::pd_coly',                 'pars': {}, 'cols_family': 'coly',       'cols_out': 'coly',           'type': 'coly'         },
            #{'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            #{'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            # {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair',  'type': 'cross'}
        ],
               }
        },

      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                        },

      'data_pars': { 'n_sample' : n_sample,
          'cols_input_type' : cols_input_type_2,
          ### family of columns for MODEL  #########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate', 'coltext',
          'cols_model_group': [ 'colnum',
                                'colcat_onehot',
                                # 'coltext',
                                # 'coldate',
                                # 'colcross_pair'
                              ]

          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name )
    return model_dict



#####################################################################################
########## Profile data #############################################################
def data_profile(path_data_train="", path_model="", n_sample= 5000):
   from source.run_feature_profile import run_profile
   run_profile(path_data   = path_data_train,
               path_output = path_model + "/profile/",
               n_sample    = n_sample,
              )


###################################################################################
########## Preprocess #############################################################
### def preprocess(config='', nsample=1000):
from core_run import preprocess



##################################################################################
########## Train #################################################################
## def train(config=None, nsample=None):
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
"""
python  outlier_predict.py  data_profile
python  outlier_predict.py  preprocess  --nsample 100
python  outlier_predict.py  train       --nsample 200
python  outlier_predict.py  check
python  outlier_predict.py  predict


"""
if __name__ == "__main__":
    import fire
    fire.Fire()
    

