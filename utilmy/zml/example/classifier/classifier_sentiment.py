# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
You can put hardcode here, specific to titatinic dataet
All in one file config

  python classifier_sentiment.py  preprocess

  python classifier_sentiment.py  train    > zlog/log-airbnb.txt 2>&1
  python classifier_sentiment.py  check    > zlog/log-airbnb.txt 2>&1
  python classifier_sentiment.py  predict  > zlog/log-airbnb.txt 2>&1





"""
import warnings, copy, os, sys
warnings.filterwarnings('ignore')

###################################################################################
from source import util_feature


####################################################################################
###### Path ########################################################################
print( os.getcwd())
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)

dir_data  = os.path.abspath( root + "/data/" ) + "/"
dir_data  = dir_data.replace("\\", "/")
print(dir_data)


def global_pars_update(model_dict,  data_name, config_name):
    m                      = {}
    model_name             = model_dict['model_pars']['model_class']
    m['config_path'] = root + f"/{config_file}"
    m['model_class']       = config_name

    m['path_data_train']   = f'data/input/{data_name}/train/'
    m['path_data_test']    = f'data/input/{data_name}/test/'

    m['path_model']        = f'data/output/{data_name}/{model_name}/'
    m['path_output_pred']  = f'data/output/{data_name}/pred_{config_name}/'
    m['n_sample']          = model_dict['data_pars'].get('n_sample', 5000)

    model_dict[ 'global_pars'] = m
    return model_dict


def os_get_function_name():
    import sys
    return sys._getframe(1).f_code.co_name


####################################################################################
config_file     = "classifier_sentiment.py"
config_default  = 'sentiment_lightgbm'


#####################################################################################


####################################################################################
##### Params########################################################################
cols_input_type_1 = {
	 "coly"     : ""
	,"colid"    : ""
	,"colcat"   :  [  ]
	,"colnum"   :  [       ]
	,"coltext"  :  [  ]
	,"coldate"  :  [  ]
	,"colcross" : [  ]
}






####################################################################################
def sentiment_lightgbm(path_model_out="") :
    """

    """
    data_name    = "sentiment"   ###in data/
    model_name   = 'LGBMClassifier'
    n_sample     = 1000

    def post_process_fun(y):
        return int(y)


    def pre_process_fun(y):
        return int(y)


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
            # {'uri': 'source/prepro.py::pd_colnum_bin',           'pars': {}, 'cols_family': 'colnum',     'cols_out': 'colnum_bin',     'type': ''             },
            # {'uri': 'source/prepro.py::pd_colnum_binto_onehot',  'pars': {}, 'cols_family': 'colnum_bin', 'cols_out': 'colnum_onehot',  'type': ''             },
            # {'uri': 'source/prepro.py::pd_colcat_bin',           'pars': {}, 'cols_family': 'colcat',     'cols_out': 'colcat_bin',     'type': ''             },
            # {'uri': 'source/prepro.py::pd_colcat_to_onehot',     'pars': {}, 'cols_family': 'colcat_bin', 'cols_out': 'colcat_onehot',  'type': ''             },
            # {'uri': 'source/prepro.py::pd_colcross',             'pars': {}, 'cols_family': 'colcross',   'cols_out': 'colcross_pair_onehot',  'type': 'cross'}
        ],
        }
    },

    'compute_pars': { 'metric_list': ['root_mean_squared_error', 'mean_absolute_error',   #### sklearm names
                                      'explained_variance_score', 'r2_score', 'median_absolute_error']
                    },

    'data_pars': {
          'cols_input_type' : cols_input_type_1


         #"colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
         #"colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
         #'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
         # 'coldate', #'coltext',

         ,'cols_model_group': [  'colnum'
                                #,'colcat_bin'
                                #,'coltext'
                              ]


         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }   ### Filter data

         }}


    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
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
def preprocess(config=None, nsample=None):
    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_preprocess, run_preprocess_old
    run_preprocess.run_preprocess(config_name=  config_name,
                                  path_data         =  m['path_data_train'],
                                  path_output       =  m['path_model'],
                                  path_config_model =  m['config_path'],
                                  n_sample          =  nsample if nsample is not None else m['n_sample'],
                                  mode              =  'run_preprocess')


##################################################################################
########## Train #################################################################
def train(config=None, nsample=None):

    config_name  = config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict)

    from source import run_train
    run_train.run_train(config_name=  config_name,
                        path_data_train=  m['path_data_train'],
                        path_output       =  m['path_model'],
                        config_path=  m['config_path'],
                        n_sample          =  nsample if nsample is not None else m['n_sample']
                        )


###################################################################################
######### Check data ##############################################################
def check():
   pass




####################################################################################
####### Inference ##################################################################
def predict(config=None, nsample=None):
    config_name  =  config  if config is not None else config_default
    mdict        = globals()[config_name]()
    m            = mdict['global_pars']
    print(mdict['data_pars']['cols_input_type'])
    print(m)

    from source import run_inference,run_inference
    run_inference.run_predict(config_name,
                              path_model  = m['path_model'],
                              path_data   = m['path_data_test'],
                              path_output = m['path_output_pred'],
                              cols_group  = mdict['data_pars']['cols_input_type'],
                              n_sample    = nsample if nsample is not None else m['n_sample']
                              )


def run_all():
    data_profile()
    preprocess()
    train()
    check()
    predict()




###########################################################################################################
###########################################################################################################
"""
python  classifier_sentiment.py  preprocess
python  classifier_sentiment.py  train
python  classifier_sentiment.py  check
python  classifier_sentiment.py  predict
python  classifier_sentiment.py  run_all
"""
if __name__ == "__main__":
        import fire
        fire.Fire()













def sentiment_elasticnetcv(path_model_out=""):
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



def sentiment_bayesian_pyro(path_model_out="") :
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

                        ### Pipeline for data processing.
                       'pipe_list'  : [ 'filter', 'label', 'dfnum_bin', 'dfnum_hot',  'dfcat_bin', 'dfcat_hot', 'dfcross_hot', ]
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
    return model_dict


