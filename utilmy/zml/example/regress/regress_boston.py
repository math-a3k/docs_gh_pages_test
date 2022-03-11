# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
  python regress_boston.py  train
  python regress_boston.py  check
  python regress_boston.py  predict

https://causalnex.readthedocs.io/en/stable/causalnex.structure.DAGRegressor.html

https://www.splunk.com/en_us/blog/platform/causal-inference-determining-influence-in-messy-data.html


"""
import warnings
warnings.filterwarnings('ignore')
import os, sys, pandas as pd, copy, pdb
#####################################################################################
from source import util_feature



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


####################################################################################
config_default = 'boston_lightgbm'


cols_input_type_1 = {
     "coly"     : "SalePrice"
    ,"colid"    : "Id"
    ,"colcat"   : []
    ,"colnum"   : []
    ,"coltext"  : []
    ,"coldate"  : []   # ["YearBuilt", "YearRemodAdd", "GarageYrBlt"]
    ,"colcross" : []

},




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
def boston_lightgbm(path_model_out="") :
    """
        Huber Loss includes L1  regurarlization
        We test different features combinaison, default params is optimal
    """
    data_name         = "boston"
    model_name        = 'LGBMRegressor'
    n_sample          = 10**5

    def post_process_fun(y):
        return y_norm(y, inverse=True, mode='boxcox')

    def pre_process_fun(y):
        return y_norm(y, inverse=False, mode='boxcox')


    model_dict = {'model_pars':
        {'model_class': model_name
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


def boston_causalnex(path_model_out="") :
    """
       Contains all needed informations for Light GBM Classifier model,
       used for titanic classification task
    """
    data_name    = "boston"         ### in data/input/
    model_class  = 'DAGRegressor'  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 1000

    def post_process_fun(y):
        ### After prediction is done
        return  int(y)

    def pre_process_fun(y):
        ### Before the prediction is done
        return  int(y)


    model_dict = {'model_pars': {
        'model_path'       : path_model_out

        ### LightGBM API model   #######################################
        ,'model_class': model_class
        ,'model_pars' : {
                'alpha' : 0.1,
                'beta' : 0.9,
                'fit_intercept' :True,
                'hidden_layer_units': None,
                'dependent_target' : True,
                'enforce_dag' :True
        }

        ### After prediction  ##########################################
        , 'post_process_fun' : post_process_fun


        ### Before training  ##########################################
        , 'pre_process_pars' : {'y_norm_fun' :  pre_process_fun ,


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

      'compute_pars': { 'metric_list': ['accuracy_score','average_precision_score']
                      },

      'data_pars': { 'n_sample' : n_sample,
          'cols_input_type' : cols_input_type_1,

          ### family of columns for MODEL  ########################################################
          #  "colnum", "colnum_bin", "colnum_onehot", "colnum_binmap",  #### Colnum columns
          #  "colcat", "colcat_bin", "colcat_onehot", "colcat_bin_map",  #### colcat columns
          #  'colcross_single_onehot_select', "colcross_pair_onehot",  'colcross_pair',  #### colcross columns
          #  'coldate',  'coltext',
          'cols_model_group': [ 'colnum_bin',
                                'colcat_bin',
                                # 'coltext',
                                # 'coldate',
                                # 'colcross_pair'
                              ]

          ### Filter data rows   ##################################################################
         ,'filter_pars': { 'ymax' : 2 ,'ymin' : -1 }

         }
      }

    ##### Filling Global parameters    ############################################################
    model_dict        = global_pars_update(model_dict, data_name, config_name=os_get_function_name() )
    return model_dict











#!/usr/bin/env python
# coding: utf-8
"""


    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.                                                      
                                                                                   
    :Attribute Information (in order):                                             
        - CRIM     per capita crime rate by town                                   
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.                                                                                   
        - INDUS    proportion of non-retail business acres per town                
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)                                                                              
        - NOX      nitric oxides concentration (parts per 10 million)              
        - RM       average number of rooms per dwelling                            
        - AGE      proportion of owner-occupied units built prior to 1940          
        - DIS      weighted distances to five Boston employment centres            
        - RAD      index of accessibility to radial highways                       
        - TAX      full-value property-tax rate per $10,000                        
        - PTRATIO  pupil-teacher ratio by town                                     
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  
        - LSTAT    % lower status of the population                                
        - MEDV     Median value of owner-occupied homes in $1000's     




"""

# # Sklearn Tutorial

# This notebook walks through using the sklearn style DAGRegressor and DAGClassifier models.

# ___
# ## DAGRegressor
# This section demonstrates the performance of the DAGRegressor on a real-world dataset. The run_train things to note in this section are:
# - The scale sensitivity of the algorithm
# - Interpretability of nonlinear `.coef_`
# ### The Data: Boston Housing
# 
# The boston housing dataset is a classic benchmark regression task. The objective is to predict a set of house prices given a small set of features.
# 
# The meaning of the set of avaliable features is shown below.

# In[47]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
print(load_boston(return_X_y=False)["DESCR"])


# Lets initially benchmark the performance of an `ElasticNetCV` fitted across the entire dataset.

# In[48]:


from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
y = (y - y.mean()) / y.std()


reg = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=True)

from sklearn.model_selection import KFold
scores = cross_val_score(reg, X, y, cv=KFold(shuffle=True, random_state=42))
print(f'MEAN R2: {np.mean(scores).mean():.3f}')


# ### Linear DAGRegressor
# 
# The DAGRegressor has several parameters which can be used to better fit a more complicated noisy DAG:
# - `alpha`: The l1 (lasso) regularisation parameter. Increasing this creates a sparser DAG.
# - `beta`: The l2 (ridge) regularisation parameter.
# It was decided to use `alpha` and `beta` rather than `alpha` and `l1_ratio` like in sklearn elasticnet to uncouple the parameters during optimisation.
# 
# There are several parameters which are also of interest which have good defaults, but we highlight here:
# - `dependent_target`: This forces the target variable y to be only a child node. This is important for performance because in some cases `X -> y` is indistinguishable from `y -> X`. Enabling this (default enabled) ensures that the regressor performance at least matches linear regression. The trade-off is that the learned structure might be less accurate if y does cause other features.
# - `enforce_dag`: This thresholds the learned structure model until the system is a DAG. This is useful for removing the small straggler connections which enables the DAG to be visualised easier. It does not impact performance, because the regressor still uses those connections under the hood.

# In[50]:


from sklearn.datasets import load_boston
data = load_boston()
X, y = data.data, data.target
names = data["feature_names"]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
y = (y - y.mean()) / y.std()

from causalnex.structure.pytorch import DAGRegressor
reg = DAGRegressor(
                alpha=0.1,
                beta=0.9,
                fit_intercept=True,
                hidden_layer_units=None,
                dependent_target=True,
                enforce_dag=True,
                 )

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
scores = cross_val_score(reg, X, y, cv=KFold(shuffle=True, random_state=42))
print(f'MEAN R2: {np.mean(scores).mean():.3f}')

X = pd.DataFrame(X, columns=names)
y = pd.Series(y, name="MEDV")
reg.fit(X, y)
print(pd.Series(reg.coef_, index=names))
reg.plot_dag(enforce_dag=True)



from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
y = (y - y.mean()) / y.std()



reg = DAGRegressor(
                alpha=0.1,
                beta=0.9,
                fit_intercept=True,
                hidden_layer_units=None,
                dependent_target=True,
                enforce_dag=True,
                 )

from sklearn.model_selection import KFold
scores = cross_val_score(reg, X, y, cv=KFold(shuffle=True, random_state=42))
print(f'MEAN R2: {np.mean(scores).mean():.3f}')

X_pd = pd.DataFrame(X, columns=load_boston(return_X_y=False)["feature_names"])
y_pd = pd.Series(y, name="price")
reg.fit(X_pd, y_pd)
print(pd.Series(reg.coef_, index=load_boston(return_X_y=False)["feature_names"]))
reg.plot_dag(True)


# ### NonLinear DAGRegressor
# 
# Specifying a nonlinear model is extremely simple, only a single parameter needs to be altered: `hidden_layer_units`
# 
# `hidden_layer_units` takes _any_ **iterable** of **integers**: 
# - The value specifies the number of perceptrons to use in each nonlinear MLP layer:
# - The number of elements in the iterable determines the number of hidden layers. 
# The more layers and more perceptrons per layer, the more complicated the function which can be fit. The trade off is a greater tendency to overfit, and slower fitting.
# 
# A good default starting argument is ~[5]. This is unlikely to overfit, and usually demonstrates immidiately whether the DAG has nonlinear components.
# 
# The setting of the `alpha` and `beta` parameters is very important.
# Typically `beta` is more important than `alpha` when using nonlinear layers. This is because l2 is applied across all layers, whereas l1 is only applied to the first layer.
# A good starting point is `~beta=0.5`.
# 
# **NOTE it is very important to scale your data!**
# 
# The nonlinear layers contain sigmoid nonlinearities which can become saturated with unscaled data. Also, unscaled data means that regularisation parameters do not impact weights across features equally.
# 
# For convnenience, setting `standardize=True` scales both the X and y data during fit. It also inverse transforms the y on predict similar to the sklearn `TransformedTargetRegressor`.

# In[54]:


# from causalnex.structure.sklearn import DAGRegressor
from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

reg = DAGRegressor(threshold=0.0,
                    alpha=0.0,
                    beta=0.5,
                    fit_intercept=True,
                    hidden_layer_units=[5],
                    standardize=True,
                 )

from sklearn.model_selection import KFold
scores = cross_val_score(reg, X, y, cv=KFold(shuffle=True, random_state=42))
print(f'MEAN R2: {np.mean(scores).mean():.3f}')

X_pd = pd.DataFrame(X, columns=load_boston(return_X_y=False)["feature_names"])
y_pd = pd.Series(y, name="price")
reg.fit(X_pd, y_pd)

reg.plot_dag(True)


# #### Interpereting the Nonlinear DAG
# 
# For nonlinear analysis, understanding the impact of one feature on another is not as simple as taking the mean effect as in the linear case.
# Instead, a combination of `reg.coef_` and `reg.feature_importances` should be used:
# 
# - `reg.coef_` provides the mean **directional** effect of all the features on the target. This gives average directional information, but can be misleading in terms of magnitude if the feature has a positive _and_ negative effect on the target.
# 
# - `reg.feature_importances_` provides the mean **magnitude** effect of the features on the target. These values will be _strictly larger_ than the `reg.coef_` because there are no cancellation effects due to sign differences. 
# 
# The magnitude difference between the `reg.coef_` and `reg.feature_importances_` values can give insight into the _degree of directional variability_ of the parameter:
# - A large difference means that the parameter has **large positive and negative effects** on the target. 
# - A zero difference means that the parameter always has the same directional impact on the target.

# In[56]:


# from causalnex.structure.sklearn import DAGRegressor
from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

reg = DAGRegressor( alpha=0.0,
                    beta=1.0,
                    fit_intercept=True,
                    hidden_layer_units=[8, 8, 8],
                    standardize=True,
                 )

from sklearn.model_selection import KFold
scores = cross_val_score(reg, X_pd.values, y_pd.values, cv=KFold(shuffle=True, random_state=42))
print(f'MEAN R2: {np.mean(scores).mean():.3f}')

reg.fit(X_pd, y_pd)
print("MEAN EFFECT DIRECTIONAL:")
print(pd.Series(reg.coef_, index=load_boston(return_X_y=False)["feature_names"]))
print("MEAN EFFECT MAGNITUDE:")
print(pd.Series(reg.feature_importances_, index=load_boston(return_X_y=False)["feature_names"]))

reg.plot_dag(True)


# The `reg.get_edges_to_node` method allows for analysis of other edges in the graph easily.
# 
# Passing in `data="weight"` returns the mean effect magnitude of the variables on the requested node. It is equivalent to the `reg.feature_importances` return for the target node.
# 
# Passing in `data="mean_effect"` returns the mean directional effect.
# 
# Below is a good example of a large difference between the magnitude and directional effects: 
# - The feature RAD has overall a large effect on the presence of NOX. 
# - However, the _directional_ effect of this feature is highly variable, which leads the mean_effect to be an order of magnitude smaller than the mean effect magnitude!

# In[57]:


vals = reg.get_edges_to_node("NOX", data="weight").copy()
vals[vals.abs() < 0.01] = 0
vals


# In[58]:


vals = reg.get_edges_to_node("NOX", data="mean_effect")
vals[vals.abs() < 0.01] = 0
vals


# #### Dependent Target
# 
# Setting the `dependent_target=False` has an impact on performance as shown below, but can give better insight into the overall nonlinear structure of the data.
# 
# This is effectively the same as fitting causalnex on the data using from_pandas, but using the sklearn interface provides a set of useful convenience functions not present in the base causalnex implementation.

# In[60]:


# from causalnex.structure.sklearn import DAGRegressor
from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

reg = DAGRegressor( alpha=0.0,
                    beta=1.0,
                    fit_intercept=True,
                    hidden_layer_units=[5],
                    standardize=True,
                    dependent_target=True,
                 )

from sklearn.model_selection import KFold
scores = cross_val_score(reg, X_pd.values, y_pd.values, cv=KFold(shuffle=True, random_state=42))
print(f'MEAN R2: {np.mean(scores).mean():.3f}')

reg.fit(X_pd, y_pd)
print("MEAN EFFECT DIRECTIONAL:")
print(pd.Series(reg.coef_, index=load_boston(return_X_y=False)["feature_names"]))
print("MEAN EFFECT MAGNITUDE:")
print(pd.Series(reg.feature_importances_, index=load_boston(return_X_y=False)["feature_names"]))

reg.plot_dag(True)


# ___
# ## DAGClassifier
# This section demonstrates the performance of the algorithm on a real-world dataset.
# 
# The interface is very similar to the DAGRegressor so key details should be found there.
# ### The Data: Breast Cancer

# In[1]:


from sklearn.datasets import load_breast_cancer
print(load_breast_cancer(return_X_y=False)["DESCR"])


# In[17]:


from causalnex.structure import DAGClassifier
from sklearn.model_selection import cross_val_score

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
names = load_breast_cancer(return_X_y=False)["feature_names"]

reg = DAGClassifier(
                alpha=0.1,
                beta=0.5,
                hidden_layer_units=[0],
                fit_intercept=True,
                standardize=True
                 )
from sklearn.model_selection import KFold
scores = cross_val_score(reg, X, y, cv=KFold(shuffle=True, random_state=42))
print(f'MEAN Score: {np.mean(scores).mean():.3f}')

X_pd = pd.DataFrame(X, columns=names)
y_pd = pd.Series(y, name="NOT CANCER")
reg.fit(X_pd, y_pd)
print("MEAN EFFECT DIRECTIONAL:")
print(pd.Series(reg.coef_, index=names).sort_values(ascending=False))
reg.plot_dag(True)


# In[ ]:





# In[ ]:




