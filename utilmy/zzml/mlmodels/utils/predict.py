"""
 python offline/amodel_export.py --do shopid_json   --config offline.config.prod.shopid_model  --mode prod

"""
import os, sys, gc, glob
import pandas as pd, numpy as np, traceback
import time, msgpack
import matplotlib.pyplot as plt

####################################################################################################
agg_level_data=None;agg_level_model=None;colsX=None;colsX_cat=None;colsX_num=None;compute_pars=None;date_now_jp=None;debug=None;exp_save_all_params=None;iimax=None;iimin=None;metric_name=None;metrics=None;metrics_cols=None;model_date=None;model_group=None;model_name=None;model_pars=None;model_tag=None;modelx=None;path_key_all=None;pos_path=None;pred_date=None;pred_end=None;pred_group=None;pred_path=None;pred_path_input=None;pred_path_output=None;pred_prefix=None;pred_start=None;pred_tag=None;root=None;root_model=None;shop_list=None;tag_all=None;test_period=None;train_path=None;train_path_X=None;train_path_y=None;verbose=None;ytarget=None;

from offline.util import *
from offline.util import (pd_to_onehot, pd_histo, log, config_import, pd_filter, key_load_all,get_itemlist,
list_intersection, pd_read_file2, pd_read_file, train_split_time, pd_to_file, log_error, log_pd,
os_variable_check, os_clean_memory, to_timekey, todatetime, from_timekey, pd_cartesian  )
                          
from zlocal import root

####################################################################################################
def cli_load_argument(config_file=None):
    """ arg = to_namespace({ "do": root + "/data/pos/",   
                           "config" : "offline.config.prod.itemid_model" ,
                           "mode" : "prod", "verbose": True })    
         vars(arg)      
    """
    import argparse
    p = argparse.ArgumentParser()
    def add(*w, **kw):  p.add_argument(*w, **kw)

    add("--do", default="", help="train, create_metrics ")
    add("--mode", default="test", help=" ")
    add("--config", default="offline.config.ztest_model", help=" ")
    add("--verbose", default=True, type=bool, help=" ")
    
    arg = p.parse_args()
    return arg

arg = cli_load_argument(config_file=None)


####################################################################################################
##### Input data  ##################################################################################
#### Load configuration from offline/config/   "offline.config.genre_l1_model"
log(" amodel_train.py  ####################", mode="title")
log("CLI Params:  ok2 ", vars(arg))
config_import( arg.config, globs=globals(), verbose=verbose)


####################################################################################################
####### Transform ##################################################################################
from offline.util import os_variable_exist


####################################################################################################
if arg.do == "shopid_json" :
  ###  python offline/amodel_export.py --do shopid_json   --config offline.config.prod.shopid_model  --mode prod

  #### Export to JSON/DB   
  from offline.config.prod.shopid_model import shopid_pred_export   
  shopid_pred_export( shop_list,  date0=pred_date , verbose=verbose)    
  sys.exit(0)




    
    
    

"""


### Get All keys   ####################################################
Nb sub-models 373
### dfX0 : X_input Features     #######################################


Pool 0,Pool 1,Pool 2,

python offline/amodel_train.py --do mae_summary --config offline.config.prod.itemid_model --mode prod
CLI Params: {'do': 'mae_summary', 'mode': 'prod', 'config': 'offline.config.prod.itemid_model', 'verbose': True}
Importing:,agg_level_data,agg_level_model,colsX,compute_pars,date_now_jp,debug,exp_save_all_params,iimax,iimin,metric_name,metrics,metrics_cols,model_date,model_group,model_name,model_pars,model_tag,modelx,path_key_all,pd,pos_path,pred_date,pred_end,pred_group,pred_path,pred_path_input,pred_path_output,pred_prefix,pred_start,pred_tag,root,root_model,shop_list,test_period,train_path,train_path_X,train_path_y,verbose,ytarget,zlocal,
###### Paths ###################################################################
Train Input:  /a/adigcb301/ipsvols05/offline/test//data/train/all/
Model Output:  /a/adigcb301/ipsvols05/offline/test//model//itemid_20200802/shop_id-RandomForestRegressordaily/





"""
