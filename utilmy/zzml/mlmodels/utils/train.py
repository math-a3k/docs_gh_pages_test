"""
######## Create manual file in 
  offline/config/ 

### Train
  python offline/amodel_train.py --do train  --config offline.config.ztest_model 

  python offline/amodel_train.py --do train  --config offline.config.genre_l2_model 


### Metrics Summary
python offline/amodel_train.py --do create_metrics_summary  --config offline.config.genre_l2_model 

### MAE summary
python offline/amodel_train.py --do create_mae_summary  --config offline.config.genre_l1_model



--config offline.config.genre_l1_model
--config offline.config.genre_l2_model
--config offline.config.itemid_model


python offline/amodel_train.py --do train        --config offline.config.prod.itemid_units --mode prod 



########### Prod :
python offline/amodel_train.py --do train        --config offline.config.prod.genre_l1_model 
python offline/amodel_train.py --do mae_summary  --config offline.config.prod.genre_l1_model 


python offline/amodel_train.py --do train        --config offline.config.prod.genre_l2_model 
python offline/amodel_train.py --do mae_summary  --config offline.config.prod.genre_l2_model 


python offline/amodel_train.py --do train        --config offline.config.prod.itemid_model --mode prod 
python offline/amodel_train.py --do mae_summary  --config offline.config.prdo.itemid_model --mode prod




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

       
if arg.mode == "test" :
    model_tag = model_tag +"_test"   ### Apppend test to all foldera
    shop_list = [16 ]; iimin= 0; iimax  = 5  ; verbose=True; debug=True
    log("Mode test activated", model_tag, iimax, mode="title")

os_variable_check([ "colsX_cat", "colsX",  ], globals())


####################################################################################################
####### Transform ##################################################################################
from offline.util import os_variable_exist
    
if not os_variable_exist( "y_transform" ,globals()) :
    from offline.util import y_transform


if not os_variable_exist( "X_transform" ,globals()) :
    from offline.util import X_transform



def create_metrics_summary(path_model, im=40, verbose=True):
   """function create_metrics_summary
   Args:
       path_model:   
       im:   
       verbose:   
   Returns:
       
   """
   import matplotlib.pyplot as plt 
   path_model = path_model + "/"
   to_path    = path_model + "/metrics/"
   os.makedirs(to_path, exist_ok=True)
   path_list  = [  f for f in os.listdir( path_model )  ]
   
   if verbose : log(path_list)
   for ii, fname in enumerate(path_list) :  
    try: 
      if fname in ['metrics'] : continue  
      dfe2 = pd_read_file( path_model + fname + "/df_error*" )   
      colt = [ i  for i,t in enumerate(dfe2.columns) if  ("porder" in t  or  "order_id" in t ) and  "_diff" not in t ]
      log("Metrics: ", fname, colt) 
      
      if "item_id" in dfe2.columns :
        dfe2 = dfe2[dfe2['item_id'] == dfe2["item_id"].values[-1] ]
        
        
      dfe2.iloc[-im:,colt].plot()
      plt.savefig(to_path + f"/plot_out_{fname}.png"); plt.close()

      dfe2.iloc[:-im,colt].plot()      
      plt.savefig(to_path + f"/plot_in_{fname}.png"); plt.close()

      dfe2['model_sub'] = fname
      dfe2.to_csv(to_path + f"/df_error_all.csv", mode="a" if ii > 0 else "w", index=True,
                  header= 0 if ii > 0 else 1 )
      
      dfe2.iloc[-im:,:].to_csv(to_path + f"/df_error_out.csv", mode="a" if ii > 0 else "w",
               header= False if ii > 0 else True   , index=True)
      
    except Exception as e:
      log("Error",fname, e) 



def create_mae_summary( path , path_modelgroup, tag="", ytarget="porder_s2", agg_level= None, verbose=True) :
    """function create_mae_summary
    Args:
        path:   
        path_modelgroup:   
        tag:   
        ytarget:   
        agg_level:   
        verbose:   
    Returns:
        
    """
    import json
    to_path = path # path +"/metrics/"
    dfea    = pd.read_csv(to_path + "/df_error_out.csv" )
    cols    = list(dfea.columns)
    if verbose : log_pd(dfea)    

    ### Target colum
    cola = ytarget
    if ytarget is None :       #'porder_s2_pred' 
      cola = [ t for t in dfea.columns if ("porder" in t  or  "order_id" in t ) and "_diff" not in t and "_pred" not in t  ][0]
    colb, col0 = cola + "_pred"  , cola + "_diff" 

    ### Agg by each sub-model
    dfs = dfea.groupby( agg_level ).agg({ cola : "sum", colb : "sum"  }).reset_index()
    
    dfs['diff_40d'] = dfs[cola]  -  dfs[colb] 
    mae  = np.mean(np.abs( dfs['diff_40d'] ))
    med  = dfs[cola].median()
    mape = mae / med
    log( "Average MAE, MAPE" , med,  mape  )
    
    pd_histo( np.abs(dfs['diff_40d']), to_path +"/dist_err_40d.png" , 30) 
    pd_histo( np.abs(dfea[ col0 ]) , to_path +"/dist_err_1d.png", 30) 
    
    dd = { "mape_40d" :mape, "mae_40d" :mae  }
    json.dump( dd, open( to_path +"/metrics.json" , mode="w" ))
    
    
    #### Copy to main folder
    import shutil
    os.makedirs(path_modelgroup, exist_ok=True)
    for t in [ "metrics.json", "dist_err_40d.png", "dist_err_1d.png"   ] :
        log(shutil.copyfile( to_path +"/" + t, path_modelgroup +  f"/{tag}_{t}" )  )
    log(" Success :", to_path)
    return None


def pd_check_na(name, dfXy, verbose = False, debug=False, train_path="ztmp/") :
  """function pd_check_na
  Args:
      name:   
      dfXy:   
      verbose :   
      debug:   
      train_path:   
  Returns:
      
  """
  log(name, "checking")
  # if verbose :  log_pd(dfXy, n_tail=10)
  # if debug :    dfXy.to_csv( train_path + f"/debug_{name}.csv" )  
  n = len(dfXy)
  for x in dfXy.columns :
    xna   = dfXy[x].isna().sum()
    ratio = xna / n + 0.0 
    if ratio > 0.1 :
       log("Warning : too much NA, Potential bug ", x, ratio)


def train_enhance(dfi, colsref, ytarget, n_sample=5):
    """function train_enhance
    Args:
        dfi:   
        colsref:   
        ytarget:   
        n_sample:   
    Returns:
        
    """
    return dfi
    import copy
    ### Cardianl date X n_item : for missing dates    
    keyref = ['time_key'] + colsref
    df2a = dfi[[ 'time_key']].drop_duplicates(['time_key']).sort_values('time_key')
    df2b = dfi[colsref].drop_duplicates( colsref ).sort_values(colsref)
    df2  = pd_cartesian(df2a, df2b)  
    # log_pd(df2)
    df2 = df2.join( dfi.set_index(keyref), on= keyref, how="left"  , rsuffix="b" )
    for t in df2.columns :
        if t not in keyref : df2[t] = df2[t].fillna(0.0)
        if t  in keyref    : df2[t] = df2[t].astype("int64")

    ### Add Noise in the label data: 
    df3 = copy.deepcopy(df2)
    for i in range(n_sample-1) :
      dfj           = copy.deepcopy(df2)
      mval          = dfj[ytarget].mean()
      dfj[ytarget]  = dfj[ytarget] + np.random.normal(0.0, 0.1 * mval  , len(dfj)) 
      df3           = pd.concat((df3, dfj)) 
    return df3

    
def add_dates(df):
    """function add_dates
    Args:
        df:   
    Returns:
        
    """
    return df
    #### new date
    """
      dfdates
    
    """
    

    
####################################################################################################
log("###### Paths ###################################################################")
log("Train Input: ", train_path)
log("Model Output: ", f"{root_model}/{model_group}/shop_id-{model_name}{model_tag}/"   )    
tmax = to_timekey( todatetime( model_date)  )
log('tmax time_key', tmax, from_timekey(tmax))
agg_level_data_time =['time_key'] + agg_level_data
            
if arg.do == "train" :
  log("###### Loading keys ##########################################################")
  key_level_all    = [  "shop_id", "dept_id", "l1_genre_id", "l2_genre_id", 'item_id'   ]  
  key_allref       = key_load_all(key_level_all, path_key_all)  
  log("Shop_list", shop_list, "key_list", key_allref.shape)
  
  # shop_list = [16]
  for shop_id in shop_list :
    try :    
        log( f"### Get All keys , shop_id : {shop_id}  ###########################################")
        key_select_ref = key_allref[ key_allref["shop_id"] == shop_id]
        if globals().get('l2_genre_id_filter', None) :
           ll3 = globals()['l2_genre_id_filter'] 
           # ll3 = [15041, 20101, 20501, 20502, 20504, 20601, 20901, 21501, 40303, 51103, 80103, 90506, 90403]        
           key_select_ref = key_allref[ key_allref["l2_genre_id"].isin(ll3) ]            

        key_select  = key_select_ref[ key_level_all  ].drop_duplicates(agg_level_model).values.tolist()
        log("Nb sub-models", len(key_select), key_select[:3])    

        log( f"### dfX0 : X_input {train_path_X}     #############################################")
        dfX0        = pd_read_file2( train_path_X, cols=None, shop_id= shop_id, n_pool=1)
        dfX0        = dfX0[dfX0['shop_id'] == shop_id]  if "shop_id" in list(dfX0.columns) else dfX0         
        agg_level_X = list_intersection( list(dfX0.columns), agg_level_data)
        if debug    : log_pd(  dfX0 , n_tail=10)

        log( f"### dfy0 : Porder with level2 or item_id agg level {train_path_y} #################")  
        dfy0 = pd_read_file2( train_path_y , cols=  agg_level_data  + ["time_key", ytarget], shop_id= shop_id, n_pool=1)
        
        # dfy0.to_parquet( root       + "/data/porder/y-porder_all.parquet" )
        
        dfy0 = dfy0[ dfy0['shop_id'] == shop_id]
        dfy0 = dfy0.drop_duplicates( agg_level_data_time ).sort_values(["time_key"]) 
        if debug : log_pd(  dfy0, n_tail=10 )
        log( f"dfX0, dfy0 loadded: ", dfX0.shape, dfy0.shape)  
        
        if len(dfy0) < 50 or len(dfX0) < 50  :
           log("dfy0, dfX0 too small", shop_id, dfy0.shape, dfX0.shape ) 
           continue

        model_save   = f"{shop_id}-{model_name}_{model_tag}"
    except Exception as e :
        log_error( f"Error {shop_id}", exception=e) 
        continue
    
    log("\n####  Loop over genre1, genre2. item_id  ##############################################")
    # key_select = [  [16, 5, 511, 51103, 5360001 ]  ]
    for ii, key_val in enumerate(key_select) :    
        try :
            id_global                      = time.time()
            (shop_id, dept_id, g1,g2, g0 ) = key_val               # key_val = (11,0,0,62201,0)
            key_val_dict   =  {  t : key_val[i] for i,t in enumerate(key_level_all) if t in agg_level_model } 
            key_val_dict_X =  {  t : key_val[i] for i,t in enumerate(key_level_all) if t in agg_level_X }     ### BUG, too much reduced, dfy and dfX : same size

            if ii < iimin : continue
            if ii > iimax : break     
            if g1 == -1   : continue

            model_sub   = "_".join([ str(key_val_dict.get(t,0)) for t in key_level_all ] )
            model_tag2  = "{model_save}_{model_sub}"        
            model_path  = root_model + f"/{model_group}/{model_save}/{model_sub}/"
            log("\n", ii, key_val_dict, model_path)

            ### Target y to predict  ###############################################
            dfy = pd_filter(dfy0, filter_dict = key_val_dict )    
            if len(dfy) < 50 : 
                log(ii, key_val, "Too few samples < 50")
                continue
                        
            ### Merge X , y with  join : Bug in fitleration
            os.makedirs( model_path, exist_ok=True )
            dfX  = pd_filter(dfX0, filter_dict = key_val_dict if "item_id" in agg_level_data else key_val_dict_X )  # key_val_dict_X )
            dfX  = dfX.set_index(["time_key"]  +  agg_level_X)    #### item_key
            if verbose : pd_check_na( "dfX", dfX, verbose, debug, train_path )     
            
            n_sample = 1 
            dfXy0 = dfy.join( dfX, on=["time_key"] + agg_level_X, how="left"  )
            dfXy0 = add_dates(dfXy0)  ## Extend of 30days
            dfXy  = train_enhance( dfXy0,  agg_level_data, ytarget, n_sample=n_sample)  ### Add samples, noise
            dfXy  = dfXy[dfXy['time_key'] <= tmax ]  ### Limit in time
            del dfy, dfX; gc.collect()
            if verbose : pd_check_na( "dfXy", dfXy, verbose, debug, train_path  )     ### Validate before training


            if len(dfXy) < test_period*2 :   ### Just Interpolation
                dfXytrain, dfXytest = dfXy, dfXy
            else : 
                dfXytrain, dfXytest = train_split_time(dfXy, test_period, cols = agg_level_data,
                                                       coltime= "time_key", minsize=5, sort=False, n_sample=n_sample) 

            ##Pb of cat category for item_id
            itemid_list =  get_itemlist(key_allref, shop_id, g1, g2)   if 'item_id' in colsX  else None

            if verbose : log( "dfXytrain/test  Shape", dfXytrain.shape , dfXytest.shape , "colsx_cat:", colsX_cat  )
            ##### Data processing for X,y   #######################################
            ytrain    = y_transform( dfXytrain[ ytarget ] , inverse=0 )
            Xtrain    = X_transform( dfXytrain[colsX]     , colsX , itemid_list, colsX_cat, verbose=verbose )    #One Hot encoding 
            ytest     = y_transform( dfXytest[ ytarget ]  , inverse=0 )
            Xtest     = X_transform( dfXytest[colsX]      , colsX,   itemid_list, colsX_cat, verbose=verbose )    #One Hot encoding 
            if debug  : log( "Xtrain", Xtrain.shape , 'Xtest', Xtest.shape ,  )

            ### Validation dataset
            dfr = dfXy0 #[ agg_level_data_time + [ytarget] ].set_index(  agg_level_data_time  )
            X   = X_transform( dfXy0[colsX]   , colsX , itemid_list, colsX_cat )
            del dfXy; del dfXy0;  gc.collect()
            data_pars = {"type"  :  "ram",  ##In memory dataest        
                         "train" :  { key:globals()[key] for key in [ "Xtrain", "Xtest", "ytrain", "ytest"]    },
                         "eval"  :  {  "X" : Xtest , "y": ytest  }     }     
            if debug  : log( "X shape", X.shape, "Xtrain shape", Xtrain.shape, "Xtest Shape", Xtest.shape )  

            ######### Model fit, eval,   ##########################################
            modelx.reset()                    
            modelx.init(model_pars, compute_pars= compute_pars)
            modelx.fit(data_pars, compute_pars)
            stats = modelx.fit_metrics(data_pars, compute_pars)  ### on eval dataset 
            modelx.save(model_path , stats)
            # del Xtrain, Xtest, ytrain, ytest, data_pars  ; gc.collect()
            model_pars["model_name"] = model_name
            if verbose : log(str(modelx.model.model))


            ### Prediction   #####################################################
            if debug : log('X.shape, dfr.shape', X.shape, dfr.shape)
            data_pars2              = {  "predict" : { "X" : X}    }   
            ypred                   = modelx.predict(data_pars2 )
            dfr[ ytarget + '_pred'] = y_transform( ypred, inverse=1 )
            dfr[ ytarget + '_diff'] = dfr[ytarget+'_pred'] - dfr[ytarget] 
            
            ### Add on dates
            # dfr[ ytarget + '_pred_season'] = y_transform( modelx.predict(data_pars2, compute_pars={"season": 1}), inverse=1 )            
            # dfr[ ytarget + '_pred_price']  = y_transform( modelx.predict(data_pars2, compute_pars={"price": 1}), inverse=1 )            
            
            dfr[ 'key']             = str(key_val)
            metric_val = np.sum(np.abs(dfr[ ytarget + '_diff'] ))   ###Final values
            if verbose : log_pd(dfr)

            ### Saving on disk statistics  #######################################
            pd_to_file(X.iloc[:1,:], model_path +"/cols.csv", check="NO",    verbose=False)                    
            pd_to_file(dfr, model_path + "/df_error.parquet", check="check", verbose=verbose)        
            dfr[[ ytarget, ytarget     + "_pred" ]].iloc[:,:].plot()    
            plt.savefig(model_path     + "/graph_pred.png"); plt.close()

            # xinfo =  ";".join([  str(colsX), str(len(X) - test_period), str(X.shape), model_path, str(test_period),    ] )  
            # metrics = pd_add(metrics, metrics_cols,  [ agg_level_model, model_save, model_sub,  stats, ytarget, metric_name, metric_val,
            #                  dfr[ ytarget  ].median(),  id_global   ])
            
        except Exception as e :
            log_error("Error ", ii, key_val, exception=e)
            # os.system( f" rm -rf {model_path}" )   ### Delete In-complete model
    
    try :
      log("### create_metrics_summary  #######################")     
      # pd.DataFrame(metrics).to_csv( root_model + f"/metrics_global.csv", mode="a", header=False )        
      create_metrics_summary(  root_model + f"/{model_group}/{shop_id}-{model_name}_{model_tag}/", im= 40  )
      os_clean_memory( ["dfX0", "dfy0",  "X", "dfr"   ] , globals() )

    except Exception as e:
      log(e)



####################################################################################################
########## Add metrics Summary #####################################################################
if arg.do == "metrics_summary" :
  for shop_id in shop_list :
     create_metrics_summary(  root_model + f"/{model_group}/{shop_id}-{model_name}_{model_tag}/", im= 40 )
  sys.exit(0)




####################################################################################################
########## Add metrics Summary #####################################################################
if arg.do == "mae_summary" :
    for shop_id in shop_list :
      root_model2 = root_model + f"/{model_group}/{shop_id}-{model_name}_{model_tag}/" 
      log(root_model2)
      # ytarget     = "porder_go"
      agg_level   = ["model_sub"]
      df_maee = create_mae_summary( path            = root_model2 + "/metrics/", 
                                    path_modelgroup = root_model  + f"/{model_group}/metrics/",   
                                    tag = f"{model_group}-{shop_id}-{model_name}_{model_tag}",                                   
                                    ytarget   = ytarget,
                                    agg_level = agg_level )     
    sys.exit(0)
    
    
if arg.do == "mae_summary_loop" :
    root_model2 = root_model + f"/{model_group}/" 
    # ytarget     = "porder_go"
    agg_level   = ["model_sub"]
    flist       = os.listdir( root_model2 )
    for fp in flist :  
      try :  
        log(fp)  
        dfs2 = create_mae_summary( path = root_model2 + fp + "/metrics/", 
                                   path_modelgroup = root_model2 , 
                                   tag = model_group,
                                   cola = ytarget,
                                   agg_level = agg_level ) 
      except : pass
    sys.exit(0)  

    
######## Export Pivot Metrics ####################################################################
if arg.do == "metrics_global" :
  metrics2 = pd.read_csv( root_model + f"/metrics_global.csv", mode="a", header=False )
  metrics2['metric_val'] = metrics2['metric_val'].astype("float")
  metrics2 = pd.pivot_table(metrics, values="metric_val", columns="model_name", index="model_sub",
                          aggfunc = "mean")
  metrics2.to_csv( root_model + f"/metrics_global_pivot.csv", mode="a" )
  sys.exit(0)


####################################################################################################
############ Prediction Shopid ###################################################################
if arg.do == "shopid_train_predict" :
  ###  python offline/amodel_train.py --do shopid_train_predict   --config offline.config.prod.shopid_model  --mode prod
  log("\n####### Starting ", arg.do,  " model date", model_date, to_timekey(model_date), ytarget )
  os_variable_check([ "shopid_train_predict", "shop_params", 'ytarget'  ],  globals())   
  from offline.util import to_datetime
  cols       = ['time_key', 'shop_id', ytarget]      # ytarget    = "order_id_s"
    
  df         = pd_read_file2( train_path_y, cols=cols, n_pool=3, drop_duplicates= ['time_key', 'shop_id' ], verbose=verbose)
  # df       = df[ df['time_key'] <= to_timekey(model_date) ]
  
  df         = df[cols].drop_duplicates( ['time_key', 'shop_id' ]  ).sort_values(['shop_id', 'time_key'])
  df['date'] = df['time_key'].apply(lambda t :  to_datetime(from_timekey(t)) )
  df['key']  = df['shop_id'].apply( lambda t : f"{t},0,0,0" )

  for shop_id in    shop_list :
     mpars  = shop_params[ shop_id if shop_id in shop_params.keys() else -1  ]    ###3 Model params
     dfi    = df[ df.shop_id == shop_id]
     ilimit = len(  dfi[ dfi['time_key'] <= to_timekey(model_date) ])  
     log("\nTime Max", len(dfi), "Time train", ilimit )    
     log("model pars", shop_id, mpars)
     model, dfp = shopid_train_predict(dfi, ytarget= ytarget, 
                              n_cutoff  = test_period, n_future = n_future,   
                              mpars     = mpars,
                              covariate_list = [],  
                              model_date = ilimit,         
                              model_path = root_model       + f"/{model_group}/{shop_id}_{model_tag}_{ytarget}_/",
                              pred_path  = pred_path_output + f"/pred_{shop_id}_shopid_{ytarget}_{pred_date}.parquet",
                              verbose    = verbose )
  
  export_json = True
  if export_json :
  #  # from util.config.shopid_model import shopid_pred_export 
     shopid_pred_export(dir_input= "", dir_output="",  model_group="shop_", shop_list= shop_list, 
                        date0=model_date , ytarget=ytarget, verbose=False) 
    
  #### Export to JSON/DB   
  # from offline.config.prod.shopid_model import shopid_pred_export   
  # shopid_pred_export( shop_list,  date0=pred_date , verbose=verbose)    
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
