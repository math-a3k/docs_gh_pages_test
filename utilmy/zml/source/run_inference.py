# -*- coding: utf-8 -*-
"""
python source/run_inference.py  run_predict  --n_sample 1000  --config_name lightgbm  --path_model /data/output/a01_test/   --path_output /data/output/a01_test_pred/     --path_data_train /data/input/train/
"""
import warnings,sys, os, json, importlib, copy, gc, glob
warnings.filterwarnings('ignore')
####################################################################################################
from utilmy import global_verbosity, os_makedirs
verbosity = global_verbosity(__file__, "/../config.json" ,default= 5)

def log(*s):
    if verbosity >= 1 : print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    if verbosity >= 3 : print(*s, flush=True)

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
log(root)


####################################################################################################
from util_feature import load, load_function_uri, load_dataset

def model_dict_load(model_dict, config_path, config_name, verbose=True):
    """ Load the model dict from the python config file.
       ### Issue wiht passing function durin pickle on disk
    :param model_dict:
    :param config_path:
    :param config_name:
    :param verbose:
    :return:
    """
    if model_dict is None :
      log("#### Model Params Dynamic loading  ###############################################")
      model_dict_fun = load_function_uri(uri_name=config_path + "::" + config_name)
      model_dict    = model_dict_fun()   ### params 

    else :
        ### Passing dict 
        ### Due to Error when saving on disk the model, function definition is LOST, need dynamic loca
        path_config = model_dict[ 'global_pars']['config_path']

        p1 = path_config + "::" + model_dict['model_pars']['post_process_fun'].__name__
        model_dict['model_pars']['post_process_fun'] = load_function_uri( p1)   

        p1 = path_config + "::" + model_dict['model_pars']['pre_process_pars']['y_norm_fun'] .__name__ 
        model_dict['model_pars']['pre_process_pars']['y_norm_fun'] = load_function_uri( p1 ) 

    return model_dict


def map_model(model_name="model_sklearn:MyClassModel"):
    """ Get the module of the model stored in source/models/
    :param model_name:   model_sklearn
    :return: model module
    """
    ##### Custom folder
    if ".py" in model_name :
       model_file = model_name.split(":")[0] 
       ### Asbolute path of the file
       path = os.path.dirname(os.path.abspath(model_file))
       sys.path.append(path)
       mod    = os.path.basename(model_file).replace(".py", "")
       modelx = importlib.import_module(mod)
       return modelx

    ##### Repo folder
    model_file = model_name.split(":")[0]
    if  'optuna' in model_name : model_file = 'optuna_lightgbm'

    try :
       ##  'models.model_bayesian_pyro'   'model_widedeep'
       mod    = f'models.{model_file}'
       modelx = importlib.import_module(mod)

    except :
       ### SKLEARN API by default
       mod    = 'models.model_sklearn'
       modelx = importlib.import_module(mod)

    return modelx


##################################################################################################
def run_predict_batch(config_name, config_path, n_sample=-1,
                path_data=None, path_output=None, pars={}, model_dict=None, return_mode='file'):

    model_dict = model_dict_load(model_dict, config_path, config_name, verbose=True)
    m          = model_dict['global_pars']
    path_model       = m['path_pred_model']    
    path_data        = m['path_pred_data']   if path_data   is None else path_data
    path_output      = m['path_pred_output'] if path_output is None else path_output
    log(path_data, path_model, path_output)

    flist = glob.glob( path_data + "/*.parquet")
    for i, fi in enumerate(flist) :
       log(fi) 
       model_dict['global_pars']['path_pred_data'] = fi
       run_predict(config_name, config_path, n_sample,
                   path_data, 
                   path_output + f"/pred_{i}*", 
                   pars, model_dict, return_mode='file')


def predict(model_dict, dfX, cols_family, post_process_fun=None):
    """
    :param model_dict:  dict containing params
    :param dfX:  pd.DataFrame
    :param cols_family: dict of list containing column names
    :param post_process_fun:
    :return: dfXtrain , dfXval  DataFrame containing prediction
    """
    model_pars, compute_pars = model_dict['model_pars'], model_dict['compute_pars']
    data_pars                = model_dict['data_pars']
    model_name, model_path   = model_pars['model_class'], model_dict['global_pars']['path_train_model']
    # metric_list              = compute_pars['metric_list']


    assert  'cols_model_type2' in data_pars, 'Missing cols_model_type2, split of columns by data type '
    log2(data_pars['cols_model_type2'])

    log("#### Model Input preparation ################################")    
    colsX  = data_pars['cols_model']
    coly   = data_pars['coly']
    # colsX       = load(model_path + "/model/colsX.pkl")   ## column name
    # coly  = load( model_path + "/model/coly.pkl"   )
    colsX       = load(model_path + "/colsX.pkl")   ## column name
    assert colsX        is not None, "cannot load colsx, " + model_path


    log("#### Load model  ############################################")
    modelx = map_model(model_name)
    modelx.reset()
    log2(modelx, model_path)
    sys.path.append( root)    #### Needed due to import source error

    log2(model_path + "/model/model.pkl")
    modelx.load_model(model_path )
    assert modelx.model is not None, "cannot load modelx, " + model_path
    log2("### modelx\n", modelx.model.model)


    log("### Prediction  ############################################")
    #dfX1  = dfX.reindex(columns=colsX)   #reindex included
    ypred, ypred_proba  = modelx.predict(dfX[colsX],
                           data_pars    = data_pars,
                           compute_pars = compute_pars )

    dfX[coly + '_pred']  = ypred 
    dfX[coly + '_pred']  = dfX[coly + '_pred'].apply(lambda  x : post_process_fun(x) )
    log("Pred    : ",  dfX[[ coly + '_pred'  ]])

    if ypred_proba is None : 
         pass 
    elif len(ypred_proba.shape) == 1  :  #### Single dim proba
        dfX[coly + '_proba'] = ypred_proba
        log3(coly + '_proba', dfX[coly + '_proba'])
        
    elif len(ypred_proba.shape) > 1 :   ## Muitple proba
        from util_feature import np_conv_to_one_col
        dfX[coly + '_proba'] = np_conv_to_one_col(ypred_proba, ";")  ### merge into string "p1,p2,p3,p4"
        log3(coly + '_proba', dfX[coly + '_proba'])

    stats = {}
    return dfX, stats 


####################################################################################################
############CLI Command ############################################################################
def run_predict(config_name, config_path, n_sample=-1,
                path_data=None, path_output=None, pars={}, model_dict=None, return_mode='file'):

    model_dict = model_dict_load(model_dict, config_path, config_name, verbose=True)

    m          = model_dict['global_pars']
    path_data        = m['path_pred_data']   if path_data   is None else path_data
    path_pipeline    = m['path_pred_pipeline']    #   path_output + "/pipeline/" )
    path_model       = m['path_pred_model']
    path_output      = m['path_pred_output'] if path_output is None else path_output
    log(path_data, path_model, path_output)


    log("#### load raw data column family  ###############################################")
    pars = {'cols_group': model_dict['data_pars']['cols_input_type'],
            'pipe_list' : model_dict['model_pars']['pre_process_pars']['pipe_list']}


    log("#### Preprocess  ################################################################")
    from run_preprocess import preprocess_inference   as preprocess
    colid            = load(f'{path_pipeline}/colid.pkl')
    df               = load_dataset(path_data, path_data_y=None, colid=colid, n_sample=n_sample)
    dfX, cols        = preprocess(df, path_pipeline, preprocess_pars=pars)


    log("#### Extract column names  ######################################################")
    ### Actual column names for Model Input :  label y and Input X (colnum , colcat), remove duplicate names
    model_dict['data_pars']['coly']       = cols['coly']
    model_dict['data_pars']['cols_model'] = list(set(sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , []) ))


    #### Col Group by column type : Sparse, continuous, .... (ie Neural Network feed Input, remove duplicate names
    ## 'coldense' = [ 'colnum' ]     'colsparse' = ['colcat' ]
    model_dict['data_pars']['cols_model_type2'] = {}
    for colg, colg_list in model_dict['data_pars'].get('cols_model_type', {}).items() :
        model_dict['data_pars']['cols_model_type2'][colg] = list(set(sum([  cols[colgroup] for colgroup in colg_list ]   , [])))


    log("##### Predict + Proba ###########################################################")
    log(str(model_dict)[:1000])
    post_process_fun      = model_dict['model_pars']['post_process_fun']
    dfX, stats            = predict(model_dict, dfX, cols, post_process_fun)


    log("##### Export ####################################################################")
    dfX   = dfX.reset_index()
    log2(dfX)
    if return_mode == 'dict' :
        return { 'dfXy' : dfX, 'stats' : stats   }


    log("##### Saving prediction  #######################################################")
    os.makedirs(path_output, exist_ok=True)
    log(path_output)
    dfX.sample(n=500, replace=True).to_csv(f"{path_output}/pred_all_sample.csv")
    dfX.to_parquet(f"{path_output}/pred_all.parquet")
    colexport = [cols['colid'], cols['coly'] + "_pred"]
    if cols['coly'] + '_proba' in  dfX.columns :
        colexport.append( cols['coly'] + '_proba' )

    dfX[ colexport ].sample(n=500, replace=True).to_csv(f"{path_output}/pred_only_sample.csv")
    dfX[ colexport ].to_parquet(f"{path_output}/pred_only.parquet")



def run_data_check(path_data, path_data_ref, path_model, path_output, sample_ratio=0.5):
    """
     Calcualata Dataset Shift before prediction.
    """
    from run_preprocess import preprocess_inference   as preprocess
    path_output   = root + path_output
    path_data     = root + path_data
    path_data_ref = root + path_data_ref
    path_pipeline = root + path_model + "/pipeline/"

    os.makedirs(path_output, exist_ok=True)
    colid          = load(f'{path_pipeline}/colid.pkl')

    df1                = load_dataset(path_data_ref,colid=colid)
    dfX1, cols_family1 = preprocess(df1, path_pipeline)

    df2                = load_dataset(path_data,colid=colid)
    dfX2, cols_family2 = preprocess(df2, path_pipeline)

    colsX       = cols_family1["colnum_bin"] + cols_family1["colcat_bin"]
    dfX1        = dfX1[colsX]
    dfX2        = dfX2[colsX]

    from util_feature import pd_stat_dataset_shift
    nsample     = int(min(len(dfX1), len(dfX2)) * sample_ratio)
    metrics_psi = pd_stat_dataset_shift(dfX2, dfX1,
                                        colsX, nsample=nsample, buckets=7, axis=0)
    metrics_psi.to_csv(f"{path_output}/prediction_features_metrics.csv")
    log(metrics_psi)



###########################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()




