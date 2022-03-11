# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
  cd analysis
  run preprocess

  ipython tseries.py  train      --config  config1  --pdb

"""
import warnings, sys, gc, os, sys, json, copy, pandas as pd
warnings.filterwarnings('ignore')
####################################################################################################
from utilmy import global_verbosity, os_makedirs
verbosity = global_verbosity(__file__, "/../config.json" ,default= 5)

def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    if verbosity >= 1 : print(*s, flush=True)

def log2(*s):
    """function log2
    Args:
        *s:   
    Returns:
        
    """
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    """function log3
    Args:
        *s:   
    Returns:
        
    """
    if verbosity >= 3 : print(*s, flush=True)


####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
log(root)


####################################################################################################
def log_pd(df, *s, n=0, m=1):
    """function log_pd
    Args:
        df:   
        *s:   
        n:   
        m:   
    Returns:
        
    """
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump,  df.head(n), flush=True)


####################################################################################################
from util_feature import  save, load_function_uri, load_dataset

def save_features(df, name, path=None):
    """ Save dataframe on disk
    """
    if path is not None :
       os.makedirs( f"{path}/{name}" , exist_ok=True)
       if isinstance(df, pd.Series):
           df0=df.to_frame()
       else:
           df0=df
       log( f"{path}/{name}/features.parquet" )
       log(df0, list(df0.columns))
       df0.to_parquet( f"{path}/{name}/features.parquet")
    else:
       log("No saved features, path is none")


def load_features(name, path):
    """function load_features
    Args:
        name:   
        path:   
    Returns:
        
    """
    try:
        return pd.read_parquet(f"{path}/{name}/features.parquet")
    except:
        log("Not available", path, name)
        return None


def model_dict_load(model_dict, config_path, config_name, verbose=True):
    """ Load the model dict from the python config file.
       ### Issue wiht passing function durin pickle on disk
    :return:
    """
    if model_dict is None :
      log("#### Model Params Dynamic loading  ###############################################")
      model_dict_fun = load_function_uri(uri_name=config_path + "::" + config_name)
      model_dict     = model_dict_fun()   ### params

    else :
        ### Passing dict
        ### Due to Error when saving on disk the model, function definition is LOST, need dynamic load
        path_config = model_dict[ 'global_pars']['config_path']

        p1 = path_config + "::" + model_dict['model_pars']['post_process_fun'].__name__
        model_dict['model_pars']['post_process_fun'] = load_function_uri( p1)

        p1 = path_config + "::" + model_dict['model_pars']['pre_process_pars']['y_norm_fun'] .__name__
        model_dict['model_pars']['pre_process_pars']['y_norm_fun'] = load_function_uri( p1 )

    return model_dict


####################################################################################################
####################################################################################################
def preprocess_batch(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={}, path_features_store=None):
   """
       Process by mini batch of files

   """
   import glob
   flist = glob.glob(path_train_X)
   dfXy_list = []
   for i,file_i in enumerate(flist) :
       dfXi, cols_familiy_i = preprocess(file_i, "", path_pipeline_export, cols_group, n_sample,
                                         preprocess_pars, path_features_store)

       dfXy_list.append( [ dfXi, cols_familiy_i ] )

   cols_family = cols_familiy_i    ### should be all same
   return dfXy_list, cols_family



def preprocess(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={}, path_features_store=None):
    """ Used for trainiing only, Save params on disk
    :param path_train_X:
    :param path_train_y:
    :param path_pipeline_export:
    :param cols_group:
    :param n_sample:
    :param preprocess_pars:
    :param path_features_store:
    :return:
    """
    ##### column names for feature generation #####################################################
    log(cols_group)
    coly            = cols_group['coly']  # 'salary'
    colid           = cols_group['colid']  # "jobId"
    colcat          = cols_group['colcat']  # [ 'companyId', 'jobType', 'degree', 'major', 'industry' ]
    colnum          = cols_group['colnum']  # ['yearsExperience', 'milesFromMetropolis']
    os.makedirs(path_pipeline_export, exist_ok=True)
    log(path_pipeline_export)
    save(colid, f'{path_pipeline_export}/colid.pkl')

    ### Pipeline Execution ##########################################
    pipe_default = [
      {'uri' : 'source/prepro.py::pd_coly',                'pars': {}, 'cols_family': 'coly',        'type': 'coly'  },
      {'uri' : 'source/prepro.py::pd_colnum_bin',          'pars': {}, 'cols_family': 'colnum',      'type': ''      },
      {'uri' : 'source/prepro.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'colnum_bin',  'type': ''      },
      {'uri':  'source/prepro.py::pd_colcat_bin',          'pars': {}, 'cols_family': 'colcat',      'type': ''      },
      {'uri':  'source/prepro.py::pd_colcat_to_onehot',    'pars': {}, 'cols_family': 'colcat_bin',  'type': ''      },
      {'uri' : 'source/prepro.py::pd_colcross',            'pars': {}, 'cols_family': 'colcross',    'type': 'cross' }
    ]

    pipe_list    = preprocess_pars.get('pipe_list', pipe_default)
    pipe_list_X  = [ task for task in pipe_list  if task.get('type', '')  not in ['coly', 'filter']  ]
    pipe_list_y  = [ task for task in pipe_list  if task.get('type', '')   in ['coly']  ]
    pipe_filter  = [ task for task in pipe_list  if task.get('type', '')   in ['filter']  ]

    log("##### Load data ######################################################################")
    df = load_dataset(path_train_X, path_train_y, colid, n_sample= n_sample)


    ##### Generate features ##########################################################################
    dfi_all          = {} ### Dict of all features
    cols_family_all  = {'colid' : colid, 'colnum': colnum, 'colcat': colcat}


    if len(pipe_filter) > 0 :
        log("#####  Filter  #########################################################################")
        pipe_i       = pipe_filter[ 0 ]
        pipe_fun     = load_function_uri(pipe_i['uri'])
        df, col_pars = pipe_fun(df, list(df.columns), pars=pipe_i.get('pars', {}))


    if len(pipe_list_y) > 0 :
        log("#####  coly  ###########################################################################")
        pipe_i       = pipe_list_y[ 0 ]
        pipe_fun     = load_function_uri(pipe_i['uri'])
        pars                         = pipe_i.get('pars', {})
        pars['path_features_store']  = path_features_store
        pars['path_pipeline_export'] = path_pipeline_export
        df, col_pars                 = pipe_fun(df, cols_group['coly'], pars=pars)   ### coly can remove rows
        log3("----------dfy----------\n", df)

    ### save dfy  #####################################################
    dfi_all['coly']              = df[cols_group['coly'] ]
    cols_family_all['coly']      = cols_group['coly']
    save_features(df[cols_group['coly'] ], "coly", path_features_store)  ### already saved
    save(coly, f'{path_pipeline_export}/coly.pkl')


    #####  Processors  ###############################################################################
    for pipe_i in pipe_list_X :
       log("###################", pipe_i, "##########################################################")
       pipe_fun    = load_function_uri(pipe_i['uri'])    ### Load the code definition  into pipe_fun
       cols_name   = pipe_i['cols_family']
       col_type    = pipe_i['type']

       pars        = pipe_i.get('pars', {})
       pars['path_features_store']  = path_features_store    ### intermdiate dataframe
       pars['path_pipeline_export'] = path_pipeline_export   ### Store pipeline

       if col_type == 'cross':
           log("###################  Adding Cross ###################################################")
           pars['dfnum_hot']       = dfi_all['colnum_onehot']  ### dfnum_hot --> dfcross
           pars['dfcat_hot']       = dfi_all['colcat_onehot']
           pars['colid']           = colid
           pars['colcross_single'] = cols_group.get('colcross', [])


       pars['coly'] = cols_group['coly']
       pars['dfy']  = dfi_all[ 'coly' ]  ### add dfy by default

       ### Input columns or prevously Computed Columns ( colnum_bin ), prevent duplicates
       cols_list  = cols_group[cols_name] if cols_name in cols_group else list(dfi_all[cols_name].columns)
       df_        = df[ cols_list]        if cols_name in cols_group else dfi_all[cols_name]

       dfi, col_pars = pipe_fun(df_, cols_list, pars= pars)


       log3("dfi input", df_.columns, df_.shape, cols_name)
       log3("dfi new",   dfi.columns, dfi.shape, col_pars)

       ## Check Index are matching for Later JOIN: Issues with Sampler, re-index !!!!!!
       if 'sampler' not in pipe_i['uri'] :
          dfi.index = df_.index


       ### Concatenate colnum, colnum_bin into cols_family_all , dfi_all  ###########################
       for colj, colist in  col_pars['cols_new'].items() :
          ### Merge sub-family
          cols_family_all[colj] = cols_family_all.get(colj, []) + colist
          dfi_all[colj]         = pd.concat((dfi_all[colj], dfi), axis=1)  if colj in dfi_all else dfi
          # save_features(dfi_all[colj], colj, path_features_store)


    ######  Merge AlL int dfXy  ##################################################################
    dfXy = df[ [coly] + colnum + colcat ]

    for t in dfi_all.keys():
        if t not in [ 'coly', 'colnum', 'colcat' ] :
           dfXy = pd.concat((dfXy, dfi_all[t] ), axis=1)
    save_features(dfXy, 'dfX', path_features_store)


    colX = list(dfXy.columns)
    colX.remove(coly)    ##### Only X columns
    if len(colid)>0:
        cols_family_all['colid']=colid
    cols_family_all['colX'] = colX


    ####  Cols group for model input  ###########################################################
    save(colX,             f'{path_pipeline_export}/colsX.pkl' )
    save(cols_family_all,  f'{path_pipeline_export}/cols_family.pkl' )

    ###### Return values  #######################################################################
    return dfXy, cols_family_all



def preprocess_inference(df, path_pipeline="data/pipeline/pipe_01/", preprocess_pars={}, cols_group=None):
    """
       At Inference time, load model, params and preprocess data.
       Not saving the data, only output final dataframe
    :param df: input dataframe
    :param path_pipeline:  path where processors are stored
    :param preprocess_pars: dict of params specific to preprocessing
    :param cols_group:  dict of column family
    :return: dfXy  Final dataframe,
             cols_family_full : dict of column family
    """
    from util_feature import load, load_function_uri, load_dataset

    #### Pipeline Execution  ####################################################
    pipe_default = [
      {'uri' : 'source/prepro.py::pd_colnum_bin',          'pars': {}, 'cols_family': 'colnum',     'type': ''      },
      {'uri' : 'source/prepro.py::pd_colnum_binto_onehot', 'pars': {}, 'cols_family': 'colnum_bin', 'type': ''      },
      {'uri':  'source/prepro.py::pd_colcat_bin',          'pars': {}, 'cols_family': 'colcat',     'type': ''      },
      {'uri':  'source/prepro.py::pd_colcat_to_onehot',    'pars': {}, 'cols_family': 'colcat_bin', 'type': ''      },
      {'uri' : 'source/prepro.py::pd_colcross',            'pars': {}, 'cols_family': 'colcross',   'type': 'cross' }
    ]
    pipe_list    = preprocess_pars.get('pipe_list', pipe_default)
    pipe_list_X  = [ task for task in pipe_list  if task.get('type', '')  not in ['coly', 'filter']  ]
    pipe_filter  = [ task for task in pipe_list  if task.get('type', '')   in ['filter']  ]


    log("########### Load column by column type ##################################")
    cols_group      = preprocess_pars['cols_group']
    log(cols_group)   ### list of model columns familty
    colid           = cols_group['colid']   # "jobId"
    coly            = cols_group['coly']
    colcat          = cols_group['colcat']  # [ 'companyId', 'jobType', 'degree', 'major', 'industry' ]
    colnum          = cols_group['colnum']  # ['yearsExperience', 'milesFromMetropolis']


    ##### Generate features ########################################################################
    dfi_all          = {} ### Dict of all features
    cols_family_full = {'coly':coly}

    if len(pipe_filter) > 0 :
        log("#####  Filter  #######################################################################")
        pipe_i       = pipe_filter[ 0 ]
        pipe_fun     = load_function_uri(pipe_i['uri'])
        df, col_pars = pipe_fun(df, list(df.columns), pars=pipe_i.get('pars', {}))


    #####  Processors  #############################################################################
    for pipe_i in pipe_list_X :
       log("###################", pipe_i, "#######################################################")
       pipe_fun    = load_function_uri(pipe_i['uri'])    ### Load the code definition  into pipe_fun
       cols_name   = pipe_i['cols_family']
       col_type    = pipe_i['type']
       pars        = pipe_i.get('pars', {})

       ### Load data from disk : inference time
       pars['path_pipeline'] = path_pipeline

       cols_list  = cols_group[cols_name]       if cols_name in cols_group else  cols_family_full[cols_name]
       df_        = df[ cols_group[cols_name]]  if cols_name in cols_group else  dfi_all[cols_name]
       log3(df_, cols_list)

       if col_type == 'cross':
           pars['dfnum_hot']       = dfi_all['colnum_onehot']  ### dfnum_hot --> dfcross
           pars['dfcat_hot']       = dfi_all['colcat_onehot']
           pars['colid']           = colid
           pars['colcross_single'] = cols_group.get('colcross', [])

       dfi, col_pars             = pipe_fun(df_, cols_list, pars= pars)

       ## Check Index are matching for Later JOIN: Issues with Sampler, re-index !!!!!!
       log3("dfi input", df_.columns, df_.shape, cols_name)
       log3("dfi new",   dfi.columns, dfi.shape, col_pars)       
       if 'sampler' not in pipe_i['uri'] :
          dfi.index = df_.index

       ### Concatenate colnum, colnum_bin into cols_family_all
       for colj, colist in  col_pars['cols_new'].items() :
          ### Merge sub-family
          cols_family_full[colj] = cols_family_full.get(colj, []) + colist
          dfi_all[colj]          = pd.concat((dfi_all[colj], dfi), axis=1)  if colj in dfi_all else dfi


    log("######  Merge AlL int dfXy  #############################################################")
    dfXy = df[  colnum + colcat ]
    for t in dfi_all.keys():
        if t not in [  'colnum', 'colcat'] :
           dfXy = pd.concat((dfXy, dfi_all[t] ), axis=1)


    colXy = list(dfXy.columns)
    if len(colid)>0:
        cols_family_full['colid']=colid
    cols_family_full['colX'] = colXy

    return dfXy, cols_family_full


def preprocess_load(path_train_X="", path_train_y="", path_pipeline_export="", cols_group=None, n_sample=5000,
               preprocess_pars={},  path_features_store=None):
    """
        Load pre-computed dataframe
    :param path_train_X:
    :param path_train_y:
    :param path_pipeline_export:
    :param cols_group:
    :param n_sample:
    :param preprocess_pars:
    :param path_features_store:
    :return:
    """
    from source.util_feature import load
    from utilmy  import pd_read_file

    dfXy    = pd_read_file(path_features_store + "/dfX/*.parquet")

    try :
       dfy  = pd_read_file(path_features_store + "/dfy/*.parquet")
       dfXy = dfXy.join(dfy, on= cols_group['colid']  , how="left")

    except :
       log('Error no label', path_features_store + "/dfy/features.parquet")

    cols_family = load(f'{path_pipeline_export}/cols_family.pkl')

    return  dfXy, cols_family



####################################################################################################
############CLI Command ############################################################################
def run_preprocess(config_name, config_path, n_sample=5000,
                   mode='run_preprocess', model_dict=None):     #prefix "pre" added, in order to make if loop possible
    """
    :param config_name:   titanic_lightgbm
    :param config_path:   titanic_classifier.py
    :param n_sample:     nb of rows used
    :param mode:     'run_preprocess'  / 'load_prerocess'
    :param model_dict:  Optional provide the dict model
    :return: None,  only show and save dataframe
    """
    model_dict = model_dict_load(model_dict, config_path, config_name, verbose=True)

    m = model_dict['global_pars']
    path_data         = m['path_data_preprocess']
    path_train_X      = m.get('path_data_prepro_X', path_data + "/features.zip") # ### Can be a list of zip or parquet files
    path_train_y      = m.get('path_data_prepro_y', path_data + "/target.zip")   # ### Can be a list of zip or parquet files

    path_output         = m['path_train_output']
    path_pipeline       = m.get('path_pipeline',       path_output + "/pipeline/" )
    path_features_store = m.get('path_features_store', path_output + '/features_store/' )  #path_data_train replaced with path_output, because preprocessed files are stored there
    path_check_out      = m.get('path_check_out',      path_output + "/check/" )
    log2(path_output)


    log("#### load input column family  ###################################################")
    cols_group = model_dict['data_pars']['cols_input_type']  ### the model config file


    log("#### Preprocess  #################################################################")
    preprocess_pars = model_dict['model_pars']['pre_process_pars']

    if mode == "run_preprocess" :
        dfXy, cols      = preprocess(path_train_X, path_train_y, path_pipeline, cols_group, n_sample,
                                 preprocess_pars,  path_features_store)

    elif mode == "load_preprocess" :
        dfXy, cols      = preprocess_load(path_train_X, path_train_y, path_pipeline, cols_group, n_sample,
                                 preprocess_pars,  path_features_store)
    model_dict['data_pars']['coly'] = cols['coly']

    ### Generate actual column names from colum groups  INTO a single list of columns
    model_dict['data_pars']['cols_model'] = sum([  cols[colgroup] for colgroup in model_dict['data_pars']['cols_model_group'] ]   , [])
    log2(  model_dict['data_pars']['cols_model'] , model_dict['data_pars']['coly'])


    log("#### Save data on disk ############################################################")
    dfXy.to_parquet( path_output  +"/dfXy.parquet"  )
    save(model_dict, path_output  +"/model_dict.pkl")


    log("######### finish ################################################################", )


if __name__ == "__main__":
    import fire
    fire.Fire()
