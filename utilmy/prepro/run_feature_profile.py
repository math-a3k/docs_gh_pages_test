# -*- coding: utf-8 -*- 
"""
python classifier_adfraud.py  data_profile  --path_data_train data/input/adfraud/raw/raw_10m.zip
"""
import gc, os, logging
from datetime import datetime
import warnings, numpy as np, pandas as pd
import pandas_profiling as pp
import sys, json


###################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")
import util_feature


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


def log(*s, n=0, m=0):
    """function log
    Args:
        *s:   
        n:   
        m:   
    Returns:
        
    """
    sspace = "#" * n
    sjump = "\n" * m
    ### Implement pseudo Logging
    print(sjump, sspace, s, sspace, flush=True)



############CLI Command ############################################################################
def run_profile(path_data=None,  path_output="data/out/ztmp/", n_sample=5000):
    """
      Use folder , filename are fixed.
    """
    path_output = path_output
    os.makedirs(path_output, exist_ok=True)
    log(path_output)

    if ".zip" in path_data  or "gz" in path_data :
        path_train_X   = path_data
        path_train_y   = ""
    else :
        path_train_X   = path_data   + "/features*"
        path_train_y   = path_data   + "/target*"

    try :
        log("#### load input column family  ###################################################")
        cols_group = json.load(open(path_data + "/cols_group.json", mode='r'))
        log(cols_group)


        ##### column names for feature generation ###############################################
        log(cols_group)
        coly            = cols_group['coly']  # 'salary'
        colid           = cols_group['colid']  # "jobId"
        colcat          = cols_group['colcat']  # [ 'companyId', 'jobType', 'degree', 'major', 'industry' ]
        colnum          = cols_group['colnum']  # ['yearsExperience', 'milesFromMetropolis']

        colcross_single = cols_group.get('colcross', [])   ### List of single columns
        #coltext        = cols_group.get('coltext', [])
        coltext         = cols_group['coltext']
        coldate         = cols_group.get('coldate', [])
        colall          = colnum + colcat + coltext + coldate
        log(colall)

    except :
        log("######## Generating cols_json   ############################")
        from util_feature import pd_read_file
        df     = pd_read_file( path_train_X )
        colall = list( df.columns)
        dtype = df.dtypes

        with open( path_output + "cols_group.json" , mode='w')  as fp :
            js    = {
               'colnum'  :  [] ,
               'colcat'  :  [] ,
               'coltext' :  [] ,
               'coldate' :  [] ,
               'colall'  :  colall
            }
            json.dump( js, fp)
            fp.write(str(dtype))
        log( path_output + "cols_group.json"  )
        return None


    #### Pandas Profiling for features in train  ###############################
    df = pd.read_csv( path_train_X ) # path + f"/new_data/Titanic_Features.csv")
    try :
        dfy = pd.read_csv(path_train_y)  # + f"/new_data/Titanic_Labels.csv")
        df  = pd.merge(df, dfy, on =colid,  how="left")
    except : 
          pass

    df = df.set_index(colid)
    for x in colcat:
       df[x] = df[x].factorize()[0]


    profile = df.profile_report(title='Profile data')
    profile.to_file(output_file=path_output + "/00_features_report.html")
    log( path_output + "/00_features_report.html")
    log("######### finish #################################", )




if __name__ == "__main__":
    import fire
    fire.Fire()




"""
    #### Test dataset  ################################################
    df = pd.read_csv(path + f"/new_data/Titanic_test.csv")
    df = df.set_index(colid)
    for x in colcat:
        df[x] = df[x].factorize()[0]
    profile = df.profile_report(title='Profile Test data')
    profile.to_file(output_file=path + "/analysis/00_features_test_report.html")
    log("#### Preprocess  #################################################################")
    preprocess_pars = model_dict['model_pars']['pre_process_pars']
    filter_pars     = model_dict['data_pars']['filter_pars']    
    dfXy, cols = preprocess(path_train_X, path_train_y, path_pipeline_out, cols_group, n_sample, 
                            preprocess_pars, filter_pars)
    model_dict['data_pars']['coly'] = cols['coly']
    
    log("######### export #################################", )
    os.makedirs(path_check_out, exist_ok=True)
    colexport = [cols['colid'], cols['coly'], cols['coly'] + "_pred"]
    dfXy[colexport].to_csv(path_check_out + "/pred_check.csv")  # Only results
    #dfXy.to_parquet(path_check_out + "/dfX.parquet")  # train input data
    dfXy.to_csv(path_check_out + "/dfX.csv")  # train input data
    #dfXytest.to_parquet(path_check_out + "/dfXtest.parquet")  # Test input data
    dfXytest.to_csv(path_check_out + "/dfXtest.csv")  # Test input data
 
"""
