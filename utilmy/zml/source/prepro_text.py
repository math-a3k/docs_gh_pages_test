# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
  Text pre-processing

"""
import warnings
warnings.filterwarnings('ignore')
import sys, gc, os, pandas as pd, json, copy
import numpy as np

####################################################################################################
#### Add path for python import
sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/")


#### Root folder analysis
root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
print(root)


#### Debuging state (Ture/False)
DEBUG_=True

####################################################################################################
####################################################################################################
def log(*s, n=0, m=1):
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

def logs(*s):
    """function logs
    Args:
        *s:   
    Returns:
        
    """
    if DEBUG_:
        print(*s, flush=True)


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


from util_feature import  save, load_function_uri, load, save_features
####################################################################################################
####################################################################################################

def pd_coltext_clean( df, col, stopwords= None , pars=None):
    """function pd_coltext_clean
    Args:
        df:   
        col:   
        stopwords:   
        pars:   
    Returns:
        
    """
    import string, re
    ntoken= pars.get('n_token', 1)
    df      = df.fillna("")
    dftext = df
    log(dftext)
    log(col)
    list1 = col

    def coltext_remove_stopwords(text, stopwords=None, sep=" "):
        tokens = text.split(sep)
        if stopwords != None:
            tokens = [t.strip() for t in tokens if t.strip() not in stopwords]
        return " ".join(tokens)

    # list1 = []
    # list1.append(col)
    # fromword = [ r"\b({w})\b".format(w=w)  for w in fromword    ]
    # print(fromword)
    for col_n in list1:
        dftext[col_n] = dftext[col_n].fillna("")
        dftext[col_n] = dftext[col_n].apply(lambda x : str(x))
        dftext[col_n] = dftext[col_n].str.lower()
        dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.punctuation))
        dftext[col_n] = dftext[col_n].apply(lambda x: x.translate(string.digits))
        dftext[col_n] = dftext[col_n].apply(lambda x: re.sub("[!@,#$+%*:()'-]", " ", x))
        dftext[col_n] = dftext[col_n].apply(lambda x: coltext_remove_stopwords(x, stopwords=stopwords))
    return dftext



def pd_coltext_wordfreq(df, col, stopwords, ntoken=100):
    """
    :param df:
    :param coltext:  text where word frequency should be extracted
    :param nb_to_show:
    :return:
    """
    sep=" "
    logs('----col-----\n', col)
    coltext_freq = df[col].apply(str).apply(lambda x: pd.value_counts(x.split(sep))).sum(axis=0).reset_index()
    coltext_freq.columns = ["word", "freq"]
    coltext_freq = coltext_freq.sort_values("freq", ascending=0)
    log(coltext_freq)

    word_tokeep  = coltext_freq["word"].values[:ntoken]
    word_tokeep  = [  t for t in word_tokeep if t not in stopwords   ]

    return coltext_freq, word_tokeep


def nlp_get_stopwords():
    """function nlp_get_stopwords
    Args:
    Returns:
        
    """
    import json
    import string
    stopwords = json.load(open("source/utils/stopwords_en.json") )["word"]
    stopwords = [ t for t in string.punctuation ] + stopwords
    stopwords = [ "", " ", ",", ".", "-", "*", '€', "+", "/" ] + stopwords
    stopwords =list(set( stopwords ))
    stopwords.sort()
    print( stopwords )
    stopwords = set(stopwords)
    return stopwords


def pd_coltext(df, col, pars={}):
    """
    df : Datframe
    col : list of columns
    pars : dict of pars

    """
    from utils import util_text, util_model

    #### Load pars ###################################################################
    path_pipeline        = pars.get('path_pipeline', None)
    word_tokeep_dict_all = load(  path_pipeline + "/word_tokeep_dict_all.pkl" )  if path_pipeline is not None else {}
    # dftext_tdidf_all = load(f'{path_pipeline}/dftext_tdidf.pkl') if  path_pipeline else None
    # dftext_svd_list_all      = load(f'{path_pipeline}/dftext_svd.pkl')   if  path_pipeline else None
    dimpca       = pars.get('dimpca', 2)
    word_minfreq = pars.get('word_minfreq', 3)

    #### Process  ####################################################################
    stopwords           = nlp_get_stopwords()
    dftext              = pd_coltext_clean(df, col, stopwords= stopwords , pars=pars)
    dftext_svd_list_all = None
    dftext_tdidf_all    = None

    ### Processing each of text columns to create a bag of word/to load the bag of word -> tf-idf -> svd
    for col_ in col:

            if path_pipeline is not None:
                ### If it is in Inference step, use the saved bag of word for the column `col_`
                word_tokeep = word_tokeep_dict_all[col_]

            else:
                ### If it is not, create a bag of word
                coltext_freq, word_tokeep = pd_coltext_wordfreq(df, col_, stopwords, ntoken=100)  ## nb of words to keep
                word_tokeep_dict_all[col_] = word_tokeep  ## save the bag of wrod for `col_` in a dict

            dftext_tdidf_dict, word_tokeep_dict = util_text.pd_coltext_tdidf(dftext, coltext=col_, word_minfreq= word_minfreq,
                                                                             word_tokeep = word_tokeep,
                                                                             return_val  = "dataframe,param")

            dftext_tdidf_all = pd.DataFrame(dftext_tdidf_dict) if dftext_tdidf_all is None else pd.concat((dftext_tdidf_all,pd.DataFrame(dftext_tdidf_dict)),axis=1)
            log(word_tokeep_dict)

            ###  Dimesnion reduction for Sparse Matrix
            dftext_svd_list, svd_list = util_model.pd_dim_reduction(dftext_tdidf_dict,
                                                           colname        = None,
                                                           model_pretrain = None,
                                                           colprefix      = col_ + "_svd",
                                                           method         = "svd",  dimpca=dimpca,  return_val="dataframe,param")

            dftext_svd_list_all = dftext_svd_list if dftext_svd_list_all is None else pd.concat((dftext_svd_list_all,dftext_svd_list),axis=1)
    #################################################################################

    ###### Save and Export ##########################################################
    if 'path_features_store' in pars:
            save_features(dftext_svd_list_all, 'dftext_svd' + "-" + str(col), pars['path_features_store'])
            # save(dftext_svd_list_all,  pars['path_pipeline_export'] + "/dftext_svd.pkl")
            # save(dftext_tdidf_all,     pars['path_pipeline_export'] + "/dftext_tdidf.pkl" )
            save(word_tokeep_dict_all,     pars['path_pipeline_export'] + "/word_tokeep_dict_all.pkl" )

    col_pars = {}
    col_pars['cols_new'] = {
     # 'coltext_tdidf'    : dftext_tdidf_all.columns.tolist(),       ### list
     'coltext_svd'      : dftext_svd_list_all.columns.tolist()      ### list
    }

    dftext_svd_list_all.index = dftext.index
    # return pd.concat((dftext_svd_list_all,dftext_svd_list_all),axis=1), col_pars
    return dftext_svd_list_all, col_pars


def pd_coltext_universal_google(df, col, pars={}):
    """
     # Universal sentence encoding from Tensorflow
       Text ---> Vectors
    from source.preprocessors import  pd_coltext_universal_google
    https://tfhub.dev/google/universal-sentence-encoder-multilingual/3

    #latest Tensorflow that supports sentencepiece is 1.13.1
    !pip uninstall --quiet --yes tensorflow
    !pip install --quiet tensorflow-gpu==1.13.1
    !pip install --quiet tensorflow-hub
    pip install --quiet tf-sentencepiece, simpleneighbors
    !pip install --quiet simpleneighbors

    # df : dataframe
    # col : list of text colnum names
    pars
    """
    prefix = "coltext_universal_google"
    if 'path_pipeline' in  pars  :   ### Load during Inference
       coltext_embed = load( pars['path_pipeline'] + "/{prefix}.pkl" )
       pars_model    = load( pars['path_pipeline'] + "/{prefix}_pars.pkl" )

    ####### Custom Code ###############################################################
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text
    #from tqdm import tqdm #progress bar
    uri_list = [
    ]
    url_default = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    url         = pars.get("model_uri", url_default )
    model       = hub.load( url )
    pars_model  = {}
    dfall       = None
    for coli in col[:1] :
        X = []
        for r in (df[coli]):
            if pd.isnull(r)==True :
                r=""
            emb = model(r)
            review_emb = tf.reshape(emb, [-1]).numpy()
            X.append(review_emb)

        dfi   = pd.DataFrame(X, columns= [ coli + "_" + str(i) for i in range( len(X[0]))   ] ,
                             index = df.index)
        dfall = pd.concat((dfall, dfi))  if dfall is not None else dfi

    coltext_embed = list(dfall.columns)


    ##### Export ####################################################################
    if 'path_features_store' in pars and 'path_pipeline_export' in pars:
       save_features(dfall, 'dftext_embed', pars['path_features_store'])
       save(coltext_embed,  pars['path_pipeline_export'] + "/{prefix}.pkl" )
       save(pars_model,     pars['path_pipeline_export'] + "/{prefix}_pars.pkl" )
       # save(model,          pars['path_pipeline_export'] + "/{prefix}_model.pkl" )
       # model_uri = pars['path_pipeline_export'] + "/{prefix}_model.pkl"


    # col_pars = {'model_uri' :  model_uri, 'pars': pars_model}
    col_pars = {'model_uri' :  url , 'pars': pars_model} # model_uri
    col_pars['cols_new']      = {
       'coltext_universal_google' :  coltext_embed ### list
    }
    return dfall, col_pars


if __name__ == "__main__":
    import fire
    fire.Fire()
