# -*- coding: utf-8 -*-
import os
import yaml
import pyspark
from loguru import logger

##########################################################################################
################### Logs Wrapper #########################################################
def logger_setdefault():
    """
    Returns:
    """
    # Linux / OSX
    cmd = "export LOGURU_FORMAT='{time} | <lvl>{message}</lvl>'"

    # Windows
    cmd = 'setx LOGURU_DEBUG_COLOR "<green>"'



def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    """function log
    Args:
        *s:   
    Returns:
        
    """
    logger.info(",".join([ str(t) for t in s  ]) )

def log2(*s):
    """function log2
    Args:
        *s:   
    Returns:
        
    """
    logger.warning(",".join([ str(t) for t in s  ]) )

def log3(*s):
    """function log3
    Args:
        *s:   
    Returns:
        
    """
    logger.debug(",".join([ str(t) for t in s  ]) )


def log(*s):
    print(*s)

def log_sample(*s):
    """function log_sample
    Args:
        *s:   
    Returns:
        
    """
    print(*s)


##########################################################################################
def config_load(config_path:str):
    """  Load Config file into a dict
    Args:
        config_path: path of config
    Returns: dict config
    """
    #Load the yaml config file
    with open(config_path, "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    dd = {}
    for x in config_data :
        for key,val in x.items():
           dd[key] = val

    return dd


def spark_check(df:pyspark.sql.DataFrame, conf:dict=None, path:str="", nsample:int=10 ,
                save=True, verbose=True , returnval=False):
    """ Snapshot checkpoint for dataframe
    Args:
        conf:  Configuration in dict
        df:
        path:
        nsample:
        save:
        verbose:
        returnval:
    Returns:
    """
    if conf is not None :
        confc = conf.get('Check', {})
        path = confc.get('path_check', path)
        save = confc.get('save', save)
        returnval = confc.get('returnval', returnval)
        verbose = confc.get('verbose', verbose)

    if save or returnval or verbose:
        df1 =   df.limit(nsample).toPandas()

    if save :
        ##### Need HDFS version
        os.makedirs( path , exist_ok=True )
        df1.to_csv(path + '/table.csv', sep='\t', index=False)

    if verbose :
        log(df1.head(2).T)
        log( df.printSchema() )

    if returnval :    
        return df1



##########################################################################################
class to_namespace(object):
    def __init__(self, d):
        """ to_namespace:__init__
        Args:
            d:     
        Returns:
           
        """

        self.__dict__ = d







