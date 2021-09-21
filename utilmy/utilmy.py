# -*- coding: utf-8 -*-
HELP= """



"""
import os, sys, time, datetime,inspect, json, yaml, gc

def log(*s):
    print(*s, flush=True)

def log2(*s, verbose=1):
    if verbose >0 : print(*s, flush=True)


def help():
    ss  = ""


    ss += HELP
    print(ss)


def help_get_codesource(func):
    """ Extract code source from func name"""
    import inspect
    try:
        lines_to_skip = len(func.__doc__.split('\n'))
    except AttributeError:
        lines_to_skip = 0
    lines = inspect.getsourcelines(func)[0]
    return ''.join( lines[lines_to_skip+1:] )



###################################################################################################
###### Pandas #####################################################################################
from utilmy.parallel import (
    pd_read_file,    ### parallel reading
    pd_read_file2,
    pd_groupby_parallel,
)



from utilmy.ppandas import (
    pd_random,
    pd_merge,
    pd_plot_multi,
    pd_filter,
    pd_to_file,
    pd_sample_strat,
    pd_cartesian,
    pd_plot_histogram,
    pd_col_bins,
    pd_dtype_reduce,
    pd_dtype_count_unique,
    pd_dtype_to_category,
    pd_dtype_getcontinuous,
    pd_del,
    pd_add_noise,
    pd_cols_unique_count,
    pd_show
)




#########################################################################################################
##### Utils numpy, list #################################################################################
from utilmy.keyvalue import  (
   diskcache_load,
   diskcache_save,
   diskcache_save2,
   db_init, db_size, db_flush
)



###################################################################################################
###### Parallel #####################################################################################
from utilmy.parallel import (
    multithread_run,
    multiproc_run
)




###################################################################################################
####### Base tyoe #################################################################################
class dict_to_namespace(object):
    #### Dict to namespace
    def __init__(self, d):
        self.__dict__ = d


def to_dict(**kw):
  ## return dict version of the params
  return kw


def to_timeunix(datex="2018-01-16"):
  if isinstance(datex, str)  :
     return int(time.mktime(datetime.datetime.strptime(datex, "%Y-%m-%d").timetuple()) * 1000)

  if isinstance(datex, datetime)  :
     return int(time.mktime( datex.timetuple()) * 1000)


def to_datetime(x) :
  import pandas as pd
  return pd.to_datetime( str(x) )


def np_list_intersection(l1, l2) :
  return [x for x in l1 if x in l2]


def np_add_remove(set_, to_remove, to_add):
    # a function that removes list of elements and adds an element from a set
    result_temp = set_.copy()
    for element in to_remove:
        result_temp.remove(element)
    result_temp.add(to_add)
    return result_temp


def to_float(x):
    try :
        return float(x)
    except :
        return float("NaN")

def to_int(x):
    try :
        return int(x)
    except :
        return float("NaN")

def is_int(x):
    try :
        int(x)
        return True
    except :
        return False    

def is_float(x):
    try :
        float(x)
        return True
    except :
        return False   




########################################################################################################



##### OS, cofnfig ######################################################################################
from utilmy.oos import(
    os_getsize,
    os_path_split,
    os_file_check,
    os_file_replacestring,
    os_walk,
    z_os_search_fast,
    os_search_content,
    os_get_function_name,
    os_variable_exist,
    os_variable_init,
    os_import,
    os_variable_check,
    os_clean_memory,
    os_system_list,
    os_to_file,
    os_platform_os,
    os_platform_ip,
    os_memory,
    os_sleep_cpu,
    os_cpu,
    os_ram_object,
    os_copy,
    os_removedirs,
    os_getcwd,
    os_system,
    os_makedirs
)




################################################################################################
################################################################################################
def config_load(config_path:str = None,
                path_default:str=None,
                config_default:dict=None):
    """Load Config file into a dict  from .json or .yaml file
    TODO .cfg file
    1) load config_path
    2) If not, load default from HOME USER
    3) If not, create default on in python code
    Args:
        config_path: path of config or 'default' tag value
    Returns: dict config
    """
    import json, yaml
    import pathlib

    path_default        = pathlib.Path.home() / ".mygenerator" if path_default is None else path_default
    config_path_default = path_default / "config.yaml"
    if config_default is None :
        config_default = {
            "current_dataset": "mnist",
            "datasets": {
                "mnist": {
                    "url": "/mnist_png.tar.gz",
                    "path": str(path_default / "mnist_png" / "training"),
                }
            },
        }

    #####################################################################
    if config_path is None or config_path == "default":
        log(f"Using config: {config_path_default}")
        config_path = config_path_default
    else :
        config_path = pathlib.Path(config_path)

    try:
        log("loading config", config_path)
        if ".yaml" in config_path :
            cfg = yaml.load(config_path.read_text(), Loader=yaml.Loader)
            dd = {}
            for x in cfg :   ### Map to dict
                for key,val in x.items():
                   dd[key] = val
            return cfg

        if ".json" in config_path :
           return json.load(config_path.read_text())

    except Exception as e:
        log(f"Cannot read yaml/json file {config_path}", e)


    log("#### Using default configuration")
    log(config_default)
    log(f"Creating config file in {config_path_default}")
    os.makedirs(path_default, exist_ok=True)
    with open(config_path_default, mode="w") as fp:
        yaml.dump(config_default, fp)
    return config_default



def global_verbosity(cur_path, path_relative="/../../config.json",
                   default=5, key='verbosity',):
    """ Get global verbosity
    verbosity = global_verbosity(__file__, "/../../config.json", default=5 )

    verbosity = global_verbosity("repo_root", "config/config.json", default=5 )

    :param cur_path:
    :param path_relative:
    :param key:
    :param default:
    :return:
    """
    try   :
      if 'repo_root' == cur_path  :
          cur_path =  git_repo_root()

      if '.json' in path_relative :
         dd = json.load(open(os.path.dirname(os.path.abspath(cur_path)) + path_relative , mode='r'))

      elif '.yaml' in path_relative or '.yml' in path_relative :
         import yaml
         dd = yaml.load(open(os.path.dirname(os.path.abspath(cur_path)) + path_relative , mode='r'))

      else :
          raise Exception( path_relative + " not supported ")
      verbosity = int(dd[key])

    except Exception as e :
      verbosity = default
      #raise Exception(f"{e}")
    return verbosity






######################################################################################################
######## External IO #################################################################################
from utilmy.io import (
 hdfs_put,
 hdfs_get,
 hdfs_walk
)




######################################################################################################
########Git ##########################################################################################
def git_repo_root():
    try :
      cmd = "git rev-parse --show-toplevel"
      mout, merr = os_system(cmd)
      path = mout.split("\n")[0]
      if len(path) < 1:  return None
    except : return None
    return path


def git_current_hash(mode='full'):
   import subprocess
   # label = subprocess.check_output(["git", "describe", "--always"]).strip();
   label = subprocess.check_output([ 'git', 'rev-parse', 'HEAD' ]).strip();
   label = label.decode('utf-8')
   return label




######################################################################################################
###### Plot ##########################################################################################
from utilmy.viz.vizhtml import (
  images_to_html   ### folder of images to HTML

)



################################################################################################
################################################################################################
class Session(object) :
    """ Save Python Interpreter session on disk
      from util import Session
      sess = Session("recsys")
      sess.save( globals() )
    """
    def __init__(self, dir_session="ztmp/session/",) :
      os.makedirs(dir_session, exist_ok=True)
      self.dir_session =  dir_session
      self.cur_session =  None
      print(self.dir_session)

    def show(self) :
       import glob
       flist = glob.glob(self.dir_session + "/*" )
       print(flist)

    def save(self, name, glob=None, tag="") :
       path = f"{self.dir_session}/{name}{tag}/"
       self.cur_session = path
       os.makedirs(self.cur_session, exist_ok=True)
       self.save_session(self.cur_session, glob)

    def load(self, name, glob:dict=None, tag="") :
      path = f"{self.dir_session}/{name}{tag}/"
      self.cur_session = path
      print(self.cur_session)
      self.load_session(self.cur_session , glob )


    def save_session(self, folder , globs, tag="" ) :
      import pandas as pd
      os.makedirs( folder , exist_ok= True)
      lcheck = [ "<class 'pandas.core.frame.DataFrame'>", "<class 'list'>", "<class 'dict'>",
                 "<class 'str'>" ,  "<class 'numpy.ndarray'>" ]
      lexclude = {   "In", "Out" }
      gitems = globs.items()
      for x, _ in gitems :
         if not x.startswith('_') and  x not in lexclude  :
            x_type =  str(type(globs.get(x) ))
            fname  =  folder  + "/" + x + ".pkl"
            try :
              if "pandas.core.frame.DataFrame" in x_type :
                  pd.to_pickle( globs[x], fname)

              elif x_type in lcheck or x.startswith('clf')  :
                  save( globs[x], fname )

              print(fname)
            except Exception as e:
                  print(x, x_type, e)


    def load_session(self, folder, globs=None) :
      """
      """
      print(folder)
      for dirpath, subdirs, files in os.walk( folder ):
        for x in files:
           filename = os.path.join(dirpath, x)
           x = x.replace(".pkl", "")
           try :
             globs[x] = load(  filename )
             print(filename)
           except Exception as e :
             print(filename, e)


def save(dd, to_file="", verbose=False):
  import pickle, os
  os.makedirs(os.path.dirname(to_file), exist_ok=True)
  pickle.dump(dd, open(to_file, mode="wb") , protocol=pickle.HIGHEST_PROTOCOL)
  #if verbose : os_file_check(to_file)


def load(to_file=""):
  import pickle
  dd =   pickle.load(open(to_file, mode="rb"))
  return dd




###################################################################################################
###### Debug ######################################################################################
from utilmy.debug import (
    print_everywhere,

    log10,
    log_trace,  ###(msg="", dump_path="", globs=None)  Debug with full trace message


    profiler_start,
    profiler_stop
)



###################################################################################################
if __name__ == "__main__":
    import fire ;
    fire.Fire()




