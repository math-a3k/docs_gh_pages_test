# -*- coding: utf-8 -*-
HELP= """



"""
import os, sys, time, datetime,inspect, json, yaml, gc

def log(*s):
    print(*s, flush=True)

def log2(*s, verbose=1):
    if verbose >0 : print(*s, flush=True)


def help():
    suffix = "\n\n\n###############################"
    ss     = help_create(modulename='utilmy', prefixs=None) + suffix
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


def import_function(fun_name=None, module_name=None):
    import importlib

    if isinstance(module_name, str):
       module1 = importlib.import_module(module_name)
       func = getattr(module1, fun_name)
    else :
       func = globals()[fun_name]

    return func


def help_create(modulename='utilmy.nnumpy', prefixs=None):
    """
       Extract code source from test code
    """
    import importlib
    prefixs = ['test']
    module1 = importlib.import_module(modulename)
    ll      = dir(module1)
    ll  = [ t for t in ll if prefixs[0] in t]
    ss  = ""
    for fname in ll :
        fun = import_function(fname, modulename)
        ss += help_get_codesource(fun)
    return ss



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
#from utilmy.keyvalue import  (
#   diskcache_load,
#   diskcache_save,
#   diskcache_save2,
#   db_init, db_size, db_flush
#)



###################################################################################################
###### Parallel #####################################################################################
from utilmy.parallel import (
    multithread_run,
    multiproc_run
)




###################################################################################################
####### Numpy compute #############################################################################
from utilmy.nnumpy import (

    dict_to_namespace,
    to_dict,
    to_timeunix,
    to_datetime,
    np_list_intersection,
    np_add_remove,
    to_float,
    to_int,
    is_int,
    is_float

)



##### OS, cofnfig ######################################################################################
from utilmy.oos import(
    os_path_size,
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
    # os_ram_object,
    os_copy,
    os_removedirs,
    os_getcwd,
    os_system,
    os_makedirs
)




################################################################################################
########  Configuration  #######################################################################
from utilmy.configs.util_config import (
 config_load,
 global_verbosity


)


######################################################################################################
######## External IO #################################################################################
from utilmy.iio import (
 hdfs_put,
 hdfs_get,
 hdfs_walk
)


######################################################################################################
###### Plot ##########################################################################################
#from utilmy.viz.vizhtml import (
#  images_to_html,   ### folder of images to HTML
#  test_getdata
# )



###################################################################################################
###### Debug ######################################################################################
from utilmy.debug import (
    print_everywhere,

    log10,
    log_trace,  ###(msg="", dump_path="", globs=None)  Debug with full trace message


    profiler_start,
    profiler_stop
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
if __name__ == "__main__":
    import fire ;
    fire.Fire()




