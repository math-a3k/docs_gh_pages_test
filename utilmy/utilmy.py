# -*- coding: utf-8 -*-
HELP= """



"""
import os, sys, time, datetime,inspect, json, yaml, gc, random

verbose = 3   ### Global setting

def log(*s, **kw):
    print(*s, flush=True, **kw)

def log2(*s, **kw):
    if verbose >=2 : print(*s, flush=True, **kw)

def log3(*s, **kw):
    if verbose >=3 : print(*s, flush=True, **kw)

        
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
    """ Extract code source from test code
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


def pd_random(ncols=7, nrows=100):
   import pandas as pd
   ll = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
   df = pd.DataFrame(ll, columns = [str(i) for i in range(0,ncols)])
   return df


def pd_generate_data(ncols=7, nrows=100):
    """ Generate sample data for function testing categorical features
    """
    import numpy as np, pandas as pd
    np.random.seed(444)
    numerical    = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
    df = pd.DataFrame(numerical, columns = [str(i) for i in range(0,ncols)])
    df['cat1']= np.random.choice(  a=[0, 1],  size=nrows,  p=[0.7, 0.3]  )
    df['cat2']= np.random.choice(  a=[4, 5, 6],  size=nrows,  p=[0.5, 0.3, 0.2]  )
    df['cat1']= np.where( df['cat1'] == 0,'low',np.where(df['cat1'] == 1, 'High','V.High'))
    return df


def pd_getdata(verbose=True):
    """data = test_get_data()
    df   = data['housing.csv']
    df.head(3)
    https://github.com/szrlee/Stock-Time-Series-Analysis/tree/master/data
    """
    import pandas as pd
    flist = [
        'https://raw.githubusercontent.com/samigamer1999/datasets/main/titanic.csv',
        'https://github.com/subhadipml/California-Housing-Price-Prediction/raw/master/housing.csv',
        'https://raw.githubusercontent.com/AlexAdvent/high_charts/main/data/stock_data.csv',
        'https://raw.githubusercontent.com/samigamer1999/datasets/main/cars.csv',
        'https://raw.githubusercontent.com/samigamer1999/datasets/main/sales.csv',
        'https://raw.githubusercontent.com/AlexAdvent/high_charts/main/data/weatherdata.csv'
    ]
    data = {}
    for url in flist :
       fname =  url.split("/")[-1]
       print( "\n", "\n", url, )
       df = pd.read_csv(url)
       data[fname] = df
       if verbose: print(df)
       # df.to_csv(fname , index=False)
    print(data.keys() )
    return data


def glob_glob(dirin, nfile=1000):
    import glob
    flist  = sorted( glob.glob(dirin  ))
    flist  = flist[:nfile]
    log('Nfile: ', len(flist), str(flist)[:100])
    return flist


def sys_exit(msg="exited",  err_int=0):
    import os, sys
    print(msg)         
    ### exit with no error msg 
    # sys.stderr = open(os.devnull, 'w')
    sys.stdout = sys.__stdout__ = open(os.devnull, 'w')
    sys.exit(err_int)         

    
def sys_install(cmd=""):
   import os, sys, time  
   print("Installing  ")
   print( cmd +"  \n\n ...") ; time.sleep(7)
   os.system(cmd )
   print( "\n\n\n############### Please relaunch python  ############## \n")   
   print('Exit \n\n\n')


class Index0(object):
    """
    ### to maintain global index, flist = index.read()  index.save(flist)
    """
    def __init__(self, findex:str="ztmp_file.txt"):
        self.findex = findex        
        os.makedirs(os.path.dirname(self.findex), exist_ok=True)
        if not os.path.isfile(self.findex):
            with open(self.findex, mode='a') as fp:
                fp.write("")              

    def read(self,):            
        with open(self.findex, mode='r') as fp:
            flist = fp.readlines()

        if len(flist) < 1 : return []    
        flist2 = []
        for t  in flist :
            if len(t) > 5 and t[0] != "#"  :
              flist2.append( t.strip() )
        return flist2    

    def save(self, flist:list):
        if len(flist) < 1 : return True
        ss = ""
        for fi in flist :
          ss = ss + fi + "\n"                
        with open(self.findex, mode='a') as fp:
            fp.write(ss )
        return True   

###################################################################################################
###### Test #####################################################################################
def test_all():
   import utilmy as m

   ###################################################################################
   log("\n##### git_repo_root  ")
   log(m.git_repo_root())
   assert not m.git_repo_root() == None, "err git repo"


   log("\n##### Doc generator: help_create  ")
   for name in [ 'utilmy.parallel', 'utilmy.utilmy', 'utilmy.ppandas'  ]:
      log("\n#############", name,"\n", m.help_create(name))


   ###################################################################################
   log("\n##### global_verbosity  ")
   print('verbosity', m.global_verbosity(__file__, "config.json", 40,))
   print('verbosity', m.global_verbosity('../', "config.json", 40,))
   print('verbosity', m.global_verbosity(__file__))

   verbosity = 40
   gverbosity = m.global_verbosity(__file__)
   assert gverbosity == 5, "incorrect default verbosity"
   gverbosity =m.global_verbosity(__file__, "config.json", 40,)
   assert gverbosity == verbosity, "incorrect verbosity "

   ###################################################################################




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
    pd_plot_histogram,
    pd_filter,
    pd_to_file,
    pd_sample_strat,
    pd_cartesian,
    pd_col_bins,
    pd_dtype_reduce,
    pd_dtype_count_unique,
    pd_dtype_to_category,
    pd_dtype_getcontinuous,
    pd_del,
    pd_add_noise,
    pd_cols_unique_count,
    pd_show,
    pd_to_hiveparquet,
    pd_to_mapdict
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
    os_makedirs,
    
    toFileSafe
)

### Alias
os_remove = os_removedirs
to_file   = os_to_file



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
      lexclude = {   "In", "Out", "get_ipython", 'exit','quit', 'Session',  }
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




