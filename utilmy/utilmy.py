# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect, json, yaml



################################################################################################
def pd_read_file(path_glob="*.pkl", ignore_index=True,  cols=None,
                 verbose=False, nrows=-1, concat_sort=True, n_pool=1, drop_duplicates=None, col_filter=None,
                 col_filter_val=None, dtype=None,  **kw):
  """  Read file in parallel from disk : very Fast
  :param path_glob: list of pattern, or sep by ";"
  :return:
  """
  import glob, gc,  pandas as pd, os
  def log(*s, **kw):
      print(*s, flush=True, **kw)      
  readers = {
          ".pkl"     : pd.read_pickle,
          ".parquet" : pd.read_parquet,
          ".tsv"     : pd.read_csv,
          ".csv"     : pd.read_csv,
          ".txt"     : pd.read_csv,
          ".zip"     : pd.read_csv,
          ".gzip"    : pd.read_csv,
          ".gz"      : pd.read_csv,
   }
  from multiprocessing.pool import ThreadPool

  #### File
  if isinstance(path_glob, list):  path_glob = ";".join(path_glob)
  path_glob  = path_glob.split(";")
  file_list = []
  for pi in path_glob :    
      file_list.extend( sorted( glob.glob(pi) ) )
  file_list = sorted(list(set(file_list)))
  n_file    = len(file_list)
  if verbose: log(file_list)

  #### Pool count
  if n_pool < 1 :  n_pool = 1
  if n_file <= 0:  m_job  = 0
  elif n_file <= 2:
    m_job  = n_file
    n_pool = 1
  else  :
    m_job  = 1 + n_file // n_pool  if n_file >= 3 else 1
  if verbose : log(n_file,  n_file // n_pool )

  pool   = ThreadPool(processes=n_pool)
  dfall  = pd.DataFrame()
  for j in range(0, m_job ) :
      if verbose : log("Pool", j, end=",")
      job_list = []
      for i in range(n_pool):
         if n_pool*j + i >= n_file  : break
         filei         = file_list[n_pool*j + i]
         ext           = os.path.splitext(filei)[1]
         if ext == None or ext == '':
           continue

         pd_reader_obj = readers[ext]
         if pd_reader_obj == None:
           continue

         job_list.append( pool.apply_async(pd_reader_obj, (filei, )))
         if verbose : log(j, filei)

      for i in range(n_pool):
        if i >= len(job_list): break
        dfi   = job_list[ i].get()
        
        if dtype is not None      : dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32') 
        if col_filter is not None : dfi = dfi[ dfi[col_filter] == col_filter_val ]
        if cols is not None :       dfi = dfi[cols]
        if nrows > 0        :       dfi = dfi.iloc[:nrows,:]
        if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates)
        gc.collect()

        dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
        #log("Len", n_pool*j + i, len(dfall))
        del dfi; gc.collect()

  if m_job>0 and verbose : log(n_file, j * n_file//n_pool )
  return dfall




def pd_dtype_reduce(dfm, int0 ='int32', float0 = 'float32') :
    for c in dfm.columns :
        if dfm[c].dtype ==  np.dtype(np.int32) :       dfm[c] = dfm[c].astype( int0 )
        elif   dfm[c].dtype ==  np.dtype(np.int64) :   dfm[c] = dfm[c].astype( int0 )  
        elif dfm[c].dtype ==  np.dtype(np.float64) :   dfm[c] = dfm[c].astype( float0 )
    return dfm


      

def pd_show(df, nrows=100, **kw):
    """
      Show from Dataframe
    """
    import pandas as pd
    fpath = 'ztmp/ztmp_dataframe.csv'
    os_makedirs(fpath)
    df.iloc[:nrows,:].to_csv(fpath, sep=",", mode='w')


    ## In Windows
    cmd = f"notepad.exe {fpath}"
    os.system(cmd)

    
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

      
################################################################################################
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












###################################################################################################
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
def os_platform_os():
    #### get linux or windows
    pass

def os_cpu():
    ### Nb of cpus cores
    pass


def os_platform_ip():
    ### IP
    pass

def os_memory():
    """ Get node total memory and memory usage in linux
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret




def os_removedirs(path):
    """
       issues with no empty Folder
    # Delete everything reachable from the directory named in 'top',
    # assuming there are no symbolic links.
    # CAUTION:  This is dangerous!  For example, if top == '/', it could delete all your disk files.
    """
    if len(path) < 3 :
        print("cannot delete root folder")
        return False

    import os
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            try :
              os.remove(os.path.join(root, name))
            except :
              pass
        for name in dirs:
            try :
              os.rmdir(os.path.join(root, name))
            except: pass
    try :
      os.rmdir(path)
    except: pass
    return True


def os_get_function_name():
    return sys._getframe(1).f_code.co_name


def os_getcwd():
    root = os.path.abspath(os.getcwd()).replace("\\", "/") + "/"
    return  root


def os_system(cmd, doprint=False):
  """ get values
       os_system( f"   ztmp ",  doprint=True)
  """
  import subprocess  
  try :
    p          = subprocess.run( cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, )
    mout, merr = p.stdout.decode('utf-8'), p.stderr.decode('utf-8')    
    if doprint: 
      l = mout  if len(merr) < 1 else mout + "\n\nbash_error:\n" + merr
      print(l)

    return mout, merr
  except Exception as e :
    print( f"Error {cmd}, {e}")

      
def os_makedirs(dir_or_file):
    if os.path.isfile(dir_or_file) or "." in dir_or_file.split("/")[-1] :
        os.makedirs(os.path.dirname(os.path.abspath(dir_or_file)), exist_ok=True)
    else :
        os.makedirs(os.path.abspath(dir_or_file), exist_ok=True)







################################################################################################
class Session(object) :
    """ Save Python session on disk
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
       save_session(self.cur_session, glob)

    def load(self, name, glob:dict=None, tag="") :
      path = f"{self.dir_session}/{name}{tag}/"
      self.cur_session = path
      print(self.cur_session)
      load_session(self.cur_session , glob )


def save_session(folder , globs, tag="" ) : 
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


def load_session(folder, globs=None) :
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




################################################################################################
class dict_to_namespace(object):
    def __init__(self, d):
        self.__dict__ = d











##################################################################################################
def test():
   from utilmy import (os_makedirs, Session, global_verbosity, os_system  
                       
                      )

  
if __name__ == "__main__":
    import fire
    fire.Fire()




