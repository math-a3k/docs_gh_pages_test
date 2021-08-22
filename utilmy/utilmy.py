# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect, json, yaml, gc


def log(*s):
    print(*s, flush=True)

def log2(*s, verbose=1):
    if verbose >0 : print(*s, flush=True)

    
###################################################################################################
###### Pandas #####################################################################################
def pd_merge(df1, df2, on=None, colkeep=None):
  ### Faster merge
  cols = list(df2.columns) if colkeep is None else on + colkeep
  return df1.join( df2[ cols   ].set_index(on), on=on, how='left', rsuffix="2")


def pd_plot_multi(df, plot_type=None, cols_axe1:list=[], cols_axe2:list=[],figsize=(8,4), spacing=0.1, **kwargs):
    from pandas import plotting
    from pandas.plotting import _matplotlib
    from matplotlib import pyplot as plt


    plt.figure(figsize= figsize )
    # Get default color style from pandas - can be changed to any other color list
    if cols_axe1 is None: cols_axe1 = df.columns
    if len(cols_axe1) == 0: return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols_axe1 + cols_axe2))
    
    # Displays subplot's pair in case of plot_type defined as `pair`
    if plot_type=='pair':
        ax = df.plot(subplots=True, figsize=figsize, **kwargs)
        plt.show()
        return
    
    # First axis
    ax = df.loc[:, cols_axe1[0]].plot(label=cols_axe1[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols_axe1[0])
    ##  lines, labels = ax.get_legend_handles_labels()
    lines, labels = [], []

    i1 = len(cols_axe1)
    for n in range(1, len(cols_axe1)):
        df.loc[:, cols_axe1[n]].plot(ax=ax, label=cols_axe1[n], color=colors[(n) % len(colors)], **kwargs)
        line, label = ax.get_legend_handles_labels()
        lines  += line
        labels += label

    for n in range(0, len(cols_axe2)):
        ######### Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        df.loc[:, cols_axe2[n]].plot(ax=ax_new, label=cols_axe2[n], color=colors[(i1 + n) % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols_axe2[n])

        ######### Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0)
    plt.show()
    return ax



def pd_filter(df, filter_dict="shop_id=11, l1_genre_id>600, l2_genre_id<80311," , verbose=False) :
    """
     dfi = pd_filter2(dfa, "shop_id=11, l1_genre_id>600, l2_genre_id<80311," )
     dfi2 = pd_filter(dfa, {"shop_id" : 11} )
     ### Dilter dataframe with basic expr
    """
    #### Dict Filter
    if isinstance(filter_dict, dict) :
       for key,val in filter_dict.items() :
          df =   df[  (df[key] == val) ]
       return df

    # pd_filter(df,  ss="shop_id=11, l1_genre_id>600, l2_genre_id<80311," )
    ss = filter_dict.split(",")
    def x_convert(col, x):
      x_type = str( dict(df.dtypes)[col] )
      if "int" in x_type or "float" in x_type :
         return float(x)
      else :
          return x
    for x in ss :
       x = x.strip()
       if verbose : print(x)
       if len(x) < 3 : continue
       if "=" in x :
           coli= x.split("=")
           df = df[ df[coli[0]] == x_convert(coli[0] , coli[1] )   ]

       if ">" in x :
           coli= x.split(">")
           df = df[ df[coli[0]] > x_convert(coli[0] , coli[1] )   ]

       if "<" in x :
           coli= x.split("<")
           df = df[ df[coli[0]] < x_convert(coli[0] , coli[1] )   ]
    return df


def pd_to_file(df, filei,  check="check", verbose=True,   **kw):
  import os, gc
  from pathlib import Path
  parent = Path(filei).parent
  os.makedirs(parent, exist_ok=True)
  ext  = os.path.splitext(filei)[1]
  if ext == ".pkl" :
      df.to_pickle(filei, **kw)

  if ext == ".parquet" :
      df.to_parquet(filei, **kw)

  if ext == ".csv"  or ext == ".txt" :
      df.to_csv(filei, **kw)

  #if check == "check" :
  #  os_file_check( filei )

  # elif check =="checkfull" :
  #  os_file_check( filei )
  #  dfi = pd_read_file( filei, n_pool=1)   ### Full integrity
  #  log("#######  Reload Check: ",  filei, "\n"  ,  dfi.tail(3).T)
  #  del dfi; gc.collect()
  gc.collect()


def pd_read_file(path_glob="*.pkl", ignore_index=True,  cols=None, verbose=False, nrows=-1, concat_sort=True, n_pool=1, 
                 drop_duplicates=None, col_filter=None,  col_filter_val=None, dtype_reduce=None,  **kw):
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

         ### TODO : use with kewyword arguments
         job_list.append( pool.apply_async(pd_reader_obj, (filei, )))
         if verbose : log(j, filei)

      for i in range(n_pool):
        if i >= len(job_list): break
        dfi   = job_list[ i].get()

        if dtype_reduce is not None: dfi = pd_dtype_reduce(dfi, int0 ='int32', float0 = 'float32')
        if col_filter is not None :  dfi = dfi[ dfi[col_filter] == col_filter_val ]
        if cols is not None :        dfi = dfi[cols]
        if nrows > 0        :        dfi = dfi.iloc[:nrows,:]
        if drop_duplicates is not None  : dfi = dfi.drop_duplicates(drop_duplicates)
        gc.collect()

        dfall = pd.concat( (dfall, dfi), ignore_index=ignore_index, sort= concat_sort)
        #log("Len", n_pool*j + i, len(dfall))
        del dfi; gc.collect()
                
  pool.terminate()
  pool.join()  
  pool = None          
  if m_job>0 and verbose : log(n_file, j * n_file//n_pool )
  return dfall


def pd_sample_strat(df, col, n):
  ### Stratified sampling
  # n   = min(n, df[col].value_counts().min())
  df_ = df.groupby(col).apply(lambda x: x.sample(n = n, replace=True))
  df_.index = df_.index.droplevel(0)
  return df_


def pd_cartesian(df1, df2) :
  ### Cartesian preoduct
  import pandas as pd
  col1 =  list(df1.columns)
  col2 =  list(df2.columns)
  df1['xxx'] = 1
  df2['xxx'] = 1
  df3 = pd.merge(df1, df2,on='xxx')[ col1 + col2 ]
  del df3['xxx']
  return df3


def pd_plot_histogram(dfi, path_save=None, nbin=20.0, q5=0.005, q95=0.995, nsample= -1, show=False, clear=True) :
    ### Plot histogram
    from matplotlib import pyplot as plt
    import numpy as np, os, time
    q0 = dfi.quantile(q5)
    q1 = dfi.quantile(q95)

    if nsample < 0 :
        dfi.hist( bins=np.arange( q0, q1,  (  q1 - q0 ) /nbin  ) )
    else :
        dfi.sample(n=nsample, replace=True ).hist( bins=np.arange( q0, q1,  (  q1 - q0 ) /nbin  ) )
    plt.title( path_save.split("/")[-1] )

    if show :
      plt.show()

    if path_save is not None :
      os.makedirs(os.path.dirname(path_save), exist_ok=True)
      plt.savefig( path_save )
      print(path_save )
    if clear :
        # time.sleep(5)
        plt.close()


def pd_col_bins(df, col, nbins=5):
  ### Shortcuts for easy bin of numerical values
  import pandas as pd, numpy as np
  assert nbins < 256, 'nbins< 255'
  return pd.qcut(df[col], q=nbins,labels= np.arange(0, nbins, 1)).astype('int8')


def pd_dtype_reduce(dfm, int0 ='int32', float0 = 'float32') :
    import numpy as np
    for c in dfm.columns :
        if dfm[c].dtype ==  np.dtype(np.int32) :       dfm[c] = dfm[c].astype( int0 )
        elif   dfm[c].dtype ==  np.dtype(np.int64) :   dfm[c] = dfm[c].astype( int0 )
        elif dfm[c].dtype ==  np.dtype(np.float64) :   dfm[c] = dfm[c].astype( float0 )
    return dfm


def pd_dtype_count_unique(df, col_continuous=[]):
    """Learns the number of categories in each variable and standardizes the data.
        ----------
        data: pd.DataFrame
        continuous_ids: list of ints
            List containing the indices of known continuous variables. Useful for
            discrete data like age, which is better modeled as continuous.
        Returns
        -------
        ncat:  number of categories of each variable. -1 if the variable is  continuous.
    """
    import numpy as np
    def gef_is_continuous(data, dtype):
        """ Returns true if data was sampled from a continuous variables, and false
        """
        if dtype == "Object":
            return False

        observed = data[~np.isnan(data)]  # not consider missing values for this.
        rules = [np.min(observed) < 0,
                 np.sum((observed) != np.round(observed)) > 0,
                 len(np.unique(observed)) > n(30, len(observed)/3)]
        if any(rules):
            return True
        else:
            return False

    cols = list(df.columns)
    ncat = {}

    for coli in cols:
        is_cont = gef_is_continuous( df[coli].sample( n=min(3000, len(df)) ).values , dtype = df[coli].dtype )
        if coli in col_continuous or is_cont:
            ncat[coli] =  -1
        else:
            ncat[coli] =  len( df[coli].unique() )
    return ncat


def pd_dtype_to_category(df, col_exclude, treshold=0.5):
  """
    Convert string to category
  """
  import pandas as pd
  if isinstance(df, pd.DataFrame):
    for col in df.select_dtypes(include=['object']):
        if col not in col_exclude :
            num_unique_values = len(df[col].unique())
            num_total_values  = len(df[col])
            if float(num_unique_values) / num_total_values < treshold:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df
  else:
    print("Not dataframe")


def pd_dtype_getcontinuous(df, cols_exclude:list=[], nsample=-1) :
    ### Return continuous variable
    clist = {}
    for ci in df.columns :
        ctype   = df[ci].dtype
        if nsample == -1 :
            nunique = len(df[ci].unique())
        else :
            nunique = len(df.sample(n= nsample, replace=True)[ci].unique())
        if 'float' in  str(ctype) and ci not in cols_exclude and nunique > 5 :
           clist[ci] = 1
        else :
           clist[ci] = nunique
    return clist


def pd_del(df, cols:list):
    ### Delete columns without errors
    for col in cols :
        try:
            del df[col]
        except : pass
    return df


def pd_add_noise(df, level=0.05, cols_exclude:list=[]) :
    import numpy as np, pandas as pd
    df2 = pd.DataFrame()
    colsnum = pd_dtype_getcontinuous(df, cols_exclude)
    for ci in df.columns :
        if ci in colsnum :
           print(f'adding noise {ci}')
           sigma = level * (df[ci].quantile(0.95) - df[ci].quantile(0.05)  )
           df2[ci] = df[ci] + np.random.normal(0.0, sigma, [len(df)])
        else :
           df2[ci] = df[ci]
    return df2


def pd_cols_unique_count(df, cols_exclude:list=[], nsample=-1) :
    ### Return cadinat=lity
    clist = {}
    for ci in df.columns :
        ctype   = df[ci].dtype
        if nsample == -1 :
            nunique = len(df[ci].unique())
        else :
            nunique = len(df.sample(n= nsample, replace=True)[ci].unique())

        if 'float' in  str(ctype) and ci not in cols_exclude and nunique > 5 :
           clist[ci] = 0
        else :
           clist[ci] = nunique

    return clist


def pd_show(df, nrows=100, reader='notepad.exe', **kw):
    """ Show from Dataframe
    """
    import pandas as pd
    fpath = 'ztmp/ztmp_dataframe.csv'
    os_makedirs(fpath)
    df.iloc[:nrows,:].to_csv(fpath, sep=",", mode='w')





#########################################################################################################
##### Utils numpy, list #################################################################################
def diskcache_save(df, colkey:str, colvalue:str, db_path:str="", size_limit=50000000000, timeout=999, shards:int=1):    
    """ Create dict type on disk, < 100 Gb
       shards>1 : disk spaced is BLOCKED in advance, so high disk usage
       shards is for concurrent writes
    """
    if shards == 1 :
       import diskcache as dc
       cache = dc.Cache(db_path, size_limit= size_limit, timeout= timeout )        
    else :
       from diskcache import FanoutCache
       cache = FanoutCache( db_path, shards= shards, size_limit= size_limit, timeout= timeout )

    v  = df[[ colkey, colvalue  ]].drop_duplicates(colkey)
    v  = v.values
    for i in range(len(v)):
        cache[ v[i,0] ] = v1[i,1]        
    print('Cache size', len(cache), "\n", db_path)    
    return cache


def diskcache_load( db_path="", size_limit=50000000000,timeout=2, force_create=False ):    
    """ Load cache dict from disk and use as dict
       val = cache[mykey]
    
    """
    import diskcache as dc
    if os.path.exists(db_path) and not force_create :
      cache = dc.Cache(db_path )               
    else :   ### create new one
      cache = dc.Cache(db_path, size_limit= size_limit, timeout= timeout )           
    print('Cache size', len(cache)) 
    return cache



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
    from Path import pathlib

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



def os_path_split(fpath:str=""):
    #### Get path split
    fpath = fpath.replace("\\", "/")
    if fpath[-1] == "/":
        fpath = fpath[:-1]

    parent = "/".join(fpath.split("/")[:-1])
    fname  = fpath.split("/")[-1]
    if "." in fname :
        ext = ".".join(fname.split(".")[1:])
    else :
        ext = ""

    return parent, fname, ext



def os_file_replacestring(findstr, replacestr, some_dir, pattern="*.*", dirlevel=1):
    """ #fil_replacestring_files("logo.png", "logonew.png", r"D:/__Alpaca__details/aiportfolio",
        pattern="*.html", dirlevel=5  )
    """
    def os_file_replacestring1(find_str, rep_str, file_path):
        """replaces all find_str by rep_str in file file_path"""
        import fileinput

        file1 = fileinput.FileInput(file_path, inplace=True, backup=".bak")
        for line in file1:
            line = line.replace(find_str, rep_str)
            sys.stdout.write(line)
        file1.close()
        print(("OK: " + format(file_path)))


    list_file = os_walk(some_dir, pattern=pattern, dirlevel=dirlevel)
    list_file = list_file['file']
    for file1 in list_file:
        os_file_replacestring1(findstr, replacestr, file1)


def os_walk(path, pattern="*", dirlevel=50):
    """ dirlevel=0 : root directory
        dirlevel=1 : 1 path below

    """
    import fnmatch, os, numpy as np

    matches = {'file':[], 'dir':[]}
    dir1    = path.replace("\\", "/").rstrip("/")
    num_sep = dir1.count("/")

    for root, dirs, files in os.walk(dir1):
        root = root.replace("\\", "/")
        for fi in files :
            if root.count("/") > num_sep + dirlevel: continue 
            matches['file'].append(os.path.join(root, fi).replace("\\","/"))

        for di in dirs :
            if root.count("/") > num_sep + dirlevel: continue 
            matches['dir'].append(os.path.join(root, di).replace("\\","/") + "/")

    ### Filter files
    matches['file'] = [ t for t in fnmatch.filter(matches['file'], pattern) ] 
    return  matches



def z_os_search_fast(fname, texts=None, mode="regex/str"):
    import re
    if texts is None:
        texts = ["myword"]

    res = []  # url:   line_id, match start, line
    enc = "utf-8"
    fname = os.path.abspath(fname)
    try:
        if mode == "regex":
            texts = [(text, re.compile(text.encode(enc))) for text in texts]
            for lineno, line in enumerate(open(fname, "rb")):
                for text, textc in texts:
                    found = re.search(textc, line)
                    if found is not None:
                        try:
                            line_enc = line.decode(enc)
                        except UnicodeError:
                            line_enc = line
                        res.append((text, fname, lineno + 1, found.start(), line_enc))

        elif mode == "str":
            texts = [(text, text.encode(enc)) for text in texts]
            for lineno, line in enumerate(open(fname, "rb")):
                for text, textc in texts:
                    found = line.find(textc)
                    if found > -1:
                        try:
                            line_enc = line.decode(enc)
                        except UnicodeError:
                            line_enc = line
                        res.append((text, fname, lineno + 1, found, line_enc))

    except IOError as xxx_todo_changeme:
        (_errno, _strerror) = xxx_todo_changeme.args
        print("permission denied errors were encountered")

    except re.error:
        print("invalid regular expression")

    return res



def os_search_content(srch_pattern=None, mode="str", dir1="", file_pattern="*.*", dirlevel=1):
    """  search inside the files

    """
    import pandas as pd
    if srch_pattern is None:
        srch_pattern = ["from ", "import "]

    list_all = os_walk(dir1, pattern=file_pattern, dirlevel=dirlevel)
    ll = []
    for f in list_all["fullpath"]:
        ll = ll + z_os_search_fast(f, texts=srch_pattern, mode=mode)
    df = pd.DataFrame(ll, columns=["search", "filename", "lineno", "pos", "line"])
    return df


def os_get_function_name():
    ### Get ane,
    import sys, socket
    ss = str(os.getpid()) # + "-" + str( socket.gethostname())
    ss = ss + "," + str(__name__)
    try :
        ss = ss + "," + __class__.__name__
    except :
        ss = ss + ","
    ss = ss + "," + str(  sys._getframe(1).f_code.co_name)
    return ss


def os_variable_init(ll, globs):
    for x in ll :
        try :
          globs[x]
        except :
          globs[x] = None


def os_import(mod_name="myfile.config.model", globs=None, verbose=True):
    ### Import in Current Python Session a module   from module import *
    ### from mod_name import *
    module = __import__(mod_name, fromlist=['*'])
    if hasattr(module, '__all__'):
        all_names = module.__all__
    else:
        all_names = [name for name in dir(module) if not name.startswith('_')]

    all_names2 = []
    no_list    = ['os', 'sys' ]
    for t in all_names :
        if t not in no_list :
          ### Mot yet loaded in memory  , so cannot use Global
          #x = str( globs[t] )
          #if '<class' not in x and '<function' not in x and  '<module' not in x :
          all_names2.append(t)
    all_names = all_names2

    if verbose :
      print("Importing: ")
      for name in all_names :
         print( f"{name}=None", end=";")
      print("")
    globs.update({name: getattr(module, name) for name in all_names})


def os_variable_exist(x ,globs, msg="") :
    x_str = str(globs.get(x, None))
    if "None" in x_str:
        log("Using default", x)
        return False
    else :
        log("Using ", x)
        return True


def os_variable_check(ll, globs=None, do_terminate=True):
  import sys
  for x in ll :
      try :
         a = globs[x]
         if a is None : raise Exception("")
      except :
          log("####### Vars Check,  Require: ", x  , "Terminating")
          if do_terminate:
                 sys.exit(0)


def os_clean_memory( varlist , globx):
  for x in varlist :
    try :
       del globx[x]
       gc.collect()
    except : pass


def os_system_list(ll, logfile=None, sleep_sec=10):
   ### Execute a sequence of cmd
   import time, sys
   n = len(ll)
   for ii,x in enumerate(ll):
        try :
          log(x)
          if sys.platform == 'win32' :
             cmd = f" {x}   "
          else :
             cmd = f" {x}   2>&1 | tee -a  {logfile} " if logfile is not None else  x

          os.system(cmd)

          # tx= sum( [  ll[j][0] for j in range(ii,n)  ]  )
          # log(ii, n, x,  "remaining time", tx / 3600.0 )
          #log('Sleeping  ', x[0])
          time.sleep(sleep_sec)
        except Exception as e:
            log(e)


def os_file_check(fp):
   import os, time
   try :
       log(fp,  os.stat(fp).st_size*0.001, time.ctime(os.path.getmtime(fp)) )
   except :
       log(fp, "Error File Not exist")


def os_to_file( txt="", filename="ztmp.txt",  mode='a'):
    with open(filename, mode=mode) as fp:
        fp.write(txt + "\n")


def os_platform_os():
    #### get linux or windows
    return sys.platform


def os_cpu():
    ### Nb of cpus cores
    return os.cpu_count()


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


def os_sleep_cpu(priority=300, cpu_min=50, sleep=10):
    #### Sleep until CPU becomes normal usage
    import psutil, time

    aux = psutil.cpu_percent()
    while aux > cpu_min:
        #print("CPU:", aux, time.time())
        time.sleep(priority)
        aux = psutil.cpu_percent()
        time.sleep(sleep)
        aux = 0.5 * (aux + psutil.cpu_percent())
    return aux


def os_ram_object(o, ids, hint=" deep_getsizeof(df_pd, set()) "):
    """ deep_getsizeof(df_pd, set())
    Find the memory footprint of a Python object
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    """
    from collections import Mapping, Container
    from sys import getsizeof

    _ = hint

    d = os_ram_object
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, str):
        r = r

    if isinstance(o, Mapping):
        r = r + sum(d(k, ids) + d(v, ids) for k, v in o.items())

    if isinstance(o, Container):
        r = r + sum(d(x, ids) for x in o)

    return r * 0.0000001



def os_copy(src, dst, overwrite=False, exclude=""):
    import shutil
    def ignore_pyc_files(dirname, filenames):
        return [name for name in filenames if name.endswith('.pyc')]


    patterns = exclude.split(";")
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, ignore = shutil.ignore_patterns(*patterns))



def os_removedirs(path):
    """  issues with no empty Folder
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


def os_getcwd():
    ## This is for Windows Path normalized As Linux /
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










######################################################################################################
######## External IO #################################################################################
def hdfs_put(from_dir="", to_dir="",  verbose=True, n_pool=25, dirlevel=50,  **kw):
    """ 
     hdfs_put LocalFile into HDFS in multi-thread
    from_dir = "hdfs://nameservice1/user/
    to_dir   = "data/"
    
    """
    import glob, gc,os, time, pyarrow as pa
    from multiprocessing.pool import ThreadPool

    def log(*s, **kw):
      print(*s, flush=True)
    
    #### File ############################################   
    hdfs      = pa.hdfs.connect()  
    hdfs.mkdir(to_dir  )  

    from utilmy import os_walk    
    dd = os_walk(from_dir, dirlevel= dirlevel, pattern="*") 
    fdirs, file_list = dd['dir'], dd['file']
    file_list = sorted(list(set(file_list)))
    n_file    = len(file_list)
    log('Files', n_file)

    file_list2 = [] 
    for i, filei in enumerate(file_list) :
        file_list2.append( (filei,   to_dir + filei.replace(from_dir,"")   )  )


    ##### Create Target dirs  ###########################
    fdirs = [ t.replace(from_dir,"") for t in fdirs]
    for di in fdirs :
        hdfs.mkdir(to_dir + "/" + di )   

    #### Input xi #######################################    
    xi_list = [ []  for t in range(n_pool) ]     
    for i, xi in enumerate(file_list2) :
        jj = i % n_pool 
        xi_list[jj].append( xi )
        
    #### function #######################################
    def fun_async(xlist):
      for x in xlist :   
         try :
           with open(x[0], mode='rb') as f:
                hdfs.upload(x[1], f,)                        
         except :
            try :
               time.sleep(60)
               with open(x[0], mode='rb') as f:
                  hdfs.upload(x[1], f,)  
            except : print('error', x[1])
                  
    #### Pool execute ###################################
    pool     = ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
         job_list.append( pool.apply_async(fun_async, (xi_list[i], )))
         if verbose : log(i, xi_list[i] )

    res_list = []            
    for i in range(n_pool):
        if i >= len(job_list): break
        res_list.append( job_list[ i].get() )
        log(i, 'job finished')


    pool.terminate() ; pool.join()  ;  pool = None          
    log('n_processed', len(res_list) )
    



def hdfs_walk(path="hdfs://nameservice1/user/", dirlevel=3, hdfs=None):   ### python  prepro.py hdfs_walk 
    import pyarrow as pa
    hdfs = pa.hdfs.connect() if hdfs is None else hdfs
    path = "hdfs://nameservice1/" + path if 'hdfs://' not in path else path   
         
    def os_walk(fdirs):
        flist3 = []
        for diri  in fdirs :
            flist3.extend( [ t for t in hdfs.ls(diri) ]  )
        fdirs3 = [ t   for t in flist3 if hdfs.isdir(t) ]  
        return flist3, fdirs3
    
    flist0, fdirs0   = os_walk([path])
    fdirs = fdirs0
    for i in range(dirlevel):
       flisti, fdiri = os_walk(fdirs)     
       flist0 =  list(set(flist0  + flisti ))
       fdirs0 =  list(set(fdirs0  + fdiri ))
       fdirs  = fdiri    
    return {'file': flist0, 'dir': fdirs0}
    
    
    
def hdfs_get(from_dir="", to_dir="",  verbose=True, n_pool=20,   **kw):
    """ 
    import fastcounter
    counter = fastcounter.FastWriteCounter,()
    counter.increment(1)    
    cnt.value
    """
    import glob, gc,os, time
    from multiprocessing.pool import ThreadPool

    def log(*s, **kw):
      print(*s, flush=True, **kw)
    
    #### File ############################################
    os.makedirs(to_dir, exist_ok=True)    
    import pyarrow as pa
    hdfs      = pa.hdfs.connect()  
    # file_list = [ t for t in hdfs.ls(from_dir) ] 
    file_list = hdfs_walk(from_dir, dirlevel=10)['file']
    

    def fun_async(xlist):
      for x in xlist :   
         try :
            hdfs.download(x[0], x[1])
            # ktot = ktot + 1   ### Not thread safe
         except :
            try :
               time.sleep(60)
               hdfs.download(x[0], x[1])
               # ktot = ktot + 1
            except : pass
            
    ######################################################    
    file_list = sorted(list(set(file_list)))
    n_file    = len(file_list)
    log('Files', n_file)
    if verbose: log(file_list)

    xi_list = [ []  for t in range(n_pool) ]     
    for i, filei in enumerate(file_list) :
        jj = i % n_pool 
        xi_list[jj].append( (filei,   to_dir + "/" + filei.split("/")[-1]   )  )
        
    #### Pool count
    pool     = ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
         job_list.append( pool.apply_async(fun_async, (xi_list[i], )))
         if verbose : log(i, xi_list[i] )

    res_list = []            
    for i in range(n_pool):
        if i >= len(job_list): break
        res_list.append( job_list[ i].get() )
        log(i, 'job finished')

    pool.terminate()  ;  pool.join()  ; pool = None          
    log('n_processed', len(res_list) )
    log('n files', len(os.listdir(to_dir)) )


    
    
    





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
def plot_to_html(dir_input="*.png", out_file="graph.html", title="", verbose=False):
    """
      plot_to_html( model_path + "/graph_shop_17_past/*.png" , model_path + "shop_17.html" )

    """
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    import glob
    html = f'<html><body><h2>{title}</h2>'
    flist = glob.glob(dir_input)
    flist.sorted()
    for fp in flist :
        if verbose : print(fp,end=",")
        with open(fp, mode="rb" ) as fp2 :
            tmpfile =fp2.read()
        encoded = base64.b64encode( tmpfile ) .decode('utf-8')
        html =  html + f'<p><img src=\'data:image/png;base64,{encoded}\'> </p>\n'
    html = html + "</body></html>"
    with open(out_file,'w') as f:
        f.write(html)







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
def print_everywhere():
    """
    https://github.com/alexmojaki/snoop
    """
    txt ="""
    import snoop; snoop.install()  ### can be used anywhere
    
    @snoop
    def myfun():
    
    from snoop import pp
    pp(myvariable)
    
    
    
    """
    import snoop
    snoop.install()  ### can be used anywhere"
    print("Decaorator @snoop ")
    
    
def log10(*s, nmax=60):
    """ Display variable name, type when showing,  pip install varname
    
    """
    from varname import varname, nameof
    for x in s :
        print(nameof(x, frame=2), ":", type(x), "\n",  str(x)[:nmax], "\n")
        
    
def log5(*s):
    """    ### Equivalent of print, but more :  https://github.com/gruns/icecream
    pip install icrecream
    ic()  --->  ic| example.py:4 in foo()
    ic(var)  -->   ic| d['key'][1]: 'one'
    
    """
    from icecream import ic
    return ic(*s)
    
    
def log_trace(msg="", dump_path="", globs=None):
    print(msg)
    import pdb;
    pdb.set_trace()


def profiler_start():
    ### Code profiling
    from pyinstrument import Profiler
    global profiler
    profiler = Profiler()
    profiler.start()


def profiler_stop():
    global profiler
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))




###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()




