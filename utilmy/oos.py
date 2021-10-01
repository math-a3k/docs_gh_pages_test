# -*- coding: utf-8 -*-
HELP="""
https://github.com/uqfoundation/pox/tree/master/pox


"""
import os, sys, time, datetime,inspect, json, yaml, gc

#################################################################
def log(*s):
    print(*s, flush=True)

def log2(*s, verbose=1):
    if verbose >0 : print(*s, flush=True)

def help():
    ss = HELP
    print(ss)

    
        
################################################################3#        
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
def os_path_size(path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size



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


def os_copy_safe(dirin=None, dirout=None, nlevel=10, nfile=100000, cmd_fallback=""):  
    """ Copy Safely/slowly between drive    
    
    """
    import shutil, time, os, glob       
    flist = [] ; dirinj = dirin        
    for j in range(nlevel): 
        dirinj = dirinj + "/*"
        tmp = glob.glob(dirinj )
        if len(tmp) < 1 : break
        flist  = flist + tmp        
        
    flist = flist[:nfile]            
    log('n files', len(flist))
    kk = 1 ; ntry = 0
    for i in range(0, len(flist)) :
        fi  = flist[i]
        fi2 = fi.replace(dirin, dirout)
        if not os.path.isfile(fi2) and os.path.isfile(fi) :
             kk = kk + 1
             if kk > nfile   : return 1   
             if kk % 50 == 0 : time.sleep(0.5)             
             if kk % 10 :      log(i, fi2)
             os.makedirs(os.path.dirname(fi2), exist_ok=True)
             try :
                shutil.copy(fi, fi2)
                ntry = 0
             except Exception as e:
                log(e)
                time.sleep(10)
                log(cmd_fallback)
                os.system(cmd_fallback)
                time.sleep(10)
                i    = i - 1
                ntry = ntry + 1

                
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


def os_sleep_cpu(cpu_min=30, sleep=10, interval=5, msg= "", verbose=True):
    #### Sleep until CPU becomes normal usage
    import psutil, time
    aux = psutil.cpu_percent(interval=interval)  ### Need to call 2 times
    while aux > cpu_min:
        ui = psutil.cpu_percent(interval=interval)
        aux = 0.5 * (aux +  ui)
        if verbose : log( 'Sleep sec', sleep, ' Usage %', aux, ui, msg )
        time.sleep(sleep)        
    return aux


def os_sizeof(o, ids, hint=" deep_getsizeof(df_pd, set()) "):
    """ deep_getsizeof(df_pd, set())
    Find the memory footprint of a Python object
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    """
    from collections import Mapping, Container
    from sys import getsizeof

    _ = hint

    d = os_sizeof
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




