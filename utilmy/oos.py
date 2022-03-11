# -*- coding: utf-8 -*-
MNAME= "utilmy.dates"
HELP="""
https://github.com/uqfoundation/pox/tree/master/pox


"""
import os, sys, time, datetime,inspect, json, yaml, gc, pandas as pd, numpy as np


#################################################################
from utilmy.utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    ss = help_create(MNAME) + HELP
    print(ss)


#################################################################
def test_all():
    """
    #### python test.py   test_oos
    """
    return 1  
    log("Testing oos.py............................")
    from utilmy import oos as m
    from utilmy import pd_random


    from utilmy.oos import os_makedirs, os_system, os_removedirs
    os_makedirs('ztmp/ztmp2/myfile.txt')
    os_makedirs('ztmp/ztmp3/ztmp4')
    os_makedirs('/tmp/one/two')
    os_makedirs('/tmp/myfile')
    os_makedirs('/tmp/one/../mydir/')
    os_makedirs('./tmp/test')
    os.system("ls ztmp")

    path = ["/tmp/", "ztmp/ztmp3/ztmp4", "/tmp/", "./tmp/test","/tmp/one/../mydir/"]
    for p in path:
       f = os.path.exists(os.path.abspath(p))
       assert  f == True, "path " + p

    rev_stat = os_removedirs("ztmp/ztmp2")
    assert not rev_stat == False, "cannot delete root folder"

    ############################################################
    res = os_system( f" ls . ",  doprint=True)
    log(res)
    res = os_system( f" ls . ",  doprint=False)

    from utilmy import os_platform_os
    assert os_platform_os() == sys.platform


    ############################################################
    def test_log():
        from utilmy.oos import log, log2, log5
        log("Testing logs ...")
        log2("log2")
        log5("log5")
    
    
    def int_float_test():
        log("Testing int/float ..")
        from utilmy.oos import is_float,to_float,is_int,to_int
        int_ = 1
        float_ = 1.1
        is_int(int_)
        is_float(float_)
        to_float(int_)
        to_int(float_)
    
    def os_path_size_test():
        log("Testing os_path_size() ..")
        from utilmy.oos import os_path_size
        size_ = os_path_size()
        log("total size", size_)
    
    def os_path_split_test():
        log("Testing os_path_split() ..")
        from utilmy.oos import os_path_split
        result_ = os_path_split("test/tmp/test.txt")
        log("result", result_)
    
    def os_file_replacestring_test():
        log("Testing os_file_replacestring() ..")
        
    def os_walk_test():
        log("Testing os_walk() ..")
        from utilmy.oos import os_walk
        import os
        cwd = os.getcwd()
        # log(os_walk(cwd))
    
    def os_copy_safe_test():
        log("Testing os_copy_safe() ..")
        from utilmy.oos import os_copy_safe
        os_copy_safe("./testdata/tmp/test", "./testdata/tmp/test_copy/")
    
    def z_os_search_fast_test():
        log("Testing z_os_search_fast() ..")
        from utilmy.oos import z_os_search_fast
        with open("./testdata/tmp/test/os_search_test.txt", 'a') as file:
            file.write("Dummy text to test fast search string")
        res = z_os_search_fast("./testdata/tmp/test/os_search_test.txt", ["Dummy"],mode="regex")
        print(res)
    
    def os_search_content_test():
        log("Testing os_search_content() ..")
        from utilmy.oos import os_search_content
        with open("./testdata/tmp/test/os_search_content_test.txt", 'a') as file:
            file.write("Dummy text to test fast search string")
        import os
        cwd = os.getcwd()
        '''TODO: for f in list_all["fullpath"]:
            KeyError: 'fullpath'
        res = os_search_content(srch_pattern= "Dummy text",dir1=os.path.join(cwd ,"tmp/test/"))
        log(res)
        '''
    
    def os_get_function_name_test():
        log("Testing os_get_function_name() ..")
        from utilmy.oos import os_get_function_name
        log(os_get_function_name())
    
    def os_variables_test():
        log("Testing os_variables_test ..")
        from utilmy.oos import os_variable_init, os_variable_check, os_variable_exist, os_import, os_clean_memory
        ll = ["test_var"]
        globs = {}
        os_variable_init(ll,globs)
        os_variable_exist("test_var",globs)
        os_variable_check("other_var",globs,do_terminate=False)
        os_import(mod_name="pandas", globs=globs)
        os_clean_memory(["test_var"], globs)
        log(os_variable_exist("test_var",globs))

    def os_system_list_test():
        log("Testing os_system_list() ..")
        from utilmy.oos import os_system_list
        cmd = ["pwd","whoami"]
        os_system_list(cmd, sleep_sec=0)
    
    def os_file_check_test():
        log("Testing os_file_check()")
        from utilmy.oos import os_to_file, os_file_check
        os_to_file(txt="test text to write to file",filename="./testdata/tmp/test/file_test.txt", mode="a")
        os_file_check("./testdata/tmp/test/file_test.txt")
    
    def os_utils_test():
        log("Testing os utils...")
        from utilmy.oos import os_platform_os, os_cpu, os_memory,os_getcwd, os_sleep_cpu,os_copy,\
             os_removedirs,os_sizeof, os_makedirs
        log(os_platform_os())
        log(os_cpu())
        log(os_memory())
        log(os_getcwd())
        os_sleep_cpu(cpu_min=30, sleep=1, interval=5, verbose=True)
        os_makedirs("./testdata/tmp/test")
        with open("./testdata/tmp/test/os_utils_test.txt", 'w') as file:
            file.write("Dummy file to test os utils")
            
        os_makedirs("./testdata/tmp/test/os_test")
        from utilmy.oos import os_file_replacestring
        with open("./testdata/tmp/test/os_test/os_file_test.txt", 'a') as file:
            file.write("Dummy text to test replace string")

        os_file_replacestring("text", "text_replace", "./testdata/tmp/test/os_test/")

        #os_copy(os.path.join(os_getcwd(), "tmp/test"), os.path.join(os_getcwd(), "tmp/test/os_test"))
        os_removedirs("./testdata/tmp/test/os_test")
        pd_df = pd_random()
        log(os_sizeof(pd_df, set()))

    def os_system_test():
        log("Testing os_system()...")
        from utilmy.oos import os_system
        os_system("whoami", doprint=True)


    test_log()
    int_float_test()
    #os_path_size_test()
    #os_path_split_test()
    #os_file_replacestring_test()
    # os_walk_test()
    os_copy_safe_test()
    #z_os_search_fast_test()
    #os_search_content_test()
    #os_get_function_name_test()
    #os_variables_test()
    #os_system_list_test()
    #os_file_check_test()
    #os_utils_test()
    #os_system_test()


    
def test0(): 
    """function test0
    Args:
    Returns:
        
    """
    os_makedirs('ztmp/ztmp2/myfile.txt')
    os_makedirs('ztmp/ztmp3/ztmp4')
    os_makedirs('/tmp/one/two')
    os_makedirs('/tmp/myfile')
    os_makedirs('/tmp/one/../mydir/')
    os_makedirs('./tmp/test')
    os.system("ls ztmp")

    path = ["/tmp/", "ztmp/ztmp3/ztmp4", "/tmp/", "./tmp/test","/tmp/one/../mydir/"]
    for p in path:
       f = os.path.exists(os.path.abspath(p))
       assert  f == True, "path " + p

    rev_stat = os_removedirs("ztmp/ztmp2")
    assert not rev_stat == False, "cannot delete root folder"

    res = os_system( f" ls . ",  doprint=True)
    log(res)
    res = os_system( f" ls . ",  doprint=False)
    assert os_platform_os() == sys.platform

def test1():
    """function test1
    Args:
    Returns:
        
    """
    int_ = 1
    float_ = 1.1
    is_int(int_)
    is_float(float_)
    to_float(int_)
    to_int(float_)

def test2():
    """function test2
    Args:
    Returns:
        
    """
    size_ = os_path_size()
    log("total size", size_)
    result_ = os_path_split("test/tmp/test.txt")
    log("result", result_)
    with open("./testdata/tmp/test/os_file_test.txt", 'a') as file:
        file.write("Dummy text to test replace string")

    os_file_replacestring("text", "text_replace", "./testdata/tmp/test/")
    os_copy_safe("./testdata/tmp/test", "./testdata/tmp/test_copy/")

    with open("./testdata/tmp/test/os_search_test.txt", 'a') as file:
        file.write("Dummy text to test fast search string")
    res = z_os_search_fast("./testdata/tmp/test/os_search_test.txt", ["Dummy"],mode="regex")

    with open("./testdata/tmp/test/os_search_content_test.txt", 'a') as file:
        file.write("Dummy text to test fast search string")
    cwd = os.getcwd()
    '''TODO: for f in list_all["fullpath"]:
        KeyError: 'fullpath'
    res = os_search_content(srch_pattern= "Dummy text",dir1=os.path.join(cwd ,"tmp/test/"))
    log(res)
    '''

def test4():
    """function test4
    Args:
    Returns:
        
    """
    log(os_get_function_name())
    cwd = os.getcwd()
    log(os_walk(cwd))
    cmd = ["pwd","whoami"]
    os_system_list(cmd, sleep_sec=0)
    ll = ["test_var"]
    globs = {}
    os_variable_init(ll,globs)
    os_variable_exist("test_var",globs)
    os_variable_check("other_var",globs,do_terminate=False)
    os_import(mod_name="pandas", globs=globs)
    os_clean_memory(["test_var"], globs)
    log(os_variable_exist("test_var",globs))
    
    os_to_file(txt="test text to write to file",filename="./testdata/tmp/test/file_test.txt", mode="a")
    os_file_check("./testdata/tmp/test/file_test.txt")

def test5():
    """function test5
    Args:
    Returns:
        
    """
    log("Testing os utils...")
    from utilmy import pd_random
    log(os_platform_os())
    log(os_cpu())
    log(os_memory())
    log(os_getcwd())
    os_sleep_cpu(cpu_min=30, sleep=1, interval=5, verbose=True)
    pd_df = pd_random()
    log(os_sizeof(pd_df, set()))


    
    
    

########################################################################################################
########################################################################################################
class dict_to_namespace(object):
    #### Dict to namespace
    def __init__(self, d):
        """ dict_to_namespace:__init__
        Args:
            d:     
        Returns:
           
        """
        self.__dict__ = d


def to_dict(**kw):
  """function to_dict
  Args:
      **kw:   
  Returns:
      
  """
  ## return dict version of the params
  return kw


def to_timeunix(datex="2018-01-16"):
  """function to_timeunix
  Args:
      datex:   
  Returns:
      
  """
  if isinstance(datex, str)  :
     return int(time.mktime(datetime.datetime.strptime(datex, "%Y-%m-%d").timetuple()) * 1000)

  if isinstance(datex, datetime)  :
     return int(time.mktime( datex.timetuple()) * 1000)


def to_datetime(x) :
  """function to_datetime
  Args:
      x:   
  Returns:
      
  """
  import pandas as pd
  return pd.to_datetime( str(x) )


def np_list_intersection(l1, l2) :
  """function np_list_intersection
  Args:
      l1:   
      l2:   
  Returns:
      
  """
  return [x for x in l1 if x in l2]


def np_add_remove(set_, to_remove, to_add):
    """function np_add_remove
    Args:
        set_:   
        to_remove:   
        to_add:   
    Returns:
        
    """
    # a function that removes list of elements and adds an element from a set
    result_temp = set_.copy()
    for element in to_remove:
        result_temp.remove(element)
    result_temp.add(to_add)
    return result_temp


def to_float(x):
    """function to_float
    Args:
        x:   
    Returns:
        
    """
    try :
        return float(x)
    except :
        return float("NaN")


def to_int(x):
    """function to_int
    Args:
        x:   
    Returns:
        
    """
    try :
        return int(x)
    except :
        return float("NaN")


def is_int(x):
    """function is_int
    Args:
        x:   
    Returns:
        
    """
    try :
        int(x)
        return True
    except :
        return False    

def is_float(x):
    """function is_float
    Args:
        x:   
    Returns:
        
    """
    try :
        float(x)
        return True
    except :
        return False   


########################################################################################################
##### OS, cofnfig ######################################################################################
def os_monkeypatch_help():
    """function os_monkeypatch_help
    Args:
    Returns:
        
    """
    print( """
    https://medium.com/@chipiga86/python-monkey-patching-like-a-boss-87d7ddb8098e
    
    
    """)
    
    
def os_module_uncache(exclude='os.system'):
    """Remove package modules from cache except excluded ones.
       On next import they will be reloaded.  Useful for monkey patching
    Args:
        exclude (iter<str>): Sequence of module paths.
        https://medium.com/@chipiga86/python-monkey-patching-like-a-boss-87d7ddb8098e
    """
    import sys
    pkgs = []
    for mod in exclude:
        pkg = mod.split('.', 1)[0]
        pkgs.append(pkg)

    to_uncache = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + '.'):
                to_uncache.append(mod)
                break

    for mod in to_uncache:
        del sys.modules[mod]

        
        
def os_file_date_modified(dirin, fmt="%Y%m%d-%H:%M", timezone='Asia/Tokyo'):
    """last modified date
    """
    import datetime 
    from pytz import timezone as tzone
    try :
      mtime  = os.path.getmtime(dirin)
      mtime2 = datetime.datetime.utcfromtimestamp(mtime)
      mtime2 = mdate2.astimezone(tzone(timezone))
      return mtime2.strftime(fmt)
    except:
      return ""  


def date_to_timezone(tdate,  fmt="%Y%m%d-%H:%M", timezone='Asia/Tokyo'):
    """function date_to_timezone
    Args:
        tdate:   
        fmt="%Y%m%d-%H:   
        timezone:   
    Returns:
        
    """
    # "%Y-%m-%d %H:%M:%S %Z%z"
    from pytz import timezone as tzone
    import datetime
    # Convert to US/Pacific time zone
    now_pacific = tdate.astimezone(tzone('Asia/Tokyo'))
    return now_pacific.strftime(fmt)



def os_process_list():
     """  List of processes
     #ll = os_process_list()
     #ll = [t for t in ll if 'root' in t and 'python ' in t ]
     ### root   ....  python run          
     """
     import subprocess
     ps = subprocess.Popen('ps -ef', shell=True, stdout=subprocess.PIPE)
     ll = ps.stdout.readlines()
     ll = [ t.decode().replace("\n", "") for t in ll ]
     return ll


def os_wait_processes(nhours=7):        
    """function os_wait_processes
    Args:
        nhours:   
    Returns:
        
    """
    t0 = time.time()
    while (time.time() - t0 ) < nhours * 3600 :
       hdfs_export()            
       ll = os_process_list()
       ll = [t for t in ll if 'scoupon' in t and 'python ' in t ]
       if len(ll) < 2 : break   ### Process are not running anymore 
       log("sleep 30min", ll)     
       time.sleep(3600* 0.5)

    
    
class toFileSafe(object):
   def __init__(self,fpath):
      """ Thread Safe file writer
        tofile = toFileSafe('mylog.log)
        tofile.w("msg")
      """
      logger = logging.getLogger('logsafe')
      logger.setLevel(logging.INFO)
      ch = logging.FileHandler(fpath)
      ch.setFormatter(logging.Formatter('%(message)s'))
      logger.addHandler(ch)     
      self.logger = logger
      
   def write(self, msg):   
        """ toFileSafe:write
        Args:
            msg:     
        Returns:
           
        """
        self.logger.info( msg)
    
   def log(self, msg):   
        """ toFileSafe:log
        Args:
            msg:     
        Returns:
           
        """
        self.logger.info( msg)    

   def w(self, msg):   
        """ toFileSafe:w
        Args:
            msg:     
        Returns:
           
        """
        self.logger.info( msg)   

        
        
def os_path_size(path = '.'):
    """function os_path_size
    Args:
        path :   
    Returns:
        
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def os_path_split(fpath:str=""):
    """function os_path_split
    Args:
        fpath ( str ) :   
    Returns:
        
    """
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


def os_copy_safe(dirin:str=None, dirout:str=None,  nlevel=5, nfile=5000, logdir="./", pattern="*", exclude="", force=False, sleep=0.5, cmd_fallback="",
                 verbose=True):  ### 
    """ Copy safe
    """
    import shutil, time, os, glob

    flist = [] ; dirinj = dirin
    for j in range(nlevel):
        ztmp   = sorted( glob.glob(dirinj + "/" + pattern ) )
        dirinj = dirinj + "/*/"             
        if len(ztmp) < 1 : break
        flist  = flist + ztmp

    flist2 = []    
    for x in exclude.split(","):
        if len(x) <=1 : continue
        for t in flist :
            if  not x in t :
                flist2.append(t)
    flist = flist2

    log('n files', len(flist), dirinj, dirout ) ; time.sleep(sleep)
    kk = 0 ; ntry = 0 ;i =0
    for i in range(0, len(flist)) :
        fi  = flist[i]
        fi2 = fi.replace(dirin, dirout)

        if not fi.isascii(): continue
        if not os.path.isfile(fi) : continue

        if (not os.path.isfile(fi2) )  or force :
             kk = kk + 1
             if kk > nfile   : return 1
             if kk % 50 == 0  and sleep >0 : time.sleep(sleep)
             if kk % 10 == 0  and verbose  : log(fi2)
             os.makedirs(os.path.dirname(fi2), exist_ok=True)
             try :
                shutil.copy(fi, fi2)
                ntry = 0
                if verbose: log(fi2)
             except Exception as e:
                log(e)
                time.sleep(10)
                log(cmd_fallback)
                os.system(cmd_fallback)
                time.sleep(10)
                i = i - 1
                ntry = ntry + 1
    log('Scanned', i, 'transfered', kk)

### Alias       
os_copy = os_copy_safe


def os_merge_safe(dirin_list=None, dirout=None, nlevel=5, nfile=5000, nrows=10**8,  cmd_fallback = "umount /mydrive/  && mount /mydrive/  ", sleep=0.3):
    """function os_merge_safe
    Args:
        dirin_list:   
        dirout:   
        nlevel:   
        nfile:   
        nrows:   
        cmd_fallback :   
        sleep:   
    Returns:
        
    """
    ### merge file in safe way
    nrows = 10**8
    flist = []
    for fi in dirin_list :
        flist = flist + glob.glob(fi)
    log(flist); time.sleep(2)    

    os_makedirs(dirout)            
    fout = open(dirout,'a')
    for fi in flist :    
        log(fi)             
        ii   = 0
        fin  = open(fi,'r')
        while True:
            try :
              ii = ii + 1
              if ii % 100000 == 0 : time.sleep(sleep)
              if ii > nrows : break      
              x = fin.readline()
              if not x: break        
              fout.write(x.strip()+"\n")
            except Exception as e:
              log(e)
              os.system(cmd_fallback)
              time.sleep(10)
              fout.write(x.strip()+"\n") 
        fin.close()    






    
    
def z_os_search_fast(fname, texts=None, mode="regex/str"):
    """function z_os_search_fast
    Args:
        fname:   
        texts:   
        mode:   
    Returns:
        
    """
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
    """function os_get_function_name
    Args:
    Returns:
        
    """
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
    """function os_variable_init
    Args:
        ll:   
        globs:   
    Returns:
        
    """
    for x in ll :
        try :
          globs[x]
        except :
          globs[x] = None


def os_import(mod_name="myfile.config.model", globs=None, verbose=True):
    """function os_import
    Args:
        mod_name:   
        globs:   
        verbose:   
    Returns:
        
    """
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
    """function os_variable_exist
    Args:
        x:   
        globs:   
        msg:   
    Returns:
        
    """
    x_str = str(globs.get(x, None))
    if "None" in x_str:
        log("Using default", x)
        return False
    else :
        log("Using ", x)
        return True


def os_variable_check(ll, globs=None, do_terminate=True):
  """function os_variable_check
  Args:
      ll:   
      globs:   
      do_terminate:   
  Returns:
      
  """
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
  """function os_clean_memory
  Args:
      varlist:   
      globx:   
  Returns:
      
  """
  for x in varlist :
    try :
       del globx[x]
       gc.collect()
    except : pass


def os_system_list(ll, logfile=None, sleep_sec=10):
   """function os_system_list
   Args:
       ll:   
       logfile:   
       sleep_sec:   
   Returns:
       
   """
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
   """function os_file_check
   Args:
       fp:   
   Returns:
       
   """
   import os, time
   try :
       log(fp,  os.stat(fp).st_size*0.001, time.ctime(os.path.getmtime(fp)) )
   except :
       log(fp, "Error File Not exist")


def os_to_file( txt="", filename="ztmp.txt",  mode='a'):
    """function os_to_file
    Args:
        txt:   
        filename:   
        mode:   
    Returns:
        
    """
    with open(filename, mode=mode) as fp:
        fp.write(txt + "\n")


def os_platform_os():
    """function os_platform_os
    Args:
    Returns:
        
    """
    #### get linux or windows
    return sys.platform


def os_cpu():
    """function os_cpu
    Args:
    Returns:
        
    """
    ### Nb of cpus cores
    return os.cpu_count()


def os_platform_ip():
    """function os_platform_ip
    Args:
    Returns:
        
    """
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
    """function os_sleep_cpu
    Args:
        cpu_min:   
        sleep:   
        interval:   
        msg:   
        verbose:   
    Returns:
        
    """
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






def os_removedirs(path, verbose=False):
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
              if verbose: log(name)
            except Exception as e :
              log('error', name, e)
            
        for name in dirs:
            try :
              os.rmdir(os.path.join(root, name))
              if verbose: log(name)
            except  Exception as e:
              log('error', name, e)
            
    try :
      os.rmdir(path)
    except: pass
    return True


def os_getcwd():
    """function os_getcwd
    Args:
    Returns:
        
    """
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
    """function os_makedirs
    Args:
        dir_or_file:   
    Returns:
        
    """
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
    """function log_trace
    Args:
        msg:   
        dump_path:   
        globs:   
    Returns:
        
    """
    print(msg)
    import pdb;
    pdb.set_trace()


def profiler_start():
    """function profiler_start
    Args:
    Returns:
        
    """
    ### Code profiling
    from pyinstrument import Profiler
    global profiler
    profiler = Profiler()
    profiler.start()


def profiler_stop():
    """function profiler_stop
    Args:
    Returns:
        
    """
    global profiler
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))




###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()




