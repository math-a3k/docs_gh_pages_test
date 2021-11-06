HELP="""All related to distributed compute and atomic read/write
   Thread Safe
   Process Safe
   Lock Mechanism
   
"""
import os, sys, socket, platform, time, gc,logging, random

###############################################################################################
from utilmy.utilmy import log, log2

def help():
    from utilmy import help_create
    ss  = help_create("utilmy.distributed", prefixs= [ 'test'])  #### Merge test code
    ss += HELP
    print(ss)


def log_mem(*s):
    try:
        # print(*s, "\n", flush=True)
        import psutil
        log2('mem check', str(psutil.virtual_memory()))
        # print(s)
    except:
        pass



################################################################################################
# Test functions
def test_functions():
    """Check that list function is working.
    os_lock_releaseLock, os_lock_releaseLock, os_lock_execute
    Basic test on only 1 thread
    """
    # test function
    def running(fun_args):
        print(f'Function running with arg: {fun_args}')

    # test that os_lock_execute is working
    os_lock_execute(running, 'Test_args', plock='tmp/plock.lock')
    os_lock_execute(running, [1, 2, 3], plock='tmp/plock.lock')


def test_funtions_thread():
    """Check that list function is working.
    os_lock_releaseLock, os_lock_releaseLock, os_lock_execute
    Multi threads
    How the test work.
    - Create and run 5 threads. These threads try to access and use 1 function `running`
    with os_lock_execute. So in one 1, only 1 thread can access and use this function.
    """
    import threading

    # define test function
    def running(fun_args):
        print(f'Function running in thread: {fun_args} START')
        time.sleep(fun_args* 0.2)
        print(f'Function running in thread: {fun_args} END')

    # define test thread
    def thread_running(number):
        print(f'Thread {number} START')
        os_lock_execute(running, number, plock='tmp/plock2.lock')
        print(f'Thread {number} sleeping in {number*3}s')
        time.sleep(number* 0.5)
        print(f'Thread {number} END')

    # Create thread
    for i in range(3):
        t = threading.Thread(target=thread_running, args=(i+1, ))
        t.start()


def test_index():
    """Check that class IndexLock is working
    Multi threads
    How the test work.
    - The test will create the INDEX with the file using plock
    - Create 100 threads that try to write data to this INDEX file lock
    - This test will make sure with this INDEX file log
        only 1 thread can access and put data to this file.
        Others will waiting to acquire key after thread release it.
    """
    import threading

    file_name = "./test.txt"
    #file_lock = "tmp/plock3.lock"

    INDEX = IndexLock(file_name, file_lock=None)

    #1. Create test file
    #with open(file_name, mode='w+') as fp:
    #    pass

    # define test thread
    def thread_running(number):
        print(f'Thread {number} START')
        INDEX.put(f'Thread {number}')
        INDEX.save_filter(f'Thread {number}')
        print( INDEX.get() )

        print(f'Thread {number} END')

    # Create thread
    for i in range(3):
        t = threading.Thread(target=thread_running, args=(i+1, ))
        t.start()


def test_tofilesafe():
   pass



def test_all():
    test_functions()
    test_funtions_thread()
    test_index()

#########################################################################################################
####### Atomic File writing ##############################################################################
class toFile(object):
   def __init__(self,fpath):
      """
       Thread Safe file writer
      """
      logger = logging.getLogger('log')
      logger.setLevel(logging.INFO)
      ch = logging.FileHandler(fpath)
      ch.setFormatter(logging.Formatter('%(message)s'))
      logger.addHandler(ch)
      self.logger = logger

   def write(self, msg):
        self.logger.info( msg)


def to_file_safe(msg:str, fpath:str):
   ss = str(msg)
   logger = logging.getLogger('log')
   logger.setLevel(logging.INFO)
   ch = logging.FileHandler(fpath)
   ch.setFormatter(logging.Formatter('%(message)s'))
   logger.addHandler(ch)

   logger.info( ss)


#########################################################################################################
####### Atomic File Index  read/writing #################################################################
class IndexLock(object):
    """Keep a Global Index of processed files.
      INDEX = IndexLock(findex)

      flist = index.save_isok(flist)  ## Filter out files in index and return available files
      ### only process correct files
      

    """
    ### Manage Invemtory Index with Atomic Write/Read
    def __init__(self, findex, file_lock=None, min_size=5, skip_comment=True, ntry=20):
        self.findex= findex
        os.makedirs(os.path.dirname( os.path.abspath(self.findex)), exist_ok=True)

        if file_lock is None:
            file_lock = os.path.dirname(findex) +"/"+ findex.split("/")[-1].replace(".", "_lock.")
        self.plock = file_lock

        ### Initiate the file
        if not os.path.isfile(self.findex):
            with open(self.findex, mode='a') as fp:
                fp.write("")

        self.min_size=min_size
        self.skip_comment=True
        self.ntry =ntry


    def read(self,): ### alias
        return self.get()


    def save_isok(self, flist:list):   ### Alias
        return put(self, val)

    def save_filter(self, val:list=None):
        return put(self, val)


    ######################################################################
    def get(self, **kw):
        ## return the list of files
        with open(self.findex, mode='r') as fp:
            flist = fp.readlines()

        if len(flist) < 1 : return []

        flist2 = []
        for t  in flist :
            if len(t) < self.min_size: continue
            if self.skip_comment and t[0] == "#"  : continue
            flist2.append( t.strip() )
        return flist2


    def put(self, val:list=None):
        """ Read, check if the insert values are there, and save the files
          flist = index.check_filter(flist)   ### Remove already processed files
          if  len(flist) < 1 : continue   ### Dont process flist

          ### Need locking mechanism Common File to check for Check + Write locking.

        """
        import random, time
        if val is None : return True

        if isinstance(val, str):
            val = [val]

        i = 1
        while i < self.ntry :
            try :
                lock_fd = os_lock_acquireLock(self.plock)

                ### Check if files exist  #####################
                fall =  self.read()
                val2 = [] ; isok= True
                for fi in val:
                    if fi in fall :
                        print('exist in Index, skipping', fi)
                        isok =False
                    else :
                        val2.append(fi)

                if len(val2) < 1 : return []

                #### Write the list of files on Index: Wont be able to use by other processes
                ss = ""
                for fi in val2 :
                  x  = str(fi)
                  ss = ss + x.strip() + "\n"

                with open(self.findex, mode='a') as fp:
                    fp.write( ss )

                os_lock_releaseLock(lock_fd)
                return val2

            except Exception as e:
                log2(f"file lock waiting {i}s")
                time.sleep( random.random() * i )
                i += 1




class Index0(object):
    """
    ### to maintain global index, flist = index.read()  index.save(flist)
    """
    def __init__(self, findex:str="ztmp_file.txt", ntry=10):
        self.findex = findex
        os.makedirs(os.path.dirname(self.findex), exist_ok=True)
        if not os.path.isfile(self.findex):
            with open(self.findex, mode='a') as fp:
                fp.write("")

        self.ntry= ntry

    def read(self,):
        import time
        try :
           with open(self.findex, mode='r') as fp:
              flist = fp.readlines()
        except:
           time.sleep(5)
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


    def save_filter(self, val:list=None):
        """
          isok = index.save_isok(flist)
          if not isok : continue   ### Dont process flist
          ### Need locking mechanism Common File to check for Check + Write locking.

        """
        import random, time
        if val is None : return True

        if isinstance(val, str):
            val = [val]

        i = 1
        while i < self.ntry :
            try :
                ### Check if files exist  #####################
                fall =  self.read()
                val2 = [] ; isok= True
                for fi in val:
                    if fi in fall :
                        print('exist in Index, skipping', fi)
                        isok =False
                    else :
                        val2.append(fi)

                if len(val2) < 1 : return []

                #### Write the list of files on disk
                ss = ""
                for fi in val2 :
                  x  = str(fi)
                  ss = ss + x + "\n"

                with open(self.findex, mode='a') as fp:
                    fp.write( ss )

                return val2

            except Exception as e:
                print(f"file lock waiting {i}s")
                time.sleep( random.random() * i )
                i += 1



#####################################################################################################
######  Atomic Execution ############################################################################
def os_lock_acquireLock(plock:str="tmp/plock.lock"):
    ''' acquire exclusive lock file access, return the locker
    '''
    import fcntl
    os.makedirs(os.path.dirname(os.path.abspath(plock)), exist_ok=True)
    locked_file_descriptor = open( plock, 'w+')
    fcntl.flock(locked_file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return locked_file_descriptor


def os_lock_releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    import fcntl
    fcntl.flock(locked_file_descriptor, fcntl.LOCK_UN)
    # locked_file_descriptor.close()


def os_lock_execute(fun_run, fun_args=None, ntry=5, plock="tmp/plock.lock", sleep=5):
    """ Run a function in an atomic way :
         Write on disk  exclusively on COMMON File.
    """
    i = 0
    while i < ntry :
        try :
            lock_fd = os_lock_acquireLock(plock)
            fun_run(fun_args)
            os_lock_releaseLock(lock_fd)
            break
        except Exception as e:
            # log2(e)
            # reduce sleep time
            log2("file lock waiting", sleep, 'sec')
            time.sleep(sleep)
            i += 1



################################################################################################
def date_now(fmt = "%Y-%m-%d %H:%M:%S %Z%z"):
    from pytz import timezone
    from datetime import datetime
    # Current time in UTC
    now_utc = datetime.now(timezone('UTC'))
    # Convert to US/Pacific time zone
    now_pacific = now_utc.astimezone(timezone('Asia/Tokyo'))
    return now_pacific.strftime(fmt)


def time_sleep_random(nmax=5):
    import random, time
    time.sleep( random.randrange(nmax) )


def save(dd, to_file="", verbose=False):
  import pickle, os
  os.makedirs(os.path.dirname(to_file), exist_ok=True)
  pickle.dump(dd, open(to_file, mode="wb") , protocol=pickle.HIGHEST_PROTOCOL)
  #if verbose : os_file_check(to_file)


def load(to_file=""):
  import pickle
  dd =   pickle.load(open(to_file, mode="rb"))
  return dd


def load_serialize(name):
     global pcache
     #import diskcache as dc
     log2("loading ", pcache)
     cache = load(pcache)
     return cache
     # return {'a' : {'b': 2}}

def save_serialize(name, value):
     global pcache
     #import diskcache as dc
     log2("inserting ", pcache)
     save(value, pcache)







if __name__ == '__main__':
    import fire
    fire.Fire()
