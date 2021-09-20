"""
All related to distributed compute and atomic read/write

   Thread Safe
   Process Safe
   Lock Mechanism





"""
import os, sys, socket, platform, time, gc

###############################################################################################
def log2(*s):
    print(*s, flush=True)


def log_mem(*s):
    try:
        # print(*s, "\n", flush=True)
        import psutil
        log2('mem check', str(psutil.virtual_memory()))
        # print(s)
    except:
        pass


###############################################################################################
def os_lock_acquireLock(plock:str="tmp/plock.lock"):
    ''' acquire exclusive lock file access, return the locker

    '''
    import fcntl
    os.makedirs(os.path.dirname( os.path.abspath(plock)  ))
    locked_file_descriptor = open( plock, 'w+')
    fcntl.flock(locked_file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return locked_file_descriptor


def os_lock_releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    import fcntl
    fcntl.flock(locked_file_descriptor, fcntl.LOCK_UN)
    # locked_file_descriptor.close()

    
def os_lock_execute(fun_run, fun_args=None, ntry=5, plock="tmp/plock.lock"):
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
        except :
            log2("file lock waiting", 20*i, 'sec')
            time.sleep(20*i)
            i += 1



class IndexLock(object):
    """
      Keep a Global Index of processed files.
      INDEX = IndexLock(findex, plock)
    
    """
    ### Manage Invemtory Index with Atomic Write/Read
    def __init__(self, findex, plock):
        self.findex= findex
        self.plock = plock
        
        
    def get(self, val="", ntry=5, plock="tmp/plock.lock"):    
        with open(self.findex, mode='r') as fp:
            fall = fp.readlines()  
        return fall
    
            
    def put(self, val="", ntry=100, plock="tmp/plock.lock"):
        ### Need locking mechanism Common File to check for Check + Write locking.
        i = 1
        while i < ntry :
            try :
                lock_fd = os_lock_acquireLock(self.plock)

                with open(self.findex, mode='r') as fp:
                    fall = fp.readlines()    

                if val in set(fall) :  return False

                with open(self.findex, mode='a') as fp:
                    fp.write( val.strip() + "\n" )

                os_lock_releaseLock(lock_fd)
                return True

            except :
                log2("file lock waiting", i)
                time.sleep(5*i*i)
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


  
  
