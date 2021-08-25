"""
All related to distributed compute and atomic read/write





"""


def log_mem(*s):
    try:
        # print(*s, "\n", flush=True)
        import psutil
        log2('mem check', str(psutil.virtual_memory())) 
        # print(s)
    except:
        pass    
    


def date_now(fmt = "%Y-%m-%d %H:%M:%S %Z%z"):
    from pytz import timezone
    from datetime import datetime
    # Current time in UTC
    now_utc = datetime.now(timezone('UTC'))
    # Convert to US/Pacific time zone
    now_pacific = now_utc.astimezone(timezone('Asia/Tokyo'))
    return now_pacific.strftime(fmt)

def sleep_random(nmax=5):
    import random, time
    time.sleep( random.randrange(nmax) )

       
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


def os_lock_acquireLock(plock):
    import fcntl
    ''' acquire exclusive lock file access '''
    locked_file_descriptor = open( plock, 'w+')
    fcntl.flock(locked_file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return locked_file_descriptor


def os_lock_releaseLock(locked_file_descriptor):
    import fcntl
    ''' release exclusive lock file access '''
    fcntl.flock(locked_file_descriptor, fcntl.LOCK_UN)
    # locked_file_descriptor.close()

    
def os_lock_execute(fun_run, pars, ntry=5, plock="tmp/plock.lock"):    
    ### Need locking mechanism Common File to check for locking.
    ntry = 1
    while ntry < ntry :
        try :
            lock_fd = acquireLock(plock)
            fun_run(pars)                      
            releaseLock(lock_fd)
            break
        except :
            log2("file lock waiting", ntry)
            time.sleep(20*ntry)
            ntry += 1



class IndexLock(object):
    """
    
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
    
            
    def put(self, val="", ntry=5, plock="tmp/plock.lock"):    
        ### Need locking mechanism Common File to check for Check + Write locking.
        ntry = 1
        while ntry < ntry :
            try :
                lock_fd = os_lock_acquireLock(self.plock)

                with open(self.findex, mode='r') as fp:
                    fall = fp.readlines()    

                if val in set(fall) :  return False

                with open(self.findex, mode='a') as fp:
                    fp.write( fpath.strip() + "\n" )

                os_lock_releaseLock(lock_fd)
                return True

            except :
                print("file lock waiting", ntry)
                time.sleep(5*ntry*ntry)
                ntry += 1
            

                        
  
  
