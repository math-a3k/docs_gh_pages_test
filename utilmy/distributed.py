HELP="""
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



################################################################################################
# Test functions
def test1_functions():
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


def test2_funtions_thread():
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
        time.sleep(fun_args*3)
        print(f'Function running in thread: {fun_args} END')

    # define test thread
    def thread_running(number):
        print(f'Thread {number} START')
        os_lock_execute(running, number, plock='tmp/plock2.lock')
        print(f'Thread {number} sleeping in {number*3}s')
        time.sleep(number*3)
        print(f'Thread {number} END')

    # Create thread
    for i in range(5):
        t = threading.Thread(target=thread_running, args=(i+1, ))
        t.start()


def test3_index():
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

    file_name = "test.txt"
    file_lock = "tmp/plock3.lock"

    INDEX = IndexLock(file_name, file_lock)

    #1. Create test file
    with open(file_name, mode='w+') as fp:
        pass

    # define test thread
    def thread_running(number):
        print(f'Thread {number} START')
        INDEX.put(f'Thread {number}')
        print(f'Thread {number} END')

    # Create thread
    for i in range(100):
        t = threading.Thread(target=thread_running, args=(i+1, ))
        t.start()


def test_all():
    test1_funtions()
    test2_funtions_thread()
    test3_index()
      
      
###############################################################################################
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
        except Exception as e:
            # log2(e)
            # reduce sleep time
            log2("file lock waiting", 20, 'sec')
            time.sleep(20)
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
        
        
    def get(self):    
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

            except Exception as e:
                # reduce waiting time
                log2(f"file lock waiting {i}s")
                # time.sleep(5*i*i)
                time.sleep(i)
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
    test_all()
