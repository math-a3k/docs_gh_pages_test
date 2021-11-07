HELP= """ Utils for easy batching




"""
import os, sys, socket, platform, time, gc,logging, random, datetime, logging


from utilmy.utilmy import log, log2
from utilmy import pd_read_file



def now_weekday_isin(day_week=[  0,1,2  ]) :
    ### 0 is sunday, 1 is monday 
    return False
    
    return True



def now_hour_between(hour1="12:45", hour2="13:45", timezone="jp") :
    ### Daily Batch time is between 2 time. 
    return False
    
    return True


def now_daymonth_isin(day_month=[ 1,2  ], timezone="jp") :
    ### 1th day of month
    return False
    
    return True


def now_weekday_isin(day_week=[  0,1,2  ], timezone="jp") :
    ### 0 is sunday, 1 is monday 
    return False
    
    return True



#########################################################################################
def batchLog(object):    
    def __init__(self,dirlog="log/batch_log", date_fmt="%Y%m%d", format="", timezone="jp")
          pass
    
    def save(self, name,  tag="end/start", info="", **kw)    
        """  Log on file , termination of file
             Thread Safe writing.    
            dt\t prog_name\t name \t tag \t info
            
            One file per day

        """
        pass


    def isdone(self, name="*",  tag="end/start", info="",  tstart="*", tend="*",  )
        """  Find if a task was done or not.
        """
        pass

    
    def getall(self, date="*" )
        """  Find if a task was done or not.
        """
        df = pd_read_file(  self.dirlog + "/" + date, sep="\t" )
        return df
        
        
    


#########################################################################################
def os_wait_cpu_ram_lower(cpu_max, ram_max):
    #### Wait until Server CPU and ram are lower than threshold.    
    return True



def os_wait_program_end(prog_name, max_wait=86400):
    #### Wait until one program finishes
    
    return True


def os_wait_isfile_exist(dirin, ntry_max=30): 
    import glob, time
    log('####### Check if file ready', "\n", dirin,)
    ntry=0
    while ntry < ntry_max :
       fi = glob.glob(dirin )
       if len(fi) >= 1: break
       ntry += 1
       time.sleep(60*5)    
    log('File is ready:', dirin)    


    
date_now = date_now_jp  ### alias


def date_now_jp(fmt="%Y%m%d", add_days=0, add_hours=0, timezone='Asia/Tokyo'):
    # "%Y-%m-%d %H:%M:%S %Z%z"
    from pytz import timezone as tzone
    import datetime
    # Current time in UTC
    now_utc = datetime.datetime.now(tzone('UTC'))
    now_new = now_utc+ datetime.timedelta(days=add_days, hours=add_hours)

    if timezone == 'utc':
       return now_new.strftime(fmt)
      
    else :
       # Convert to US/Pacific time zone
       now_pacific = now_new.astimezone(tzone(timezone))
       return now_pacific.strftime(fmt)

      
      
        
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
     
        
def time_sleep_random(nmax=5):
    import random, time
    time.sleep( random.randrange(nmax) )
        
        

if __name__ == '__main__':
    import fire
    fire.Fire()        
        
        
        
        
