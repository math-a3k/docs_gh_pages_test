import os, sys, time, datetime,inspect


def verbosity_get(cur_path, path_relative="/../../config.json", key='verbosity', default=5):
  try   : 
    verbosity = int(json.load(open(os.path.dirname(os.path.abspath(cur_path)) + path_relative , mode='r'))[key])
  except Exception as e : 
    verbosity = default
  return verbosity
  #raise Exception(f"{e}")

  
def os_makedirs(dir_or_file):
    if os.path.isfile(dir_or_file) :os.makedirs(os.path.dirname(os.path.abspath(dir_or_file)), exist_ok=True)
    else : os.makedirs(os.path.abspath(dir_or_file), exist_ok=True)


class Session(object) :
   """ Save Python session on disk
      from util import Session
      sess = Session("recsys")
      sess.save( globals() )
   """
   def __init__(self,  name="default", dir_session="ztmp/",) :
      self.dir_session =  "ztmp/session/"  if dir_session is None else dir_session
      self.name = name
      self.cur_session = self.dir_session + "/" + name + "/"
      print(self.cur_session)

   def save(self, tag="", glob) : 
       path = f"{self.dir_session}/{self.name}_{tag}/"  if tag != "" else self.cur_session
       save_session(path, glob)

   def load(self, name, glob) :
      self.dir_session = "ztmp/session/"  if self.dir_session is None else self.dir_session
      self.name = name
      self.cur_session = self.dir_session + "/" + name + "/"
      print(self.cur_session)
      load_session(self.cur_session , glob )


def save_session(folder , globs, tag="" ) :    
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
  if verbose : os_file_check(to_file)


def load(to_file=""):
  import pickle  
  dd =   pickle.load(open(to_file, mode="rb"))
  return dd



if __name__ == "__main__":
    import fire
    fire.Fire()






      
