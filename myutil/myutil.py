import os, sys, time, datetime,inspect


verbosity_get(cur_path, path_relative="/../../config.json", key='verbosity'):
  try   : verbosity = int(json.load(open(os.path.dirname(os.path.abspath(cur_path)) + path_relative , mode='r'))[key])
  except Exception as e : verbosity = 2
  return verbosity
  #raise Exception(f"{e}")

  
def os_makedirs(dir_or_file):
    if os.path.isfile(dir_or_file) :os.makedirs(os.path.dirname(os.path.abspath(dir_or_file)), exist_ok=True)
    else : os.makedirs(os.path.abspath(dir_or_file), exist_ok=True)


