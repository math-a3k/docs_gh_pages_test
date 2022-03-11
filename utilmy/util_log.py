


"""
Need log generator
  get_logger('info', verbose=verbose)
  return log_fun
  
  log, logd, = logger_get(['info', verbose)

"""
def os_getenv_dict():
  """
    utilmy
    set utilmy_pars = "$utilmy_pars;  verbose:1"  
  """
  import os
  def to_int(x, val=-1):
     try:     return int(x)
     except : return val
  
  llvars = os.getenv('utilmy_pars', {}).split(";") 
  ddict = {'common': [] }
  for x in llvars:
     ll = [ t.strip() for t in  x.split(":") ]
     if len(ll) > 1:     ddict[ ll[0] ] = ll[1]
     elif len(ll) == 1:  ddict['common'].append( ll[0] ) 
  
  if 'verbose' in ddict :  ddict['verbose'] = to_int(  ddict['verbose'], 0 )
  return ddict 
  
  
VERBOSE = os.getenv('verbose', 1)


