# -*- coding: utf-8 -*-
""" 
Created on Thu Jan 26 17:29:50 2017
"""

############################################################################################  
fmt='YYYY-MM-DD'
def date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH:mm:SS') :
   dd= arrow.get(str_t1, fmt) - arrow.get(str_t0, fmt) 
   return dd.total_seconds()

   
def date_diffstart(t) : return date_diffsecond(str_t1=t, str_t0=t0)
def date_diffend(t) :   return date_diffsecond(str_t1=t1, str_t0=t)
   

def np_dict_tolist(dd) :
    return [ val  for _, val in dd.items() ]
            
def np_dict_tostr_val(dd) :
    return ','.join([ str(val)  for _, val in dd.items() ])
         
def np_dict_tostr_key(dd) :
    return ','.join([ str(key)  for key,_ in dd.items() ])

    
###################Faster one   ############################################################
#'YYYY-MM-DD    HH:mm:ss'
#"0123456789_10_11
import arrow, copy
def day(s):    return int(s[8:10])
def month(s):  return int(s[5:7])
def year(s):   return int(s[0:4])
def hour(s):   return int(s[11:13])


cache_weekday= {}
def weekday(s, fmt='YYYY-MM-DD', i0=0, i1=10):
  ###Super Fast because of caching
  s2= s[i0:i1]
  try : 
     return  cache_weekday[s2]
  except KeyError:
    wd= arrow.get(s2, fmt).weekday()
    cache_weekday[s2]= wd
  return wd

def season(d):
  m=  int(d[5:7])
  if m > 3 and m  < 10: return 1
  else: return 0 

def daytime(d):
  h= int(d[11:13])
  if   h < 11 :   return 0
  elif h < 14 : return 1     # lunch
  elif h < 18 : return 2     # Afternoon
  elif h < 21 : return 3     # Dinner
  else :        return 4     # Night


def pd_date_splitall(df, coldate='purchased_at') :
   df= copy.deepcopy(df)
   df['year']=  df[coldate].apply(year)   
   df['month']= df[coldate].apply(month)   
   df['day']=   df[coldate].apply(day)   
   df['weekday']=   df[coldate].apply(weekday)   
   df['daytime']=   df[coldate].apply(daytime)   
   df['season']=   df[coldate].apply(season)   
   return df


