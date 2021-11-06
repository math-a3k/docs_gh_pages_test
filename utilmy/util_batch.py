HELP= """ Utils for easy batching




"""

form utilmy.utilmy import log, log2



def os_wait_until(dirin, ntry_max=30): 
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

      
      
