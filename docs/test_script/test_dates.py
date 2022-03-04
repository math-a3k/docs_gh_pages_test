# -*- coding: utf-8 -*-
MNAME= "utilmy.dates"
HELP=""" dates utilities

"""
import os, sys, time, datetime,inspect, json, yaml, gc, numpy as np, pandas as pd

#############################################################################################
from utilmy.utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    print(  HELP + help_create(MNAME) )


####################################################################################################
def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    log("Testing dates.py ...")
    date_ = date_generate(start='2021-01-01', ndays=100)
    date_weekyear_excel('20210317')
    date_weekday_excel('20210317')
    #TODO:
    #date_is_holiday([ pd.to_datetime("2015/1/1") ] * 10)
    #date_now(fmt="%Y-%m-%d %H:%M:%S %Z%z", add_days=0, timezone='Asia/Tokyo')
    df = pd.DataFrame(columns=[ 'Gender', 'Birthdate'])
    df['Gender'] = random_genders(10)
    df['Birthdate'] = random_dates(start=pd.to_datetime('1940-01-01'), end=pd.to_datetime('2008-01-01'), size=10)
    # TODO:
    #pd_date_split(df,coldate="Birthdate")

def random_dates(start, end, size):
    """function random_dates
    Args:
        start:   
        end:   
        size:   
    Returns:
        
    """
    divide_by = 24 * 60 * 60 * 10**9
    start_u = start.value // divide_by
    end_u = end.value // divide_by
    return pd.to_datetime(np.random.randint(start_u, end_u, size), unit="D")
    
def random_genders(size, p=None):
    """function random_genders
    Args:
        size:   
        p:   
    Returns:
        
    """
    if not p:
        p = (0.49, 0.49, 0.01, 0.01)
    gender = ("M", "F", "O", "")
    return np.random.choice(gender, size=size, p=p)


####################################################################################################
##### Utilities for date  ##########################################################################
def pd_date_split(df, coldate =  'time_key', prefix_col ="",sep="/" ,verbose=False ):
    """function pd_date_split
    Args:
        df:   
        coldate :   
        prefix_col :   
        sep:   
        verbose:   
    Returns:
        
    """
    import pandas as pd

    df = df.drop_duplicates(coldate)
    df['date'] =  pd.to_datetime( df[coldate] )

    ############# dates
    df['year']          = df['date'].apply( lambda x : x.year   )
    df['month']         = df['date'].apply( lambda x : x.month   )
    df['day']           = df['date'].apply( lambda x : x.day   )
    df['weekday']       = df['date'].apply( lambda x : x.weekday()   )
    df['weekmonth']     = df['date'].apply( lambda x : date_weekmonth(x)   )
    df['weekmonth2']    = df['date'].apply( lambda x : date_weekmonth2(x)   )
    df['weekyeariso']   = df['date'].apply( lambda x : x.isocalendar()[1]   )
    df['weekyear2']     = df['date'].apply( lambda x : date_weekyear2( x )  )
    df['quarter']       = df.apply( lambda x :  int( x['month'] / 4.0) + 1 , axis=1  )

    def merge1(x1,x2):
        if sep == "":
            return int(str(x1) + str(x2))
        return str(x1) + sep + str(x2)

    df['yearweek']      = df.apply(  lambda x :  merge1(  x['year']  , x['weekyeariso'] )  , axis=1  )
    df['yearmonth']     = df.apply( lambda x : merge1( x['year'] ,  x['month'])         , axis=1  )
    df['yearquarter']   = df.apply( lambda x : merge1( x['year'] ,  x['quarter'] )         , axis=1  )

    df['isholiday']     = date_is_holiday(df['date'])

    exclude = [ 'date', coldate]
    df.columns = [  prefix_col + x if not x in exclude else x for x in df.columns]
    if verbose : log( "holidays check", df[df['isholiday'] == 1].tail(15)  )
    return df


def date_to_timezone(tdate,  fmt="%Y%m%d-%H:%M", timezone='Asia/Tokyo'):
    """
       dt = datetime.datetime.now(timz('UTC'))
    """
    from pytz import timezone as tzone
    import datetime
    # Convert to US/Pacific time zone
    now_pacific = tdate.astimezone(tzone('Asia/Tokyo'))
    return now_pacific.strftime(fmt)


def date_now(fmt="%Y-%m-%d %H:%M:%S %Z%z", add_days=0, timezone='Asia/Tokyo'):
    """function date_now
    Args:
        fmt="%Y-%m-%d %H:   
        add_days:   
        timezone:   
    Returns:
        
    """
    from pytz import timezone as timz
    import datetime
    # Current time in UTC
    now_utc = datetime.datetime.now(timz('UTC'))
    now_new = now_utc+ datetime.timedelta(days=add_days)

    # Convert to US/Pacific time zone
    now_pacific = now_new.astimezone(timz(timezone))
    return now_pacific.strftime(fmt)


def date_is_holiday(array):
    """
      is_holiday([ pd.to_datetime("2015/1/1") ] * 10)
    """
    import holidays , numpy as np
    jp_holidays = holidays.CountryHoliday('JP')
    return np.array( [ 1 if x in jp_holidays else 0 for x in array]  )


def date_weekmonth2(d):
     """function date_weekmonth2
     Args:
         d:   
     Returns:
         
     """
     w = (d.day-1)//7+1
     if w < 0 or w > 5 :
         return -1
     else :
         return w


def date_weekmonth(date_value):
     """  Incorrect """
     w = (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)
     if w < 0 or w > 6 :
         return -1
     else :
         return w


def date_weekyear2(dt) :
 """function date_weekyear2
 Args:
     dt:   
 Returns:
     
 """
 return ((dt - datetime.datetime(dt.year,1,1)).days // 7) + 1


def date_weekday_excel(x) :
 """function date_weekday_excel
 Args:
     x:   
 Returns:
     
 """
 import datetime
 date = datetime.datetime.strptime(x,"%Y%m%d")
 wday = date.weekday()
 if wday != 7 : return wday+1
 else :    return 1


def date_weekyear_excel(x) :
 """function date_weekyear_excel
 Args:
     x:   
 Returns:
     
 """
 import datetime
 date = datetime.datetime.strptime(x,"%Y%m%d")
 return date.isocalendar()[1]


def date_generate(start='2018-01-01', ndays=100) :
 """function date_generate
 Args:
     start:   
     ndays:   
 Returns:
     
 """
 from dateutil.relativedelta import relativedelta
 start0 = datetime.datetime.strptime(start, "%Y-%m-%d")
 date_list = [start0 + relativedelta(days=x) for x in range(0, ndays)]
 return date_list


###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()



