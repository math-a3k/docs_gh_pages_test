# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect, json, yaml, gc


def log(*s):
    print(*s)

####################################################################################################
##### Utilities for date  ##########################################################################
def date_now(fmt="%Y-%m-%d %H:%M:%S %Z%z", add_days=0, timezone='Asia/Tokyo'):
    from pytz import timezone
    from datetime import datetime
    # Current time in UTC
    now_utc = datetime.now(timezone('UTC'))
    now_new = now_utc+ datetime.timedelta(days=add_days)

    # Convert to US/Pacific time zone
    now_pacific = now_new.astimezone(timezone(timezone))
    return now_pacific.strftime(fmt)


def date_is_holiday(array):
    """
      is_holiday([ pd.to_datetime("2015/1/1") ] * 10)

    """
    import holidays , numpy as np
    jp_holidays = holidays.CountryHoliday('JP')
    return np.array( [ 1 if x.astype('M8[D]').astype('O') in jp_holidays else 0 for x in array]  )


def date_weekmonth2(d):
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
 return ((dt - datetime.datetime(dt.year,1,1)).days // 7) + 1


def date_weekday_excel(x) :
 import arrow
 wday= arrow.get( str(x) , "YYYYMMDD").isocalendar()[2]
 if wday != 7 : return wday+1
 else :    return 1


def date_weekyear_excel(x) :
 import arrow
 dd= arrow.get( str(x) , "YYYYMMDD")
 wk1= dd.isocalendar()[1]

 # Excel Convention
 # dd0= arrow.get(  str(dd.year) + "0101", "YYYYMMDD")
 dd0_weekday= date_weekday_excel( dd.year *10000 + 101  )
 dd_limit= dd.year*10000 + 100 + (7-dd0_weekday+1) +1

 ddr= arrow.get( str(dd.year) + "0101" , "YYYYMMDD")
 # print dd_limit
 if    int(x) < dd_limit :
    return 1
 else :
     wk2= 2 + int(((dd-ddr ).days  - (7-dd0_weekday +1 ) )   /7.0 )
     return wk2


def date_generate(start='2018-01-01', ndays=100) :
 from dateutil.relativedelta import relativedelta
 start0 = datetime.datetime.strptime(start, "%Y-%m-%d")
 date_list = [start0 + relativedelta(days=x) for x in range(0, ndays)]
 return date_list


###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()




