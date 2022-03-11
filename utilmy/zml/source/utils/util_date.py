# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
import datetime
datetime.datetime.strptime('20-Nov-2002','%d-%b-%Y').strftime('%Y%m%d')
'20021120'
Formats -

%d - 2 digit date
%b - 3-letter month abbreviation
%Y - 4 digit year
%m - 2 digit month
%a

df = DataFrame(dict(date = date_range('20130101',periods=10)))
https://python-utils.readthedocs.io/en/latest/usage.html#quickstart
https://dateutil.readthedocs.io/en/stable/examples.html



"""

from datetime import datetime

import dateutil
import numpy as np
import pandas as pd


def pd_datestring_split(
        dfref, coldate, fmt="%Y-%m-%d %H:%M:%S", return_val="split"):
    """
      Parsing date
      'Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p'
    :param datelist:
    :param fmt:
    :return:
    """
    fmt = None if fmt in [None, "auto"] else fmt
    if not isinstance(coldate, str):
        raise Exception("codlate must be string")
    df = pd.DataFrame(dfref[coldate])

    coldt = coldate + "_dt"
    df[coldt] = pd.to_datetime(
        df[coldate], errors="coerce", format=None, infer_datetime_format=True, cache=True
    )

    df[coldate + "_year"] = df[coldt].apply(lambda x: x.year)
    df[coldate + "_month"] = df[coldt].apply(lambda x: x.month)
    df[coldate + "_day"] = df[coldt].apply(lambda x: x.day)

    if return_val == "split":
        return df
    else:
        return df[[coldate + "_year", coldate + "_month", coldate + "_day"]]


def datestring_todatetime(datelist, fmt="%Y-%m-%d %H:%M:%S"):
    """
      Parsing date
      'Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p'
    :param datelist:
    :param fmt:
    :return:
    """
    datenew = []
    if fmt == "auto":
        if isinstance(datelist, list):
            for x in datelist:
                try:
                    datenew.append(dateutil.parser.parse(x))
                except Exception as e:
                    datenew.append(pd.NaT)

            return datenew
        else:
            return dateutil.parser.parse(datelist)
    else:
        if isinstance(datelist, list):
            return [datetime.strptime(x, fmt) for x in datelist]
        else:
            return datetime.strptime(datelist, fmt)


def datetime_tostring(datelist, fmt="%Y-%m-%d %H:%M:%S"):
    """
  https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
  :param x:
  :param fmt:
  :return:
  """
    if isinstance(datelist, list):
        ll = [datetime.strftime(x, fmt) for x in datelist]
        return ll
    else:
        return datetime.strftime(datelist, fmt)


def datetime_tointhour(datelist):
    """function datetime_tointhour
    Args:
        datelist:   
    Returns:
        
    """
    if not isinstance(datelist, list):
        x = datelist
        y = (
            x.year * 10000 * 10000 * 100
            + x.month * 10000 * 10000
            + x.day * 100 * 10000
            + x.hour * 10000
            + x.minute * 100
            + x.second
        )
        return y
    yy2 = []
    for x in datelist:
        yy2.append(
            x.year * 10000 * 10000 * 100
            + x.month * 10000 * 10000
            + x.day * 100 * 10000
            + x.hour * 10000
            + x.minute * 100
            + x.second
        )
    return np.array(yy2)


def datetime_toint(datelist):
    """function datetime_toint
    Args:
        datelist:   
    Returns:
        
    """
    if not isinstance(datelist, list):
        x = datelist
        return x.year * 10000 + x.month * 100 + x.day
    yy2 = []
    for x in datelist:
        yy2.append(x.year * 10000 + x.month * 100 + x.day)
    return np.array(yy2)


def datetime_to_milisec(datelist):
    """function datetime_to_milisec
    Args:
        datelist:   
    Returns:
        
    """
    if not isinstance(datelist, list):
        return (datelist - datetime(1970, 1, 1)).total_seconds()
    else:
        ll = [(t - datetime.datetime(1970, 1, 1)).total_seconds() for t in datelist]
        return ll


def datetime_weekday(datelist):
    """function datetime_weekday
    Args:
        datelist:   
    Returns:
        
    """
    if not isinstance(datelist, list):
        return int(datelist.strftime("%w"))
    else:
        return [int(x.strftime("%w")) for x in datelist]


dd_weekday_cache = {}


def datetime_weekday_fast(dateval):
    """
      date values
    :param dateval:
    :return:
    """
    try:
        return dd_weekday_cache[dateval]
    except:
        d = datetime_weekday(dateval)
        dd_weekday_cache[dateval] = d
        return d


def datetime_quarter(datetimex):
    """function datetime_quarter
    Args:
        datetimex:   
    Returns:
        
    """
    m = datetimex.month
    return int(m // 3) + 1


def dateime_daytime(datetimex):
    """function dateime_daytime
    Args:
        datetimex:   
    Returns:
        
    """
    h = datetimex.hour
    if h < 11:
        return 0
    elif h < 14:
        return 1  # lunch
    elif h < 18:
        return 2  # Afternoon
    elif h < 21:
        return 3  # Dinner
    else:
        return 4  # Night


def datenumpy_todatetime(tt, islocaltime=True):
    """function datenumpy_todatetime
    Args:
        tt:   
        islocaltime:   
    Returns:
        
    """
    #  http://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
    if type(tt) == np.datetime64:
        if islocaltime:
            return datetime.fromtimestamp(tt.astype("O") / 1e9)
        else:
            return datetime.utcfromtimestamp(tt.astype("O") / 1e9)
    elif type(tt[0]) == np.datetime64:
        if islocaltime:
            v = [datetime.fromtimestamp(t.astype("O") / 1e9) for t in tt]
        else:
            v = [datetime.utcfromtimestamp(t.astype("O") / 1e9) for t in tt]
        return v
    else:
        return tt  # datetime case


def datetime_tonumpydate(t, islocaltime=True):
    """function datetime_tonumpydate
    Args:
        t:   
        islocaltime:   
    Returns:
        
    """
    #  http://stackoverflow.com/questions/29753060/how-to-convert-numpy-datetime64-into-datetime
    return np.datetime64(t)


"""
def date_diffsecond(str_t1, str_t0, fmt='YYYY-MM-DD HH:mm:SS') :
    d d= arrow.get(str_t1, fmt) - arrow.get(str_t0, fmt)
    return dd.total_seconds()
"""


def np_dict_tolist(dd):
    """function np_dict_tolist
    Args:
        dd:   
    Returns:
        
    """
    return [val for _, val in list(dd.items())]


def np_dict_tostr_val(dd):
    """function np_dict_tostr_val
    Args:
        dd:   
    Returns:
        
    """
    return ",".join([str(val) for _, val in list(dd.items())])


def np_dict_tostr_key(dd):
    """function np_dict_tostr_key
    Args:
        dd:   
    Returns:
        
    """
    return ",".join([str(key) for key, _ in list(dd.items())])


"""
>>> import datetime
>>> datetime.datetime.strptime('20-Nov-2002','%d-%b-%Y').strftime('%Y%m%d')
'20021120'
Formats -

%d - 2 digit date
%b - 3-letter month abbreviation
%Y - 4 digit year
%m - 2 digit month
%a

Weekday as locale’s abbreviated name.
Sun, Mon, …, Sat (en_US);
So, Mo, …, Sa (de_DE)
(1)


%A
Weekday as locale’s full name.
Sunday, Monday, …, Saturday (en_US);
Sonntag, Montag, …, Samstag (de_DE)
(1)

%w
Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.
0, 1, …, 6

%d
Day of the month as a zero-padded decimal number.
01, 02, …, 31
(9)

%b
Month as locale’s abbreviated name.
Jan, Feb, …, Dec (en_US);
Jan, Feb, …, Dez (de_DE)
(1)

%B
Month as locale’s full name.
January, February, …, December (en_US);
Januar, Februar, …, Dezember (de_DE)
(1)

%m
Month as a zero-padded decimal number.
01, 02, …, 12

(9)
%y
Year without century as a zero-padded decimal number.
00, 01, …, 99

(9)
%Y
Year with century as a decimal number.
0001, 0002, …, 2013, 2014, …, 9998, 9999

(2)

%H

Hour (24-hour clock) as a zero-padded decimal number.

00, 01, …, 23

(9)

%I

Hour (12-hour clock) as a zero-padded decimal number.

01, 02, …, 12

(9)

%p

Locale’s equivalent of either AM or PM.

AM, PM (en_US);
am, pm (de_DE)
(1), (3)

%M

Minute as a zero-padded decimal number.

00, 01, …, 59

(9)

%S

Second as a zero-padded decimal number.

00, 01, …, 59

(4), (9)

%f

Microsecond as a decimal number, zero-padded on the left.

000000, 000001, …, 999999

(5)

%z

UTC offset in the form ±HHMM[SS[.ffffff]] (empty string if the object is naive).

(empty), +0000, -0400, +1030, +063415, -030712.345216

(6)

%Z

Time zone name (empty string if the object is naive).

(empty), UTC, EST, CST

%j

Day of the year as a zero-padded decimal number.

001, 002, …, 366

(9)

%U

Week number of the year (Sunday as the first day of the week) as a zero padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0.

00, 01, …, 53

(7), (9)

%W

Week number of the year (Monday as the first day of the week) as a decimal number. All days in a new year preceding the first Monday are considered to be in week 0.

00, 01, …, 53

(7), (9)

%c

Locale’s appropriate date and time representation.

Tue Aug 16 21:30:00 1988 (en_US);
Di 16 Aug 21:30:00 1988 (de_DE)
(1)

%x

Locale’s appropriate date representation.

08/16/88 (None);
08/16/1988 (en_US);
16.08.1988 (de_DE)
(1)

%X

Locale’s appropriate time representation.

21:30:00 (en_US);
21:30:00 (de_DE)
(1)

%%

A literal '%' character.

%

"""
