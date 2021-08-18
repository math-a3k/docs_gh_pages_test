# -*- coding: utf-8 -*-
"""database check """
%load_ext autoreload
%autoreload 2
import portfolio as pf, util
import numpy as np,  pandas as pd, copy, numexpr as ne, scipy as sci, matplotlib.pyplot as plt
import sklearn as sk
from sklearn import linear_model; from sklearn import covariance
import datetime; from datetime import datetime
#------------------------------------------------------------------------------------




################################################################################
#---------------  Import Google ---> CSV Folder --------------------------------
# All Topix  Google ---> csv files
pf.imp_googleQuoteList(tpx2000list, '20160101', '20160629', 
                    inter=300, tframe=600, dircsv='E:/_data/stock/csv/topix', intraday1=1)

# All Nikkei Import in csv
pf.imp_googleQuoteList(nk225list, '20160601', '20160615', 
                    inter=300, tframe=200, dircsv='E:/_data/stock/csv/nk225', intraday1=1)

# NK400 Import List 
pf.imp_googleQuoteList(nk400list, '20160601', '20160621', 
                    inter=300, tframe=200, dircsv='E:/_data/stock/csv/nk400', intraday1=1)


pf.imp_googleQuoteList(jpetf, '20160401', '20160625', 
                    inter=300, tframe=200, dircsv='E:/_data/stock/csv/nk400', intraday1=1)





#-------------------Insert Intraday CSV ---> Pandas ---------------------------
dircsv= r'E:/_data/stock/csv/nk400'
dbfile2=  r'F:/_data/stock/intraday/jp_intraday_google.h5'
pf.imp_hdfs_db_updatefromcsv(dircsv, filepd= dbfile, fromtimezone='Japan')

pf.imp_hdfs_removeDuplicate(dbfile)  #Remove duplicate
dbinfo2, dberr2=   pf.imp_hdfs_db_dumpinfo(dbfile2)
# imp_hdfs_mergedb(dbfile1, dbfile2 )
# pf.imp_hdfs_removeDuplicate(dbfile2)

  












################################################################################
################################################################################
dbfile=  r'F:/_data/stock/intraday/jp_intraday_google.h5'


symjp1= [ 'jp'+x for x in nk400list ]; symjp1+= ['gbpjpy' ]

symjp1= [ 'jp7203', 'jp2269', 'jp9627' ]; symjp1+= ['gbpjpy' ]


now1= '20160629'
quotesjp5min, date0, symjp1, errsym= pf.imp_hdfs_getListquote(symjp1, close1='Close',
                                                              start='2016-05-09 00:00:00',
                                                              end=now1, freq='0d0h05min', filepd= dbfile, tozone='Japan')
errsym
pf.imp_pd_checkquote(quotesjp5min)

dbinfo2, dberr2=   pf.imp_hdfs_db_dumpinfo(dbfile2)




close2, dateref2= pf.date_align(quotesjp5min, type1="close") #Get the data and align dates

util.a_cleanmemory()




q= quotesjp5min[0]


pf.imp_hdfs_removeDuplicate(dbfile)  #Remove duplicate




dbinfo2, dberr2=   pf.imp_hdfs_db_dumpinfo(dbfile)

spdateref2= util.datetime_convertzone1_tozone2(spdateref2, fromzone='Japan', tozone='US/Eastern')


spintraday.loc[:,'date']= util.datetime_convertzone1_tozone2(spintraday.date, fromzone='GMT', tozone='US/Eastern')






#####################################################################################
#---------------------             --------------------
ccylist=['usdjpy', 'gbpusd', 'gbpjpy', 'cadjpy', 'audjpy', 'nzdjpy', 'chfjpy', 'usdmxn', 'mxnjpy', 'usdcad','usdaud', 'eurusd', 'eurjpy', 'eurchf', 'usdbrl', 'jpybrl', '132030'  ]
pf.imp_googleQuoteList(ccylist, '20160101', '20160623', 
                    inter=300, tframe=600, dircsv='E:/_data/stock/csv/intraday', intraday1=1)



# Import of Currency are in local Japan Time  ---> Upload in Database
dircsv= r'E:/_data/stock/csv/jp/20160828'
dbfile=  r'F:/_data/stock/intraday/jp_intraday_google.h5'
pf.imp_hdfs_db_updatefromcsv(dircsv, filepd= dbfile, fromtimezone='Japan')

pf.imp_hdfs_removeDuplicate(dbfile)  #Remove duplicate

dbinfo2, dberr2=   pf.imp_hdfs_db_dumpinfo(dbfile)


#####################################################################################
symjp1= [ 'jp6981', 'jp4461', 'jp9022' ]


now1= '20160629'
quotesjp5min, date0, symjp1, errsym= pf.imp_hdfs_getListquote(symjp1, close1='Close',
                                                              start='2016-05-09 09:00:00',
                                                              end=now1, freq='0d0h05min', filepd= dbfile, timezone='Japan')
errsym
pf.imp_pd_checkquote(quotesjp5min)


q0= quotesjp5min[0]
q0.date.values[0]


close2, dateref2= pf.date_align(quotesjp5min, type1="close") #Get the data and align dates
dateref2= [util.datenumpy_todatetime(t) for t in dateref2  ]
ret_close2= pf.getret_fromquotes(close2, timelag=1)
price2= pf.price_normalize100(ret_close2)
pf.plot_pricedate(dateref2, symjp1, price2)


pf.plot_price( price2,   sym= symjp1, tickperday=58*5)


type(dateref2[0])



util.a_cleanmemory()
dateref2= [util.datenumpy_todatetime(t) for t in dateref2  ]

















#####################################################################################











#####################################################################################
#---------------------             --------------------








#####################################################################################





















#####################################################################################
#---------------------             --------------------






























