# -*- coding: utf-8 -*-
#utilities for portfolio data management
import calendar
import copy
import os
import re
import sys
from datetime import datetime

import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy as sci
import sklearn as sk
from dateutil import parser
from matplotlib.finance import quotes_historical_yahoo_ochl
from numba import jit, int32, float32, float64, int64
from tabulate import tabulate

import  util
from fin import technical_indicator as ta
from util import date_getspecificdate

DIRCWD=  'D:/_devs/Python01/project27/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
__path__= DIRCWD +'/aapackage/'



############## Data List #######################################################################
#runfile('D:/_devs/Python01/aapackage/alldata.py', wdir='D:/_devs/Python01/project27')
#  Get the list of Tickers
execfile(DIRCWD+'/aapackage/alldata.py')


def data_jpsector():
 #jpsector= pd.read_csv('E:/_data/stock/histo/jp_sector.csv')
 #util.save_obj(jpsector,'jpsector')
 print 'jpsector'
 return util.load_obj('jpsector')




#########################Date manipulation ##########################################        
def date_earningquater(t1):
 if (t1.month==10 and t1.day >= 14) or (t1.month==1 and t1.day < 14) or t1.month in [11,12] :
    if t1.month in [10,11,12] : qdate= datetime.datetime(t1.year+1 ,1,14)
    else : qdate= datetime.datetime(t1.year, 1,14)
    quater= 4
    
 if (t1.month==1 and t1.day >= 14) or (t1.month==4 and t1.day < 14) or t1.month in [2,3] :
    qdate= datetime.datetime(t1.year ,4,14); quater= 1
    
 if (t1.month==4 and t1.day >= 14) or (t1.month==7 and t1.day < 14) or t1.month in [5,6] :
    qdate= datetime.datetime(t1.year ,7,14); quater= 2    

 if (t1.month==7 and t1.day >= 14) or (t1.month==10 and t1.day < 14) or t1.month in [8,9] :
    qdate= datetime.datetime(t1.year ,10,14) ; quater= 3
      
 nbday= (qdate-t1).days;  
 return quater, nbday, qdate


def date_is_3rdfriday(s):
    d = datetime.strptime(s, '%b %d, %Y')
    return d.weekday() == 4 and 14 < d.day < 22


def date_option_expiry(date):
    #day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
    if date.day <= day :
        nbday= day-date.day
        datexp= datetime.datetime(date.year, date.month, day)
    else :
        if date.month== 12 :
         day = 21 - (calendar.weekday(date.year+1, 1, 1) + 2) % 7
         datexp= datetime.datetime(date.year+1, 1 , day)
        else :
         day = 21 - (calendar.weekday(date.year, date.month+1, 1) + 2) % 7
         datexp= datetime.datetime(date.year, date.month+1 , day)          

        nbday= (datexp - date).days          
   
    return nbday, datexp

def date_find_kday_fromintradaydate(kintraday, intradaydate, dailydate) :
   return util.np_find(datetime.date(intradaydate[kintraday]), dailydate) 
  
  
def date_find_kintraday_fromdate(d1, intradaydate1, h1=9, m1=30) :
  
   d1= datetime.datetime(d1.year, d1.month, d1.day, h1, m1)
   return util.np_find( d1, intradaydate1) 
   
   
   
def date_find_intradateid( datetimelist, stringdate=['20160420223000']) :
 for t in stringdate :
   tt= datestring_todatetime(t)
   k= util.np_find(tt, datetimelist)
   print str(k)+',', tt
   
#kday= date_find_kday_fromintradaydate(k, spdateref2, spdailyq.date) 




def datetime_convertzone1_tozone2(tt, fromzone='Japan', tozone='US/Eastern') :
 import pytz, dateutil
 tz = pytz.timezone(fromzone)
 tmz= pytz.timezone(tozone)
 
 if type(tt)==datetime : 
    localtime = tz.localize(tt).astimezone(tmz)
    return dateutil.parser.parse(localtime.strftime("%Y-%m-%d %H:%M:%S"))
 
 t2=[]    
 for t in tt:    
   localtime = tz.localize(t).astimezone(tmz)
   t2.append(dateutil.parser.parse(localtime.strftime("%Y-%m-%d %H:%M:%S")))
 return t2


 #--------- Find Daily Open / Close Time  --------------------------------------
def date_extract_dailyopenclosetime(spdateref1, market='us') :
 if market=='us' : 
  topenh= 9;  topenm=30;   tcloseh= 16;  tclosem=00
  spdailyopendate= []
  for k,t in enumerate(spdateref1) :
    if t.hour== topenh and t.minute== topenm :     spdailyopendate.append(k)
  spdailyopendate= np.array(spdailyopendate)

  spdailyclosedate= []
  for k,t in enumerate(spdateref1) :
    if t.hour== tcloseh and t.minute== tclosem  :   spdailyclosedate.append(k)
  spdailyclosedate= np.array(spdailyclosedate)
  
  return  spdailyopendate, spdailyclosedate
 else:
   return None, None   
   



def date_finddateid(date1, dateref) :
  i= util.np_findfirst(date1, dateref)
  if i==-1 : i= util.np_findfirst(date1+1, dateref) 
  if i==-1 : i= util.np_findfirst(date1-1, dateref)     
  if i==-1 : i= util.np_findfirst(date1+2, dateref)    
  if i==-1 : i= util.np_findfirst(date1-2, dateref) 
  if i==-1 : i= util.np_findfirst(date1+3, dateref)    
  if i==-1 : i= util.np_findfirst(date1-3, dateref) 
  if i==-1 : i= util.np_findfirst(date1+5, dateref)    
  if i==-1 : i= util.np_findfirst(date1-5, dateref)  
  if i==-1 : i= util.np_findfirst(date1+7, dateref)    
  if i==-1 : i= util.np_findfirst(date1-7, dateref)  
  return i


def date_alignfromdateref(array1, dateref):  #2 column array time, data
 #--------Align the array1= date/raw  with same date than dateref---------------------
 masset, _ = np.shape(array1) 
 tmax= len(dateref)
 close= np.zeros(( masset,tmax), dtype="float32")

 for k in range(1, masset):
#  df= array1[k]
  for t in range(0,tmax):
      ix= (np.argwhere(array1[0,:]== np.float(dateref[t])))  #Find the date index
      if len(ix) > 0  :  #If found
        ix1= ix[0,0]
        close[k,t]= array1[k,ix1]  #Update the close value
        close[0,t]= array1[0,ix1]  #Update the date
      
 #Use Previous values to fill Zero value
 for k in range(1, masset):
  for t in range(0,tmax):
   if close[k,t] == 0 : close[k,t] = close[k,t-1]

 return close
 


@jit(numba.float32[:](numba.int32[:], numba.int32[:], int32, numba.float32[:]))
def _date_align(dateref,datei,tmax, closei ) :
  close2= np.zeros(tmax, dtype=np.float16)
  for t in range(0,tmax):
      ix= (np.argwhere(datei == dateref[t]))  #Find the date index
      if len(ix) > 0  :  #If found
        ix0= ix[0]
        close2[t]= closei[ix0]  #Update the close value          
        
  for t in range(0,tmax):
    if close2[t] == 0 : close2[t] = close2[t-1]
  return close2


#@jit
def date_align(quotes, dateref=None, datestart=19550101, type1="close"):
 ''' #Aligne the price with the same dates date	year	month	day	d	open	close	high	low	volume	aclose '''
 df0= quotes[0]
 if dateref is None:  dateref= np.array(df0.date.values)
   
 isnotint1= not isint(dateref[0]) 
 if isnotint1 : dateref= np.array(util.datetime_toint(dateref))
 if datestart != 19550101:  dateref= dateref[dateref > datestart]
   
 masset= len(quotes); tmax= len(dateref)
 close= np.zeros(( masset,tmax), dtype="float16")
   
 print("Period: ", dateref[0], dateref[-1], len(dateref)) 
 for k in range(0, masset):
   df= quotes[k] ;  
   priceid= util.find(type1, df.columns)   # close price column
   closei= df.iloc[:,priceid].values       # !!!!! Otherwise 2  Columns Time Series
   if not isint(df.date.values[0]) : datei=  np.array(util.datetime_toint(df.date.values))
   else :                            datei=  np.array(df.date.values)
   #  print dateref, datei, tmax, closei
   close[k,:]=   _date_align(dateref,datei,tmax, closei )
 return np.array(close, dtype=np.float32), dateref


'''
  for t in range(0,tmax):
      ix= (np.argwhere(datei == dateref[t]))  #Find the date index
      if len(ix) > 0  :  #If found
        ix1= ix[0,0]
        close[k,t]= df.iloc[ix1,priceid]  #Update the close value
  #        close[k,t]= closei[ix[0,0]]  #Update the close value
        
  #Use Previous values to fill Zero value
  for k in range(0, masset):
  for t in range(0,tmax):
      if close[k,t] == 0 : close[k,t] = close[k,t-1]
 '''

# if   type(dateref[0]) == float :           dateref2=  np.array(dateref)
# elif type(dateref[0]) == np.datetime64  :  dateref2=  datenumpy_todatetime(dateref)
# else:  dateref2= dateref
 # else :   dateref2= np.array(datetime_toint(dateref))
# return dateref, datei

 
 

import operator
def min_withposition(values):
 min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
 return min_index, min_value
 
def max_withposition(values):
 max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
 return max_index, max_value
############################################################################## 


def _reshape(x):
  if len(np.shape(x)) < 2 :  return  np.reshape(x, (1, np.size(x))) 
  else : return x

def _notnone(x):  return not x is None

##################################################################################
#----------------- Plotting functions------------------------------------------------
def plot_price(asset,y2=None,y3=None,y4=None,y5=None, sym=None, savename1='', tickperday=20, date1=None,  graphsize=(10,5), label=('title','Time','Y'), legendloc="upper left", dpi=150, isband=0) :
 asset= _reshape(asset) 
 if _notnone(y2) :  asset= np.concatenate((asset, _reshape(y2) )) 
 if _notnone(y3) :  asset= np.concatenate((asset, _reshape(y3) ))    
 if _notnone(y4) :  asset= np.concatenate((asset, _reshape(y4)) )  
 if _notnone(y5) :  asset= np.concatenate((asset, _reshape(y5) ))        

 sh= np.shape(asset)   
 if sh[0]==1 :
    masset= 1; tmax = sh[1]
    tt1= np.arange(0, tmax-1)
    
    try :
     lband= np.zeros(tmax)
     res=  ta_lowbandtrend1(asset[0,tt1],0)
     lband= res.x[0] * (tt1) + res.x[1]

     hband= np.zeros(tmax)
     res=  ta_highbandtrend1(asset[0,tt1],0)
     hband= res.x[0] * (tt1) + res.x[1]
    except: pass
  
    plt.figure(figsize=(10,5))
    plt.plot(tt1,asset[0,tt1], color='b') # plotting t,b separately 

    plt.grid(True);  plt.title(label[0]); plt.xlabel(label[1]);  plt.ylabel(label[2])  

    if isband :
     plt.plot(tt1,lband, color='r')    
     plt.plot(tt1,hband, color='g') 
    
    tick_locs = np.arange(0, tmax, tickperday)  #58 ticks for 1 day, 5min data
    if date1 is None :
        tick_lbls = np.arange(0, int(tmax/tickperday),1)
    else :
        tick_lbls = date1[np.arange(0, tmax, tickperday)]      
    plt.xticks(tick_locs, tick_lbls)    

    if len(savename1) > 1 :   
      plt.savefig(savename1, dpi=150,  bbox_inches='tight')
    else :      plt.show()

      

 else:
   masset= sh[0]; tmax = sh[1]
   tt1= np.arange(0, tmax,1)

   if sym==None : sym= np.arange(0,masset)      
      
   fig= plt.figure(figsize=(10,5),  dpi=300)
   color=['b','g', 'r','c','m','y','k', '0.5', '0.75', '0.25', '0.66', '0.33' ]
   for k in range(0, min(12,masset)):
     plt.plot(tt1,asset[k,:], color=color[k], label=sym[k]) # plotting t,b separately 
#     
     #fig= plt.gcf()     
   fig.savefig('plot1.png')
     

   tick_locs = np.arange(0, tmax, tickperday)  #58 ticks for 1 day, 5min data
   
   if date1 is None : tick_lbls = np.arange(0, int(tmax/tickperday),1)
   else :  tick_lbls= date1[np.arange(0, int(tmax),tickperday)]
   plt.xticks(tick_locs, tick_lbls)

   plt.legend(loc=legendloc) #lower left
   plt.grid(True);  plt.title(label[0]); plt.xlabel(label[1]);  plt.ylabel(label[2])  
   #plt.show()



   if len(savename1) > 1 :   
      plt.savefig(savename1, dpi=dpi,  bbox_inches='tight')

 



 ################### Plot Intraday Cont  
def plot_priceintraday(data) :
 from matplotlib.finance import candlestick
 from matplotlib.dates import num2date

 # data in a text file, 5 columns: time, opening, close, high, low
 # note that I'm using the time you formated into an ordinal float
 data = np.loadtxt('finance-data.txt', delimiter=',')

 # determine number of days and create a list of those days
 ndays = np.unique(np.trunc(data[:,0]), return_index=True)
 xdays =  []
 for n in np.arange(len(ndays[0])):
    xdays.append(datetime.date.isoformat(num2date(data[ndays[1],0][n])))

 # creation of new data by replacing the time array with equally spaced values.
 # this will allow to remove the gap between the days, when plotting the data
 data2 = np.hstack([np.arange(data[:,0].size)[:, np.newaxis], data[:,1:]])

 # plot the data
 fig = plt.figure(figsize=(10, 5))
 ax = fig.add_axes([0.1, 0.2, 0.85, 0.7])
    # customization of the axis
 ax.spines['right'].set_color('none')
 ax.spines['top'].set_color('none')
 ax.xaxis.set_ticks_position('bottom')
 ax.yaxis.set_ticks_position('left')
 ax.tick_params(axis='both', direction='out', width=2, length=8,
               labelsize=12, pad=8)
 ax.spines['left'].set_linewidth(2)
 ax.spines['bottom'].set_linewidth(2)
    # set the ticks of the x axis only when starting a new day
 ax.set_xticks(data2[ndays[1],0])
 ax.set_xticklabels(xdays, rotation=45, horizontalalignment='right')

 ax.set_ylabel('Value', size=20)
 ax.set_ylim([177, 196])

 candlestick(ax, data2, width=0.5, colorup='g', colordown='r')
 plt.show()


#-----Plot Check Price
def plot_check(close,tt0i=20140102, tt1i=20160815, dateref=[], sym=[], tickperday=120) :
  t0i,t1i= util.find(tt0i, dateref),util.find(tt1i, dateref)
  print 'backtest period ', tt0i,tt1i, t0i, t1i
  close_ret= getret_fromquotes(close,1)
  price= price_normalize100(close_ret[:,t0i:t1i])
  plot_price(price[:,:], date1=dateref[t0i:t1i+1], tickperday=tickperday, sym=sym)
 


############## Plot Date, Label and Using Pandas  
def plot_pricedate(date1, sym1, asset1, sym2=None, bsk1=None, verticaldate=None, savename1='', graphsize=(10,5), tickperday=5) :
  if type(date1[0]) == str  : date1= datestring_todatetime(date1)
    
  if bsk1 != None :  
    asset1= np.row_stack((asset1,bsk1))
    sym1= np.concatenate((sym1, sym2))

  df_asset= array_todataframe(asset1, sym1, date1) 

  ax= df_asset.plot(figsize=graphsize)
  if verticaldate != None : #Create vertical date 
    for vdate in verticaldate:
      ax.axvline(pd.to_datetime(verticaldate), color='r', linestyle='--', lw=2)

  tmax= len(date1)
  
  
  tick_locs = np.arange(0, tmax, tickperday)  #58 ticks for 1 day, 5min data
  tick_lbls = np.arange(0, int(tmax/tickperday),1)

#  ax.set_xticks(tick_locs)
#  ax.set_xticklabels(tick_lbls, rotation='vertical')

#  plt.legend(loc='upper left')
#  plt.grid(True);   
 # plt.show()


  if len(savename1) > 1 :   ax.savefig(savename1, dpi=150,  bbox_inches='tight')
 



 # http://bokeh.pydata.org/en/latest/docs/gallery/stocks.html
def  showprice_javascript() :
 from bokeh.plotting import figure, show, output_file, vplot
 from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT

 def datetime(x):
    return np.array(x, dtype=np.datetime64)

 p1 = figure(x_axis_type = "datetime")
 p1.title = "Stock Closing Prices"
 p1.grid.grid_line_alpha=0.3
 p1.xaxis.axis_label = 'Date'
 p1.yaxis.axis_label = 'Price'

 p1.line(datetime(AAPL['date']), AAPL['adj_close'], color='#A6CEE3', legend='AAPL')
 p1.line(datetime(GOOG['date']), GOOG['adj_close'], color='#B2DF8A', legend='GOOG')
 p1.line(datetime(IBM['date']), IBM['adj_close'], color='#33A02C', legend='IBM')
 p1.line(datetime(MSFT['date']), MSFT['adj_close'], color='#FB9A99', legend='MSFT')

 aapl = np.array(AAPL['adj_close'])
 aapl_dates = np.array(AAPL['date'], dtype=np.datetime64)

 window_size = 30
 window = np.ones(window_size)/float(window_size)
 aapl_avg = np.convolve(aapl, window, 'same')

 p2 = figure(x_axis_type="datetime")
 p2.title = "AAPL One-Month Average"
 p2.grid.grid_line_alpha = 0
 p2.xaxis.axis_label = 'Date'
 p2.yaxis.axis_label = 'Price'
 p2.ygrid.band_fill_color = "olive"
 p2.ygrid.band_fill_alpha = 0.1

 p2.circle(aapl_dates, aapl, size=4, legend='close',
          color='darkgrey', alpha=0.2)

 p2.line(aapl_dates, aapl_avg, legend='avg', color='navy')

 output_file("stocks.html", title="stocks.py example")

 show(vplot(p1,p2))  # open a browser






####-----Create Vertical Separator for Graph ---------------------------------
def generate_sepvertical(asset1, tt, tmax, start=None, datebar=None) :
 vv=  np.zeros((1,(tmax)))
 if datebar != None:
   tt= datestring_todatetime(datebar) - datestring_todatetime(start)
   tt= tt.days

 vv[0,tt]=np.max(asset1)+5
 return vv

##############################################################################



##############################################################################
#----------------- Conversion I/O from file------------------------------------------
def save_asset_tofile(file1,asset1, asset2=None, asset3=None, date1=None, title1=None) :
  #Save the asset together, data must be aligned
  masset1, tmax1= np.shape(asset1)
  tmax2=0; tmax3=0; masset2=0; masset3=0
  if asset2 != None  :
    shape2= np.shape(asset2)
    if len(shape2)==1 : masset2=1; tmax2= shape2[0]
    else:  masset2=shape2[0]; tmax2= shape2[1]
    asset2= np.reshape(asset2,(masset2, tmax2))

  if asset3 != None :
    shape3= np.shape(asset3) 
    if len(shape3)==1 : masset3=1; tmax3= shape3[0]
    else:  masset2=shape3[0]; tmax2= shape3[1]
    asset3= np.reshape(asset2,(masset3, tmax3))

  if (asset2!= None and tmax1 != tmax2) or (asset3 != None and tmax1 != tmax3) :
    print("Time are not the same"); exit

  assetall= np.zeros((1+masset1+masset2+masset3, tmax1))
  if date1 != None: 
    for t in range(0,tmax1):      assetall[0,t]= int(date1[t])   #Date

  for t in range(0,tmax1):  
    for k in range(0, masset1):   assetall[1+k,t]= asset1[k,t]   #1st Asset
      
    if asset2 != None:
     for k in range(0, masset2):  assetall[1+masset1+k,t]= asset2[k,t]
        
    if asset3 != None:
     for k in range(0, masset3):  assetall[1+masset2+k,t]= asset3[k,t]        

  if title1 != None:   table1= np.row_stack((title1, assetall.T))
  else :               table1= assetall.T
  
  np.savetxt(file1, table1,  delimiter=",",  fmt="%s")  #format



def load_asset_fromfile(file1) :
 return (np.loadtxt(file1, skiprow=0, delimiter=",")).T


def array_todataframe(price, symbols=None, date1=None):
   sh= np.shape(price)
   if len(sh) > 1 :
     if sh[0] < sh[1] :   #  masset x time , need Transpose
         return pd.DataFrame(data= price.T, index= date1,  columns=symbols)  
     else :
         return pd.DataFrame(data= price.T, index= date1,  columns=symbols)  
   else :
     return pd.DataFrame(data= price, index= date1,  columns=symbols)


def pd_dataframe_toarray(df):
  date1= df.index
  array1= (df.reset_index().values)[1:,:]
  column_name= df.columns
  return column_name, date1, array1
   
      
##############################################################################
#------------  Util -----------------------------------------------------------------
def isfloat(value):
  try:    
   float(value)
   if value == np.inf : return False
   else : return True
  except :    return False      
      
      
def isint(x): return isinstance(x, ( int, long, np.int, np.int64, np.int32 ) )
      



##############################################################################
#------------Correlation / CoVariance / Regression-----------------------------------
def correlation_mat(Xmat, type1="robust", type2="correl") :
 from sklearn.covariance import MinCovDet
 from sklearn.covariance import OAS
 
 x = Xmat.copy().T
 #x= (x - x.mean(axis=0)) / x.std(axis=0) 
 
 if type2== "correl" :
   x= (x - x.mean(axis=0)) / x.std(axis=0)     #Z scores scalling, STD_dev
   # x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))   # Min-Max scaling
 #Otherwise this covariance

 if type1=="empirical" :
    correl= np.corrcoef(x) 
    correl_inv=0; partial_correl=0

 if type1=="empirical2" :
    model= sk.covariance.EmpiricalCovariance(assume_centered=True).fit(x)  
    correl= model.covariance_; correl_inv= model.precision_
 
 if type1== "spearman" :
     correl= sci.stats.spearmanr(x, axis=0).correlation
     correl_inv=0; partial_correl=0
     
 if type1== "robust" :
    model= MinCovDet(assume_centered=True).fit(x)
    correl= model.covariance_; correl_inv= model.precision_
    
 if type1== "shrinkage" :
    model= OAS(assume_centered=True).fit(x)  
    correl= model.covariance_; correl_inv= model.precision_

 if type1== "lasso" :  #Sparse Matrix
    model= sk.covariance.GraphLassoCV().fit(x)
    correl= model.covariance_; correl_inv= model.precision_
    
# print type1, type2
 if not type1 in ["empirical", "spearman"] :
  partial_correl = correl_inv.copy()
  d = 1 / np.sqrt(np.diag(partial_correl))
  partial_correl *= d; partial_correl *= d[:, np.newaxis]
 
 return correl, correl_inv, partial_correl
 
 
#correl, correl_inv, correl_partial= correlation_mat(ret_close1, type1="lasso")  
 

def correl_reducebytrigger(correl2, trigger):
  ''' Put Zero below trigger  '''
  correl = correl2.copy()
  n,m= correl.shape
  for i in range(0,n) :
     for j in range(0,m) :
         if abs(correl[i,j]) < trigger: correl[i,j]= 0

  return correl
   

 
def sk_cov_fromcorrel(correl, ret_close1): 
 volx= volhisto_fromret(ret_close1, tt, tt, axis=1)
 volxx= np.diag(volx) 
 cov= np.dot(np.dot(volxx, correl), volxx)
 return cov
 
def cointegration(x,y) :
  """ Output :
    coint_t :t-statistic of unit-root test on residuals
    pvalue :   MacKinnon's approximate p-value based on MacKinnon (1994)
    crit_value : Critical values for the test statistic at the 1 %, 5 %, and 10 %  Signifiance levels.

    The Null hypothesis is that there is no cointegration, the alt hypothesis is that there is cointegrating relationship. 
    If the pvalue is small, below a critical size:  Reject Null Hypothesis -->  Cointegration
           pvalue is high --> No Cointegration    
    
    Significance level, also denoted as alpha or α, is the probability of rejecting  the null hypothesis when it is true.
    Probability of incorrectly rejecting a true null hypothesis

    P=0.05  , At least 23% (and typically close to 50%)
    P=0.01  , At least 7% (and typically close to 15%

    (-2.9607012342275936,
    0.038730981052330332, 0,
    249,
    {'1%': -3.4568881317725864,
       '10%': -2.5729936189738876,
     '5%': -2.8732185133016057},
      601.96849256295991)
        It can be seen that the calculated test statistic of -2.96 is smaller than the 5% critical value of -2.87, 
        which means that we can reject the null hypothesis that  there isn't a cointegrating relationship at the 5% level.
  """

  import statsmodels  as sm
  from statsmodels.tsa.stattools import  coint
  
  res= sm.tsa.stattools.coint(x, y, regression="c")
  res2= np.zeros((3,2), dtype=np.object)
  res2[0,0]=  "Proba_CoInte_99%_confid";  res2[0,1]= (True if  res[0] <  res[2][0] else False )  #Proba
  res2[1,0]=  "Proba_CoInte_95%_confid";  res2[1,1]= (  res[0] <  res[2][1] )  #Proba
  res2[2,0]=  "Proba_CoInte_90%_confid";  res2[2,1]= (  res[0] <  res[2][2] )  #Proba
  return res2
  

def causality_y1_y2(price2,price1, maxlag):
  # http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.tsa.stattools.grangercausalitytests.html
 import statsmodels  as sm
 from statsmodels.tsa.stattools import  grangercausalitytests
 res= sm.tsa.stattools.grangercausalitytests(price1, price2, maxlag)
 return res
 
 
def reg_slope(close, dateref, tlag, type1='elasticv') :
 slope= np.ones((len(dateref),3), dtype= np.float32)
 slope[0:tlag,0]= dateref[0:tlag]
 for t0 in xrange(tlag,len(dateref)) :
  #  print t0,   len(close), len(dateref)
  res= regression(close[(t0-tlag):(t0)], np.arange(0,(tlag)), type1=type1 )
  slope[t0,0]= dateref[t0]
  slope[t0,1]= res[0][0]
  slope[t0,2]= res[1]
  
 return slope  



#----------Mearn Reversion Co-Integeration ----------------------------------
def rolling_cointegration(x,y):
 import statsmodels as sm
 import statsmodels.api as sm
 data = sm.datasets.macrodata.load_pandas().data
 def rolling_coint(x, y):
    yy = y[:len(x)]  
    return sm.tsa.coint(x, yy)[1]    # returns only the p-value

 historical_coint = pd.expanding_apply(data.realgdp, rolling_coint, min_periods=36, args=(data.realdpi,))
#--------------------------------------------------------------------------- 




 #-----Multivariate regression ---------------------------------------------
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
def regression(yreturn, xreturn, type1="elasticv"):
 '''Y = X* reg.coef_  +  reg.intercept_, r2 '''
 
 if len(np.shape(yreturn)) < 2 : yreturn= yreturn.ravel()
 if len(np.shape(xreturn)) < 2 : xreturn= xreturn.reshape(-1,1) 
 
 if type1=="elasticv" :
   reg= sk.linear_model.ElasticNetCV()
   reg.fit(xreturn, yreturn)    #Close1 = F(close2)
   r2= reg.score(xreturn, yreturn)
   return reg.coef_  ,  reg.intercept_, r2

     
 if type1=="ridgecv" :
   reg= sk.linear_model.RidgeCV()
   reg.fit(xreturn, yreturn)    #Close1 = F(close2)
   r2= reg.score(xreturn, yreturn)
   return reg.coef_  ,  reg.intercept_, r2     
     

 if type1=="linear" :
   reg= sk.linear_model.LinearRegression()
   reg.fit(xreturn, yreturn)    #Close1 = F(close2)
   r2= reg.score(xreturn, yreturn)
   return reg.coef_  ,  reg.intercept_, r2     
     

'''
regression(ret_close1[assetk,:].T,  ret_close2.T, type1="elasticv")   
   
 y= (ret_spy[0,0:tmax]).T

x= np.reshape((ret_close2[0,0:tmax]).T, (tmax,1))
regression( (ret_spy[0,0:tmax]).T, x, type1="elasticv")   
   
'''

def regression_fixedsymbolstock(sym, ret_close2, tsstart, tsample, ret_spy, spyclose, regonly=True) :
 ''' Price, Weight, R2, Error  '''
 nsym= len(sym)
 x0=np.reshape((ret_close2[sym[0],tsstart:tsample]).T, (tsample-tsstart,1))
 for k in range(1, nsym) :
    xnew= np.reshape((ret_close2[sym[k],tsstart:tsample]).T, (tsample-tsstart,1))
    x0= np.column_stack((x0,xnew))

 y= (ret_spy[0, tsstart:tsample]).T

 reg1= regression( y, x0, type1="elasticv")   
 ww01=reg1[0]; r2= reg1[2] #Regression Weights

 if regonly : return 0,ww01,r2,0
 else :
   reg01= sym; 
   tahead= 100 #Error of [0,100] days
   spyreg, retreg= getpriceret_fromregression(spyclose, ww01, reg01, ret_close2, tsample)
   err= np.max(np.abs(spyclose[0, tsample:tsample+tahead]-spyreg[0, tsample:tsample+tahead]))

   return spyreg, ww01, r2,  err


def regression_getpricefromww(spyclose, ww01, regasset01, ret_close2, tstart, tlag=1) :
 spyreg= np.copy(spyclose)
 _,tmax=np.shape(spyreg); tmax= tmax-2
 
 retreg= np.zeros((1, tmax))

 #Return Calc
 for t in range(0, tmax):
  ret=0
  for k in range(0, len(regasset01)):
    ret+= ww01[k] * ret_close2[regasset01[k], t]
  retreg[0,t]= ret  

 #Price calc
 for t in range(tstart, tmax-tlag):   
  spyreg[0,t+tlag]= spyreg[0,t]*(1+retreg[0,t])    
    
 return spyreg, retreg   



def regression_allstocks_vs_riskfactors(symstock, pricestock, symriskfac,  priceriskfac, nlaglist ) : 
 '''Make Regression on all stocks ''' 
 _,tmax= np.shape(pricestock) 
 nrisk,tmax2= np.shape(priceriskfac)
 
 if tmax != tmax2 : 
   print('Error Tmax Stock:'+str(tmax) +' !=' + 'Tmax Risk:'+str(tmax2) ); exit
 sym_riskid= np.arange(0,nrisk) 

 open1adj= pricestock
 close2adj= priceriskfac
 stocklist01= symstock
 symriskfac= symriskfac  
 del  symstock, pricestock,   priceriskfac
 
 strsymriskfac= ','.join(symriskfac) 
 statsjp= np.empty((len(stocklist01), 5, len(nlaglist)), dtype=np.object)
 
 #Pandas Part
 colname= ['ticker', 'lag', 'R2']; colname= colname + symriskfac
 arr= np.empty(( len(stocklist01)*len(nlaglist), 3 + len(symriskfac) ), dtype=np.object)
 index1= np.arange(0,len(stocklist01)*len(nlaglist) )

 jj=0
 for ilag,nlag in enumerate(nlaglist) :  #Time lag 
   ret_open1=  getlogret_fromquotes(open1adj,nlag)
   ret_close2= getlogret_fromquotes(close2adj,nlag)
   _,tmax= np.shape(ret_open1)  

   #------------- Custom regression over all the stock  --------------------------------
   for k in range(0, len(stocklist01)):
     #try :
       ret_stockjp= np.reshape(ret_open1[k,:],(1,len(ret_open1[k,:])))
       _, ww01, r2, _= regression_fixedsymbolstock(sym_riskid, ret_close2, 0, tmax,
                                                      ret_stockjp, None, regonly=True) 

       statsjp[k,0,ilag]= stocklist01[k]      
       statsjp[k,1, ilag]= r2
       statsjp[k,2, ilag]= nlag;               
       statsjp[k,3, ilag]= ww01
       statsjp[k,4, ilag]= strsymriskfac
       
       arr[jj,0]= stocklist01[k]      
       arr[jj,1]=  nlag     
       arr[jj,2]=  r2
       arr[jj,3:]=  ww01
       jj+=1
       
     #except : print('Error, '+ str(k) + ','+ str(stocklist01[k]))
   statsjp[:,:,ilag]= util.np_sortbycolumn(statsjp[:,:,ilag], 1, asc=False) #Sort by R2

   df= util.pd_array_todataframe(arr, colname, index1)

 util.print_object(statsjp[0:10,0:2,0],'')
 return statsjp, df

'''
sym_riskid= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17]
sym01= nk400list
stocklist01= sym01
statsjp= np.empty((len(stocklist01), 4,5), dtype=np.object)

for ilag,nlag in enumerate([1,5,21,42,58]) :  #Time lag 
 ret_open1=  getdailylogret_fromquotes(open1adj,nlag)
 ret_close2= getdailylogret_fromquotes(close2adj,nlag)


#------------- Custom regression over all the stock  --------------------------------
 for k in range(0, len(stocklist01)):
  #try :
    k_jpstock= k
    ret_stockjp= np.reshape(ret_open1[k_jpstock,:],(1,len(ret_open1[k_jpstock,:])))
    stockclose= np.reshape(open1adj[k_jpstock,nlag:],(1,len(open1adj[k_jpstock,nlag:])))
 
    _, ww01, r2, _= regression_fixedsymbolstock(sym_riskid, ret_close2, 0, 280,
                                                   ret_stockjp, stockclose, regonly=True) 

    statsjp[k,0,ilag]= stocklist01[k]
    statsjp[k,1, ilag]= r2
    statsjp[k,2, ilag]= nlag
    statsjp[k,3, ilag]= ww01
 # except : print('Error, '+ str(k) + ','+ str(stocklist01[k]))

  #statsjp= util.np_sortbycolumn(statsjp, 1, asc=False)
  #util.print_object(statsjp[:,0:2],'')

 print(nlag)
'''


###############################################################################



    
##############################################################################
#----------------- Risk calculation ------------------------------------------
def getdiff_fromquotes(close, timelag):
   sh= np.shape(close)[1]
   return close[:,timelag:(sh)] - close[:, 0:(sh-timelag)] 
   
def getret_fromquotes(close, timelag=1):
   sh= np.shape(close)
   if len(sh) > 1 :
     sh= sh[1]
     return close[:,timelag:(sh)] / close[:, 0:(sh-timelag)] - 1 
   else :
     sh= sh[0]
     return close[timelag:(sh)] / close[0:(sh-timelag)] - 1 
    
    

def getlogret_fromquotes(close, timelag=1):
   sh= np.shape(close)[1]
   return np.log(close[:,timelag:(sh)] / close[:, 0:(sh-timelag)] )


def getprice_fromret(ret, normprice=100):
 # Generate Price from Return  to 100
 masset, tmax = np.shape(ret)
 asset= np.zeros((masset,tmax+1), dtype="float32")
 #for k in range(0,masset) : asset[k,0]= normprice:
 asset[:,0]= normprice

 for t in range(1,tmax+1) :
  for k in range(0,masset) :  asset[k,t]= asset[k,t-1] * (1+ret[k,t-1])
 return asset

 
def price_normalize100(ret, normprice=100):
 # Normalize Price  to 100
 sh= np.shape(ret)
 if len(sh) < 2 : 
   masset=1; tmax = sh[0]
   ret= np.reshape(ret,(1,tmax))
 else :  masset, tmax = np.shape(ret)
 
 asset= np.zeros((masset,tmax+1), dtype=np.float32)
 for k in range(0,masset) : asset[k,0]= normprice
        
 asset[:,1:]= 1+ret[:,:]     
 asset= np.cumprod(asset, axis = 1)  #exponen product st = st-1 *st    

# for t in range(1,tmax+1):
#  for k in range(0,masset) :
#    asset[k,t]= asset[k,t-1] * (1+ret[k,t-1])
 return asset


def price_normalize_1d(ret, normprice=100, dtype1= np.float32):
 # 1D Vector normalize to 100
 asset= np.empty((len(ret)+1), dtype= dtype1)
 asset[0]= normprice        
 asset[1:]= 1+ret[:]     
 asset= np.cumprod(asset, axis= 0)  #exponen product st = st-1 *st    
 return asset


def norm_fast(y, ny):
  my= ne.evaluate("sum(y)")
  y-= my/nx
  vy= ne.evaluate("sum(y*y)")
  return y/(ny*np.sqrt(vy))



def correl_fast(xn,y,nx):
  my= ne.evaluate("sum(y)")
  y-= my/nx
  vy= ne.evaluate("sum(y*y)") 
  crr= ne.evaluate("sum(xn * y)")
  return crr / (nx*np.sqrt(vy))


def volhisto_fromret(retbsk,t, volrange, axis=0):
 if len(np.shape(retbsk)) > 1 :
    return np.std(retbsk[:,(max(0,t-volrange)):(t)], axis=axis) * 15.8745078664 
 else :
    return np.std(retbsk[(max(0,t-volrange)):(t)], axis=axis) * 15.8745078664    
   

def volhisto_fromprice(price,t, volrange, axis=0):
  if len( np.shape(price)) > 1 :
    retbsk= price[:,1:-1] / price[:,0:-2] - 1 
    return np.std(retbsk[:,(max(0,t-volrange)):(t)], axis=axis) * 15.8745078664 
  else :
    retbsk= price[1:-1] / (0.001+price[0:-2]) - 1 
    return np.std(retbsk[(max(0,t-volrange)):(t)], axis=axis) * 15.8745078664    
   
   
def volhistorolling_fromprice(price, volrange):
  sh= np.shape(price)
  if len( sh) > 1 :
    masset, tmax= sh[0], sh[1]
    volhisto= np.zeros((masset, tmax))
    for t in xrange(volrange, tmax):
      volhisto[:,t]= volhisto_fromprice(price,t, volrange, axis=0)
  else :
    tmax= sh[0]
    volhisto= np.zeros(( tmax))
    for t in xrange(volrange, tmax):
      volhisto[t]= volhisto_fromprice(price,t, volrange, axis=0)
      
  return volhisto




#Technical Indicator on Daily return  
def rsk_calc_all_TA(df='panda_dataframe') :
  '''Add All TA RMI, RSI To the '''
 #try :
  df= ta.MA(df,200); df= ta.MA(df, 50);  df= ta.MA(df, 20);   df= ta.MA(df, 5);   df= ta.MA(df, 3)
  df= ta.distance(df, "MA_200");   df= ta.distance(df, "MA_50");    df= ta.distance(df, "MA_20");    df= ta.distance(df, "MA_5") ;     df= ta.distance(df, "MA_3");

  df= ta.RET(df, 1);  df= ta.RET(df, 2);   df= ta.RET(df, 3);  df= ta.RET(df, 5);  df= ta.RET(df, 20); df= ta.RET(df, 60)

  df= ta.RMI(df, 14,10);  df= ta.RMI(df, 7,5)
  df= ta.STDDEV(df, 60);  df= ta.STDDEV(df, 120)  
 
  df= ta.nbday_low(df,252);    df= ta.nbday_high(df,252)   
  df= ta.RSI(df,14);  
  
  df= ta.qearning_dist(df)  ;  df= ta.optionexpiry_dist(df)
  
  '''     
  df= ta.MACD(df, 12, 26)
  df= ta.CCI(df, 14)
  df= ta.STO(df, 14 )  #Stochastic 
  df= ta.ADX(df, 14, 3)  #ADX

  df= ta.PPSR(df)
  df= ta.Vortex(df, 14)
  df= ta.TSI(df, 20, 5)
  df= ta.ACCDIST(df, 20)
  df= ta.Chaikin(df)
  df= ta. MFI(df, 7)
  df= ta.OBV(df, 5)
  df= ta.FORCE(df,10)
  df= ta.EOM(df, 14)
  df= ta.COPP(df, 14)
  df= ta.KELCH(df, 14)
  df= ta.ULTOSC(df)
  df= ta.DONCH(df, 14)
 #except: pass 
  '''
  return df


def ta_lowbandtrend1(close2, type1=0) :
  '''Get lower band trend '''
  def linearreg2(ww):
    v= vmax1 -  ww[0]*vmax0  -   ww[1]  
    v2= np.array([ 1000000 if x < 0.0  else x for x in v])
    return np.sum(v2)
 
  vmax= util.np_findlocalmin2(close2,3)

  if len(vmax) > 2 :
    vmax= util.sort(vmax,0,asc=0)
    kmin1,pmin1= util.np_find_minpos(vmax[:,1])
    kmax1,pmax1= util.np_find_maxpos(vmax[:,1])       

    trend= (pmax1-pmin1) / ((vmax[kmax1,0] - vmax[kmin1,0]))    
    if np.abs(trend/pmax1) < 0.01    : trend=0
    
    if trend > 0.0 :     # trend up
      if -kmax1 < - kmin1 :   kix0= kmin1+1;   kix1= 0;     
      if -kmin1 < - kmax1:   
        if vmax[kmax1,0] > 99 : #Close in time to the Last value
            kix0= kmin1+1;   kix1= kmax1    
        else :  kix0= kmin1+1;   kix1= 0;   

    if trend < 0.0 :     #Trend Down
      if -kmax1 < - kmin1 :   kix0= kmax1+1;   kix1= kmin1;     

      if -kmin1 < - kmax1:    kix0= kmax1+1;   kix1= 0;  
    elif trend == 0 :  kix1= 0; kix0= len(vmax)

    vmax1= vmax[kix1:kix0,1]; vmax0=vmax[kix1:kix0,0]
    res= sci.optimize.differential_evolution(linearreg2,[(-2.0,2.0),(0,pmin1*3)],tol=0.1)  
  else : res= [0.0,0.0]  
    
  if type1==1 :   
      return res,vmax,vmax1, vmax0
  else :           
      return res  
  



def ta_highbandtrend1(close2, type1=0) :
  def linearreg(ww):
   v= ww[0]*vmax0  +   ww[1]  - vmax1
   v2= np.array([ 1000000 if x < 0.0  else x for x in v]) 
   return np.sum(v2)
 
  vmax= util.np_findlocalmax2(close2,3)
        
  if len(vmax) > 2 :
    vmax= util.sort(vmax,0,asc=0)
    kmin1,pmin1= util.np_find_minpos(vmax[:,1])
    kmax1,pmax1= util.np_find_maxpos(vmax[:,1])       
    
    trend= (pmax1-pmin1) / (vmax[kmax1,0] - vmax[kmin1,0])    
    if np.abs(trend/pmax1) < 0.01    : trend=0    
    
    if trend > 0.0 :     # trend up
      if -kmax1 < -kmin1 :   kix0= kmin1+1;   kix1= 0;     
      if -kmin1 < -kmax1:   
        if vmax[kmax1,0] > 99 : #Close in time to the Last value
            kix0= kmin1+1;   kix1= kmax1    
        else :  kix0= kmin1+1;   kix1= 0;   

    if trend < 0.0 :     #Trend Down
       if -kmax1 < -kmin1 :              
         if vmax[kmin1,0] > 99 : #Close in time to the Last value
            kix0= kmax1+1;   kix1= kmin1   
         else :  kix0= kmax1+1;   kix1= 0 

       if -kmin1 < -kmax1:    kix0= kmax1+1;   kix1= 0  
    elif trend == 0 :  kix1= 0; kix0= len(vmax)
      
    vmax1= vmax[kix1:kix0,1]; vmax0=vmax[kix1:kix0,0]
    res= sci.optimize.differential_evolution(linearreg,[(-2.0,2.0),(0.0,pmax1*3)],tol=0.1)  
    
  else : res= [0.0,0.0]  
    
  if type1==1 :   
      return res,vmax,vmax1, vmax0
  else :           
      return res 
#####################################################################################





#####################################################################################
#----------------- Strategy/ Portfolio generation------------------------------------
def pd_transform_asset(q0, q1, type1="spread") :
 date0= util.pd_date_intersection( [q0,q1])
 q0= q0[q0.date.isin(date0)];  q1= q1[q1.date.isin(date0)]
 qn= util.pd_array_todataframe(q0[['date', 'month', 'day']].values, ['date', 'month', 'day'])
 if type1=="spread" :
   colval= 100*(q0.close.values/q0.close.values[0] - q1.close.values / q1.close.values[0])
 qn= util.pd_insertcol(qn, 'close', colval)
 qn= util.pd_insertcol(qn, 'high', colval)
 qn= util.pd_insertcol(qn, 'low', colval)

 return qn


#------------------Basket Table value at all time-----------------------------------
def calcbasket_table(wwvec, ret, type1="table", wwtype="constant",
                     rebfreq=1, costbps= 0.000):
  #if wwtype=="constant" : wwvec= wwvec[:-1]  #Remove Last weight, w= 1-sum(w)
  masset, tmax= np.shape(ret)
  bsk= np.zeros(tmax+1,dtype="float32"); bsk[0]=100.0
  ww0= np.ones(masset, dtype="float16") / masset
  ww2= ww0
  wwall= np.zeros((masset,tmax+1),dtype="float32")
  wwall[:,0]= ww2
  
  for t in range(1,tmax+1):
    #Weight Calc--------------------------------------------------
    hedgecost=0
    if np.mod(t, rebfreq)==0 :                  #rebalancing only every 5 days.
 #      ww2i= weightcalc_generic(wwvec,t, masset, closez, wwtype)   
        ww2i= wwvec   
        #print ww2i, t
        hedgecost= np.sum(np.abs(ww2-ww2i)) * bsk[t-1] *costbps 
        ww2=ww2i   
    
    bsk[t]=  bsk[t-1] * ( 1 + np.sum(ww2*ret[:,t-1]) ) - hedgecost
    wwall[:,t]= ww2

  if type1=="table" : return bsk, wwall.T #Return table of Price/weights
 
 
#Creation of RiskPa assets
def folio_createvolta_asset(close, vol=0.12, volrange=120, lev=1.0):
 closepa= np.zeros_like(close); masset,_ = np.shape(close)
 for i in xrange(0, masset) :
   riskpa= folio_volta(close[i,:], vol, volrange, lev)
   closepa[i,:]= riskpa
 return closepa
 

def folio_lowcorrelation(sym01, nstock, periodlist, dateref, close1, kbenchmark, badlist, costbppa=0.02, showgraph=True) :
 t1list= periodlist;  nmax= nstock; 
    
 def ww_selection(nmax1, ret_close1, badlist, *args) :
  t1= args[0];  # t2= args[3];  
  tcorrelrange= args[1];  tvolrange=args[2]

  correlr= np.corrcoef(ret_close1[:, (t1-126):t1])  #Previous Correl
  correlr= np.nan_to_num(correlr)
  # correlr= correl1        #Use longer LAG for Co-Integration issues 
  rankcorrel_perstk= np.zeros((nstock, 3))
  for k in range(0, nstock): 
    rankcorrel_perstk[k,0]=  k
    rankcorrel_perstk[k, 1]=  (np.sum(correlr[k,:]) -1)/(nstock-1)  
    rankcorrel_perstk[k, 2]=  (np.sum(np.abs(correlr[k,:])) -1)/(nstock-1)
  rankcorrel_perstk= util.np_sortbycol(rankcorrel_perstk,2, asc=True)

  stk_select=[] #1st Stock
  for i in range(0,100):
    k= int(rankcorrel_perstk[i,0] )   
    if not k in badlist and np.std(ret_close1[k,max(0,(t1-50)):(t1)]) > 0.005 and not np.isnan(np.sum(ret_close1[k,t1:(t2)])) :  #No zero Volatility :
     stk_select= [ k ]; break;
      
  for ii in range(0, nmax-1) : #2nd stock select
    scormin= 500; kmin= None
    for k in range(0, nstock) :
      if not k in badlist and util.find(k, stk_select)==-1 :
        scor= np.sum( correlr[k, stk_select] )   #Lower Correlation
        if scor < scormin :
          if np.std(ret_close1[k,max(0,(t1-252)):(t1)]) > 0.005 and  np.sum(ret_close1[k,t1:(t2)]) != 0 :  #No zero Volatility
            scormin = scor;          kmin= k
    stk_select.append(int(kmin))

  volx= volhisto_fromret(ret_close1, t1, 252, axis=1)
  # 1.0/nmax+np.zeros(nmax)
  ww= 1/volx[stk_select]
  ww= ww/ sum(ww)
  if np.isnan(sum(ww)) :  ww= 1/len(stk_select) +np.zeros(len(stk_select))
  return stk_select, ww
 
 
 #------------- Bsk Calculation   --------------------------------------------------- 
 ret_close1= getret_fromquotes(close1[:, :], timelag=1)
 nstock=  len(sym01)
 periodmax= len(t1list) 
 risktable= np.zeros((periodmax, 20))
 wwtable= np.empty((periodmax, 5), dtype=np.object)
 
 bskfull= []
 benchfull= np.ones((1,1))

 wwfull= np.zeros((15,1))
 wwselect= np.zeros((15,1))
 
 
 #-----------Start Loop--------------------------------------------------------------  
 jj=0; bskt0= 100.0; bskt03= 0.0; bencht0= 100.0
 for ss, t1  in enumerate(t1list) : #All sub-period
  if ss+1> periodmax-1 :  break;
  t2= t1list[ss+1]   #Include Last date

  stk_select, ww= ww_selection(nmax, ret_close1, badlist , *(t1, 126, 252, t2))
  nmax1=  len(stk_select)

  bsk,wwall= calcbasket_table(ww, ret_close1[stk_select,t1:t2]) # Basket
  bench1= price_normalize100(ret_close1[kbenchmark, t1:t2])     # Benchmark
  
 #Weight, Stock Selection
  wwfull=    np.column_stack((wwfull, wwall.T))
  stk_select2= np.zeros_like(wwall); stk_select= np.array(stk_select)
  stk_select2[:,:]= stk_select[ np.newaxis,:]
  wwselect=  np.column_stack((wwselect, stk_select2.T)) 

 #Long Basket 
  bsk2= bsk * bskt0/bsk[0]
  bskt0= bsk2[-1]  
  bskfull= np.concatenate((bskfull, bsk2))

 #Short Basket
  price2=  bench1 * bencht0/bench1[0,0]
  bencht0= price2[0,-1]  
  benchfull= np.column_stack((benchfull, price2))

  # print t1, t2, bsk[0:5], ww, stk_select
  if showgraph :   
    print( 'Period Start :' + str(dateref[t1])  )
    plot_price(bsk-100, bench1[0,:]-100, sym= ['BSK', 'Benchmark' ],
              tickperday=5,   label=('Daily Variation ', 'Weeks', 'Variation in %'))

    plot_price(bsk-bench1[0,:], bench1[0,:]-100, sym= ['L/S wth Benchmark', 'Benchmark' ],
              tickperday=5,   label=('Daily Variation ', 'Weeks', 'Variation in %'))

    for k in range(0, nmax1) :
       print sym01[int(stk_select[k])][:] + ' JP Equity ; ',     a_nk400name[util.find(sym01[stk_select[k]], a_nk400list)] +';',  str(ww[k]) +';'


  print( 'Period Start :' + str(dateref[t1])  +'; Perf: ' + str(bsk[-1]-bench1[0,-1]) + '  ; NKPerf:' + str(bench1[0,-1]/bench1[0,0]-1) )
    

  #--------------------- Risk  Storage --------------------------------------------- 
  risktable[jj,0] = dateref[t1];        risktable[jj,1] = dateref[t2]
  risktable[jj,2] = bsk[-1]  ;          risktable[jj,3] = bench1[0,-1]
  risktable[jj,4] = bsk[-1]-100 ;       risktable[jj,5] = bench1[0,-1]-100
  risktable[jj,6] = bsk[-1]-bench1[0,-1]
  risktable[jj,7] = volhisto_fromprice(bsk[:], -1,60)
  risktable[jj,8] = volhisto_fromprice(bench1[0,:], -1, 60)
  
  risktable[jj,9] = volhisto_fromprice(100 + bsk[:] - bench1[0,:], -1, 50)
  risktable[jj,10] = close1[kbenchmark, t1]
  risktable[jj,11] = close1[kbenchmark, t2]

  wwtable[jj,0] = dateref[t1];     wwtable[jj,1] = dateref[t2]
  wwtable[jj,2] = [ sym01[int(x)]  for x in stk_select ]
  wwtable[jj,3] = ww
  wwtable[jj,4] = bsk
  
  jj+=1
  print( '----------------------------------------------' )
 #END for 
 
 #long with 2% TC
 for t in xrange(1, len(bskfull)):
    bskfull[t]=  bskfull[t]*  (1 - t*0.00 /252.0)

 #Inverse ETF
 benchfull= benchfull[:,1:]  
 invbench=  folio_inverseetf(benchfull[0,:], costpa=0.016)
 
 
 #Long Short Portfolio with Inverse ETF:
 lsfull= np.zeros(len(bskfull)); lsfull[0]= 100.0
 for t in xrange(1, len(bskfull)) :
   lsfull[t]= lsfull[t-1]*(  0.55*bskfull[t]/bskfull[t-1] + 0.45*invbench[t]/invbench[t-1]  ) * (1  - costbppa /252.0)


 return risktable, wwtable, wwfull, wwselect,  bskfull, benchfull, invbench, lsfull


def folio_leverageetf(price, lev=1.0,costpa=0.0):
    invetf = np.zeros(len(price));
    invetf[0] = 100.0
    for t in xrange(1, len(price)):
        invetf[t] = invetf[t - 1] * (1 + lev * (price[t] / price[t - 1] - 1)) * (1 - costpa / 252.0)

    return invetf


def folio_inverseetf(price, costpa=0.0):
 invetf= np.zeros(len(price)); invetf[0]= 100.0
 for t in xrange(1, len(price)):
   invetf[t]= invetf[t-1]*( 1 -  (price[t]/price[t-1] -1))*(1- costpa/252.0)   
 
 return invetf



def folio_longshort_unit(long1, short1, ww=[1, -1], costpa=0.0, tlag=1, istable=1, wwschedule=[]):
 tmax= len(long1)
 folio= np.zeros(tmax); folio[0]= 100.0
 wwall= np.zeros((tmax, 8))
 
 ww=    wwschedule if len( wwschedule) > 1 else np.zeros((tmax,2)) + ww
 
 cash1=  folio[0]
 nl =    folio[0] * ww[0,0] / long1[0]
 ns =    folio[0] * ww[0,1] / short1[0]
 cash1 = cash1 - (nl * long1[0] + ns * short1[0])

 for t in xrange(0, tmax):
   fees = (1 - costpa / 252.0)
   if t < tlag :
     folio[t] =  nl*long1[t]   + ns*short1[t]  + cash1
   else :
     if np.mod(t, tlag)==0  or tlag==1 or tlag==0   :
        nl0= nl;  ns0= ns
        nl=  folio[t-tlag] * ww[t,0] / long1[t-tlag] *  fees
        ns = folio[t-tlag] * ww[t,1] / short1[t-tlag] *  fees
        transact= (nl-nl0)*long1[t] + (ns-ns0)*short1[t]
        cash1= cash1 -  transact
     folio[t] =  nl*long1[t]   + ns*short1[t] + cash1


   wwall[t,0]=  folio[t]

   # Theoric %
   wwall[t,1]=  ww[t,0]    ;    wwall[t, 2] = ww[t,1]

   # In % of the asset
   wwall[t,3]=  nl * long1[t]  / folio[t]    ;    wwall[t, 4] =  ns * short1[t]  / folio[t]

   # In unit
   wwall[t, 5] = nl;  wwall[t, 6] = ns

 if istable :   return folio, wwall
 else :         return folio


def folio_longshort_unitfixed(long1, short1, nn=[1, -1], costpa=0.0, tlag=1 , istable=1):
 tmax= len(long1)
 invetf= np.zeros(tmax); invetf[0]= 100.0
 wwall= np.zeros((tmax, 6))

 nl= nn[0]; ns= nn[1]
 cash1= invetf[0]
 ww[0]= nl * long1[0] /  invetf[0]
 ww[0]= nl * long1[0] /  invetf[0]

 cash1 = cash1 - (nl * long1[0] + ns * short1[0])

 for t in xrange(1, tmax):
   fees = (1 - costpa / 252.0)
   if t < tlag :
       invetf[t] =  nl * long1[t]   + ns * short1[t]  + cash1
   else :
     if np.mod(t, tlag)==0  or tlag==0 or tlag==1 :
        nl0= nl;  ns0= ns
        nl=  invetf[t-tlag] * ww[0] / long1[t-tlag] *  fees
        ns = invetf[t-tlag] * ww[1] / short1[t-tlag] *  fees
        transact= (nl-nl0)*long1[t] + (ns-ns0)*short1[t]
        cash1= cash1 -  transact
     invetf[t] =  nl * long1[t]   + ns * short1[t] + cash1

   wwall[t,0]=  invetf[t]

   # In unit
   wwall[t,1]=  nl;                                wwall[t,2] = ns

   # In % of the asset
   wwall[t,3]=  nl * long1[t]  / invetf[t]    ;    wwall[t, 4] = ns * short1[t]  / invetf[t]

 if istable :   return invetf, wwall
 else :         return invetf



def folio_longshort_pct(long1, short1, ww=[1, -1], costpa=0.0):
 invetf= np.zeros(len(long1)); invetf[0]= 100.0
 for t in xrange(1, len(long1)):
   invetf[t]= invetf[t-1]*( 1 +  ww[0]*(long1[t]/long1[t-1] -1)  + ww[1] * (short1[t]/short1[t-1] -1)  )*(1- costpa/252.0)

 return invetf


def folio_histogram(close):
 ret5d= getret_fromquotes(close, 5) *100
 
 fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))
 weights = np.ones_like(ret5d)/len(ret5d)
 ax2.hist(ret5d,50, normed=0,weights=weights,  facecolor='green')
 ax2.set_title('5d return distribution  ')
 ax2.set_xlabel('5d return in %'); ax2.set_ylabel('Probability')
 ax2.axvline(x=0, color='r', linestyle='dashed', linewidth=2)
 plt.show()

 print('5d return statistics :') 
 print (sci.stats.describe(ret5d, axis=0))
 print('-------------------------------------------------------------------------')
 #20days
 ret5d= getret_fromquotes(close, 20) *100
 fig, ax2 = plt.subplots(1, 1, figsize=(6, 4))
 weights = np.ones_like(ret5d)/len(ret5d)
 ax2.hist(ret5d,50, normed=0,weights=weights,  facecolor='green')
 ax2.set_title('20d return distribution  ')
 ax2.set_xlabel('20d return in %'); ax2.set_ylabel('Probability')
 ax2.axvline(x=0, color='r', linestyle='dashed', linewidth=2)
 plt.show()

 print('20d return statistics :') 
 print (sci.stats.describe(ret5d, axis=0))
 
 
def folio_voltarget(bsk, targetvol=0.11, volrange= 90, expocap=1.5):
 return folio_volta(bsk, targetvol=0.11, volrange= 90, expocap=1.5)


def folio_volta(bsk,  targetvol=0.11, volrange= 90, cap=1.5, floor=0.0, isweight=0, voltable=[], volschedule=[], tlag=0):
 #Generate Vol Target portfolio

 istable=    0 if len(voltable) == 0  else 1
 isschedule= 0 if len(volschedule) == 0  else 1   

 tmax = np.shape(bsk)[0]
 bsk2= np.zeros(tmax, dtype="float32");   wwvol= np.zeros((tmax,2), dtype="float32");
 retbsk= bsk[1:(tmax)] / bsk[0:(tmax-1)] -1

 bsk2[0:volrange+1]=   bsk[0:volrange+1] / bsk[0]*100.0  #initilisation
 wwvol[0:volrange+1,:]=  1.0
 for t in range(volrange+1,tmax+tlag):
    vol= np.std(retbsk[(t-volrange-1):(t+1-tlag)], axis=0)* 15.8745078664

    if istable :  # Vol-Weight table
       kid= util.findhigher(vol, voltable[:,0]) -1
       wwvol[t] =  max(floor, min(cap,  voltable[kid,1] ))     
      
    else :
      if isschedule :  
        targetvol =  volschedule[t-tlag,1]  #use Schedule 
        if volschedule[t-tlag,2]==1  :
          wwvol[t,1]=  targetvol    
          wwvol[t,0] =  max(floor, min(cap, targetvol/vol))
        else :
          wwvol[t,1]=   wwvol[t-1,1]    
          wwvol[t,0] =  wwvol[t-1,0]        
      else:    
        wwvol[t,1]=  targetvol    
        wwvol[t,0] =  max(floor, min(cap, targetvol/vol))      
      
    bsk2[t]=  bsk2[t-1] * ( 1 +  wwvol[t,0] * retbsk[t-1] )

 if isweight : return bsk2, wwvol
 else :        return bsk2



def folio_volta2(bsk, riskind, par, targetvol=0.11, volrange= 90, cap=1.5, floor=0.0, costbp=0.0005):
 #Generate Vol Target portfolio
 tmax = np.shape(bsk)[0]
 bsk2= np.zeros(tmax, dtype="float32")
 retbsk= bsk[1:(tmax)] / bsk[0:(tmax-1)] -1
 tstart=0.0; wwvol0= 1.0
 bsk2[0:volrange+1]=bsk[0:volrange+1]/bsk[0]*100.0  #initilisation
 for t in range(volrange+1,tmax):
    vol= np.std(retbsk[(t-volrange-1):(t-2)], axis=0)* 15.8745078664
    wwvol =  max(floor, min(cap, targetvol/vol))
    
    if riskind[t-2,5] > par[0] :  
      wwvol= 0.0; tstart=t; 
    if wwvol0==0.0 and t-tstart < par[1] :  
      wwvol= 0.0
     
    cost= np.abs(wwvol-wwvol0)* bsk2[t-1]*costbp
    bsk2[t]=  bsk2[t-1] * ( 1 +  wwvol * retbsk[t-1] ) -cost
    wwvol0= wwvol

 return bsk2




def folio_fixedunitprice(price, fixedww, costpa=0.0):   
 masset, tmax = np.shape(price)
 bsk= np.zeros(tmax, dtype="float32"); bsk[0]=100.0

 for t in range(1,tmax) :
   retbk=0   
   for k in range(0,masset) :  retbk+=  fixedww[k] * (price[k,t] / price[k,0] -1) 
   bsk[t]=  bsk[0] * ( 1 + retbk )  * (1  - costpa /252.0)
   
 return bsk




def folio_fixedweightprice(price, fixedww, costpa=0.0):   
 masset, tmax = np.shape(price)
 bsk= np.zeros(tmax, dtype="float32"); bsk[0]=100.0

 for t in range(1,tmax) :
   retbk=0   
   for k in range(0,masset) :  retbk+=  fixedww[k] * (price[k,t] / price[k,t-1] -1) 
   bsk[t]=  bsk[t-1] * ( 1 + retbk )  * (1  - costpa /252.0)
   
 return bsk


def folio_fixedweightret(ret, fixedww):   
 masset, tmax = np.shape(ret)
 bsk= np.zeros(tmax+1, dtype="float32"); bsk[0]=100.0

 for t in range(1,tmax+1) :
   retbk=0   
   for k in range(0,masset) :  retbk+=  fixedww[k] * ret[k,t-1]
   bsk[t]=  bsk[t-1] * ( 1 + retbk ) 
   
 return bsk



def folio_cost_turnover(wwall, bsk, dateref, costbp):
 cost= 0.0; wcount1=0; wsum2=0.0
 years= len(bsk) / 252.0
 tmax= len(bsk)
 print len(dateref), len(bsk)
 nbrebal= np.zeros(tmax)
 cumcost= np.zeros(tmax)
 
 for t in xrange(1, tmax) :
   cost= cost+ np.sum(np.abs( wwall[t,:] - wwall[t-1,:])) * bsk[t-1] *costbp
   wsum2= wsum2+ np.sum(np.abs( wwall[t,:] - wwall[t-1,:])) 
   cumcost[t]= cost
  
   if wwall[t,0] != wwall[t-1,0] : wcount1+= 1
   nbrebal[t]= wcount1   
 
 plot_price(nbrebal,  date1=dateint_tostring(dateref, format1='%b-%y'), tickperday=242, sym=['Cumulative'],
    label=('Nb of rebalancing', '', ''), legendloc='upper left',isband=0)

 plot_price(cumcost,  date1=dateint_tostring(dateref, format1='%b-%y'), tickperday=242, sym=['Cumulative'],
    label=('Cumulative Transaction Cost in %', '', ''), legendloc='upper left',isband=0)    
    
    
 return  {'cumcost':cumcost, 'nbrebal': nbrebal, "costpa":cost/years ,   "% TotalWeightchange pa":wsum2/years ,     "Nb rebalancing pa": wcount1/years }
   





def folio_riskpa(ret, targetvol=0.1, volrange=90, cap=1.0):  
 masset,tmax = np.shape(ret)
 bsk2= np.zeros(tmax+1, dtype="float32"); bsk2[0]=100
 ww0= 1.0/ masset * np.ones((masset))
   
 for t in range(1,volrange):  #initilisation
   bsk2[t]=  bsk2[t-1] * ( 1 + np.sum(ww0 * ret[:,t-1]) )    

 for t in range(volrange,tmax+1):
    vol= np.std(ret[:,(t-volrange):(t-1)], axis=1)* 0.158745078664
    wwvol =  targetvol/vol
    wwvol= wwvol / np.sum(wwvol)
    wwvol= [ min(cap,x) for x in list(wwvol)]
    # print wwvol
    
    #if not isfloat(wwvol) :
     # print wwvol
      #wwvol= ww0
      
    bsk2[t]=  bsk2[t-1] * ( 1 +  np.sum(wwvol * ret[:,t-1]) )

 return bsk2    


    
###########################################################################
#--------- Folio / Index creations   --------------------------------------

class index():
  def __init__(self, id1,  sym, ww, tstart) :
     self.id, self.sym, self.ww, self.tsart= id1, sym, ww, tstart
     self.dateref= date_generatedatetime(tstart, now1)
     self.hist=None
     
  def close(self):
    pass   
    
  def updatehisto(self):  #Download Quotes
    pass
    
    
  def help(self) :
   print('''
   Class index
   id
   name
   assetid : (masset)
   ww      : (masset, tmax)
  dateref : (tmax)
  hist   (4,tmax)   high, low, open,close 
  masset, tmax
  ''')






####################################################################################
#----------------- Portfolio Simulation--------------------------------------------
class folioSimulation(object) :
  ''' Calculate Units/Orders from any list of weights/close price '''
  def __init__(self, sym, close, dateref):
    self.sym, self.close,  self.dateref= sym, close, dateref 
    self.ret= getret_fromquotes(close)     
    self.masset, self.tmax  = np.shape(self.ret)

  def  setcriteria(self, lweight,  name,  initperiod )  :
    ''' initperiod: NbOfDays_initial_Risk_Monitoring   '''
    self.wwasset=    np.array(lweight['wwasset'], dtype=np.float32)    # Allocation weight
    self.wwbasket=   lweight['wwbasket']                               # Fees, costbp,
             
    #basket details :
    self.amt=           self.wwbasket['notional'];     self.rebfreq=       self.wwbasket['rebfreq']
    self.costbp=        self.wwbasket['costbp'];       self.costfixedbp=   self.wwbasket['costfixedbp']
    self.costshortbp=   self.wwbasket['costshortbp'];  self.borrowshortbp= self.wwbasket['borrowshortbp']
    self.feesbp=        self.wwbasket['feesbp'];       self.tlagsignal=    self.wwbasket['tlagsignal']
    self.rebal_trigger= self.wwbasket['rebaltrigger']
    self.rebal_trigger_unit= self.wwbasket['rebaltrigger_unit']
    self.schedule1=     date_getspecificdate(self.dateref, self.wwbasket['schedule1'])

    #Hyper Parameters
    self.name=    name
    self.nbrange= initperiod

    #Compute data
    self.ww0unit=    self._wwpct_tounit(self.wwasset[0,:], 1.0, self.close[:,0])

    #Output data
    self.nind= 10
    #            nind indicators +  ww in Units, ww in %, ww in units for order (diff wiht previous
    self.wwind=      np.zeros((self.tmax+1, self.nind + self.masset*3),dtype="float64")
    self.wwind[:,0]= self.dateref

    self.wwindpct=      np.zeros((self.tmax+1, self.nind+self.masset*2),dtype="float64")
    self.wwindpct[:,0]= self.dateref

    self.volatility= None
    self.drawdown=   None
    self.perfband=   None
    
    #Data check    




  def calc_baskettable_pct(self, type1="table", showdetail=0):
      '''   Calc Basket Values from Input data '''
      rebfreq, costbp, ret = self.rebfreq, self.costbp, self.ret
      masset, tmax = self.masset, self.tmax

      bsk = np.zeros(tmax + 1, dtype="float32")
      bsk[0] = 1.0;      trebal = 0;
      wwpct_actual = self.wwasset[0, :]; wwpct_th= wwpct_actual
      hedgecost= np.sum(np.abs(wwpct_actual) * bsk[0]) * costbp
      self._udpate_wwindpct(0, bsk[0], hedgecost, wwpct_actual, wwpct_th)

      for t in range(1, tmax + 1):
          hedgecost = 0
          if np.mod(t, rebfreq) == 0:  # rebal of weights
              wwpct_new, wwpct_th = self._wwpct_rebal( wwpct_actual, t - 1,  trebal)  # Calc Weights
              hedgecost = np.sum(np.abs(wwpct_new - wwpct_actual) * bsk[t-1]) * costbp     # Cost 0.002
              wwpct_actual = wwpct_new
              trebal = t

          bsk[t] = bsk[t-1] * (1 + np.sum(wwpct_actual * ret[:, t - 1])) - hedgecost  # ww basket
          self._udpate_wwindpct(t, bsk[t],  hedgecost, wwpct_actual,  wwpct_th)

          if showdetail:      print self.dateref[t], bsk[t], wwpct_actual

      return bsk, self.wwindpct


  def _wwpct_rebal(self,  wwpct_actual, t, trebal):  # BskUnit= 1.0  !!!
      wwpct_th = self.wwasset[trebal, :]
      if np.min(np.abs(wwpct_actual/wwpct_th  -1 )) > self.rebal_trigger:
          wwpct_new = wwpct_th
      else:
          wwpct_new = wwpct_actual

      return wwpct_new,  wwpct_th


  def _udpate_wwindpct(self, t, bskt,  hedgecost, wwpct_actual, wwpct_th):
      masset = self.masset
      self.wwindpct[t, 1] = t
      self.wwindpct[t, 2] = bskt
      self.wwindpct[t, 3] = hedgecost
      self.wwindpct[t, self.nind + 0 * masset:self.nind + 1 * masset] = wwpct_actual
      self.wwindpct[t, self.nind + 1 * masset:self.nind + 2 * masset] = wwpct_th


  def calc_baskettable_unit(self,  type1="table",   showdetail=0):
    '''   t           --->  t+1                 ---->   t+2
            wwactual:    B[t]    -->   B[t+1]   using  w[t+1], B[t] /   S[t]
            wwnew   :    B[t+1]  -->   B[t+2]   using  w[t+2], B[t+1] /   S[t+1]
    '''
    _= self.calc_baskettable_pct(type1="table", showdetail=0)      #Calculate pct weight
    rebfreq, costbp, close=  self.rebfreq, self.costbp, self.close
    masset, tmax, nind=      self.masset, self.tmax, self.nind
    
    bsk= np.zeros(tmax+1,dtype="float64");
    bsk[0]= self.amt
    wwunit_actual= self.ww0unit;  wwunit_new= wwunit_actual;
    wwpct_actualinv = wwunit_actual / (bsk[0]) * self.close[:, 0]
    transact=  np.sum( wwunit_actual * close[:,0] )
    hedgecost= transact * costbp
    cash= self.amt - transact - hedgecost
    self._udpate_wwind(0,  bsk, cash, transact,  hedgecost, wwunit_actual, wwpct_actualinv )

    for t in range(0,tmax) :
      hedgecost=0; transact= 0
      if np.mod(t, rebfreq)==0 and t < tmax-1 :
         trebal = t+rebfreq
         wwunit_new=    self._wwunit_rebal(bsk,   t,trebal)   # wwnew : used from (t+1-->t+2)
         transact=      np.sum( (wwunit_new - wwunit_actual) * close[:,trebal] )      # wwactual ---> wwnew at (t+1)
         hedgecost=     np.sum(np.abs(wwunit_new - wwunit_actual ) * close[:,trebal] )  * costbp   # Cost 0.002
         cash=          cash - transact - hedgecost

      bsk[t+1]=  np.sum(wwunit_actual * close[:,t+1] )  + cash
      wwpct_actualinv = wwunit_actual / (bsk[t] ) * self.close[:, t]   # used in t---> t+1
      self._udpate_wwind(t+1,  bsk,  cash, transact, hedgecost, wwunit_actual, wwpct_actualinv )  # t---> t+1
      wwunit_actual = wwunit_new

   #   print t, bsk[t], cash, transact
   #   if t > 4 : sys.exit()
    #Diff in Basket Value
    wwdiff=      np.zeros((self.tmax+1, self.nind+self.masset),dtype="float64")
    wwdiff[:,0]= self.dateref
    wwdiff[:,1]= 100.0*(self.wwindpct[:,2] / self.wwindpct[0,2]  - self.wwind[:,2] / self.wwind[0,2] )
    wwdiff[:,nind:nind + masset]= self.wwindpct[:,nind:nind + masset] - self.wwind[:,nind + masset:nind + 2*masset]

    return bsk, self.wwind, wwdiff


  def get_orderlist(self):
    order1= np.zeros((self.tmax, 6), dtype=np.object) #dateint, ticker, units, unitprice, ordertype Buy/Sell
    
    order1 =[]
    for t in xrange(0,self.tmax):
      if  np.sum( np.abs(self.wwind[t,self.nind + self.masset*2:] )) > 5 :  #Only meanful rebalancing
        for k in xrange(0,self.masset) :
            wunit= self.wwind[t,self.nind + self.masset*2+k]
            # if wunit != 0 :
                # order1[t,0]=  self.dateref[t]
                # order1[t,1] = self.sym[k]
                # order1[t,2] = abs(wunit)
                # order1[t,3] = self.close[k,t]
                # order1[t,4] = 'Buy' if wunit > 0 else 'Sell'
                
            order1.append([self.dateref[t], self.sym[k], abs(wunit), self.close[k,t],  'Buy' if wunit > 0 else 'Sell'       ])  
              
    df= util.pd_createdf(order1, ['date','symbol','unit','unitprice','Type'])            
    return df


  def get_orderlist_tocsv(self, file1):
     df= self.get_orderlist()
     df.to_csv(file1, sep=",")


  def _wwunit_rebal(self,bsk,  t, trebal)  :       #  wunit: t+1 ---->  t+2,     wwpct[t+2],  B[t+1], S[t+1
     wwpct_new=      self.wwindpct[t+2, self.nind:self.nind + self.masset]      #at t+2, B[t+2] uses ww[t+2]
     wwunit_new =    np.round( wwpct_new * (self.amt * self.wwindpct[t+1,2] ) / self.close[:,t+1],  0)      #B[t+1] uses  close[t
     return wwunit_new

  def _wwpct_tounit(self,wwpct, bskt, closet) :    return np.round( (self.amt * bskt ) / closet * wwpct, 0) #Unitaire

  def _wwunit_topct(self,wwunit, bskt, closet) :   return wwunit / (self.amt * bskt ) * closet

  def _udpate_wwind(self,t, bsk,  cash, transact, hedgecost, wwunit_actual, wwpct_actualinv) :
        masset= self.masset        
        self.wwind[t,1]= t;                       self.wwind[t,2]= bsk[t]
        self.wwind[t,3]= cash  #cumulative
        self.wwind[t,4]= transact;                self.wwind[t,5]= hedgecost
        self.wwind[t, self.nind:self.nind+masset]=            wwunit_actual       # in Units
        self.wwind[t, self.nind+masset:self.nind+2*masset]=   wwpct_actualinv     # Pct
        self.wwind[t,
             self.nind + 2*masset:self.nind + 3 * masset] =   wwunit_actual- self.wwind[t-1, self.nind:self.nind+masset]  #Diff of Units











#Generate Performance/ Indicators from the schedule
def folio_perfreport(sym, dateref, close, wwind, t0) :
   tmax, nind= np.shape(wwind)

   # Perf of Each Components:

   table1 = df.describe()
   table1 = table1.to_html().replace('<table border="1" class="dataframe">', '<table class="table table-striped">')  # use bootstrap styling



   schedule1= generate_schedule(dateref, scheduleperiod)
   outind= np.zeros((len(schedule1), nind))
   for i,ti in enumerate(schedule1) :
      outind[:,i]= self.wwind[ti,:]

   return outind

'''


Performance of the basket:


Positions:


Rebalancing orders:



'''

table1= ''

html_string = '''
<html>
    <head>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
        <style>body{ margin:0 100; background:whitesmoke; }</style>
    </head>
    <body>
        <h1>Report Strategy/h1>

        <!-- *** Section 1 *** --->
        <h2>Section 1: Apple Inc. (AAPL) stock in 2014</h2>
        <p>Apple stock price rose steadily through 2014.</p>

        <!-- *** Section 2 *** --->
        <h2>Section 2: AAPL compared to other 2014 stocks</h2>
        <p>GE had the most predictable stock price in 2014. IBM had the highest mean stock price. \
The red lines are kernel density estimations of each stock price - the peak of each red lines \
corresponds to its mean stock price for 2014 on the x axis.</p>
        <h3>Reference table: stock tickers</h3>
        ''' + table1 + '''
        <h3>Summary table: 2014 stock statistics</h3>
        ''' + table1 + '''
    </body>
</html>'''




# Generate Performance/ Indicators from the schedule
def folio_perfreport_schedule(sym, dateref, close, wwind, t0, scheduleperiod="1monthend"):
    tmax, nind = np.shape(wwind)

    schedule1 = generate_schedule(dateref, scheduleperiod)
    outind = np.zeros((len(schedule1), nind))
    for i, ti in enumerate(schedule1):
        outind[:, i] = self.wwind[ti, :]

    return outind


####################################################################################
#----------------- Portfolio Calculation -------------------------------------------
class folioCalc(object):
    def __init__(self, sym, close, dateref):
        self.sym = sym;   self.close = close;   self.ret = getret_fromquotes(close)
        masset, tmax = np.shape(self.ret)
        self.masset = masset;  self.tmax = tmax
        self.ww0 = np.ones(masset, dtype="float16") / masset
        self.dateref = dateref

        # Output data
        self.bsk = None;   self.wwbest = None;    self.res = None

        if tmax + 1 != len(dateref): print("Dateref and Close length mistmatch")
        print(str(masset) + " assets", dateref[0], dateref[-1])

    def set_symclose(self, sym, close, dateref):
        self.sym = sym;  self.close = close ;    self.ret = getret_fromquotes(close)
        masset, tmax = np.shape(self.ret)
        self.masset = masset;    self.tmax = tmax
        self.ww0 = np.ones(masset, dtype=np.float32) / masset
        self.dateref = dateref
        if tmax + 1 != len(dateref): print("Dateref and Close length mistmatch")
        print(str(masset) + " assets", dateref[0], dateref[-1])

    def setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid="spprice",
                    lfun=None):
        ''' optimcrit:  perf / perfyearly / drawdownyearly / sharpe/ volatility/ linear
            wwtype:     constant / regime /regimExtra
            initperiod: NbOfDays_initial_Risk_Monitoring
            riskid:     spprice/spperf/multi
        '''
        self.wwasset = np.array(lweight[0], dtype=np.float16)  # Allocation weight
        self.wwrisk = np.array(lweight[1], dtype=np.float16)  # Risk Trigger
        self.wwbasket = lweight[2]  # Fees, costbp,
        self.wwregime = np.array(lweight[3], dtype=np.float16)
        self.wwasset0 = np.array(lweight[4], dtype=np.float32)  # Default weighted
        self.nwwasset = len(self.wwasset);
        self.nwwrisk = len(self.wwrisk)
        print len(self.wwrisk), len(self.wwasset)

        # State data
        self.riskind = np.array(statedata[0], dtype=np.float64)
        self.risk = 2.0

        # basket details :
        self.rebfreq = self.wwbasket['rebfreq']
        self.costbp = self.wwbasket['costbp']
        self.costbpshort = self.wwbasket['costbpshort']
        self.feesbp = self.wwbasket['feesbp']
        self.maxweight = self.wwbasket['maxweight']
        self.date_31dec = self.wwbasket['schedule1']
        self.tlagsignal = self.wwbasket['tlagsignal']

        # Hyper Parameters
        self.name = name
        self.criteria = optimcrit
        self.wwtype = wwtype
        self.ismultiperiod = 0
        self.riskid = riskid
        self.model = None
        self.nbrange = initperiod
        self.nbregime = nbregime
        self.nyears = len(self.date_31dec)
        self.tstart = 0

        # Function mapped outside
        self._mapping_calc_risk = lfun['mapping_calc_risk']
        self._mapping_risk_ww =   lfun['mapping_risk_ww']

        # Data check
        # if self.nbrange < 2 : print('ERROR initperiod=2 days')


    # ------------------------- Weight Calculation--------------------------------------
    def _weightcalc_generic(self, wwvec, t):
        if self.wwtype == "constant":      return self._weightcalc_constant(wwvec, t)
        if self.wwtype == "regime":        return self._weightcalc_regime(wwvec, self.wwrisk, t)
        if self.wwtype == "regimeExtra":   return self._weightcalc_regime(self.wwasset, wwvec, t)
        if self.wwtype == "regimeBoth":
            return self._weightcalc_regime(
                wwvec[self.nwwrisk:self.nwwrisk + self.nwwasset], wwvec[0:self.nwwrisk], t)
        else:
            print("Error no weight style selected")

    # ------ Asset weight  Optimization ------------------------------------------------
    def _weightcalc_regime(self, wwvec, wwextra, t):
        if t < self.nbrange:
            return self.ww0
        else:
            risk = self._regimecalc(t, wwextra)  # Risk indicator
            # print np.shape(wwvec),  self.masset, self.nbregime
            wwmat = np.reshape(wwvec, (self.masset, self.nbregime))  # Reshape into matrix asset x nbRegime
            ww2 = self._mapping_risk_ww(risk, wwmat, self.wwasset0)  # Market Cycle/Regime from Risk Indicator Level
            return ww2

    def _regimecalc(self, t, wwextra):  # Risk Indicator
        if self.riskid == "spprice":   return self.riskind[0, t - 1]
        if self.riskid == "spperf":   return 100.0 * (
        self.riskind[0, t - 1] / self.riskind[0, t - 1 - self.nbrange] - 1)
        if self.riskid == "multi":
            self.risk = self._mapping_calc_risk(self.riskind, wwextra, t, self.risk)
            return self.risk


    def _weightcalc_constant(self, ww2, t):
        return ww2 / np.sum(ww2, axis=0)


    def getweight(self):
        wwbest2 = np.reshape(self.wwasset, (self.masset, self.nbregime))
        wwbest2 = wwbest2 / np.sum(wwbest2, axis=0)
        print wwbest2
        return wwbest2

    # Calc Basket Values from Input data
    def calc_baskettable(self, wwvec, ret, type1="table", wwtype="constant", rebfreq=1, costbps=0.000, showdetail=0):
        tlagsignal = self.tlagsignal
        masset, tmax = np.shape(ret)
        nriskfactor = np.shape(self.riskind)[1]
        wwall = np.zeros((masset + nriskfactor + 2, tmax + 1), dtype="float64")
        wwall[0, :] = self.dateref
        # wwall[1,:]= risk indicator
        wwall[2:2 + nriskfactor, :] = self.riskind[:, :].T
        wwall[2 + nriskfactor:, 0] = self.ww0

        bsk = np.zeros(tmax + 1, dtype="float32");
        bsk[0] = 100.0
        ww2 = self.ww0
        for t in range(1, tmax + 1):
            hedgecost = 0
            if np.mod(t, rebfreq) == 0:  # rebabacing of weights
                ww2i = self._weightcalc_generic(wwvec, t - tlagsignal)  # Calc Weights
                if costbps != 0: hedgecost = np.sum(np.abs(ww2 - ww2i)) * bsk[t - 1] * costbps  # Cost 0.002
                ww2 = ww2i

            bsk[t] = bsk[t - 1] * (1 + np.sum(ww2 * ret[:, t - 1])) - hedgecost
            wwall[1, t - 1] = self.risk
            wwall[2 + nriskfactor:, t - 1] = ww2

            if showdetail:   print( self.dateref[t], self.risk, self.riskind[t, 0], self.riskind[t, 1], self.riskind[t, 2])

        if type1 == "table": return bsk, wwall.T  # Return table of Price/weights

    def plot(self, wwvec=None, show1=1, tickperday=60):
        if wwvec is None: wwvec = self.wwbest
        bsk, wwall = self.calc_baskettable(wwvec, self.ret, type1="table", wwtype=self.wwtype, rebfreq=self.rebfreq,
                                           costbps=self.costbp)

        if show1 == 1: 
          plot_price(bsk, tickperday=tickperday, date1=self.dateref)
          print("Final BSK Val", bsk[-1])
        return np.array(bsk, dtype=np.float32), wwall


    def multiperiod_ww(self, t):
        kid = self.kmaxmodel - 1;
        kdate = 0;
        tt = self.dateref[t]
        for k in xrange(0, self.kmaxmodel - 1):
            #    if tt > self.model[k,1][0] and  tt < self.model[k,1][1]  :        #In sample
            if tt >= self.model[k, 1][1] and tt <= self.model[k + 1, 1][1]:        # OutSample
                kdate = self.model[k, 1][1]
                kid = k

        print self.dateref[t], self.model[kid, 1]
        return self.model[kid, 2]

    def help(self):
        print(''' ''')




####################################################################################
#----------------- Portfolio Optimization USD Elvis---------------------------------
#@autojit
class folioRiskIndicator(object) :
  def __init__(self, sym, close, dateref):
    self.sym= sym; self.close= close
    self.ret= getret_fromquotes(close)
    masset, tmax= np.shape(self.ret)      
    self.masset= masset;  self.tmax= tmax
    self.ww0= np.ones(masset, dtype="float16") / masset    
    self.dateref= dateref
  
    #Output data
    self.bsk= None; self.wwbest= None; self.res= None

    if tmax+1 != len(dateref): print("Dateref and Close length mistmatch")
    print(str(masset) + " assets", dateref[0], dateref[-1])

  def set_symclose(self, sym, close, dateref):
    self.sym= sym; self.close= close
    self.ret= getret_fromquotes(close)
    masset, tmax= np.shape(self.ret)
    self.masset= masset;  self.tmax= tmax
    self.ww0= np.ones(masset) / masset
    self.dateref= dateref
    if tmax + 1 != len(dateref): print("Dateref and Close length mistmatch")
    print(str(masset) + " assets", dateref[0], dateref[-1])

  def setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid="spprice", lfun=None  ) :
    ''' optimcrit:  perf / perfyearly / drawdownyearly / sharpe/ volatility/ linear
        wwtype:     constant / regime /regimExtra
        initperiod: NbOfDays_initial_Risk_Monitoring
        riskid:     spprice/spperf/multi
    '''    
    self.wwasset=    np.array(lweight[0], dtype=np.float16)   # Allocation weight
    self.wwrisk=     np.array(lweight[1], dtype=np.float16)   # Risk Trigger
    self.wwbasket=   lweight[2]                               # Fees, costbp, 
    self.wwregime=   np.array(lweight[3], dtype=np.float16)
    self.wwasset0=   np.array(lweight[4], dtype=np.float16)   # Default weighted
    self.nwwasset=   len(self.wwasset);            self.nwwrisk=    len(self.wwrisk)
    print len(self.wwrisk), len(self.wwasset)


#    self.boundsasset=  np.array(lbounds[0], dtype=np.float16) 
#    self.boundsrisk=   np.array(lbounds[1], dtype=np.float16)
#    self.boundsbasket= np.array(lbounds[2], dtype=np.float16)
#    self.boundsregime= np.array(lbounds[3], dtype=np.float16)
       
    #basket details :
    try :    
      self.rebfreq=      self.wwbasket['rebfreq'];        self.date_31dec=   self.wwbasket['schedule1']
      self.tlagsignal=   self.wwbasket['tlagsignal'];     self.nrisk=   self.wwbasket['nrisk'];
    except: pass    

    
    #State data
    self.riskind=       np.array(statedata[0], dtype=np.float64)    #In 
    self.riskind_out=   np.zeros((self.tmax+1,self.nrisk+1),dtype=np.float64)      #Out risk

    #Function mapped outside  
    self._mapping_calc_risk=  lfun['mapping_calc_risk']     
    self._mapping_risk_ww=    lfun['mapping_risk_ww']   

     

    #Hyper Parameters  
    self.name= name; self.criteria= optimcrit;     self.wwtype=wwtype;  self.ismultiperiod= 0    
    self.riskid= riskid;                           self.model=None
    self.nbrange= initperiod;                      self.nbregime= nbregime;
    self.nyears= len(self.date_31dec);             self.tstart=0
    
    #Data check    
   # if self.nbrange < 2 : print('ERROR initperiod=2 days')
#    if len(self.boundsasset) != self.masset *  self.nbregime:  print('error Bounds Size', len(self.boundsasset) )


  def calcrisk(self,wwvec=[], initval=1) :
    TMAX, tlagsignal= self.tmax,   self.tlagsignal
                         
    self.riskind_out[:,0]= self.dateref
    self.riskind_out[0,1:]= initval
    
    for t in xrange(1,TMAX+1):  
      self._weightcalc_generic(wwvec,t)   
 
    return np.column_stack((self.riskind_out, self.riskind))
       
  #------------------------- Weight Calculation--------------------------------------
  def _weightcalc_generic(self,wwvec,t):
    if self.wwtype== "regime"   :       return self._weightcalc_regime(wwvec,        self.wwrisk,  t)
    if self.wwtype== "regimeExtra"  :   return self._weightcalc_regime(self.wwasset, wwvec,        t)     
    if self.wwtype== "regimeBoth"   :   return self._weightcalc_regime(
                        wwvec[ self.nwwrisk:self.nwwrisk+self.nwwasset],  wwvec[0:self.nwwrisk],   t)
    else :                              print("Error no weight style selected")

    
  #------ Asset weight  Optimization ------------------------------------------------
  def _weightcalc_regime(self,wwvec, wwextra, t): 
    if t < self.nbrange:  pass
    else:                 self._regimecalc(t, wwextra)                       # Risk indicator   
 

  def _regimecalc(self,t, wwextra):    #Risk Indicator  
    if self.riskid== "spprice" :   return  self.riskind[0,t-1]
    if self.riskid== "spperf"  :   return   100.0*(self.riskind[0,t-1] / self.riskind[0,t-1-self.nbrange] - 1 )
    if self.riskid== "multi"   :   self.riskind_out[t,1]= self._mapping_calc_risk(self.riskind, wwextra, t, self.riskind_out)
      
  
  def calc_optimal_weight(self,maxiter=1, name1='',  isreset=1, popsize=15) :
    fbest, xbest, solver= datanalysis.sk_optim_de(self.calcbasket_obj, self.boundsoptim, maxiter=maxiter, name1= name1, solver1= None, isreset=isreset, popsize=popsize)
    self.wwbest= xbest; self.res= (copy.deepcopy(solver), xbest, fbest)  
  
  


def folio_concenfactor2(ww, masset=12):
 wwraw= getweight( ww, (masset,3),2)*100
 ss=0.0
 for k in xrange(0,3) :
   ss+=  np.sum(np.abs(wwraw[:,k] - np.mean(wwraw[:,k])    )) 
 return ss




"""
if bsk > maxprice :       maxprice= bsk
if maxprice-bsk > maxdd : maxdd=    maxprice-bsk

"""


@jit(float32(float32[:,:], int64, float64[:,:], float64[:,:], float64[:], float64[:], float64, float64[:] ), nopython=True, nogil=True)
def calcbasket_objext(RETURN, TMAX, riskind_i, wwmat, wwasset0, ww0,nbrange, criteria) :
    bsk=100.0;  var1= 0.0
    maxprice= 100.0;  maxdd= 0.01; avgdd=0.0
    for t in xrange(1,TMAX+1):  
        # if np.mod(t,rebfreq)==0 :     # rebalancing of weights
        if t < nbrange:   ww2=  ww0
        else:             
           risk=  riskind_i[t, 1]    
           if   risk== 4.0 :    ww2= wwmat[:,2]             # Risk indicator      
           elif risk== 6.0 :    ww2= wwasset0               # Super Bear
           elif risk== 0.0 :    ww2= wwmat[:,0] 
           elif risk== 2.0 :    ww2= wwmat[:,1]
           else            :    ww2= None
           
        ret=  np.sum(ww2 * RETURN[:,t-1])
        bsk=  bsk * ( 1 + ret ) 
        var1+=ret * ret

        if bsk > maxprice :       maxprice= bsk
        if maxprice - bsk> maxdd*maxprice : maxdd=    1.0 - bsk/maxprice
        # if  bsk - maxprice > maxdd2 :   maxdd2=   bsk - maxprice
        # avgdd+= 1.0 - bsk/maxprice

    penalty=0.0
    if  bsk <  criteria[1]  :    penalty= 500.0 * ( criteria[1] - bsk )**2      # Minimum return
    if  var1 > criteria[2]  :    penalty= 1000000.0 * ( var1 - criteria[2] )    # Max Vol

    crit= criteria[0]
    if   crit==0.0 :  obj1= -(bsk -100)**2 / var1 + penalty
    elif crit==1.0 :  obj1= -(bsk- 100)**2 + penalty
    elif crit==2.0 :  obj1= -(bsk -100) / var1 + penalty
    elif crit==3.0 :  obj1= -(bsk -100) / math.sqrt(var1) + penalty
    elif crit==4.0 :  obj1= -(bsk -100)**2 / (maxdd*maxdd * var1) + penalty
    elif crit==5.0 :  obj1= -(bsk -100)**2 / (maxdd * math.sqrt(var1)  ) + penalty
    elif crit==6.0 :  obj1= -(bsk -100)**2 / (avgdd*avgdd  ) + penalty
    elif crit==7.0 :  obj1= -(bsk -100) / (maxdd * math.sqrt(var1)  ) + penalty
    elif crit==8.0 :  obj1= -(bsk -100)**2 / var1  -(bsk -100)**2 / (maxdd * math.sqrt(var1)  )  + 2*penalty


    #var1=  np.round(np.std( ret , axis=0)  * 15.86, 2)
    # var1= m.sqrt(var1) *15.86
    return obj1 
   


####################################################################################
#----------------- Portfolio Optimization USD Elvis---------------------------------
#@autojit
class folioOptimizationF(object) :
  def __init__(self, sym, close, dateref):
    self.sym= sym; self.close= close
    self.ret= getret_fromquotes(close)
    masset, tmax= np.shape(self.ret)      
    self.masset= masset;  self.tmax= tmax
    self.ww0= np.ones(masset) / masset    
    self.dateref= dateref
  
    #Output data
    self.bsk= None; self.wwbest= None; self.res= None

    if tmax+1 != len(dateref): print("Dateref and Close length mistmatch")
    print(str(masset) + " assets", dateref[0], dateref[-1])

  def set_symclose(self, sym, close, dateref):
    self.sym= sym; self.close= close
    self.ret= getret_fromquotes(close)
    masset, tmax= np.shape(self.ret)
    self.masset= masset;  self.tmax= tmax
    self.ww0= np.ones(masset) / masset
    self.dateref= dateref
    if tmax + 1 != len(dateref): print("Dateref and Close length mistmatch")
    print(str(masset) + " assets", dateref[0], dateref[-1])

  def setcriteria(self, lweight, lbounds, statedata, name, optimcrit, wwtype, nbregime, initperiod, riskid="spprice", lfun=None  ) :
    ''' optimcrit:  perf / perfyearly / drawdownyearly / sharpe/ volatility/ linear
        wwtype:     constant / regime /regimExtra
        initperiod: NbOfDays_initial_Risk_Monitoring
        riskid:     spprice/spperf/multi
    '''    
    self.wwasset=    np.array(lweight[0], dtype=np.float16)   # Allocation weight
    self.wwrisk=     np.array(lweight[1], dtype=np.float16)   # Risk Trigger
    self.wwbasket=   lweight[2]                               # Fees, costbp, 
    self.wwregime=   np.array(lweight[3], dtype=np.float16)
    self.wwasset0=   np.array(lweight[4])   # Default weighted
    self.nwwasset=   len(self.wwasset);            self.nwwrisk=    len(self.wwrisk)
    print len(self.wwrisk), len(self.wwasset)


    self.boundsasset=  np.array(lbounds[0], dtype=np.float16) 
    self.boundsrisk=   np.array(lbounds[1], dtype=np.float16)
    self.boundsbasket= np.array(lbounds[2], dtype=np.float16)
    self.boundsregime= np.array(lbounds[3], dtype=np.float16)


    #State data
    self.riskind= np.array(statedata[0], dtype=np.float64)
    self.risk= 2.0



    # if len(self.wwrisk) < 1 :   trigger= np.zeros((1, np.shape(self.riskind)[1]))
    # else : trigger= np.reshape(self.wwrisk, (1,-1))
    # self.state1= util.sk_stateRule(self.riskind, trigger) 
        
    #basket details :
    try :    
      self.rebfreq=      self.wwbasket['rebfreq'];        self.costbp=       self.wwbasket['costbp']
      self.costbpshort=  self.wwbasket['costbpshort'];    self.feesbp=       self.wwbasket['feesbp']
      self.maxweight=    self.wwbasket['maxweight'];      self.date_31dec=   self.wwbasket['schedule1']
      self.tlagsignal=   self.wwbasket['tlagsignal'];     self.wwpenalty=    self.wwbasket['wwpenalty']  
    except: pass    
    
    #Hyper Parameters  
    self.name= name; self.criteria= optimcrit;     self.wwtype=wwtype;  self.ismultiperiod= 0    
    self.riskid= riskid;                           self.model=None
    self.nbrange= initperiod;                      self.nbregime= nbregime;
    self.nyears= len(self.date_31dec);             self.tstart=0
    self.concenfactor= self.wwbasket['concenfactor']
    
    #Function mapped outside  
    self._mapping_calc_risk= lfun['mapping_calc_risk']     
    self._mapping_risk_ww=    lfun['mapping_risk_ww']   
    
    #Data check    
   # if self.nbrange < 2 : print('ERROR initperiod=2 days')
    if len(self.boundsasset) != self.masset *  self.nbregime:  print('error Bounds Size', len(self.boundsasset) )

    if self.wwtype=='regimeExtra'  :  self.boundsoptim=  self.boundsrisk 
    elif self.wwtype=='regime'  :     self.boundsoptim=  self.boundsasset     
    elif self.wwtype=='regimeBoth' :  self.boundsoptim=  list(self.boundsrisk  ) +  list(self.boundsasset)  



  def calcbasket_obj2(self,wwvec) :
    masset, nbregime= self.masset, self.nbregime
    wwpenalty, concenfactor2= self.wwpenalty, self.concenfactor  
    wwmat= np.reshape(wwvec,(masset, nbregime))               # Reshape into matrix asset x nbRegime  
    wwmat= np.array(wwmat / np.sum(wwmat, axis=0) )

    concenfactor=  100*np.sum(np.abs(wwmat - 1.0/masset   ))
    if concenfactor > concenfactor2 or  concenfactor < concenfactor - 3*masset*nbregime : return concenfactor * 2.0 - 1000.0  # 4.2*masset*nbregime
    else             :        concenfactor=  round( concenfactor,0)  * 2.0

    # Penalization On Over-Weight  : float or array of values
    lossww= np.sum(np.maximum(0,np.abs(wwmat) - wwpenalty)) * 6000.0

    # if lossww > 1.0:  return lossww - 1300.0


    #Minimum Perf, Max Vol
    # crit0= 1.0 if self.criteria=="perf" else 0.0
    crit= np.array([self.wwbasket['opticriteria'], self.wwbasket['minperf'], self.wwbasket['maxvol']])

    obj1= calcbasket_objext(self.ret, self.tmax, self.riskind_input ,  wwmat, np.array(self.wwasset0) , 
                            self.ww0, float(self.nbrange), crit)  
    return round(obj1  + lossww + concenfactor,0) 
       


  def calcbasket_obj(self,wwvec) :
    rebfreq, costbp, tlagsignal= self.rebfreq, self.costbp,  self.tlagsignal
    TMAX, RETURN= self.tmax,  self.ret
    wwpenalty, concenfactor2= self.wwpenalty, self.concenfactor       
    concenfactor=folio_concenfactor2(wwvec, self.masset) * concenfactor2
  
    bsk= np.zeros(TMAX+1,dtype=np.float16); bsk[0]=100.0
    ww2= self.ww0; lossww= 0.0
    for t in xrange(1,TMAX+1):  
      hedgecost=0.0
      if np.mod(t,rebfreq)==0 :     # rebalancing of weights
         ww2i= self._weightcalc_generic(wwvec,t-tlagsignal)   
         if costbp != 0: hedgecost= np.sum(np.abs(ww2-ww2i)) * bsk[t-1] * costbp #cost 0.002
         ww2=ww2i
 
         # Penalization On Over-Weight
         lossww+= self._loss_obj(ww2, wwpenalty) 
    
      bsk[t]=  bsk[t-1] * ( 1 + np.sum(ww2 * RETURN[:,t-1]) ) - hedgecost
        
    obj1= self._objective_criteria(bsk)
    if obj1 < -1000000.0 : return 1000.0
    else:                  return obj1  + lossww + concenfactor 
   

  def _loss_obj(self, ww2, wwpenalty):
    # Penalization 1/Sum==0, Weight infinite
    # if np.sum(np.abs(ww2)) > 2.5  : lossww=  np.sum(np.abs(ww2))*20
 
    # Penalization On Over-Weight
    lossww= np.sum(np.maximum(0,np.abs(ww2) - wwpenalty))*50.0      
    return lossww
    
    
    
  #------------------------- Weight Calculation--------------------------------------
  def _weightcalc_generic(self,wwvec,t):
    if self.wwtype== "constant" :       return self._weightcalc_constant(wwvec,t)
    if self.wwtype== "regime"   :       return self._weightcalc_regime(wwvec,        self.wwrisk,  t)
    if self.wwtype== "regimeExtra"  :   return self._weightcalc_regime(self.wwasset, wwvec,        t)     
    if self.wwtype== "regimeBoth"   :   return self._weightcalc_regime(
                        wwvec[ self.nwwrisk:self.nwwrisk+self.nwwasset],  wwvec[0:self.nwwrisk],   t)
    else :                              print("Error no weight style selected")

    
  #------ Asset weight  Optimization ------------------------------------------------
  def _weightcalc_regime(self,wwvec, wwextra, t): 
    if t < self.nbrange:  return self.ww0
    else:             
      risk=  self._regimecalc(t, wwextra)                       # Risk indicator   
      # print np.shape(wwvec),  self.masset, self.nbregime     
      wwmat= np.reshape(wwvec,(self.masset, self.nbregime))     # Reshape into matrix asset x nbRegime
      ww2=   self._mapping_risk_ww(risk, wwmat, self.wwasset0)  # Market Cycle/Regime from Risk Indicator Level
      return ww2
 

  def _regimecalc(self,t, wwextra):    #Risk Indicator  
    if self.riskid== "spprice" :   return  self.riskind[0,t-1]
    if self.riskid== "spperf"  :   return   100.0*(self.riskind[0,t-1] / self.riskind[0,t-1-self.nbrange] - 1 )
    if self.riskid== "multi"   :   
      self.risk= self._mapping_calc_risk(self.riskind, wwextra, t, self.risk)
      return self.risk


  def _weightcalc_constant(self,ww2,t):
    return ww2 / np.sum(ww2, axis=0)

            
  def calc_optimal_weight(self,maxiter=1, name1='',  isreset=1, popsize=15) :
    fbest, xbest, solver= datanalysis.sk_optim_de(self.calcbasket_obj, self.boundsoptim, maxiter=maxiter, name1= name1, solver1= None, isreset=isreset, popsize=popsize)
    self.wwbest= xbest; self.res= (copy.deepcopy(solver), xbest, fbest)  
  

  def getweight(self):
      wwbest2=  np.reshape(self.wwasset,(self.masset, self.nbregime))
      wwbest2=  wwbest2 / np.sum(wwbest2, axis=0)
      print wwbest2
      return wwbest2
      

  #Calc Basket Values from Input data
  def calc_baskettable(self,wwvec, ret, type1="table", wwtype="constant", rebfreq=1, costbps= 0.000, showdetail=0): 
    tlagsignal=    self.tlagsignal     
    masset, tmax= np.shape(ret)
    nriskfactor= np.shape(self.riskind)[1]
    wwall= np.zeros((masset+nriskfactor+2,tmax+1),dtype="float64")
    wwall[0,:]= self.dateref
    # wwall[1,:]= risk indicator
    wwall[2:2+nriskfactor,:]= self.riskind[:,:].T 
    wwall[2+nriskfactor:,0]= self.ww0
    wwall[1, 0] = self.risk

    bsk= np.zeros(tmax+1,dtype="float32"); bsk[0]=100.0
    ww2= self.ww0   
    for t in range(1,tmax+1):  
      hedgecost=0
      if np.mod(t,rebfreq)==0 :     # rebabacing of weights
         ww2i= self._weightcalc_generic(wwvec,t-tlagsignal)                          # Calc Weights
         if costbps != 0: hedgecost= np.sum(np.abs(ww2-ww2i)) * bsk[t-1] * costbps   # Cost 0.002
         ww2=ww2i

      bsk[t]=  bsk[t-1] * ( 1 +np.sum(ww2 * ret[:,t-1]) ) - hedgecost
      wwall[1,t]=               self.risk
      wwall[2+nriskfactor:,t]=  ww2

      if showdetail :      print(self.dateref[t], self.risk, self.riskind[t,0], self.riskind[t,1], self.riskind[t,2]  )
    
    if type1=="table" : return bsk, wwall.T    #Return table of Price/weights

 
  def plot(self, wwvec=None, show1=1, tickperday=60):
    if wwvec is None: wwvec= self.wwbest
    bsk, wwall= self.calc_baskettable(wwvec, self.ret, type1="table", wwtype= self.wwtype, rebfreq=self.rebfreq, costbps= self.costbp)

    if show1==1 : 
      plot_price(bsk, tickperday=tickperday, date1=self.dateref)
      print("Final BSK Val", bsk[-1])
  
    return np.array(bsk, dtype=np.float32), wwall
 


  def _objective_criteria(self, bsk): 
    criteria=self.criteria

    if criteria=="sharpe":
       vol= np.std(bsk[1:-1]/bsk[0:-2]-1, axis=0)  * 15.86
       sharpe= (bsk[-1] -100) / vol   
       return -sharpe
       
    if criteria=="perf" :  return -bsk[-1]  #Total perf     

    if criteria=="perfyearly" : #sum of yearly perf    31dec  to 31dec     
       date_31dec= self.date_31dec
       sumperf= np.sum(bsk[date_31dec[1:-1]] / bsk[date_31dec[0:-2]])        
       return -sumperf

    if criteria=="drawdownstart" : #sum of yearly perf    31dec  to 31dec    
       sumperf=0
       date_31dec= self.date_31dec
       for k in range(2, self.nyears):  
         perf0= bsk[date_31dec[k]] / bsk[date_31dec[0]]
#         loss= 10 * (1.0- perf0)   if perf0 < 1.0 else 0.0   #Continuous Loss 
         loss= 10 * (1.01 - perf0)    if perf0 < 1.01 else 0.0   #Continuous Loss       

         sumperf+= perf0 - loss
       return -sumperf*100
      
    if criteria=="drawdownyearly" : #sum of yearly perf    31dec  to 31dec    
       sumperf=0
       for k in range(2, self.nyears):  
         date_31dec= self.date_31dec
         perf1= bsk[date_31dec[k]] / bsk[date_31dec[k-1]]
         loss= 100 * (1.02 - perf1 ) - 90  if perf1 < 1.02 else 0.0   #Continuous Loss       
         sumperf+= perf1 - loss
       return -sumperf*100
      


    if criteria=="volatility":
       vol= np.std(bsk[1:-1]/bsk[0:-2]-1, axis=0)  * 15.86
       return  vol*vol
 
    if criteria=="linear":
       tcrash= np.arange(0,self.tmax-1)
       rett= np.arange(0, self.tmax+1) * 0.07/252.0
       ss= np.sum(np.abs( (bsk[tcrash]-100) - rett[tcrash]*100))*10
       return  ss 

  def multiperiod_ww(self, t):
    kid= self.kmaxmodel-1; kdate=0; tt= self.dateref[t]
    for k in xrange(0, self.kmaxmodel-1) :
  #    if tt > self.model[k,1][0] and  tt < self.model[k,1][1]  :        #In sample
      if tt >= self.model[k,1][1] and    tt <= self.model[k+1,1][1]  :  # OutSample
             kdate= self.model[k,1][1]
             kid= k
             
    print self.dateref[t], self.model[kid,1]
    return self.model[kid,2]
    
    
  def help(self):
   print(''' ''')


  #Extra Params  Optimization, fixed Asset weights-----------------------------------
  ''' def _weightcalc_regimeExtra(self,wwextra,t):  
    wwvec= self.wwasset                             # Fixed weight from input    
      
    if t < self.nbrange:  return self.ww0
    else:          
      risk=  self._regimecalc(t, wwextra)                        # Risk(t-2) --> ww(t-1), used in t-1,t       
      wwmat= np.reshape(wwvec,(self.masset, self.nbregime))      # Reshape into matrix asset x nbRegime
      ww2=   self._mapping_risk_ww(risk, wwmat)                  # Market Cycle/Regime from Risk Indicator Level
     return ww2
  '''
  

  '''
    bounds= self.boundsoptim 
    if isreset==2 :  
      print('Traditionnal Optim, no saving')      
      res= sci.optimize.differential_evolution(self.calcbasket_obj, bounds=bounds, maxiter=maxiter)
      xbest, fbest, solver, i= res.x, res.fun, '', maxiter
    else :   #iterative solver
      print('Iterative Solver ')
      if name1 != '' :  # wtih file
        print '/batch/'+name1
        solver2= util.load_obj('/batch/'+name1);                  imin= int(name1[-3:])+1
        solver= sci.optimize._differentialevolution.DifferentialEvolutionSolver(self.calcbasket_obj, bounds=bounds, popsize=popsize)    
        solver.population= copy.deepcopy(solver2.population)  
        solver.population_energies= copy.deepcopy(solver2.population_energies)  
        del solver2  
      
      elif isreset==0 : # Start from zero     
        solver= copy.deepcopy(self.res[0]);            imin= self.res[1]+1     
      else :
        solver= sci.optimize._differentialevolution.DifferentialEvolutionSolver(self.calcbasket_obj, bounds=bounds, popsize=popsize); imin=0      
      
      name1= '/batch/solver_'+self.name+'_'+self.wwtype+'_'+self.criteria+'_'
      fbest0=1500000.0
      for i in xrange(imin, imin+maxiter):
        xbest, fbest = next(solver)              
        print 0,i, fbest, xbest
        self.wwbest= xbest; self.res= (copy.deepcopy(solver), i, xbest, fbest)  
        try :
         util.save_obj(solver, name1+util.date_now()+'_'+util.np_int_tostr(i))
        except :  pass      
        if np.mod(i+1, 11)==0 :
            if np.abs(fbest - fbest0) < 0.001 : break;
            fbest0= fbest  


    self.wwbest= xbest; self.res= (copy.deepcopy(solver), i, xbest, fbest)     
    if self.wwtype== "regime" :
       self.wwbest2=  np.reshape(xbest,(self.masset, self.nbregime))
       self.wwbest2=  self.wwbest2 / np.sum(self.wwbest2, axis=0)
  '''
  
  '''
  def mapping_risk_ww(self, risk, wwmat, ww2=self.wwasset0):
      if   risk== 6.0 :    return ww2                               # Super Bear
      elif risk== 0.0 :    vv= wwmat[:,0];   ww2 = vv / np.sum(vv)  
      elif risk== 2.0 :    vv= wwmat[:,1];   ww2 = vv / np.sum(vv)
      elif risk== 4.0 :    vv= wwmat[:,2];   ww2 = vv / np.sum(vv)  
      elif risk== 8.0 :    ww2 = np.array([ 0.0, 0.1, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4 ])
        
      return ww2    
  '''
  
  
  '''
  def _mapping_calc_risk(self, ss, tr, t, risk0) :
     # Pattern Regime:   Drop --> Crash --> OVerSold --> Recovery ---> Bull ---> Drop 
    
     if (ss[t, 0] < -13.0 and ss[t, 2] < -18.0) or (ss[t, 0] < -9.0 and ss[t, 2] < -25.0) :  
         risk = 6.0  # Super Bear Protection

     elif (ss[t, 0] <= 0.001 and ss[t-1, 0] >= ss[t, 0]-0.01 ) or  ss[t, 0] <= -1.0  :  #Crash Protection
       
        if ss[t, 0] <= -10.0 and ss[t, 2] < -15.0 and (risk0 == 2.0 or risk0 == 0.0 ) :  
            risk= 0.0   #Recovery from botoom, over-sold position
          
        elif ss[t, 0] == 0.0  and ss[t, 1] > 0.0 and ss[t-1, 1] < 0.01 and ss[t-2, 1] < 0.01  and risk0 == 2.0 :
            risk= 0.0   #Recovery from bottom
           
        else :  
          risk = 2.0  # Crash Protection

     elif (risk0==4.0 or risk0==2.0) and ss[t, 2] < -7.0  :  #Drawdown from Bull position
         risk = 2.0 

     else:
         risk = 4.0  # Bull Case
     return risk
     '''
  '''
     print tr[5]
     # Previous Strategy 
     if  ss[t,0] < tr[0] and ss[t,2] < tr[1] :
         risk= 6.0    # Super Bear Protection
      
     # elif  ss[t,5] > 95.0      :  # 35  10
     #    risk= 8.0    # Mean reversion period
     #    self.tstart= t
 
     # elif  (risk0==8.0   and t-self.tstart < 10  )    :  # 45   10
     #    risk= 8.0    # Mean reversion period

     elif (ss[t,2] < tr[2] and risk0==2.0) or (tr[3] < ss[t,1] < tr[4] and risk0==0.0  )  :
         risk= 0.0    # Bottom oversold

     elif ((ss[t, 0] <= tr[5]) or (risk0 == 4.0 and ss[t, 2] < tr[6]) or (ss[t, 2] < tr[7])):
         risk = 2.0  # Crash Protection

     else :
         risk= 4.0    # Bull Case 
     return risk         
     '''





#####################################################################################
#----------------- Portfolio analysis------------------------------------------------
def drawdown_calc(price): 
    n= len(price); 
    maxprice = np.zeros(n); ddowndur2 = np.zeros(n);
    dd = np.zeros(n); ddowndur = np.zeros(n); ddstart = np.zeros(n); 
    ddmin_date = np.zeros(n); 
    maxstartdate=0
    minlevel= 0
    for t in range(1, n):

        if maxprice[t-1] < price[t]: 
          maxprice[t]= price[t]  #update
          maxstartdate= t #update the start date
          minlevel=0
        else :       
          maxprice[t]= maxprice[t-1]
            
        dd[t] = price[t]/maxprice[t] -1 #drawdown level
        
        
        if dd[t] !=0 : #Increase period of same drawdown
         ddstart[t] = maxstartdate 
         ddowndur[t]=  1 + ddowndur[t-1]
         if dd[t] < minlevel : #Find lowest level
             minlevel = dd[t]
             ddowndur2[t]=  ddowndur[t]
             ddmin_date[t]=  t
         else:
             ddowndur2[t]= ddowndur2[t-1]
             ddmin_date[t]=  ddmin_date[t-1]
             
    return dd, ddstart, ddowndur, ddowndur2, ddmin_date

# dd, ddstart, ddowndur, ddowndur2, ddmindate= drawdown_calc(bsk)


def drawdown_calc2(price): 
    n= len(price); 
    maxprice = np.zeros(n); ddowndur2 = np.zeros(n);
    dd = np.zeros(n); ddowndur = np.zeros(n); ddstart = np.zeros(n); 
    ddmin_date = np.zeros(n); 
    maxstartdate=0
    minlevel= 0
    for t in range(1, n):

        if maxprice[t-1] < price[t]: 
          maxprice[t]= price[t]  #update
          maxstartdate= t #update the start date
          minlevel=0
        else :       
          maxprice[t]= maxprice[t-1]
            
        dd[t] = price[t]/maxprice[t] -1 #drawdown level
               
        if dd[t] !=0 : #Increase period of same drawdown
         ddstart[t] = maxstartdate 
         ddowndur[t]=  1 + ddowndur[t-1]
         if dd[t] < minlevel : #Find lowest level
             minlevel = dd[t]
             ddowndur2[t]=  ddowndur[t]
             ddmin_date[t]=  t
         else:
             ddowndur2[t]= ddowndur2[t-1]
             ddmin_date[t]=  ddmin_date[t-1]
             
    return dd, ddstart, ddowndur, ddowndur2, ddmin_date

# dd, ddstart, ddowndur, ddowndur2, ddmindate= drawdown_calc(bsk)




def drawup_calc(price): 
    n= len(price); 
    maxprice = np.zeros(n); ddowndur2 = np.zeros(n);
    ddup= np.zeros(n); dd = np.zeros(n); 
    ddmin_date = np.zeros(n); 
    minlevel= 0
    for t in range(1, n):

        if maxprice[t-1] < price[t]: 
          maxprice[t]= price[t]  #update
          minlevel=0
        else :       
          maxprice[t]= maxprice[t-1]
            
        dd[t] = price[t]/maxprice[t] -1 #drawdown level
               
        if dd[t] !=0 : #Increase period of same drawdown              
         if dd[t] < minlevel : #Find lowest level         
             minlevel = dd[t]
             ddup[t]= -dd[t]
             ddmin_date[t]=  t
             ddowndur2[t]=  0
         else:
             ddowndur2[t]= ddowndur2[t-1]+1
             ddmin_date[t]=  ddmin_date[t-1]
             ddup[t]= ddup[t-1]
             
    return ddup, ddmin_date,  ddowndur2,  ddup


def drawup_calc2(price): 
    n= len(price); 
    minprice = np.zeros(n); ddowndur2 = np.zeros(n);
    ddup= np.zeros(n); dd = np.zeros(n); 
    ddmin_date = np.zeros(n); 
    minlevel= 0; minprice[0]= price[0]
    for t in range(1, n):

        if minprice[t-1] > price[t]: 
          minprice[t]= price[t]  #update
          minlevel=0
        else :       
          minprice[t]= minprice[t-1]
            
        dd[t] = price[t]/minprice[t] -1 #drawdown level
               
        if dd[t] !=0 : #Increase period of same drawdown              
         if dd[t] > minlevel : #Find lowest level         
             minlevel = dd[t]
             ddup[t]= dd[t]
             ddmin_date[t]=  t
             ddowndur2[t]=  0
         else:
             ddowndur2[t]= ddowndur2[t-1]+1
             ddmin_date[t]=  ddmin_date[t-1]
             ddup[t]= ddup[t-1]
             
    return ddup, ddmin_date,  ddowndur2,  ddup


 # dd, ddstart, ddowndur, ddowndur2, ddmindate= drawdown_calc(bsk)
 #ddstart, ddowndur2,  ddminlevel1= drawup_calc(bsk)
 # bsk= np.array(asset[0,:])

def pp(x): return str(round(x,3))
def rr(x): return round(x,2)

   
def getweight(ww,size=(9,3), norm=1):
 ww2= np.reshape(ww, size)
 if norm== 1 : ww2= ww2/ np.sum(ww2, axis=0)
 return ww2



def folio_concenfactor(ww, masset=12, isnorm=1):
 wwraw= np.reshape(ww,(masset, 3)) 
 if isnorm : wwraw= wwraw / np.sum(wwraw, axis=0) 
 ss=0.0; lss=[]
 for k in xrange(0,3) :
   aux= np.sum(np.abs(wwraw[:,k] - np.mean(wwraw[:,k])      ))
   ss+= aux
   lss.append(aux)  
 return ss*100, lss  



 
def folio_metric(pf4, wwbest, noprint=1, masset=12) :
  pf4.wwbest= wwbest
  bsk, wwall= pf4.plot(tickperday=252, show1=0)
  bvolta, wwvolta= folio_volta(bsk,  0.08, 40, 1.0, isweight=1, tlag=0)

  date1= pf4.dateref
  tmax= len(date1)
  date1= dateint_todatetime(date1)
  tyear= datediff_inyear(date1[-1], date1[0])

  volfull= volhisto_fromprice(bsk,tmax-1, tmax-1, axis=0)
  perfull= bsk[-1]/bsk[0] - 1
  perfyearly=  np.power(1+perfull, 1/tyear) -1
  sharpe= perfyearly/volfull
  dd, ddstart, ddowndur, ddowndur2, ddmin_date= drawdown_calc(bsk)
  ddmax= np.min(dd)

  perfvolta= bvolta[-1]/bvolta[0] - 1

  perf3mth= bsk[-1]/bsk[-60] -  1  if tyear > 0.2 else bsk[-1]/bsk[0] - 1
  perf6mth= bsk[-1]/bsk[-120] - 1  if tyear > 0.5 else bsk[-1]/bsk[0] - 1
  vol6mth= volhisto_fromprice(bsk,tmax-1, 120, axis=0)  if tyear > 0.5 else   volhisto_fromprice(bsk,tmax-1, tmax-1, axis=0)

  cc= folio_concenfactor(wwbest, masset)

  if noprint :
     cols=  ['bsk', 'perfull', 'perfyearly', 'volfull', 'sharpe', 'concen', 'drawdown', 'perf3mth', 'perf6mth', 'vol6mth', 'extra']
     return [bsk[-1], perfull * 100, perfyearly * 100, volfull * 100, sharpe, cc[0], ddmax*100, perf3mth*100, perf6mth*100, vol6mth*100, ('perfvolta', perfvolta*100)], cols
  else :
     print(bsk[-1], perfull*100, perfyearly*100, volfull*100, sharpe, cc[0], ddmax, ('perfvolta', perfvolta*100) )



def folio_analysis(date1, bsk, tablefmt="simple") :
 tmax= len(date1)
 if len(bsk) != tmax:  print("Date and Data not aligned"); return -1;

 #----Perf, vol, perf compound, sharpe, 
 date1= dateint_todatetime(date1)
 tyear= datediff_inyear(date1[-1], date1[0])
 volfull= volhisto_fromprice(bsk,tmax-1, tmax-1, axis=0)
 vol252= volhisto_fromprice(bsk,tmax-1, 252, axis=0)
 perfull= bsk[-1]/bsk[0] - 1
 perfyearly=  np.power(1+perfull, 1/tyear) -1
 sharpe= perfyearly/volfull

 headers1=  ["Period", "Tot_Perf","Yearly_Perf", "Sharpe", "Vol_full", "Vol_1year_last"] 
 data1=[[tyear,rr(100*perfull), rr(100*perfyearly), rr(sharpe), rr(100*volfull), rr(100*vol252)]]
 print tabulate(data1, headers=headers1, tablefmt=tablefmt) ; print "\n"


 #------Yearly: perf, volatility, sharpe
 date_31dec= date_getspecificdate(date1, "yearend");  nyears= len(date_31dec)

 data2= []
 for i in range(1,nyears) :
   bsk1y= bsk[(date_31dec[i-1]):(date_31dec[i]+1)];  tmax1y= len(bsk1y)  
   perf= bsk[date_31dec[i]] / bsk[date_31dec[i-1]]-1
   vol252= volhisto_fromprice(bsk1y,tmax1y, 252, axis=0)
   sharpe= perf / vol252   
   yearend= datetime_tostring(date1[date_31dec[i]])
  # print bsk1y[-1], bsk1y[0], date1[date_31dec[i-1]], date1[date_31dec[i]]
   data2.append([yearend,rr(bsk[date_31dec[i]]), rr(100*perf), rr(100*vol252), rr(sharpe)  ])
 
 headers1=  ["YearEnd", "Bsk", "Perf",  "Vol1Y", "Sharpe" ]   
 print tabulate(data2, headers=headers1, tablefmt=tablefmt);  print "\n"


 #------Yearly: perf, volatility, sharpe
 ''' date_31dec= date_getspecificdate(date1, "yearend");  nyears= len(date_31dec)

 data2= []
 for i in range(1,nyears) :
   bsk1y= bsk[(date_31dec[i-1]):(date_31dec[i]+1)];  tmax1y= len(bsk1y)  
   perf= bsk[date_31dec[i]] / bsk[date_31dec[i-1]]-1
   vol252= volhisto_fromprice(bsk1y,tmax1y, 252, axis=0)
   sharpe= perf / vol252   
   yearend= datetime_tostring(date1[date_31dec[i]])
   print bsk1y[-1], bsk1y[0], date1[date_31dec[i-1]], date1[date_31dec[i]]
   data2.append([yearend,rr(bsk[date_31dec[i]]), rr(100*perf), rr(100*vol252), rr(sharpe)  ])
 
 headers1=  ["YearEnd", "Bsk", "Perf",  "Vol1Y", "Sharpe" ]   
 print tabulate(data2, headers=headers1, tablefmt=tablefmt);  print "\n"
  '''




 #------Drawdown: count, startDate, EndDate, duration, loss in %
 draw_pct, ddstart, ddurlast, ddurmin, ddmindate = drawdown_calc( bsk)

 data3= []
 for i in range(0,len(draw_pct)) :
  if ddstart[i]!= ddstart[i-1] and ddstart[i-1] !=0  and  ddurlast[i-1]>10 :
    data3.append([datetime_tostring(date1[int(ddstart[i-1])]),
                  datetime_tostring(date1[int(ddmindate[i-1])]),
                  rr(draw_pct[ddmindate[i-1]]), ddurmin[i-1],  ddurlast[i-1]])     
    
 headers1= ["DrawStart", "DrawWorstDate", "WorstDrawPct ", "WorstDrawDur", "DrawDur"] 
 print tabulate(data3, headers=headers1, tablefmt=tablefmt) ; print "\n\n"     


 #------DrawUp: count, startDate, EndDate, duration, loss in %
 dd,ddstart, ddurlast, ddup = drawup_calc( bsk)

 data3= []
 for i in range(0,len(ddstart)) :
  if ddstart[i]!= ddstart[i-1] and ddstart[i-1] !=0  and  ddurlast[i-1]>5 :
    data3.append([datetime_tostring(date1[int(ddstart[i-1])]),
                    rr(ddup[i-1]), ddurlast[i-1] ])     
    
   
 headers1= ["Uptart",  "UpPct ",  "UpDur"] 
 print tabulate(data3, headers=headers1, tablefmt=tablefmt) ; print "\n\n"     


          
#Rally:  count, startDate, EndDate, duration, loss in %

#dateref3= dateref2[0:6118]
# portfolio_analysis(dateref3, bsk) 




'''
 idx_max, max_draw_pct = min_withposition(draw_pct)
 max_draw_dur= ddurlast[idx_max] 
 draw_end= datetime_tostring(date1[idx_max])
 draw_start=  0 #datetime_tostring(date1[idx_max+max_draw_dur])
 
 headers1=  ["MaxDraw", "MaxDraw Dur", "Start", "End"] 
 data1=[[max_draw_pct, max_draw_dur, draw_start, draw_end]]
 print tabulate(data1, headers=headers1, tablefmt=tablefmt) 
 print "\n\n"
'''

'''



'''
##################################################################################






##############################################################################
 #------------------Drawdown calculation --------------------------------------------

#drawdown , drawdowndur= drawdown_calc(bsk)


 
''' 
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt

# Get SPY data for past several years  
def drawdown_calc(dateref, price, window = 252) :

 Roll_Max = pd.rolling_max(price, window, min_periods=1)
 Daily_Drawdown = price/Roll_Max - 1.0 
 Max_Daily_Drawdown = pd.rolling_min(Daily_Drawdown, window, min_periods=1)

# SPY = web.DataReader('SPY', 'yahoo', datetime.date(2007,1,1))

# Calculate the max drawdown in the past window days for each day in the series.
# Use min_periods=1 if you want to let the first 252 days data have an expanding window

# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window

# Plot the results
#Daily_Drawdown.plot()
#Max_Daily_Drawdown.plot()
#plt.show()
http://quanttech.co/2015/03/23/visualising-strategy-drawdowns/
'''



'''
#------------------Cython encoding of the drawdown---------------------------------
# Before the launch of cython script
import cython
%load_ext Cython

#---Execute this part---------------------------------------------------
%%cython
import numpy as np
cimport numpy as np
 
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
 
cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
def dd_cython(np.ndarray[DTYPE_t] s):
    cdef np.ndarray[DTYPE_t] maxprice = np.zeros(len(s),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] drawdown = np.zeros(len(s),dtype=DTYPE)
    cdef np.ndarray[DTYPE_t] drawdowndur = np.zeros(len(s),dtype=DTYPE)
 
    cdef int t
    for t in range(1,len(s)):
        maxprice[t] = max(maxprice[t-1], s[t])
        drawdown[t] = (maxprice[t]-s[t])
        drawdowndur[t]= (0 if drawdown[t] == 0 else drawdowndur[t-1]+1)      
    return drawdown , drawdowndur

'''






'''
http://www.r-bloggers.com/import-japanese-equity-data-into-r-with-quantmod-0-4-4/

# getSymbols.yahooj {{{
"getSymbols.yahooj" <-
    function(Symbols, env=parent.frame(), return.class='xts', index.class="Date",
             from='2007-01-01',
             to=Sys.Date(),
             ...)
    {
        importDefaults("getSymbols.yahooj")



        
        yahoo.URL <- "http://info.finance.yahoo.co.jp/history/"
        
        
        for(i in 1:length(Symbols)) {
            return.class <- getSymbolLookup()[[Symbols[[i]]]]$return.class
            return.class <- ifelse(is.null(return.class),default.return.class,
                                   return.class)
            from <- getSymbolLookup()[[Symbols[[i]]]]$from
            from <- if(is.null(from)) default.from else from
            to <- getSymbolLookup()[[Symbols[[i]]]]$to
            to <- if(is.null(to)) default.to else to

            Symbols.name <- getSymbolLookup()[[Symbols[[i]]]]$name
            Symbols.name <- ifelse(is.null(Symbols.name),Symbols[[i]],Symbols.name)


            page <- 1
            totalrows <- c()
            while (TRUE) {
                tmp <- tempfile()
                download.file(paste(yahoo.URL,
                                    "?code=",Symbols.name,
                                    "&sm=",from.m,
                                    "&sd=",sprintf('%.2d',from.d),
                                    "&sy=",from.y,
                                    "&em=",to.m,
                                    "&ed=",sprintf('%.2d',to.d),
                                    "&ey=",to.y,
                                    "&tm=d",
                                    "&p=",page,
                                    sep=''),destfile=tmp,quiet=!verbose)
                                    
                page <- page + 1
            }


            # Available columns
            cols <- c('Open','High','Low','Close','Volume','Adjusted')


            # Process from the start, for easier stocksplit management
            totalrows <- rev(totalrows)
            mat <- matrix(0, ncol=length(cols) + 1, nrow=0, byrow=TRUE)
            for(row in totalrows) {
                cells <- getNodeSet(row, "td")

                # 2 cells means it is a stocksplit row
                # So extract stocksplit data and recalculate the matrix we have so far
                if (length(cells) == 2 & length(cols) == 6 & nrow(mat) > 1) {
                    ss.data <- as.numeric(na.omit(as.numeric(unlist(strsplit(xmlValue(cells[[2]]), "[^0-9]+")))))
                    factor <- ss.data[2] / ss.data[1]

                    mat <- rbind(t(apply(mat[-nrow(mat),], 1, function(x) {
                        x * c(1, rep(1/factor, 4), factor, 1)
                    })), mat[nrow(mat),])
                }

                if (length(cells) != length(cols) + 1) next

                date <- as.Date(xmlValue(cells[[1]]), format="%Y年%m月%d日")
                entry <- c(date)
                for(n in 2:length(cells)) {
                    entry <- cbind(entry, as.numeric(gsub(",", "", xmlValue(cells[[n]]))))
                }

                mat <- rbind(mat, entry)
            }

            fr <- xts(mat[, -1], as.Date(mat[, 1]), src="yahooj", updated=Sys.time())
            symname <- paste('YJ', toupper(Symbols.name), sep="")
            colnames(fr) <- paste(symname, cols, sep='.')



  
            if(i >= 5 && length(Symbols) > 5) {
                message("pausing 1 second between requests for more than 5 symbols")
                Sys.sleep(1)


'''


'''
util.getmodule_doc("bt.backtest")

bt.algos
bt.backtest
bt.core

Module: bt.algos-------------------------------------------------
    
   +Class: Algo
    
   +Class: AlgoStack
    
   +Class: CapitalFlow
          +  __call__(self, target)
          +  __init__(self, amount)
    
   +Class: CloseDead
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: Debug
          +  __call__(self, target)
    
   +Class: LimitDeltas
          +  __call__(self, target)
          +  __init__(self, limit)
        	  	  Default_Args:(limit, 0.1)
    
   +Class: LimitWeights
          +  __call__(self, target)
          +  __init__(self, limit)
        	  	  Default_Args:(limit, 0.1)
    
   +Class: PrintDate
          +  __call__(self, target)
    
   +Class: PrintInfo
          +  __call__(self, target)
          +  __init__(self, fmt_string)
        	  	  Default_Args:(fmt_string, {full_name} {now})
    
   +Class: PrintTempData
          +  __call__(self, target)
    
   +Class: Rebalance
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: RebalanceOverTime
          +  __call__(self, target)
          +  __init__(self, n)
        	  	  Default_Args:(n, 10)
    
   +Class: Require
          +  __call__(self, target)
          +  __init__(self, pred, item, if_none)
        	  	  Default_Args:(if_none, False)
    
   +Class: RunAfterDate
          +  __call__(self, target)
          +  __init__(self, date)
    
   +Class: RunAfterDays
          +  __call__(self, target)
          +  __init__(self, days)
    
   +Class: RunDaily
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: RunEveryNPeriods
          +  __call__(self, target)
          +  __init__(self, n, offset)
        	  	  Default_Args:(offset, 0)
    
   +Class: RunMonthly
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: RunOnDate
          +  __call__(self, target)
          +  __init__(self)
        	   Positional_Args: dates
    
   +Class: RunOnce
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: RunQuarterly
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: RunWeekly
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: RunYearly
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: SelectAll
          +  __call__(self, target)
          +  __init__(self, include_no_data)
        	  	  Default_Args:(include_no_data, False)
    
   +Class: SelectHasData
          +  __call__(self, target)
          +  __init__(self, lookback, min_count, include_no_data)
        	  	  Default_Args:(lookback, <DateOffset: kwds={months: 3}>), (min_count, None), (include_no_data, False)
    
   +Class: SelectMomentum
          +  __init__(self, n, lookback, lag, sort_descending, all_or_none)
        	  	  Default_Args:(lookback, <DateOffset: kwds={months: 3}>), (lag, <DateOffset: kwds={days: 0}>), (sort_descending, True), (all_or_none, False)
    
   +Class: SelectN
          +  __call__(self, target)
          +  __init__(self, n, sort_descending, all_or_none)
        	  	  Default_Args:(sort_descending, True), (all_or_none, False)
    
   +Class: SelectRandomly
          +  __call__(self, target)
          +  __init__(self, n, include_no_data)
        	  	  Default_Args:(n, None), (include_no_data, False)
    
   +Class: SelectThese
          +  __call__(self, target)
          +  __init__(self, tickers, include_no_data)
        	  	  Default_Args:(include_no_data, False)
    
   +Class: SelectWhere
          +  __call__(self, target)
          +  __init__(self, signal, include_no_data)
        	  	  Default_Args:(include_no_data, False)
    
   +Class: StatTotalReturn
          +  __call__(self, target)
          +  __init__(self, lookback, lag)
        	  	  Default_Args:(lookback, <DateOffset: kwds={months: 3}>), (lag, <DateOffset: kwds={days: 0}>)
    
   +Class: WeighEqually
          +  __call__(self, target)
          +  __init__(self)
    
   +Class: WeighInvVol
          +  __call__(self, target)
          +  __init__(self, lookback, lag)
        	  	  Default_Args:(lookback, <DateOffset: kwds={months: 3}>), (lag, <DateOffset: kwds={days: 0}>)
    
   +Class: WeighMeanVar
          +  __call__(self, target)
          +  __init__(self, lookback, bounds, covar_method, rf, lag)
        	  	  Default_Args:(lookback, <DateOffset: kwds={months: 3}>), (bounds, (0.0, 1.0)), (covar_method, ledoit-wolf), (rf, 0.0), (lag, <DateOffset: kwds={days: 0}>)
    
   +Class: WeighRandomly
          +  __call__(self, target)
          +  __init__(self, bounds, weight_sum)
        	  	  Default_Args:(bounds, (0.0, 1.0)), (weight_sum, 1)
    
   +Class: WeighSpecified
          +  __call__(self, target)
          +  __init__(self)
        	   Keyword_Args: weights
    
   +Class: WeighTarget
          +  __call__(self, target)
          +  __init__(self, weights)
      +Func: iteritems(obj)
    	   Keyword_Args: kwargs
      +Func: run_always(f)

 
Module: bt.backtest-------------------------------------------------
   +Class: Backtest
          +  run(self)
          +  __init__(self, strategy, data, name, initial_capital, commissions, integer_positions)
        	  	  Default_Args:(name, None), (initial_capital, 1000000.0), (commissions, None), (integer_positions, True)
    
   +Class: RandomBenchmarkResult
          +  plot_histogram(self, statistic, figsize, title, bins)
        	  	  Default_Args:(statistic, monthly_sharpe), (figsize, (15, 5)), (title, None), (bins, 20)
        	   Keyword_Args: kwargs
          +  __init__(self)
        	   Positional_Args: backtests
    
   +Class: Result
          +  plot_security_weights(self, backtest, filter, figsize)
        	  	  Default_Args:(backtest, 0), (filter, None), (figsize, (15, 5))
        	   Keyword_Args: kwds
          +  display_monthly_returns(self, backtest)
        	  	  Default_Args:(backtest, 0)
          +  plot_weights(self, backtest, filter, figsize)
        	  	  Default_Args:(backtest, 0), (filter, None), (figsize, (15, 5))
        	   Keyword_Args: kwds
          +  plot_histogram(self, backtest)
        	  	  Default_Args:(backtest, 0)
        	   Keyword_Args: kwds
          +  _get_backtest(self, backtest)
          +  __init__(self)
        	   Positional_Args: backtests
      +Func: benchmark_random(backtest, random_strategy, nsim)
    	  	  Default_Args:(nsim, 100)
      +Func: deepcopy(x, memo, _nil)
    	  	  Default_Args:(memo, None), (_nil, )
    	   Positional_Args: backtests

 
Module: bt.core-------------------------------------------------
   +Class: Algo
   +Class: AlgoStack
   +Class: Node
   +Class: SecurityBase
   +Class: Strategy
   +Class: StrategyBase
      +Func: deepcopy(x, memo, _nil)   Default_Args:(memo, None), (_nil, )

'''

''' Backtesting
http://synesthesiam.com/posts/an-introduction-to-pandas.html


https://github.com/pmorissette/bt
!pip install bt

import bt
#%pylab inline
# download data
data = bt.get('aapl,msft,c,gs,ge', start='2010-01-01')

# calculate moving average DataFrame using pandas' rolling_mean
import pandas as pd
# a rolling mean is a moving average, right?
sma = pd.rolling_mean(data, 50)


# let's see what the data looks like - this is by no means a pretty chart, but it does the job
plot = bt.merge(data, sma).plot(figsize=(15, 5))


class SelectWhere(bt.Algo):
    """
    Selects securities based on an indicator DataFrame.
    Selects securities where the value is True on the current date (target.now).

    Args: * signal (DataFrame): DataFrame containing the signal (boolean DataFrame)

    Sets: * selected

    """
    def __init__(self, signal):  self.signal = signal

    def __call__(self, target):
        # get signal on target.now
        if target.now in self.signal.index:
            sig = self.signal.ix[target.now]

            # get indices where true as list
            selected = list(sig.index[sig])

            # save in temp - this will be used by the weighing algo
            target.temp['selected'] = selected

        # return True because we want to keep on moving down the stack
        return True

# first we create the Strategy
s = bt.Strategy('above50sma', [SelectWhere(data > sma),
                               bt.algos.WeighEqually(),
                               bt.algos.Rebalance()])

# now we create the Backtest
t = bt.Backtest(s, data)

# and let's run it!
res = bt.run(t)

res.plot('d')

res.display()


#------------------Financial Function in Python----------------------------
!pip install ffn

import ffn
returns = ffn.get('aapl,msft,c,gs,ge', start='2010-01-01').to_returns().dropna()
returns.calc_mean_var_weights().as_format('.2%')

'''




'''
#------------------Drawdown calculation --------------------------------------------
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as pp

# Get SPY data for past several years
SPY_Dat = web.DataReader('SPY', 'yahoo', datetime.date(2007,1,1))

# We are going to use a trailing 252 trading day window
window = 252

# Calculate the max drawdown in the past window days for each day in the series.
# Use min_periods=1 if you want to let the first 252 days data have an expanding window
Roll_Max = pd.rolling_max(SPY_Dat['Adj Close'], window, min_periods=1)
Daily_Drawdown = SPY_Dat['Adj Close']/Roll_Max - 1.0

# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
Max_Daily_Drawdown = pd.rolling_min(Daily_Drawdown, window, min_periods=1)

# Plot the results
Daily_Drawdown.plot()
Max_Daily_Drawdown.plot()
pp.show()

http://quanttech.co/2015/03/23/visualising-strategy-drawdowns/




http://quanttech.co/contact/


#----------Micro Service for application
http://quanttech.co/2015/06/23/quickly-run-up-microservices-for-your-trading-apps/


'''

#--------------------Statistical Analysis of Time Series-----------------------------

#--------------------Calculate Rank Table    ---------------------------------------
from scipy.stats import norm


def np_countretsign(x):
  s=0
  for k in range(0, len(x)-1) :
     if np.sign(x[k+1] - x[k]) >= 0 :s+= 1    
  return s/(0.0+ len(x)  )  
  

def np_trendtest(x, alpha = 0.05):
    """
    This function is derived from code originally posted by Sat Kumar Tomer (satkumartomer@gmail.com)
    See also: http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm
    The purpose of the Mann-Kendall (MK) test (Mann 1945, Kendall 1975, Gilbert 1987) is to statistically assess if there is a monotonic upward or downward trend of the variable of interest over time. A monotonic upward (downward) trend means that the variable consistently increases (decreases) through time, but the trend may or may not be linear. The MK test can be used in place of a parametric linear regression analysis, which can be used to test if the slope of the estimated linear regression line is different from zero. The regression analysis requires that the residuals from the fitted regression line be normally distributed; an assumption not required by the MK test, that is, the MK test is a non-parametric (distribution-free) test.
    Hirsch, Slack and Smith (1982, page 107) indicate that the MK test is best viewed as an exploratory analysis and is most appropriately used to identify stations where changes are significant or of large magnitude and to quantify these findings.
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)
    
    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics 
        
    Examples x = np.random.rand(100) trend,h,p,z = mk_test(x,0.05) 
    """
    n = len(x)
    # calculate S 
    s = 0
    for k in range(n-1):
        for j in range(k+1,n): s += np.sign(x[j] - x[k])
    
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    
    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
    
    if s>0:  z = (s - 1)/np.sqrt(var_s)
    elif s == 0:    z = 0
    elif s<0: z = (s + 1)/np.sqrt(var_s)
    
    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1-alpha/2) 
    
    if (z<0) and h:  trend = 1
    elif (z>0) and h: trend = -1
    else:  trend = 0
        
    return trend, h, p, z
    
    
    
    
    
def correl_rankbystock(stkid=[2,5,6], correl=[[1,0],[0,1]]) :
 """ Ranking of stocks by correl:  Stock i, Correl_with_i, Abs_Sum_correl_i    """
 avgcorrel= np.zeros((len(stkid),3), dtype=np.float32) 
 # stkid= [int(x) for x in stkid]
 for ix, i in enumerate(stkid) :
   avgcorrel[ix,0]= i
   avgcorrel[ix,1]= np.sum(correl[i, stkid]) -1
   avgcorrel[ix,2]= np.sum(np.abs(correl[i, stkid])) -1 
 avgcorrel= util.sortcol(avgcorrel, 1, asc=True)
 return avgcorrel
 




def calc_print_correlrank(close2, symjp1, nlag, refindexname, toprank2=5, customnameid=[], customnameid2=[]) :
 ''' Most correlated/Un-correlated from One Risk Factor'''
 refindex= util.np_findfirst(refindexname, symjp1)
 rank= calc_ranktable(close2, symjp1, 24, refindex=refindex, funeval=similarity_correl, funargs=['empirical'])
 nrank= np.shape(rank)[0]-1

 #----Show Best Rank + Graph--------------------------------
 print("\n\n Best Rank \n"  ) 
 toprank= 50;
 refindex= util.np_find('jp7203', symjp1)
 for kk, i in enumerate(range(0, toprank,1) ) :
   stki= int(rank[i,0]); namei= symjp1[stki][2:6]
   print kk ,stki, ';'+a_nk400name[util.np_find(namei, a_nk400list)  ]+';', namei+';', rank[i,1]  

 #----Lowest Rank -----------------------------------------
 print("\n\n Lowest Rank \n"  ) 
 for kk, i in enumerate(range(nrank-1, nrank-toprank,-1) ) :
   stki= int(rank[i,0]); namei= symjp1[stki][2:6]
   print kk ,stki, ';'+a_nk400name[util.np_find(namei, a_nk400list)  ]+';', namei+';', rank[i,1]  

 #Show Graph--------------------------------------------
 toprank= 5;
 refindex= util.np_find('jp7203', symjp1)
 if customnameid== [] :
    stk_select= util.np_mergelist(np.array(rank[0:toprank2,0], dtype=int) , [refindex]  )
 else:   stk_select= util.np_mergelist([refindex], customnameid)
 
 
 ret_close2= getret_fromquotes(close2, timelag=1)
 price1= price_normalize100(ret_close2[stk_select, :])
 plot_price(price1-100, sym= [symjp1[x] for x in stk_select  ],
              tickperday=58*1,   # 1 day ---> 58 5mins tick
              label=('Intraday Variation ', 'Days', 'Variation in %'))
              

 #Show Graph--------------------------------------------
 if customnameid2== [] :
       stk_select= util.np_mergelist(np.array(rank[nrank-toprank2:nrank,0], dtype=int) , [refindex]  )
 else: stk_select= util.np_mergelist([refindex], customnameid2)
 
 ret_close2= getret_fromquotes(close2, timelag=1)
 price1= price_normalize100(ret_close2[stk_select, :])
 plot_price(price1-100, sym= [symjp1[x] for x in stk_select  ],
              tickperday=58,   # 1 day ---> 58 5mins tick
              label=('Intraday Variation ', 'Days', 'Variation in %'))
              
 return rank
  
  


def calc_ranktable(close2, symjp1, nlag, refindex, funeval, funargs) :
  #------Run over all the stocks, Calc Evaluation, Put into Rank --------------------
  correlag = np.zeros( (len(symjp1), 2))  
  j=0; refk= refindex
  ret_close2= getlogret_fromquotes(close2, nlag)
  
  for i,stocki in enumerate(symjp1):
    try :
      correl1 = funeval(ret_close2[[i,refk],:], funargs=funargs)
      correlag[j, 0]= i  
      correlag[j, 1]= correl1
      j+=1
    except :  pass  

  correlag= correlag[correlag[:,1] != 0]
  rank= util.np_sortbycolumn(correlag  ,1, asc=False)  #Sort by rank
  
  return rank


def similarity_correl(ret_close2, funargs):
 type1= funargs[0];   
 correl1, _,_= correlation_mat(ret_close2, type1=type1)  
 return correl1[0,1]


#  Calculate most correlated stock with a given index
# rank= calc_ranktable(close2, symjp1, 5, refindex=391, funeval=similarity_correl, funargs=['empirical'])







##################################################################################### 
 #--------------------Search Engine  -------------------------------------------------
'''
#---------------------  Universe Load         ---------------------------------------
filejpstock= r'E:/_data/stock/daily/20160616/jp'
sym01= nk400list

quotes= imp_txt_getquotes(sym01, filejpstock, startdate=20090101, endate=20160616)
open1, dateref= date_align(quotes, type1="close")    #Get the data and align dates
del quotes; util.a_cleanmemory()

#util.save_obj(open1, 'close_nk400_2009_2016')
#util.save_obj(dateref, 'dateref_nk400_2009_2016')

open1= util.load_obj('close_nk400_2009_2016')
dateref= util.load_obj( 'dateref_nk400_2009_2016')

nlag=1
ret_open1 =  getdailyret_fromquotes(open1,nlag)
del open1


#-----Example of Code 
search01= searchSimilarity(sym01=nk400list[0:100], symname=nk400name, 
                           startdate= 20160101, enddate=20160601, pricetype="close")

search01.load_quotes_fromdb(picklefile=('close_nk400_2009_2016', 'dateref_nk400_2009_2016')


search01.set_searchcriteria(typesearch="pattern2", date1=20160401, date2=20160601)
search01.launch_search()

search01.show_comparison_graph(maxresult=5, show_only_different_time=True)

rank= search01.resultrank
'''



#  -----      Similarity definition 
def np_similarity(x,y, wwerr=[], type1=0):
  if type1==2 : return np_distance_l1(x,y, wwerr)
  if type1==1: return sci.spatial.distance.correlation(x, y)  
  if type1==0 : return 1-np.corrcoef(x,y)[0,1]


def np_distance_l1(x,y, wwerr) :
 return np.sum(wwerr * np.abs(x-y))  
  

class searchSimilarity() :
  def __init__(self, filejpstock=r'E:/_data/stock/daily/20160616/jp', sym01=['7203'], symname=['Toyota'], startdate= 20150101, enddate=20160601, pricetype="close"):
      self.symname=symname; self.sym01= sym01; self.filejpstock= filejpstock;
      self.dbstartdate= startdate;  self.dbenddate= enddate; self.pricetype= pricetype  
      self.ret_open1= None;  self.open1= None;  self.nlag= None

  def load_quotes_fromdb(self, picklefile='') :
      if picklefile != '': 
         open1=   util.load_obj(picklefile[0])
         dateref= util.load_obj(picklefile[1])        
      else :
         quotes= imp_txt_getquotes(self.sym01, self.filejpstock, startdate= self.dbstartdate, endate=self.dbenddate)
         self.open1, self.dateref= date_align(quotes, type1= self.pricetype)    #Get the data and align dates
         del quotes; util.a_cleanmemory()

      self.open1= open1;  self.dateref= dateref    

  def __generate_return__(self, nlag):
      self.nlag= nlag
      self.ret_open1 =  getret_fromquotes(self.open1,nlag)
    
  def __overweight__(self, px) :
    #Over-weight Local Max, Local Min
    nx= len(px)
    wwerr= np.ones(nx, dtype=np.float16)
    vvmax= util.np_findlocalmax(px); xprev=0
   # print vvmax
    if type(vvmax) !=int :
     for x in vvmax:
       if x[0] !=0.0 and np.abs(xprev-x[0]) > 6:   #Max not too close to each other
         wwerr[max(0,x[0]-1):min(nx,(x[0]+1))]= 1.5;    xprev= x[0] 

    vvmin= util.np_findlocalmin(px); xprev=0
    if type(vvmin) !=int :    
      for x in vvmin:
       if x[0] !=0.0 and np.abs(xprev-x[0.0]) > 6: 
         wwerr[max(0,x[0]-1):min(nx,(x[0]+1))]= 1.5;    xprev= x[0] 
    
    self.wwerr= wwerr
  
  
  def set_searchcriteria(self, name1='7203', date1=20160301, date2=20160601, nlag=1, searchperiodstart=20120101, typesearch="pattern2",) :
    #--------------------  Search Asset Input     -----------------------------------
    #name1= '7203' ; date1= 20160328; date2= 20160531
    if self.ret_open1 == None: 
       self.nlag=       nlag
       self.ret_open1 = getret_fromquotes(self.open1,nlag)
       self.open1=      None #Free memory
    
    twindow=    date_finddateid(date2,self.dateref) - date_finddateid(date1, self.dateref) 
    tstartsrch= max(0,date_finddateid(searchperiodstart, self.dateref))

    tstartx= util.np_findfirst(date1, self.dateref)              # Date Index
    if tstartx==-1: print('Cannot Find Search Start Date:'+str(date1) )
    stockx=  util.np_findfirst(name1, self.sym01)
    retx=    self.ret_open1[stockx, tstartx:(tstartx+twindow-1)]    # Recent Period
    px=      price_normalize_1d(retx);     nx= len(px)

    #Over-weight Local Max, Local Min
    self.__overweight__(px) 

    if typesearch=="pattern0" : self.typesearch=0 
    if typesearch=="pattern1" : self.typesearch=1 
    if typesearch=="pattern2" : self.typesearch=2 
    
    self.tstartsrch= tstartsrch
    self.tstartx=    tstartx;  self.stockx= stockx;       self.px=     px ;       
    self.nx=     nx       ;  self.twindow= twindow 


  def launch_search(self) :
     #--------------------  Search the Pattern    -----------------------------------
     tmaxx= len(self.dateref)-1;     nstock = len(self.sym01)
     twindow= self.twindow; tstartx= self.tstartx;  stockx=     self.stockx; 
     px= self.px;           wwerr=   self.wwerr;    typesearch= self.typesearch
     tstartsrch= self.tstartsrch
     
     print tstartsrch, tmaxx-twindow, tstartx
          
     similar_list= np.zeros((nstock*(tmaxx-twindow)+1, 3), np.float16)
     i=0
     for iy in range(0, nstock) :
      for t in range(tstartsrch, tmaxx-twindow):
        if iy == stockx and t > tstartx : 
          pass
        else :
         print "ok"
         rety= self.ret_open1[iy, t:(t+twindow-1)]
         py=   price_normalize_1d(rety)
         ss= np_similarity(px, py, wwerr=wwerr, type1= typesearch)
         i+=1
         similar_list[i,0]= iy    # Stock ID
         similar_list[i,1]= t     # Time ID
         similar_list[i,2]= ss

     #-------------------Clean the table --------------------------------------------
     similar_list= util.np_sortbycolumn(similar_list,2, asc=True)
     for i,x in enumerate(similar_list) :
       if similar_list[i,2]==0.0 : similar_list[i,2]= 1000         # Empty cell
       if similar_list[i,1]> tstartx-3 : similar_list[i,2]= 100    # Start Element
     similar_list= util.np_sortbycolumn(similar_list,2, asc=True)  
     self.resultrank= similar_list
   
   
  def show_comparison_graph(self, maxresult=20, show_only_different_time=True, fromid=0, fromend= 0, filenameout='')  :
      resultrank=     self.resultrank[ self.resultrank[:,2] != 0   ]  
      
      if fromend == 1 :
        nmax= np.shape(resultrank)[0]
        fromid= nmax - maxresult
        maxresult= nmax
      
      for i in range(fromid, maxresult) :
        tstarti= int(resultrank[i,1])
        stocki= int(resultrank[i,0])
 
        if show_only_different_time and tstarti > self.tstartx-15  :
          pass; print('Pass same month date')

        else :
          py= price_normalize_1d(self.ret_open1[stocki, tstarti:(tstarti+self.twindow-1)]) 
          
          namei= self.sym01[stocki]
          namefulli= self.symname[util.np_findfirst(namei, self.sym01)]
          tit= namei+ " " +namefulli+ "_from_" + str(self.dateref[tstarti])  
          if filenameout != '':  filenameout +'_'+tit
            
          plot_price(self.px-100, py-100, label=('Similar Stock: '+tit, 'Weeks','% Var'),
                        savename1=filenameout, tickperday=5)


          
  @staticmethod
  def staticmethod(self,x) :
    pass
  

  def get_rankresult(self  , filetosave='') :
    resultrank=     self.resultrank[ self.resultrank[:,0] != 0   ]   
    
    if len(filetosave) > 1 :
          np.savetxt(filetosave, self.resultrank)  
          
    nmax= np.shape(resultrank)[0]
    vv= np.array(self.resultrank, dtype=np.float32)
    vv2= np.empty((nmax,3), dtype=np.object)   
    for ii in range(0, nmax):   
      namei=     self.sym01[ int(vv[ii,0]) ]
      namefulli= self.symname[util.np_findfirst(namei, self.sym01)]       
      vv2[ii, 0]= namei
      vv2[ii, 1]= namefulli
      vv2[ii, 2]=  str(self.dateref[ int(vv[ii,1])])  
    
    for i in range(0,20): print i, vv2[i,0], vv2[i,1],  'From: '+ str(vv2[i,2]) 
   
    return  vv, vv2



  def export_results(self, filename) :
    with open(filename,'w') as f:
      txt= self.to_csv()
      f.write(txt)


'''
#####################################################################################
#---------Run over all the indexes, with time lag -----------------------------------
def np_similarity(x,y, wwerr=[], type1=0):
  if type1==2 : return np_distance_l1(x,y, wwerr)
  if type1==1: return sci.spatial.distance.correlation(x, y)  
  if type1==0 : return 1-np.corrcoef(x,y)[0,1]

def np_distance_l1(x,y, wwerr) :
 return np.sum(wwerr * np.abs(x-y))  
  
#---------------------  Universe Load         ---------------------------------------
filejpstock= r'E:/_data/stock/daily/20160616/jp'
sym01= nk400list

quotes= imp_txt_getquotes(sym01, filejpstock, startdate=20090101, endate=20160616)
open1, dateref= date_align(quotes, type1="close")    #Get the data and align dates
del quotes; util.a_cleanmemory()

#util.save_obj(open1, 'close_nk400_2009_2016')
#util.save_obj(dateref, 'dateref_nk400_2009_2016')

open1= util.load_obj('close_nk400_2009_2016')
dateref= util.load_obj( 'dateref_nk400_2009_2016')

nlag=1
ret_open1 =  getdailyret_fromquotes(open1,nlag)
del open1

#--------------------  Search Asset Input     ---------------------------------------
name1= '7203' 
date1= 20160328
date2= 20160531
twindow= util.np_findfirst(date2, dateref) - util.np_findfirst(date1, dateref) 


tstartx= util.np_findfirst(date1, dateref)          # Date Index
stockx= util.np_findfirst(name1, sym01)
retx= ret_open1[stockx, tstartx:(tstartx+twindow-1)]    # Recent Period
px= price_normalize_1d(retx)
nx= len(px)


#----OverWeight for Local Min, Local Max         -----------------------------------
wwerr= np.ones(nx, dtype=np.float16)
vvmax= util.np_findlocalmax(px); xprev=0
for x in vvmax:
  if x[0] !=0 and np.abs(xprev-x[0]) > 6: 
    wwerr[max(0,x[0]-1):min(nx,(x[0]+1))]= 1.5;    xprev= x[0] 

vvmin= util.np_findlocalmin(px); xprev=0
for x in vvmin:
  if x[0] !=0 and np.abs(xprev-x[0]) > 6: 
    wwerr[max(0,x[0]-1):min(nx,(x[0]+1))]= 1.5;    xprev= x[0] 


#--------------------  Search the Pattern       -------------------------------------
tmaxx= len(dateref)-1
similar_list= np.zeros((nstock*(tmaxx-twindow)+1, 3), np.float16)
i=0
for iy in range(0, nstock) :
 for t in range(0, tmaxx-twindow):
   if iy == stockx and t > tstartx : 
     pass
   else :
    rety= ret_open1[iy, t:(t+twindow-1)]
    py=   price_normalize_1d(rety)
    ss= np_similarity(px, py, wwerr=wwerr, type1=2)
    i+=1
    similar_list[i,0]= iy
    similar_list[i,1]= t
    similar_list[i,2]= ss


 #-------------------Clean the table -------------------------------------------------
rank= util.np_sortbycolumn(similar_list,2, asc=True)
for i,x in enumerate(rank) :
  if rank[i,2]==0.0 : rank[i,2]= 1000         # Empty cell
  if rank[i,1]> tstartx-3 : rank[i,2]= 100    # Start Element
rank= util.np_sortbycolumn(rank,2, asc=True)  


 #----- Show Similar Stocks ---------------------------------------------------------
show_only_different_time= True

for i in range(0,10) :
 tstarti= rank[i,1]
 stocki= int(rank[i,0])
 
 if show_only_different_time and tstarti < tstartx-1  :
   py= price_normalize100(ret_open1[stocki, tstarti:(tstarti+twindow-1)]) 

   namei= sym01[stocki]
   namefulli= nk400name[util.np_findfirst(namei, nk400list)]

   tit= namei+ " " +namefulli+ " " + str(dateref[tstarti])          
   plot_price(px-100, py-100, label=('Similar '+tit, 'Time','% Var') )


kix= util.np_findfirst(name1, statsjp[:,0, ilag])
stocklist01[390]


stocklist01[157] , nk400name[157] ,  dateref1adj[86]     
plot_price(price_normalize100(ret_open1[157, 86:86+twindow-1])-100,px-100   )
'''    
##################################################################################### 
#####################################################################################
  


'''   Imp_  Datasource_   Action_    SubData / Scope  / toDataSource2
Easier for Refactoring

'''

######################### USer generic interface ###############################################
def imp_a_findticker(tickerlist, sym01, symname):
 v=[] 
 for ticker in tickerlist:
  namefulli= symname[util.np_findfirst(ticker, sym01)]   
  v.append(namefulli)  
 print v; return v

def imp_a_close_dateref(sym01, sdate=20100101, edate=20160628, datasource='', typeprice="close"):
  
 if datasource =='' : # Yahoo / Google 
   quotes = imp_yahoo_getquotes(sym01, start=str(sdate), end= str(edate))

 else :             #Text File
  liststockfile= util.listallfile(datasource, "*.txt", dirlevel=3)
  stk0= liststockfile[0]; # stk2= liststockfile[2]
  liststockname=[]
  for k in range(0, len(stk0)):  liststockname.append( (stk0[k])[0:4])

  quotes= imp_txt_getquotes(sym01, datasource, startdate=sdate, endate=edate)
  #quotesdaily= util.load_obj( 'nk400_20100101_20160616')
 
 print "-------------"
 for i,q in enumerate(quotes): 
   print(str(i) + '_'+ sym01[i] + "_" + str(q.date.values[0])+ "_" + str(q.date.values[-1]) )
   
 close1, dateref= date_align(quotes, type1= typeprice) #Get the data and align dates
 del quotes; util.a_cleanmemory()

 return sym01, close1, dateref

def imp_a_get_quotes() :
   pass



#--------------------Import data from Quandl --------------------------------------------------
def imp_quandl_quotes(symbols, start="20150101", end="20160101", source="google", type1=1):
 import Quandl as qd
 d1 = datetime.datetime(int(start[0:4]), int(start[4:6]), int(start[6:8]))
 d2 = datetime.datetime(int(end[0:4]), int(end[4:6]), int(end[6:8]))
 dd1= start[0:4] + '-'+ start[4:6] + '-'+ start[6:8]

# symbols = np.array(symbol1)
 quotes= []; errorlist=[]; correctlist=[]
 for i, symbol in enumerate(symbols) :
   # if np.mod(i,400) ==0 and i != 0 :
     # print('Waiting 30s'); time.sleep(30)
   try:
     if source=="yahoo2" : 
       array1= quotes_historical_yahoo_ochl(symbol, d1, d2, asobject=True)
       df= pd.DataFrame.from_records(array1)
     elif source=="google"  :
        df= imp_googleQuote( symbol, start_date=start, end_date= end)
     else : 
       if source=="google" : df= qd.get("GOOG/"+symbol,  trim_start=dd1, authtoken='dffTSNJ4JHbRE1Csd7zZ')
       if source=="yahoo" :  df= qd.get("YAHOO/"+symbol, trim_start=dd1, authtoken='dffTSNJ4JHbRE1Csd7zZ')
       df= util.pd_addcol(df, 'date')
       df.date= util.datetime_tostring(df.index.values)

     quotes.append(df)
     correctlist.append(symbol)
   except Exception as  e:
      errorlist.append(symbol)
      print('Err: '+symbol + ", " + str(e))
 #print(errorlist)
 if type1==1 :  return quotes, correctlist
 else: return quotes, correctlist, errorlist

'''
import Quandl as qd
data = Quandl.get("YAHOO/INDEX_SPY", start_date="2015-11-13", end_date="2016-04-04")

data = qd.get("GOOG/SPY", trim_start="2015-12-12")
https://github.com/quandl/quandl-python/blob/master/quandl/get.py

OLD_TO_NEW_PARAMS = {'authtoken': 'api_key', 'trim_start': 'start_date',
                     'trim_end': 'end_date', 'transformation': 'transform',
                     'sort_order': 'order'}

def get(dataset, **kwargs):
    """Return dataframe of requested dataset from Quandl.
    :param dataset: str or list, depending on single dataset usage or multiset usage
            Dataset codes are available on the Quandl website
    :param str api_key: Downloads are limited to 50 unless api_key is specified
    :param str start_date, end_date: Optional datefilers, otherwise entire dataset is returned
    :param str collapse: Options are daily, weekly, monthly, quarterly, annual
    :param str transform: options are diff, rdiff, cumul, and normalize
    :param int rows: Number of rows which will be returned
    :param str order: options are asc, desc. Default: `asc`
    :param str returns: specify what format you wish your dataset returned as,
        either `numpy` for a numpy ndarray or `pandas`. Default: `pandas`
    :returns: :class:`pandas.DataFrame` or :class:`numpy.ndarray`
    Any other `kwargs` passed to `get` are sent as field/value params to Quandl
    with no interference.

'''



#--------------------Import data from Yahoo --------------------------------------------------
def imp_yahoo_getquotes(symbols, start="20150101", end="20160101", type1=1):
 d1 = datetime.datetime(int(start[0:4]), int(start[4:6]), int(start[6:8]))
 d2 = datetime.datetime(int(end[0:4]), int(end[4:6]), int(end[6:8]))
# symbols = np.array(symbol1)
 quotes= []; errorlist=[]; correctlist=[]
 for i, symbol in enumerate(symbols) :
   # if np.mod(i,400) ==0 and i != 0 :
     # print('Waiting 30s'); time.sleep(30)
   try:   
     array1= quotes_historical_yahoo_ochl(symbol, d1, d2, asobject=True)
     df= pd.DataFrame.from_records(array1)
     quotes.append(df)
     correctlist.append(symbol)
   except :
       errorlist.append(symbol)
       print('Err: '+symbol)
 #quotes = [quotes_historical_yahoo(symbol, d1, d2, asobject=True)  for symbol in symbols]
 #print(errorlist) 
 if type1==1 :  return quotes, correctlist
 else: return quotes, correctlist, errorlist  

              
#Find error in Ticker import
def imp_yahoo_geterrorticker(symbols, start="20150101", end="20160101"):
 d1 = datetime.datetime(int(start[0:4]), int(start[4:6]), int(start[6:8]))
 d2 = datetime.datetime(int(end[0:4]), int(end[4:6]), int(end[6:8]))
 errorlist= []
 for symbol in symbols :
   try :  
     quotes_historical_yahoo_ochl(symbol, d1, d2, asobject=True)
   except :
     errorlist.append(symbol)
     print(symbol)       
 return errorlist
 
'''
symbols0= ['KHC', 'PYPL', 'HPE', 'BXLT', 'SYF', 'WLTW', 'CFG', 'BF-B', 'WRK']

#Show list of ticker in error
import_errorticker(symbols0,"20110101","20150601"  )
'''


def imp_yahoo_financials_url(ticker_symbol, statement="is", quarterly=False):
    if statement == "is" or statement == "bs" or statement == "cf":
        url = "https://finance.yahoo.com/q/" + statement + "?s=" + ticker_symbol
        if not quarterly:
            url += "&annual"; return BeautifulSoup(requests.get(url).text, "html.parser")
    return sys.exit("Invalid financial statement code '" + statement + "' passed.")

def imp_yahoo_periodic_figure(soup, yahoo_figure):
    values = []; pattern = re.compile(yahoo_figure)

    title = soup.find("strong", text=pattern)    # works for the figures printed in bold
    if title:  row = title.parent.parent
    else:
        title = soup.find("td", text=pattern)    # works for any other available figure
        if title:  row = title.parent
        else:      sys.exit("Invalid figure '" + yahoo_figure + "' passed.")

    cells = row.find_all("td")[1:]    # exclude the <td> with figure name
    for cell in cells:
        if cell.text.strip() != yahoo_figure:    # needed because some figures are indented
            str_value = cell.text.strip().replace(",", "").replace("(", "-").replace(")", "")
            if str_value == "-": str_value = 0
            value = int(str_value) * 1000
            values.append(value)

    return values
    
# print(imp_yahoo_periodic(imp_yahoo_financials_url("AAPL", "is"), "Income Tax Expense"))

 

#--------------------Import Quotes Google  ------------------------------------------
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()
import urllib, os

dirstockcsv= 'E:\_data\stock\csv'

def imp_googleIntradayQuote(symbol, freqsec=300, nday=5):
    ''' Intraday quotes from google. Specify interval seconds and number of days
    http://www.google.com/finance/getprices?q=7261&i=300&p=3d&f=d,o,h,l,c,v
    '''
    url ="http://www.google.com/finance/getprices?q={0}".format(symbol)
    url +="&i={0}&p={1}d&f=d,o,h,l,c,v".format(freqsec, nday)
    try :      
       resp= requests.post(url)
       csv= resp.text
       csv= csv.splitlines()
    except :
       print('Err Intaday '+symbol)

    if len(csv) < 9 :
      print 'Using Daily',

      url='http://www.google.com/finance/historical?q='+symbol+'&i='+str(freqsec)+'&p=3d&f=d,o,h,l,c,v&output=csv'
      csv = urllib.urlopen(url).readlines()
      csv.reverse()
      q=[]
      for bar in xrange(0,len(csv)-1):
        ds,open_,high,low,close,volume = csv[bar].rstrip().split(',')
        if isfloat(open_) and isfloat(high) :
          open_,high,low,close = [float(x) for x in [open_,high,low,close]]
          dt = datetime.datetime.strptime(ds,'%d-%b-%y')
          q.append([dt,open_,high,low,close,volume])
      df= util.pd_array_todataframe(q, ['date','open','high','low','close','volume'])
      return df


    # print(csv)
    qq=[]
    for bar in xrange(7,len(csv)):
      if csv[bar].count(',')!=5: continue
      offset,close,high,low,open_,volume = csv[bar].split(',')
      
      if offset[0]=='a':
        day = float(offset[1:])
        offset = 0
      else:
        offset = float(offset)
      open_, high, low, close = [float(x) for x in [open_,high,low,close]]
      dt = datetime.datetime.fromtimestamp(day + (freqsec * offset))
      #print (dt,open_,high,low,close,volume)
      qq.append([dt,open_,high,low,close,volume])
    df= util.pd_array_todataframe(qq, ['date','open','high','low','close','volume'])
    return df

def imp_googleIntradayQuoteSave(symbols=['NKE'], freqsec=300, nday=2000, dircsv='', dbname=''):
 ''' Save Under various format: csv / db / output in Dataframe '''
 if isinstance(symbols, str) : symbols= [symbols]
 symout,qlist, sym_error= [], [], []
 for sym in symbols:
   try :
    q = imp_googleIntradayQuote(sym, freqsec, nday)  #interval, timeframe
    print sym,
    sym= sym.replace(':', '_')

    if dircsv != '' :
      if not os.path.isdir(dircsv) :  os.makedirs(dircsv)
      start1= util.datetime_toint(util.datenumpy_todatetime(q['date'].values[0]))
      file1= dircsv+ '\\' + sym + '_' + str(start1) + '_' + str(freqsec) + '_' + str(nday) + '.csv'

      q= util.pd_addcol(q, 'symbol'); q['symbol']= sym
      q.to_csv(file1, index=False)

    elif dbname != '' :
      import sqlalchemy as sql
      dbcon= sql.create_engine(dbname)
      if interv_sec== 300 : # 5mins
        q.to_sql('q5min', dbcon)

    else :
      qlist.append(q);   symout.append(sym)
   except:
      print('Error:'+sym, )
      sym_error.append(sym)

 return qlist, symout, sym_error

def imp_googleQuote(symbol, start_date='20160101', end_date= '20160101') : # datetime.date.today().isoformat()):
    ''' Daily quotes from google. Date format='yyyymmdd' '''
    symbol = symbol.upper()
    start = datetime.date(int(start_date[0:4]),int(start_date[4:6]),int(start_date[6:8]))
    end =   datetime.date(int(end_date[0:4]),int(end_date[4:6]),int(end_date[6:8]))
    url ="http://www.google.com/finance/historical?q={0}".format(symbol)
    url +="&startdate={0}&enddate={1}&output=csv".format( start.strftime('%b %d, %Y'),end.strftime('%b %d, %Y'))
    csv = urllib.urlopen(url).readlines()
    csv.reverse()
    # print(url)
    q=[]
    for bar in xrange(0,len(csv)-1):
      ds,open_,high,low,close,volume = csv[bar].rstrip().split(',')
      if isfloat(open_) and isfloat(high) :
        open_,high,low,close = [float(x) for x in [open_, high, low, close]  ]
        dt = datetime.datetime.strptime(ds,'%d-%b-%y')
        q.append([dt,open_,high,low,close,volume])
    return  pd.DataFrame(np.array(q), columns= ["date","open","high","low","Close","volume"])


def imp_googleQuoteSave(symbols, date1, date2, dircsv):
 if isinstance(symbols, str) : symbols= [symbols]
 for sym in symbols:
   try :
      q = imp_googleQuote(sym, date1, date2)   #interval, timeframe
   except: print('Error:'+symbol)

   sym= sym.replace(':', '_')

   if dircsv != '' :
     if not os.path.isdir(dircsv) :  os.makedirs(dircsv)
     start1= util.datetime_toint(util.datenumpy_todatetime(q['date'].values[0]))
     file1= dircsv+ '\\'+ name1+'_'+ start1 +'_'+ str(date2) +'.csv'
     q.to_csv(file1)

   if dbname != '' :
     import sqlalchemy as sql
     dbcon= sql.create_engine(dbname)
     if interv_sec== 300 : # 5mins
       q.to_sql('q5min', dbcon)



def imp_csv_dbupdate(indir='E:/_data/stock/intraday/intraday_google_usetf2.h5',
                     outdir='E:/_data/stock/intraday/q5min/us/etf/', filelist=[], intype='csv',
                     refcols=['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']) :

  for file1 in filelist :
    sym=  file1[:file1.find('_')]  #Clean Up name
    print sym,
    outfile= outdir+'/'+sym+'.csv'
    if os.path.isfile(outfile) :
      df0= imp_csv_toext(file1=outfile, fromzone='Japan', tozone='UTC', header=0, cols=refcols)
    else :
      df0= None

    if intype== 'hdfs':
      df1= imp_hdfs_getquote(indir, sym)
      if isinstance(df1, float) : df1= None

    if intype=='csv' :
      infile= indir+'/'+ file1   #csv
      df1= imp_csv_toext(file1=infile, header=0, cols=refcols, fromzone='Japan', tozone='UTC')

    if df1 is not None :
       if df0 is not None  : df1= imp_pd_merge(df1[refcols], df0[refcols])
       # return outfile, df1
       df1[refcols].to_csv(outfile, index=False)


#####################################################################################
def imp_numpy_close_fromdb(dbname='/aaserialize/store/yahoo.db', table1='', symlist=[], t0=20010101, t1=20010101, priceid="close", batchsize= 400, maxasset=2600, tmax2=2000) :
 print("Get Numpy Matrix Close from DB SQL")
 import data.hist_data_storage as yhh
 dstore= yhh.dailyDataStore(dbname)
 close2= np.zeros((maxasset,tmax2),dtype=np.float16)

 dateref0= dstore.get_histo(['SPY'], table1='daily_us_etf', start1=str(t0), end1=str(t1),  split_df=0).date.values
 
 k=0; symfull=[];   i=0
 nsym= len(symlist); nbatch= int(nsym / batchsize)+1
 for j in xrange(0, nbatch) : 
      qlist, sym= dstore.get_histo(symlist[i:min(nsym,i+batchsize)], table1=table1, start1=str(t0), end1=str(t1),  split_df=1)

      #print("batch: "+str(j), len(qlist), len(qlist[0].index) )
      #for k,q in enumerate(qlist) :print sym[k], len(q.index)
      i= i + batchsize
      symfull= np.concatenate((symfull,sym))
      close,dateref= date_align(qlist, dateref= dateref0, type1=priceid)
      print("batch: "+str(j), len(sym),dateref[0],dateref[-1] )
      qlist=None; sym=None
      masset,tmax= np.shape(close)
      if tmax2 < tmax: print("Error tmax2 < tmax, wrong dates"); break
      close2[k:k+masset,0:tmax] = close
      k=k+ masset

 maxasset= k
 tmaxx= util.find(0.0, close2[0,:])
 for k in xrange(0,maxasset):
   if close2[k,tmaxx-1] == 0.0 : close2[k,tmaxx-1]=   close2[k,tmaxx-2]
 
 return close2[0:maxasset,0:tmaxx], symfull, dateref

def imp_numpy_close_fromhdfs(dbfile, symlist=[], t0=20010101, t1=20010101, priceid="close", maxasset=2600, tmax2=2000) :
 print("Get Numpy Matrix Close from Pandas")
 close2= np.zeros((maxasset,tmax2),dtype=np.float16)
 store = pd.HDFStore(dbfile)
 
 df= store.select('SPY')
 df= df[(df.date >= t0) & (df.date <= t1 )]
 dateref0= df.date.values
 
 k=0; symfull=[]; qlist=[]; sym=[]; tmax= 2000
 for j, symbol in enumerate(store.keys()):
  symbol=  symbol[1:]

  if len(symlist)==0  or  util.find(symbol,symlist) > -1 : 
   #print symbol, util.find(symbol,symlist) 
   df= store.select(symbol) 
   if t0 !=t1 :     df= df[(df.date >= t0) & (df.date <= t1 )]
   
   qlist.append(df);   sym.append(symbol)
   if np.mod(j+1, 401) == 0 :
      print("batch: "+str(j))
      close,dateref= date_align(qlist, dateref= dateref0, type1=priceid)   
      print len(sym),dateref[0],dateref[-1] 
      symfull= symfull+sym
      qlist=[]; sym=[]
 
      masset,tmax= np.shape(close)
      if tmax2 < tmax: print("Error tmax2 < tmax, wrong dates"); break
      close2[(k):(k+masset),0:tmax] = close
      k=k+ masset   
     
 if len(qlist) > 0:
  # return qlist, qlist[0] ,  qlist[1] 
   close,dateref= date_align(qlist,type1=priceid)   
   print len(sym),dateref[0],dateref[-1] 
   symfull= symfull+sym
   masset,tmax= np.shape(close);
   close2[(k):(k+masset),0:tmax] = close
 
 maxasset= k+masset
 tmaxx= util.find(0.0, close2[0,:])
 for k in xrange(0,maxasset):
   if close2[k,tmaxx-1] == 0.0 : close2[k,tmaxx-1]=   close2[k,tmaxx-2]
 
 return close2[0:maxasset,0:tmaxx], symfull, dateref


def imp_sql_getquotes(stocklist01, dbname='sqlite:///aaserialize/store/yahoo.db', start1=20150101, end1=20160616, table1='daily_us_stock'):
    import data.hist_data_storage as yhh
    dstore = yhh.dailyDataStore(dbname)
    qlist, sym= dstore.get_histo(stocklist01, table1=table1,  start1=start1, end1=end1, split_df=1)
    return qlist, sym

def imp_txt_getquotes(stocklist01, filedir='E:/_data/stock/daily/20160610/jp', startdate=20150101, endate=20160616):
 liststockfile= util.listallfile(filedir, "*.txt", dirlevel=5)
 #print liststockfile
 if len(np.shape(liststockfile)) > 1 :
  liststockname=[ x.split(".")[0] for x in liststockfile[:,0] ]
#  for k in range(0, np.shape(liststockfile)[0]):  liststockname.append( (liststockfile[k,0])[0:4])   
   
 else: 
  stk0= liststockfile[0]; #stk2= liststockfile[2]
#  liststockname=[ x[0:4] for x in stk0 ]
  liststockname=[ x for x in stk0 ]

#  liststockname=[]
#  for k in range(0, len(stk0)):  liststockname.append( (stk0[k])[0:4])
 
 quotes=[]
 print liststockname
 for sym in stocklist01:
   kstock= util.np_findfirst(sym, liststockname)
   if kstock == -1 :
     print('Not Found '+sym)
   else :
     df= pd.read_csv(liststockfile[kstock][2])
     df.columns = ['date', 'open', 'high','low','close', 'volume', 'openint']  #Change Column Name
     df2= df[(df.date > startdate  )& ( df.date   < endate )] # Filter by date
     quotes.append(df2)
  
 return quotes 


def imp_csv_getname(name1, date1, inter, tframe):
 file1= dirstockcsv+ '\\'+ name1+'_'+ date1 +'_'+ str(inter) +'_'+ str(tframe) +'.csv'
 return file1


def imp_csv_toext(file1='SPY.csv', outputfile='.h5', fromzone='Japan', tozone='UTC', header=None,
                  cols=['date', 'time','open','high','low','close','volume', 'symbol'], coldate=[0]):
  ''' cols: column name,  coldate: position of date column   '''
  from dateutil import parser

  if os.path.getsize(file1)> 500 :
   df = pd.read_csv(file1,sep=',',header=header)  # date_parser=dateparse)  # parse_dates={'date': [] },
   df.date= [ parser.parse(x) for x in  df.date]

   #if util.find('symbol', df.columns.values) < 0 :
     #df= util.pd_addcol(df, 'symbol')
     #df['symbol']= sym

   type1= outputfile[outputfile.find('.'):]

   if outputfile=='':
     # df.columns = [  x.lower() for x in df.columns.values ]
     # df.columns = ['date', 'symbol','open','high','low','close','volume']
     df.drop(df.columns[dropcol], axis=1, inplace=True)
     df= util.pd_addcol(df, 'symbol');  df['symbol']= dfname
     df.columns = ['date', 'open','high','low','close','volume', 'symbol']
     return df[cols]

   if type1=='csv' :
     df[cols].to_csv(outputfile , index=False)  # , compression='gzip' )

   if type1== 'h5' :
      # dateparse= lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
      dateparse= lambda x: (parse(x).replace(tzinfo=from_zone).astimezone(tozone))
      # dateparse= lambda x: parse(x, tzinfos=from_zone).astimezone(to_zone)
      # df.date= [pd.to_datetime((str(x)[:-6])) for x in  df.date]
      # df.date= [x.to_datetime() for x in  df.date]

      df.drop(df.columns[[0]], axis=1, inplace=True)
      print filenameh5
      if util.find('symbol', df.columns.values) < 0 :
         df= util.pd_addcol(df, 'symbol'); df['symbol']= dfname
      store = pd.HDFStore(outputfile); store.append(dfname, df);  store.close()



def imp_hdfs_db_updatefromcsv(dircsv, filepd=r'E:\_data\stock\intraday_google.h5', fromtimezone='Japan', tozone='UTC'):
 lfile= util.os_file_listall(dircsv, pattern="*.csv", dirlevel=0)
 for x in lfile:
   name1= (x[0]).split("_")[0] #get Ticker name, splitting by _
   if util.str_isfloat(name1):  name1= 'jp'+name1  #Japanese Stocks case
   file1 = x[2];  print(name1)
   if os.path.getsize(file1)> 500:
      filepd2= filepd + '/' +name1 +'.csv'
      imp_csv_toext(file1, filepd2, name1, fromtimezone, tozone)

def imp_hdfs_db_dumpinfo(dbfile='E:\_data\stock\intraday_google.h5'):
  store = pd.HDFStore(dbfile)
  extract=[]; errsym=[]
  for symbol in store.keys():
     try:
       df= pd.read_hdf(dbfile, symbol)
       t0= df['date'].values[0]
       t1= df['date'].values[-1]

       extract.append([symbol[1:], df.shape[1],   df.shape[0], t0, t1 ])

     except: errsym.append(symbol)
  return np.array(extract), errsym

def imp_hdfs_mergedb(filepdfrom, filepdto) :
  store0 = pd.HDFStore(filepdfrom)
  store1 = pd.HDFStore(filepdto)

  for symbol in store0.keys():
    qq= imp_hdfs_getquote(filepdfrom, symbol)
    store1.append(symbol, qq)

    qq= imp_hdfs_getquote(filepdto, symbol)
    qq= qq.drop_duplicates(subset='date', take_last=True)
    qq= qq.sort('date', ascending=1)
    print(symbol)
  store0.close(); store1.close()

def imp_hdfs_storecopy(hdfs1='F:/usstock8.h5', hdfs2='F:/usstock8.h5'):
 store = pd.HDFStore(hdfs1);  store2 = pd.HDFStore(hdfs2)
 for symbol in store.keys() :
   df= imp_hdfs_getquote(hdfs1, symbol)
   symbol= symbol[1:].replace("-","_")
   store2.append(symbol, df, data_columns=True)
 store.close(); store2.close()

def imp_hdfs_getquote(filenameh5, dfname="data"):
  try: return  pd.read_hdf(filenameh5, dfname)
  except: return -1.0;

def imp_hdfs_getListquote(symbols, close1='close', start='12/18/2015 00:00:00+00:00',
                          end='3/1/2016 00:00:00+00:00', freq='0d0h10min', filepd= 'E:\_data\stock\intraday_google.h5',
                          tozone='Japan', fillna=True, interpo=True):

 datefilter= pd.date_range(start=start, end=end, freq= freq,tz=tozone).values

 errorsym=[]; quotes=[]; correctsym=[]
 for symbol in symbols:  # Issue Not same size
   qq= imp_hdfs_getquote(filepd, symbol)

   if type(qq)==np.float :  # Error qq=-1
        errorsym.append(symbol)
   else:
       qq.columns = ['date', 'symbol','open','high','low','close','volume']
       qq= qq[qq['date'].isin(datefilter)]   #Only date in the range
       #qq= qq.drop_duplicates(cols='date', take_last=True)
       #  qq= qq.sort('date', asc=True)

       #print qq.date.values[0]
       if fillna : qq= qq.fillna(method='backfill')
       if interpo : qq= qq.interpolate()
       quotes.append(qq)
       correctsym.append(symbol)

 datefilter= datenumpy_todatetime(datefilter)
 return quotes,datefilter, correctsym,   errorsym


def imp_hdfs_removeDuplicate(filepd='E:\_data\stock\intraday_google.h5') :
  #-------Clean by removing duplicate-------------------------------------------------
  store = pd.HDFStore(filepd)
  for symbol in store.keys():
    #qq= imp_hdfs_getquote(filepd, symbol);
    qq= store.select(symbol[1:])
    qq= qq.drop_duplicates(subset='date', take_last=True)
    qq= qq.sort('date', ascending=1)
    qq.index = list(np.arange(0,len(qq.index)))

    store.remove(symbol); store.append(symbol, qq);
  store.close()



def imp_pd_tohdfs(sym, qqlist, filenameh5, fromzone='Japan', tozone='UTC') :
  ''' df list to HDFS '''
  store = pd.HDFStore(filenameh5);
  for k, df2 in enumerate(qqlist) :
    df= copy.deepcopy(df2)
    df['date']= datetime_toint(df.date.values)

    symbol= sym[k].replace('-','_')
    qq= imp_hdfs_getquote(filenameh5, symbol)
    # print symbol, type(qq)
    if type(qq) ==  float or  type(qq) ==  int  :
      store.append(symbol, df)  # , data_columns=True
      # print(symbol + str(df.date.values[-1]))
    else :
      qq= pd.concat([qq, df], ignore_index=True)
      qq= qq.drop_duplicates(subset='date', keep='last')
      qq= qq.sort('date', ascending=1)
      store.append(symbol, qq)  #Too much space , data_columns=True
      print(symbol + str(qq.date.values[-1]))

  store.close()

def imp_pd_merge(df1, df2) :
  df= pd.concat([df1, df2], axis=0)
  df= df.drop_duplicates(subset='date', keep='last')
  df= df.sort_values(by=['date'], ascending=1)
  df.index = list(np.arange(0,len(df.index)))
  return df

def imp_pd_checkquote(quotes) :
   for c in quotes: print np.shape(c), c['date'].values[0], c['date'].values[-1]

def imp_pd_getclose(df, datefilter=None):
 if datefilter is  not None :
    df= df[df['date'].isin(datefilter)]   #Only date in the range
 df= df.sort('date')
 close= df.interpolate(); close= close.fillna(method='backfill')  #Interpolate
 
 close= np.array(close.values[1:,1:]).astype(np.float)            #Only Price
 close= close[(np.abs(sci.stats.zscore(close)) < 3).all(axis=1)] #Remove Outlier
 return close

def imp_pd_cashyield(q, duration=10):
 qbond= copy.deepcopy(q)
 for id1 in ['close','open','high','low','aclose'] :
   try:
     vv= q[id1].values
     pbond=  100 * np.cumprod( (1 + np.ones(len(vv)) * 0.005/365.0 ))
     qbond[id1]= pbond
   except: pass
 return qbond

def imp_pd_yield_tobond(q, duration=10):
 qbond= copy.deepcopy(q)
 for id1 in ['close','open','high','low','aclose'] :
   try:
     vv= q[id1].values
     pbond=  1/ np.power(1+ vv*0.01,duration) *100
     qbond[id1]= pbond
#     print 'ok'
   except: pass
 return qbond

def imp_pd_errordate(quotes, dateref):
 ''' Show Symbol in Error when importing '''
 for i, stock in enumerate(quotes) :
   print(i, symbols1[i],  datetime_tostring(stock[0][0]))
 print("\n\n")

 for i, stock in enumerate(quotes) :
   date1= datetime_toint(stock[0][0])
   if date1 != dateref : print(i, symbols1[i],  str(date1))

def imp_pd_fxtoprice(q):
 dfprice= copy.deepcopy(q)
 for kid in ['close','open','high','low'] :
   qret= getret_fromquotes(q[kid].values,1)
   qprice= price_normalize100(qret)
   dfprice[kid]= qprice.T
 return dfprice

def imp_pd_fxinversetoprice(q):
 dfprice= copy.deepcopy(q)
 for kid in ['close','open','high','low'] :
   qret= getret_fromquotes(1.0 / q[kid].values,1)
   qprice= price_normalize100(qret)
   dfprice[kid]= qprice.T
 return dfprice

def imp_pd_filterbydate(df, dtref=None, start='2016-06-06 00:00:00', end='2016-06-14 00:00:00', freq='0d0h05min', timezone='Japan'):
 ''' df: DateSeries or TimeSeries of Quotes   '''

 if type(df) in {pd.core.frame.DataFrame} :  #Data frame version
   if type(dtref)== str:
      dtref= pd.date_range(start=start, end=end, freq= freq,tz=timezone).values

   return df[df['date'].isin(dtref)]
 else :
   if type(dtref)== str:  #Date version
      dtref= pd.date_range(start=start, end=end, freq= freq,tz=timezone).values

   return df[df.isin(dtref)].values

def imp_pd_cleanquote(q):
 col= q.columns.values
 if isinstance(q['date'].values[0] , str) :
   q['date']= datetime_todate(datestring_todatetime(q['date'].values))

 for kid in col:
   if kid not in ['date', 'day','month','year'] :
      q[kid]= pd.to_numeric(q[kid], errors='coerce').values  #Put NA on string

 q= q.fillna(method='pad')
 return q

def imp_pd_divide(q1, q2, funapply):
 close, dateref= date_align([q1,q2], type1='close')
 m,tt= np.shape(close)

 q3= util.pd_createdf( np.zeros((tt,len(q1.columns))) ,q1.columns, dateref)
 q3['date']= dateref
 q3['close']= (close[0,:] / close[1,:])

 for kid in q1.columns :
   if kid not in ['close', 'volume','date','day','month', 'year'] :
     close, _= date_align([q1,q2], type1=kid)
     q3[kid]= (close[0,:] / close[1,:])
 return q3







'''
def imp_quote_csvdir_tofile(dircsv='', outdir1='E:/_data/stock/', fromtimezone='Japan', tozone='UTC'):
 lfile= util.os_file_listall(dircsv, pattern="*.csv", dirlevel=0)
 for x in lfile:
   name1= (x[0]).split("_")[0] #get Ticker name, splitting by _
   if util.str_isfloat(name1):  name1= 'jp'+name1  #Japanese Stocks case
   file1 = x[2];  print(name1)
   if os.path.getsize(file1)> 500:
      filepd2= outdir + '/' +name1 +'.csv'
      imp_csv_toext(file1, filepd2,  header=True, fromzone=fromtimezone, tozone=tozone)
'''


'''
close1='Close' ; start='12/18/2015 00:00:00';
end='3/1/2016 00:00:00'; freq='1d0h00min';
filepd= 'E:\_data\stock\intraday_google.h5'

datefilter= pd.date_range(start=start, end=end,   freq= freq).values

symbols, names = np.array(list(symbol_dict.items())).T
i=0; vv=[]
for symbol in symbols:  # Issue Not same size
   qq= quote_frompanda(filepd, symbol); i+=1
   if isfloat(qq) :  vv.append(symbol)
   else:
     if i==1: close = qq[['Datetime']]  #new dataframe    
     tmp= pd.DataFrame(data={symbol: qq[close1]});
     if not tmp.isnull().values.all():
      close = close.join(tmp, how='outer', rsuffix='_1')
      close= close.drop_duplicates(cols='Datetime', take_last=True)

close= close[close['Datetime'].isin(datefilter)]   #Only date in the range
close= close.sort('Datetime')
close= close.interpolate(); close= close.fillna(method='backfill')  #Interpolate
close= np.array(close.values[1:,1:]).astype(np.float)  #Only Price
close= close[(np.abs(stats.zscore(close)) < 3).all(axis=1)] #Remove Outlier
 return close, vv
'''

'''
from datetime import datetime
from dateutil import tz

from_zone = tz.gettz('Japan'); to_zone = tz.gettz('UTC')

date1 = datetime.strptime('2011-01-21 09:00:00', '%Y-%m-%d %H:%M:%S')
date1 = date1.replace(tzinfo=from_zone)
dateutc = date1.astimezone(to_zone)
date1, dateutc

Suppose you have a column 'datetime' with your string, then:
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv(infile, parse_dates=['datetime'], date_parser=dateparse)
 combine multiple columns into a single datetime column,
this merges a 'date' and a 'time' column into a single 'datetime' column:

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv(infile, parse_dates={'datetime': ['date', 'time']}, date_parser=dateparse)
'''


#date1 = "20160302"; inter= 60; tframe=80; name1= "spy"
#googleIntradayQuoteSave(name1, inter, tframe)



def ta_mar(close,t,m):  return  100*( np.mean(close[(t-m):(t+1)], axis=0) / close[t] )
def ta_ma(close,t,m):  return   np.mean(close[(t-m):(t+1)], axis=0)


############################################################################
#---------------------Calculate the Statistics         --------------------  
def calc_statestock(close2, dateref, symfull):
 def sort(x,col,asc): return   util.sortcol(x,col,asc)
 def perf(close,t0,t1):  return  100*( close[:,t1] / close[:,t0] -1)
 def and2(tuple1):  return np.logical_and.reduce(tuple1)
 def mar(close,t,m):  return  100*( np.mean(close[:,(t-m):(t+1)], axis=1) / close[:,t] )
 def ma(close,t,m):  return   np.mean(close[:,(t-m):(t+1)], axis=1)


 # a= ma(close2, tmax, 5) / ma(close2, tmax, 50)  


 def dd(t) :
  x= util.find(t,dateref)  # 607
  if x==-1 : print 'Error date not found in dateref '+str(t) ; return np.nan
  else :  return x

 def gap(close,t0,t1,lag):   
  ret= getret_fromquotes(close[:,t0:t1],lag)
  rmin= 100*np.amin(ret,axis=1)
  return rmin
 
 #--------------------Size----------------------------------------- 
 masset= len(symfull)
 ttmax= util.find(0,close2[0,:])
 if ttmax==-1 : ttmax= np.shape(close2)[1]
 close2= close2[0:masset,0:ttmax]
 m, tmax= masset, ttmax-1
 
 #correl
 #close3= np.array(close2[:,0:ttmax],dtype=np.float16)
 #close2ret= getret_fromquotes(close3)
 #close2ret= np.nan_to_num(close2ret)
 #close2ret= util.np_cleanmatrix(close2ret);  del close3

##################################################################################
#1: 1d,2:5d,3:10d,4:15d,5:20d,11:50d,12:60d,13:90d,15:1y
#16: YTD,17:From Feb,21: 2015Ret,45: Price,50,51: Min,52,53: Max
 print("Calcul the Stats " + str(dateref[-1]))
 stat= np.zeros((2600,200),dtype=np.float16)

 k=0
 stat[k:(k+m),0]= np.arange(k,k+m)
 stat[k:(k+m),1]=  perf(close2,tmax-1,tmax)  
 stat[k:(k+m),2]=  perf(close2,tmax-5,tmax)  
 stat[k:(k+m),3]=  perf(close2,tmax-10,tmax)  
 stat[k:(k+m),4]=  perf(close2,tmax-15,tmax)  
 stat[k:(k+m),5]=  perf(close2,tmax-20,tmax)  
# stat[k:(k+m),6]=  perf(close2,tmax-25,tmax)  
 stat[k:(k+m),7]=  perf(close2,tmax-30,tmax)  
# stat[k:(k+m),8]=  perf(close2,tmax-35,tmax)  
 stat[k:(k+m),9]=  perf(close2,tmax-40,tmax)  
# stat[k:(k+m),10]=  perf(close2,tmax-45,tmax)  
 stat[k:(k+m),11]=  perf(close2,tmax-50,tmax)  
 stat[k:(k+m),12]=  perf(close2,tmax-60,tmax)  
 stat[k:(k+m),13]=  perf(close2,tmax-90,tmax)  
 stat[k:(k+m),14]=  perf(close2,tmax-120,tmax)  
 stat[k:(k+m),15]=  perf(close2,tmax-250,tmax)  


# stat[k:(k+m),16]=  perf(close2, dd(20160105),tmax)   #1Jan 2016
# stat[k:(k+m),17]=  perf(close2, dd(20160222),tmax)   # 22 Feb 2016
# stat[k:(k+m),18]=  perf(close2,dd(20160105),dd(20160222))   # Crash DrawDown
 # stat[k:(k+m),19]=  perf(close2,624,tmax)   #  24 June 2016
 # stat[k:(k+m),20]=  perf(close2,623,625)   #  Crash 24 June 2016
 # stat[k:(k+m),21]=  perf(close2,dd(20151230)-250,dd(20151230))    #2015 Return

 stat[k:(k+m),22]= gap(close2,tmax-80,tmax,1) #Max 1 days Gap
 stat[k:(k+m),23]= gap(close2,tmax-80,tmax,5) #Max 5 days Gap
 stat[k:(k+m),24]= perf(close2,tmax-2,tmax)  
 stat[k:(k+m),25]= perf(close2,tmax-3,tmax)  
 stat[k:(k+m),26]= perf(close2,tmax-4,tmax)  

 stat[k:(k+m),27]= perf(close2,tmax-2,tmax-1)   
 stat[k:(k+m),28]= perf(close2,tmax-3,tmax-2)   
 stat[k:(k+m),29]= perf(close2,tmax-4,tmax-3)   
 stat[k:(k+m),30]= perf(close2,tmax-5,tmax-4)
 
 stat[k:(k+m),120]= mar(close2, tmax, 20) 
 stat[k:(k+m),121]= mar(close2, tmax, 50) 
 stat[k:(k+m),122]= 100*ma(close2, tmax, 5) / ma(close2, tmax, 50)  #OverSold /OverBought
 stat[k:(k+m),123]= mar(close2, tmax, 100)  
 stat[k:(k+m),124]= mar(close2, tmax, 10) 
# stat[k:(k+m),35]= 0 
# stat[k:(k+m),36]= 0 
# stat[k:(k+m),37]= 0 
# stat[k:(k+m),38]= 0 
# stat[k:(k+m),39]= 0 

 # stat[k:(k+m),30]= volume / AvgVol3M
 # stat[k:(k+m),30]= Nbday_to_Earnings
 #Volume/3M AvgVolume
 # stat[k:(k+m),26]=  volume(volume2,tmax-120,tmax)  

 #==============================================================================
 #-------------- Data Fundamental    -------------------------------------------
 df= util.sql_query(sqlr='SELECT ticker,shortratio,sector1_id, sector2_id, marketcap   FROM stockfundamental ', dburl='/aaserialize/store/finviz.db')
 npdf= df.values; del df
 npdf0= npdf[:,0]  
 for i in range(0,m):  
   try :
     kid= util.find(symfull[i], npdf0)
     if kid != -1 :
      stat[i,150]=  float(npdf[kid, 1])   # short ratio
      stat[i,151]=  float(npdf[kid, 2])   # sector 1      
      stat[i,152]=  float(npdf[kid, 3])   # sector 2     
      stat[i,153]=  float(npdf[kid, 4] / 100000000.0)   # market Cap    
   except : pass
 del npdf, npdf0


#==============================================================================
 ################ Volatility
 stat[k:(k+m),40]= volhisto_fromprice(close2,tmax,20,axis=1)* 100
 stat[k:(k+m),41]= volhisto_fromprice(close2,tmax,60,axis=1)* 100
 stat[k:(k+m),42]= volhisto_fromprice(close2,tmax,252,axis=1)* 100
# stat[k:(k+m),43]= stat[k:(k+m),12] / stat[k:(k+m),41] #Sharpe 3Month
# stat[k:(k+m),44]= stat[k:(k+m),15] / stat[k:(k+m),42] #Sharpe 1Year

 stat[k:(k+m),45]= close2[:,tmax] #Last Close

# stat[k:(k+m),46]= 0 
# stat[k:(k+m),47]= 0 
# stat[k:(k+m),48]= 0 
# stat[k:(k+m),49]= 0 


 ###############  Technical Indicator
 #Min of last 6 months, Max of last 6months
 for i in range(0,m):
  pp0= stat[i,45]
  kmin,pmin= util.np_find_minpos(close2[i,(tmax-120):tmax])
  stat[i,50]= kmin+(tmax-120);  stat[i,51]= 100*pp0 / pmin

  kmax,pmax= util.np_find_maxpos(close2[i,(tmax-120):tmax])
  stat[i,52]= kmax+(tmax-120);  stat[i,53]= 100*pp0 / pmax

  kmin,pmin= util.np_find_minpos(close2[i,(tmax-10):tmax])
  stat[i,66]= 100*pp0 / pmin

  kmax,pmax= util.np_find_maxpos(close2[i,(tmax-10):tmax])
  stat[i,67]= 100*pp0 / pmax

  kmin,pmin= util.np_find_minpos(close2[i,(tmax-5):tmax])
  stat[i,68]= 100*pp0 / pmin

  kmax,pmax= util.np_find_maxpos(close2[i,(tmax-5):tmax])
  stat[i,69]= 100*pp0 / pmax

 del kmax,pmax,i,k  


 #Regression from Min Price Time
 for i in range(0,m):
  t0= int(stat[i,50])
  try :
    res= regression(close2[i,t0:tmax ]  ,np.arange(t0,tmax),type1="linear")
    stat[i,54]=  res[2]  #R2
    stat[i,55]=  res[0][0]  #Slope
  except : pass

 #Regression from Max Price Time
 for i in range(0,m):  
  t0= stat[i,52]
  try :
    res= regression(close2[i,t0:tmax ] ,np.arange(t0,tmax),type1="linear")
    stat[i,56]=  res[2]  #R2
    stat[i,57]=  res[0][0]  #Slope
  except : pass

 #Regression from 5 days ago
 t0= tmax-5
 for i in range(0,m): 
  try :
    res= regression(close2[i,t0:tmax ] ,np.arange(t0,tmax),type1="linear")
    stat[i,58]=  res[2]  #R2
    stat[i,59]=  res[0][0]  #Slope
  except : pass

#Regression from 10 days ago
 t0= tmax-10
 for i in range(0,m):  
   try :
    res= regression(close2[i,t0:tmax ] ,np.arange(t0,tmax),type1="linear")
    stat[i,60]=  res[2]  #R2
    stat[i,61]=  res[0][0]  #Slope
   except : pass

 '''
#Regression from 1 month
 t0= tmax-20   
 for i in range(0,m):
   try :
    res= regression(close2[i,t0:tmax ] ,np.arange(t0,tmax),type1="linear")
    stat[i,62]=  res[2]  #R2
    stat[i,63]=  res[0][0]  #Slope
   except : pass
 '''

 '''
#Regression from 2 months ago
 t0= tmax-40
 for i in range(0,m):  
  try :
    res= regression(close2[i,t0:tmax ] ,np.arange(t0,tmax),type1="linear")
    stat[i,64]=  res[2]  #R2
    stat[i,65]=  res[0][0]  #Slope
  except : pass
 '''

#Nb of Positive days last 5d
 for i in range(0,m):  stat[i,66]=  np_countretsign(close2[i,(tmax-5):tmax])  

#Nb of Positive days last 10d
 for i in range(0,m):  stat[i,67]=  np_countretsign(close2[i,(tmax-10):tmax])  

#Nb of Positive days last 20d
# for i in range(0,m):  stat[i,68]=  np_countretsign(close2[i,(tmax-20):tmax])  


#Regression from Max Values,200days
 t0= tmax-20*10
 for i in range(0,m):  
  try :
    vmax= util.np_findlocalmax2(close2[i,t0:tmax ] ,6)
    res= regression(vmax[:,1] ,vmax[:,0],type1="linear")
    stat[i,70]=  res[2]  #R2
    stat[i,71]=  res[0][0]  #Slope
  except : pass
# del vmax


#Regression from Min Values,200days 
 t0= tmax-20*10
 for i in range(0,m):  
  try :
    vmin= util.np_findlocalmin2(close2[i,t0:tmax ] ,6)
    res= regression(vmin[:,1] ,vmin[:,0],type1="linear")
    stat[i,72]=  res[2]  #R2
    stat[i,73]=  res[0][0]  #Slope
  except : pass
# del vmin


#Regression from Max Values,100days
 t0= tmax-20*5
 for i in range(0,m):  
  try :
    vmax= util.np_findlocalmax2(close2[i,t0:tmax ] ,6)
    res= regression(vmax[:,1] ,vmax[:,0],type1="linear")
    stat[i,74]=  res[2]  #R2
    stat[i,75]=  res[0][0]  #Slope
  except : pass
 del vmax


#Regression from Min Values,100days 
 t0= tmax-20*5
 for i in range(0,m):  
  try :
    vmin= util.np_findlocalmin2(close2[i,t0:tmax ] ,6)
    res= regression(vmin[:,1] ,vmin[:,0],type1="linear")
    stat[i,76]=  res[2]  #R2
    stat[i,77]=  res[0][0]  #Slope
  except : pass
 del vmin



# Upside Trend with Higher Lows detection
 t0= tmax-20*10
 for i in range(0,m):  
  try :
   vmax= util.np_findlocalmin2(close2[i,t0:tmax],4)
   vmax= util.sort(vmax,0,asc=0)
   stat[i,80]=  -1
   if len(vmax) > 1 :
    ss=0
    for k in range(0,len(vmax)-1):
     if vmax[k,1] > vmax[k+1,1] : ss+=1   #Higher lows
     else : break;
  
    if tmax - vmax[0,0]-t0 < 30  :  
        stat[i,80]=  ss  
        stat[i,81]=  vmax[0,0] + t0
  except: pass


# Upside Trend with Higher High detection
 t0= tmax-20*10
 for i in range(0,m):  
  try :
    vmax= util.np_findlocalmax2(close2[i,t0:tmax],4)
    vmax= util.sort(vmax,0,asc=0)
    stat[i,82]=  -1
    if len(vmax) > 1 :
      ss=0
      for k in range(0,len(vmax)-1):
        if vmax[k,1] > vmax[k+1,1] : ss+=1   #Higher high
        else : break;
      if tmax - vmax[0,0]-t0 < 30  :  
        stat[i,82]=  ss  
        stat[i,83]=  vmax[0,0] + t0
  except: pass


# Downside Trend with lower Lows detection
 t0= tmax-20*10
 for i in range(0,m):  
  vmax= util.np_findlocalmin2(close2[i,t0:tmax],4)
  vmax= util.sort(vmax,0,asc=0)
  stat[i,84]=  -1
  if len(vmax) > 1 :
   ss=0
   for k in range(0,len(vmax)-1):
     if vmax[k,1] < vmax[k+1,1] : ss+=1   #Higher lows
     else : break;
  
   if tmax - vmax[0,0]-t0 < 30  :  
        stat[i,84]=  ss  
        stat[i,85]=  vmax[0,0] + t0


# Downside Trend  with lower highs detection
 t0= tmax-20*6
 for i in range(0,m):  
  vmax= util.np_findlocalmax2(close2[i,t0:tmax],4)
  vmax= util.sort(vmax,0,asc=0)
  stat[i,80]=  -1
  if len(vmax) > 1 :
   ss=0
   for k in range(0,len(vmax)-1):
     if vmax[k,1] < vmax[k+1,1] : ss+=1   #Higher lows
     else : break;
  
   if tmax - vmax[0,0]-t0 < 30  :  
        stat[i,86]=  ss    #Nb  of lower high consecutive
        stat[i,87]=  vmax[0,0] + t0

 print('Higher/Lower Band Support')
#-----higher band Support / Lower Band Support 120days-------------------------------
 t0= tmax-120
 for i in range(0,m): 
  try :
    res1= ta_highbandtrend1(close2[i,t0:tmax])
    pmax= (res1.x[0] * (tmax-t0) + res1.x[1]) 
    stat[i,90]= 100.0 * stat[i, 45] / pmax   # Higher Band

    res2=  ta_lowbandtrend1(close2[i,t0:tmax])
    pmin=  (res2.x[0] * (tmax-t0) + res2.x[1]) 
    stat[i,91]= 100.0* stat[i, 45] / pmin    # LowerBand

    stat[i,92]= 100.0* (pmax - stat[i, 45]) / (pmax - pmin)    # Price inside Range

    # pmin=  (res2.x[0] * (np.arange(t0,tmax)-t0) + res2.x[1]) 
    # breachmin= np.sum(np.sign(close2[i, t0:tmax] - pmin*1.03))
  except: pass


#-----higher band Support / Lower Band Support 200days-------------------------------
 t0= tmax-200
 for i in range(0,m): 
  try :
    res1= ta_highbandtrend1(close2[i,t0:tmax])
    pmax= (res1.x[0] * (tmax-t0) + res1.x[1]) 
    stat[i,93]= 100.0 * stat[i, 45] / pmax   # Higher Band

    res2=  ta_lowbandtrend1(close2[i,t0:tmax])
    pmin=  (res2.x[0] * (tmax-t0) + res2.x[1]) 
    stat[i,94]= 100.0 * stat[i, 45] / pmin    # LowerBand

    stat[i,95]= 100.0 * (pmax - stat[i, 45]) / (pmax - pmin)    # Price inside Range

  except: pass


#-----higher band Support / Lower Band Support 300days-------------------------------------
 t0= tmax-300
 for i in range(0,m): 
  try :
    res1= ta_highbandtrend1(close2[i,t0:tmax])
    pmax= (res1.x[0] * (tmax-t0) + res1.x[1]) 
    stat[i,96]= 100.0 * stat[i, 45] / pmax   # Higher Band

    res2=  ta_lowbandtrend1(close2[i,t0:tmax])
    pmin=  (res2.x[0] * (tmax-t0) + res2.x[1]) 
    stat[i,97]= 100.0* stat[i, 45] / pmin    # LowerBand

    stat[i,98]= 100.0* (pmax - stat[i, 45]) / (pmax - pmin)    # Price inside Range

  except: pass


  '''
  # -----higher band Support / Lower Band Support 60days-------------------------------
  t0 = tmax - 60
  for i in range(0, m):
      try:
          res1 = ta_highbandtrend1(close2[i, t0:tmax])
          pmax = (res1.x[0] * (tmax - t0) + res1.x[1])
          stat[i, 99] = 100.0 * stat[i, 45] / pmax  # Higher Band

          res2 = ta_lowbandtrend1(close2[i, t0:tmax])
          pmin = (res2.x[0] * (tmax - t0) + res2.x[1])
          stat[i, 100] = 100.0 * stat[i, 45] / pmin  # LowerBand

          stat[i, 101] = 100.0 * (pmax - stat[i, 45]) / (pmax - pmin)  # Price inside Range

          # pmin=  (res2.x[0] * (np.arange(t0,tmax)-t0) + res2.x[1])
          # breachmin= np.sum(np.sign(close2[i, t0:tmax] - pmin*1.03))
      except:
          pass
  '''

  #-----Technical Indicator------------------------------------------------
# for i in range(0,m): 
#  try :
#   res2=0
   # df= pd.read_hdf(dbfile, symfull[i])
   # stat[i,110]= ta.RMI(close2[i,:])
   # stat[i,111]= 100*close2[i,:] / ta.MA(close2[i,:],20)
   # stat[i,112]= 100*close2[i,:] / ta.MA(close2[i,:],50)   
#  except: pass


 return np.array(stat, dtype=np.float16) 


'''  
i=1964  
stat= s1
stat[i, 91]  

tt= np.arange(t0, tmax)
plot_price(close2[i,t0:tmax], (res2.x[0] * (tt-t0) + res2.x[1]) )
'''

  
#------------------- Stock Monitoring Tools ------------------------------------------------
def monitor_addrecommend(string1, dbname='stock_recommend') :
 ss= string1.replace("("," ").replace(')',' ').replace(':',' ')
 ss= ss.replace('\t', ' ').replace('\n', ' ')
 sl1= ss.split(" ")
 print sl1
 aux= [datestring_toint(util.date_now())]
 symfull= copy.deepcopy(a_us_all)
 for x in sl1:
   if len(x) < 5 :
    if util.find(x,symfull) > 0 :   aux.append(x)

 if len(aux) >  1 :
  stock_recommend= util.load_obj(dbname)
  stock_recommend.append(aux)
  util.save_obj(stock_recommend,dbname)
  print aux
 return stock_recommend



#################### Finviz  ###############################################################
from bs4 import BeautifulSoup
import csv,datetime, requests
def imp_finviz():
 #pdb.set_trace() - python step by step debugger command
 print "Finviz Overview Start"
 url = "http://www.finviz.com/screener.ashx?v=111&f=geo_usa"
 response = requests.get(url)
 html = response.content
 soup = BeautifulSoup(html)
 firstcount = soup.find_all('option')
 lastnum = len(firstcount) - 1
 lastpagenum = firstcount[lastnum].attrs['value']
 currentpage = int(lastpagenum)

 alldata = []
 templist = []
 # Overview = 111, Valuation = 121, Financial = 161, Ownership = 131, Performance = 141
 #pagesarray = [111,121,161,131,141]
 titleslist = soup.find_all('td',{"class" : "table-top"})
 titleslisttickerid = soup.find_all('td',{"class" : "table-top-s"})
 titleticker = titleslisttickerid[0].text
 titlesarray = []
 for title in titleslist:
    titlesarray.append(title.text)

 titlesarray.insert(1,titleticker)
 i = 0

 while(currentpage > 0):
    i += 1
    print str(i) + " page(s) done"
    secondurl = "http://www.finviz.com/screener.ashx?v=" + str(111) + "&f=geo_usa" + "&r=" + str(currentpage)
    secondresponse = requests.get(secondurl)
    secondhtml = secondresponse.content
    secondsoup = BeautifulSoup(secondhtml)
    stockdata = secondsoup.find_all('a', {"class" : "screener-link"})
    stockticker = secondsoup.find_all('a', {"class" : "screener-link-primary"})
    datalength = len(stockdata)
    tickerdatalength = len(stockticker)

    while(datalength > 0):
        templist = [stockdata[datalength - 10].text,stockticker[tickerdatalength-1].text,stockdata[datalength - 9].text,stockdata[datalength - 8].text,stockdata[datalength - 7].text,stockdata[datalength - 6].text,stockdata[datalength - 5].text,stockdata[datalength - 4].text,stockdata[datalength - 3].text,stockdata[datalength - 2].text,stockdata[datalength - 1].text,]
        alldata.append(templist)
        templist = []
        datalength -= 10
        tickerdatalength -= 1
    currentpage -= 20

 with open('stockoverview.csv', 'wb') as csvfile:
    overview = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=titlesarray)
    overview.writeheader()

    for stock in alldata:
        overview.writerow({titlesarray[0] : stock[0], titlesarray[1] : stock[1],titlesarray[2] : stock[2],titlesarray[3] : stock[3],titlesarray[4] : stock[4], titlesarray[5] : stock[5], titlesarray[6] : stock[6], titlesarray[7] : stock[7] , titlesarray[8] : stock[8], titlesarray[9] : stock[9], titlesarray[10] : stock[10] })

 print "Finviz Overview Completed"
 return overview
 

def imp_finviz_news():
 #pdb.set_trace() - python step by step debugger command
 print "Finviz Performance Start"
 url = "http://www.finviz.com/quote.ashx?t=intc"
 response = requests.get(url)
 html = response.content
 soup = BeautifulSoup(html)
 titleslist = soup.find_all('a',{"class" : "tab-link-news"})
 return titleslist


def imp_finviz_financials():
 import datetime
 #pdb.set_trace() - python step by step debugger command
 print datetime.datetime.now()
 print "Finviz Financial Start"
 url = "http://www.finviz.com/screener.ashx?v=161&f=geo_usa"
 response = requests.get(url)
 html = response.content
 soup = BeautifulSoup(html)
 firstcount = soup.find_all('option')
 lastnum = len(firstcount) - 1
 lastpagenum = firstcount[lastnum].attrs['value']
 currentpage = int(lastpagenum)

 alldata = []
 templist = []
 # Overview = 111, Valuation = 121, Financial = 161, Ownership = 131, Performance = 141
 #pagesarray = [111,121,161,131,141]
 titleslist = soup.find_all('td',{"class" : "table-top"})
 titleslisttickerid = soup.find_all('td',{"class" : "table-top-s"})
 titleticker = titleslisttickerid[0].text
 titlesarray = []
 for title in titleslist:
    titlesarray.append(title.text)

 titlesarray.insert(1,titleticker)
 i = 0

 while(currentpage > 0):
    i += 1
    print str(i) + " page(s) done"
    secondurl = "http://www.finviz.com/screener.ashx?v=" + str(161) + "&f=geo_usa" + "&r=" + str(currentpage)
    secondresponse = requests.get(secondurl)
    secondhtml = secondresponse.content
    secondsoup = BeautifulSoup(secondhtml)
    stockdata = secondsoup.find_all('a', {"class" : "screener-link"})
    stockticker = secondsoup.find_all('a', {"class" : "screener-link-primary"})
    datalength = len(stockdata)
    tickerdatalength = len(stockticker)

    while(datalength > 0):
        templist = [stockdata[datalength - 17].text,stockticker[tickerdatalength-1].text,stockdata[datalength - 16].text,stockdata[datalength - 15].text,stockdata[datalength - 14].text,stockdata[datalength - 13].text,stockdata[datalength - 12].text,stockdata[datalength - 11].text,stockdata[datalength - 10].text,stockdata[datalength - 9].text,stockdata[datalength - 8].text,stockdata[datalength - 7].text,stockdata[datalength - 6].text,stockdata[datalength - 5].text,stockdata[datalength - 4].text,stockdata[datalength - 3].text,stockdata[datalength - 2].text,stockdata[datalength - 1].text,]
        alldata.append(templist)
        templist = []
        datalength -= 17
        tickerdatalength -= 1
    currentpage -= 20

 with open('stockfinancial.csv', 'wb') as csvfile:
    financial = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=titlesarray)
    financial.writeheader()

    for stock in alldata:
        financial.writerow({titlesarray[0] : stock[0], titlesarray[1] : stock[1],titlesarray[2] : stock[2],titlesarray[3] : stock[3],titlesarray[4] : stock[4], titlesarray[5] : stock[5], titlesarray[6] : stock[6], titlesarray[7] : stock[7] , titlesarray[8] : stock[8], titlesarray[9] : stock[9], titlesarray[10] : stock[10],titlesarray[11] : stock[11],titlesarray[12] : stock[12],titlesarray[13] : stock[13],titlesarray[14] : stock[14],titlesarray[15] : stock[15],titlesarray[16] : stock[16],titlesarray[17] : stock[17] })

 print datetime.datetime.now()
 print "Finviz Financial Completed"



def get_price2book( symbol ):
   from bs4 import BeautifulSoup as bs
   import urllib as u
   try:
    	url = r'http://finviz.com/quote.ashx?t={}'.format(symbol.lower())
        html = u.request.urlopen(url).read()
        soup = bs(html, 'lxml')
        # Change the text below to get a diff metric
        pb =  soup.find(text = r'P/B')
        pb_ = pb.find_next(class_='snapshot-td2').text
        print( '{} price to book = {}'.format(symbol, pb_) )
        return pb_
   except Exception as e:
        print(e)







'''
 try:
   q = imp_googleQuote(name1, date1, date2)  #interval, timeframe
   name1= name1.replace(':','_')
   file1= dircsv+ '\\'+ name1+'_'+ date1 +'_'+ str(date2) +'.csv'
   q.write_csv(file1)
 except: print
'''


'''
def imp_googleQuoteList(symbols, freqsec=300, nday=1, intraday1=True) :
 qlist=[]; sym=[]
 if intraday1:
  for symbol in symbols:
     #try :
        q = imp_googleIntradayQuote(symbol, freqsec, nday)  #interval, timeframe
        qlist.append(q)
        sym.append(symbol)
     #except: print('Error:'+symbol)

 else :
  for symbol in symbols:
     try :
       q = imp_googleQuote(name1, freqsec, num_days)  #interval, timeframe
     except: print(symbol)

 return qlist, sym
'''







'''        
stock_list = ['XOM','AMZN','AAPL','SWKS']
p2b_series = pd.Series( index=stock_list )

for sym in stock_list:	p2b_series[sym] = get_price2book(sym)
'''



'''
def imp_csv_toext(file1, filenameh5, dfname='sym1', fromzone='Japan', tozone='UTC'):
 # dateparse= lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
 from dateutil.parser import parse
 if os.path.getsize(file1)> 500 :
  from_zone = tz.gettz(fromzone); tozone = tz.gettz(tozone)
  # dateparse= lambda x: (pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').replace(tzinfo=from_zone).astimezone(tozone))

  # dateparse= lambda x: (pd.datetime.strptime(x,'%m-%d-%Y %H:%M:%S').replace(tzinfo=from_zone).astimezone(tozone))

  dateparse= lambda x: (parse(x).replace(tzinfo=from_zone).astimezone(tozone))

  # dateparse= lambda x: parse(x, tzinfos=from_zone).astimezone(to_zone)
  print file1
  #df = pd.read_csv(file1,sep=',',header=None, parse_dates={'date': [1]}, date_parser=dateparse)

  df = pd.read_csv(file1,sep=',',header=0)
  #df.date= [pd.to_datetime((str(x)[:-6])) for x in  df.date]
  #df.date= [x.to_datetime() for x in  df.date]


  df.drop(df.columns[[0]], axis=1, inplace=True)
  df= util.pd_addcol(df, 'symbol')
  df['symbol']= dfname

  print filenameh5
  df.columns = ['date', 'open','high','low','close','volume', 'symbol' ]
  df.to_csv(filenameh5 , index=False)  # , compression='gzip' )

  # store = pd.HDFStore(filenameh5);
  # store.append(dfname, df);  store.close()
'''

'''
def imp_csv_toext(file1, filenameh5, dropcol=[], type1='csv', hasheader=True, dfname='sym1', fromzone='Japan', tozone='UTC'):
 if os.path.getsize(file1)> 500 :
   header= 0 if hasheader else None
   df = pd.read_csv(file1, sep=',',header=0)

   df.drop(df.columns[dropcol], axis=1, inplace=True)
   df= util.pd_addcol(df, 'symbol');  df['symbol']= dfname
   df.columns = ['date', 'open','high','low','close','volume', 'symbol' ]

   if type1=='csv' :
     df.to_csv(filenameh5 , index=False)  # , compression='gzip' )
'''






'''
     if i==1: close = qq[['Datetime']]  #new dataframe
     tmp= pd.DataFrame(data={symbol: qq[close1]});
     close = close.join(tmp, how='outer', rsuffix='_1')
     close= close.drop_duplicates(cols='Datetime', take_last=True)

 close= close[close['Datetime'].isin(datefilter)]   #Only date in the range
 close= close.sort('Datetime')
 close= close.interpolate(); close= close.fillna(method='backfill')  #Interpolate

 close= np.array(close.values[1:,1:]).astype(np.float)  #Only Price
 close= close[(np.abs(sci.stats.zscore(close)) < 3).all(axis=1)] #Remove Outlier
 return close, vv
'''



'''
class Quote(object):
  DATE_FMT = '%Y-%m-%d';  TIME_FMT = '%H:%M:%S'

  def __init__(self):
    self.symbol = '';    self.exchn=''
    self.date,self.time,self.open_,self.high,self.low,self.close,self.volume = ([] for _ in range(7))

  def append(self,dt,open_,high,low,close,volume):
    self.date.append(dt.date())
    self.time.append(dt.time())
    self.open_.append(float(open_))
    self.high.append(float(high))
    self.low.append(float(low))
    self.close.append(float(close))
    self.volume.append(int(volume))

  def to_csv(self):
    return ''.join(["{0},{1},{2},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7}\n".format(self.symbol,
              self.date[bar].strftime('%Y-%m-%d'),self.time[bar].strftime('%H:%M:%S'),
              self.open_[bar],self.high[bar],self.low[bar],self.close[bar],self.volume[bar])
              for bar in xrange(len(self.close))])

  def write_csv(self,filename):
    with open(filename,'w') as f:
      txt= self.to_csv()
      f.write(txt)

  def read_csv(self,filename):
    self.symbol = ''
    self.date,self.time,self.open_,self.high,self.low,self.close,self.volume = ([] for _ in range(7))
    for line in open(filename,'r'):
      symbol,ds,ts,open_,high,low,close,volume = line.rstrip().split(',')
      self.symbol = symbol
      dt = datetime.datetime.strptime(ds+' '+ts,self.DATE_FMT+' '+self.TIME_FMT)
      self.append(dt,open_,high,low,close,volume)
    return True

  def __repr__(self):
    return self.to_csv()
'''

 

'''
def imp_googleIntradayQuoteSave2(name1, date1, inter, tframe, dircsv='', dbname=''):
 q = imp_googleIntradayQuote(name1, inter, tframe)  #interval, timeframe
 name1= name1.replace(':','_')

 if dircsv != '' :
    file1= dircsv+ '\\'+ name1+'_'+ date1 +'_'+ str(inter) +'_'+ str(tframe) +'.csv'
    q.to_csv(file1)

 if dbname != '' :
    import sqlalchemy as sql
    dbcon= sql.create_engine(dbname)
    if inter== 300 : # 5mins
      q.to_sql('q5min', dbcon)


def imp_googleQuoteList_save(symbols, date1, date2, inter=23400, tframe=2000, dircsv='', intraday1=True) :
 if intraday1:
  for symbol in symbols:
     try :
       imp_googleIntradayQuoteSave(symbol, date1, inter, tframe, dircsv)
     except: print('Error:'+symbol)

 else :
  for symbol in symbols:
     try :
       imp_googleQuoteSave(symbol, date1, date2, dircsv)
     except: print(symbol)
'''

