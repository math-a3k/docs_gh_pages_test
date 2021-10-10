# -*- coding: utf-8 -*-
######################Technical Indicator ###########################################
import pandas as pd, numpy as np, datetime, calendar, scipy as sci
from datetime import datetime
def np_find(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in xrange(len(vec)):
        if item == vec[i]: return i
    return -1
  
import operator
def np_find_minpos(values):
 min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
 return min_index, min_value
 
def np_find_maxpos(values):
 max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
 return max_index, max_value
 
def date_earningquater(t1):   #JP Morgan Qearing date
 if (t1.month==10 and t1.day >= 14) or (t1.month==1 and t1.day < 14) or t1.month in [11,12] :
    if t1.month in [10,11,12] : qdate= datetime(t1.year+1 ,1,14)
    else : qdate= datetime(t1.year, 1,14)
    quater= 4
    
 if (t1.month==1 and t1.day >= 14) or (t1.month==4 and t1.day < 14) or t1.month in [2,3] :
    qdate= datetime(t1.year ,4,14); quater= 1
    
 if (t1.month==4 and t1.day >= 14) or (t1.month==7 and t1.day < 14) or t1.month in [5,6] :
    qdate= datetime(t1.year ,7,14); quater= 2    

 if (t1.month==7 and t1.day >= 14) or (t1.month==10 and t1.day < 14) or t1.month in [8,9] :
    qdate= datetime(t1.year ,10,14) ; quater= 3
      
 nbday= (qdate-t1).days;  
 return quater, nbday, qdate


def date_option_expiry(date):
    day = 21 - (calendar.weekday(date.year, date.month, 1) + 2) % 7
    if date.day <= day :
        nbday= day-date.day
        datexp= datetime(date.year, date.month, day)
    else :
        if date.month== 12 :
         day = 21 - (calendar.weekday(date.year+1, 1, 1) + 2) % 7
         datexp= datetime(date.year+1, 1 , day)
        else :
         day = 21 - (calendar.weekday(date.year, date.month+1, 1) + 2) % 7
         datexp= datetime(date.year, date.month+1 , day)          

        nbday= (datexp - date).days          
   
    return nbday, datexp


def linearreg(a, *args):
 x= args[1]; y= args[2]; b= args[0]
 v= a*x+b  - y
 return np.sum(v*v)

def np_sortbycolumn(arr, colid, asc=True): 
 df = pd.DataFrame(arr)
 arr= df.sort(colid, ascending=asc)   
 return arr.values


def np_findlocalmax(v):
 n=len(v)   
 v2= np.zeros((n,10))
 if n > 2:
  for i,x in enumerate(v):
    if i < n-1 and i > 0 :
     if x > v[i-1] and x > v[i+1] : 
           v2[i,0]= i;  v2[i,1]=x
  v2= np_sortbycolumn(v2,1,asc=False)
  return v2
 else : 
   max_index, max_value= np_find_maxpos(v)
   return  [[max_index, max_value]]

def findhigher(item, vec):
    for i in xrange(len(vec)):
        if item < vec[i]: return i
    return -1

def findlower(item, vec):
    for i in xrange(len(vec)):
        if item > vec[i]: return i
    return -1


def np_findlocalmin(v):
 n=len(v)   
 v2= np.zeros((n,10))
 if n > 2:
  i2=0
  for i,x in enumerate(v):
    if i < n-1 and i > 0 :
     if x < v[i-1] and x < v[i+1] :      
           v2[i2,0]= i;  v2[i2,1]=x
           i2+=1  
  v2 = np_sortbycolumn(v2[:i2],0,asc=True)
  return v2
 else : 
   
   max_index, max_value= np_find_minpos(v)
   return  [[max_index, max_value]]

   
#####################################################################################    
#Finviz Style Support and Resistance
def supportmaxmin1(df1) :
 df= df1.close.values
 qmax= np_findlocalmax(df)
 t1= len(df)
 tmax,_= np_find_maxpos(df)
#Classification of the Local Max
 for k in range(0, len(qmax)) :
  kmax= qmax[k,0]
  kmaxl=  findhigher(qmax[k,1], df[:kmax][::-1])  #Find same level of max
  kmaxr=  findhigher(qmax[k,1], df[kmax+1:])

  kmaxl= 0 if kmaxl==-1 else kmax-kmaxl
  kmaxr= t1 if kmaxr==-1 else kmaxr+kmax

  qmax[k,2]= np.abs(kmaxr-kmaxl)   #Range 
  qmax[k,3]= np.abs(kmax-tmax)   #Range of the Max After
  qmax[k,4]= 0  #Range of the Max After
  qmax[k,5]= kmax-kmaxl
  qmax[k,6]= kmaxr-kmax

 qmax =np_sortbycolumn(qmax,1,asc=False)
 tmax= qmax[0,0]; pmax=  qmax[0,1]


#Trend Line Left:  t=0 to tmax
 qmax2 =  qmax[qmax[:,2]>20,:]   #Range of days where max 
 qmax2 =  qmax2[qmax2[:,0]<= tmax ,:]  #Time BEfore

 if len(qmax2) > 10 :
   qmax2 =  qmax2[qmax2[:,5]>= 10 ,:]  #Time After 
   qmax2 =  qmax2[qmax2[:,6]>= 10 ,:]  #Time After 

 qmax2 =np_sortbycolumn(qmax2,2,asc=False)   #Order by Time
 qmax2= qmax2[0:3]                                #Only Top 3 Max Value
 qmax2 =np_sortbycolumn(qmax2,0,asc=True)   #Order by Time


 if np.shape(qmax2)[0] > 1 :
   tt= np.arange(0, tmax); b= pmax
   res= sci.optimize.fmin(linearreg,0.1,(b, qmax2[:,0]-tmax, qmax2[:,1]),disp=False)
   a= min( (df[0] - b) / (0-tmax) , res[0])  #Constraint on the Slope
   suppmax1= a * (tt-tmax) + b 
 else :
   suppmax1= np.zeros(tmax) + pmax


#Trend Line Right Side from tmax and t1  
 qmax2 =  qmax[qmax[:,2]>20,:]   #Level of days where max is max
 qmax2 =  qmax2[qmax2[:,0]>= qmax2[0,0] ,:]  #Time After 
 qmax2 =  qmax2[qmax2[:,5]>= 10 ,:]  #Time After 
 qmax2 =  qmax2[qmax2[:,6]>= 10 ,:]  #Time After 
 qmax2 =np_sortbycolumn(qmax2,2,asc=False)   #Order by Time

 qmax2= qmax2[0:3]  #Only Top 3 Max Value
 qmax2 =np_sortbycolumn(qmax2,0,asc=True)   #Order by Time


 if np.shape(qmax2)[0] > 1 :
   tt= np.arange(tmax, t1); b= pmax
   res= sci.optimize.fmin(linearreg,0.1,(b, qmax2[:,0]-tmax, qmax2[:,1]),disp=False)
   a= max( (df[t1-1] - b) / (t1-tmax) , res[0])  #Constraint on the Slope
   suppmax2= a * (tt-tmax) + b
 else :
  suppmax2= np.zeros(t1-tmax) + qmax2[-1,1]

 suppmax= np.zeros(t1)
 suppmax[0:tmax]= suppmax1
 suppmax[tmax:]= suppmax2


##############################################################################
##### Local Min Trend Line
 qmin= np_findlocalmin(df)
 t1= len(df)
 tmin,_= np_find_minpos(df)
 
 #Classification of the Local min
 for k in range(0, len(qmin)) :
  if qmin[k,1] != 0.0 :
   kmin= qmin[k,0]
   kminl=  findlower(qmin[k,1], df[:kmin][::-1])  #Find same level of min
   kminr=  findlower(qmin[k,1], df[kmin+1:])

   kminl= 0 if kminl==-1 else kmin-kminl
   kminr= t1 if kminr==-1 else kminr+kmin

   qmin[k,2]= np.abs(kminr-kminl)   #Range 
   qmin[k,3]= np.abs(kmin-tmin)   #Range of the min After
   qmin[k,4]= 0  #Range of the min After
   qmin[k,5]= kmin-kminl
   qmin[k,6]= kminr-kmin

 qmin =np_sortbycolumn(qmin,1,asc=True); 
 tmin= qmin[0,0]; pmin=  qmin[0,1]   


 #Trend Line Left:  t=0 to tmin
 qmin2 =  qmin[qmin[:,2]>20,:]   #Range of days where min 
 qmin2 =  qmin2[qmin2[:,0]<= tmin ,:]  #Time BEfore

 if len(qmin2) > 10 :
  qmin2 =  qmin2[qmin2[:,5]>= 10 ,:]  #Time After 
  qmin2 =  qmin2[qmin2[:,6]>= 10 ,:]  #Time After 

 qmin2 =np_sortbycolumn(qmin2,2,asc=False)   #Order by Time
 qmin2= qmin2[0:3]                                #Only Top 3 min Value
 qmin2 =np_sortbycolumn(qmin2,0,asc=True)   #Order by Time


 if np.shape(qmin2)[0] > 1 :
  tt= np.arange(0, tmin); b=pmin
  res= sci.optimize.fmin(linearreg,0.1,(b, qmin2[:,0]-tmin, qmin2[:,1]),disp=False)
  a= max( (df[0] - b) / (0-tmin) , res[0])  #Constraint on the Slope
  suppmin1= a * (tt-tmin) + b 
 else :
  suppmin1= np.zeros(0,tmin) + qmin2[-1,1]


 #Trend Line Right Side from tmin and t1  
 qmin2 =  qmin[qmin[:,2]>20,:]   #Level of days where min is min
 qmin2 =  qmin2[qmin2[:,0]>= qmin2[0,0] ,:]  #Time After 
 qmin2 =  qmin2[qmin2[:,5]>= 10 ,:]  #Time After 
 qmin2 =  qmin2[qmin2[:,6]>= 10 ,:]  #Time After 
 qmin2 =np_sortbycolumn(qmin2,2,asc=False)   #Order by Time

 qmin2= qmin2[0:3]  #Only Top 3 min Value
 qmin2 =np_sortbycolumn(qmin2,0,asc=True)   #Order by Time


 if np.shape(qmin2)[0] > 1 :
  tt= np.arange(tmin, t1); b= pmin
  res= sci.optimize.fmin(linearreg,0.1,(b, qmin2[:,0]-tmin, qmin2[:,1]), disp=False)
  a= min( (df[t1-1] - b) / (t1-tmin) , res[0])  #Constraint on the Slope
  suppmin2= a * (tt-tmin) + b
 else :
   suppmin2= np.zeros(t1-tmin) + qmin2[0,1]

 suppmin= np.zeros(t1)
 suppmin[0:tmin]= suppmin1
 suppmin[tmin:]= suppmin2

 suppmax = pd.Series(suppmax, name = 'suppmax_1') 
 df1 = df1.join(suppmax)  
 suppmin = pd.Series(suppmin, name = 'suppmin_1') 
 df1 = df1.join(suppmin)  

 return df1




#RETURN
def RET(df,n):
    n=n+1
    M = df['close'].diff(n - 1)  
    N = df['close'].shift(n - 1)  
    ROC = pd.Series(100*M / N, name = 'RET_' + str(n-1))  
    df = df.join(ROC)  
    return df


def qearning_dist(df):
  d1= df['date'].values  
  quarter= np.zeros(len(d1));   nbday= np.zeros(len(d1));
  for i,t in enumerate(d1) :
    q1, nday,_= date_earningquater(datetime(t.year, t.month, t.day))
    quarter[i]= q1
    nbday[i]= nday   
  smin1 = pd.Series(quarter, name = 'qearning_per')  
  smin2 = pd.Series(nbday, name = 'qearning_day')  
  df = df.join(smin1);   df = df.join(smin2);
  return df
  
  
def optionexpiry_dist(df):
  d1= df['date'].values  
  nbday= np.zeros(len(d1));
  for i,t in enumerate(d1) :
    nday,_= date_option_expiry(datetime(t.year, t.month, t.day))
    nbday[i]= nday   
  smin2 = pd.Series(nbday, name = 'optexpiry_day')  
  df = df.join(smin2);
  return df
  

def nbtime_reachtop(df,n, trigger=0.005):
  '''nb of days from 1 year low '''
  close= df['close'].values
  nnbreach= np.zeros(len(close))
  for i in range(n, len(close)) :
    kid, max1= np_find_maxpos(close[i-n:i])
    dd= np.abs(close[i-n:i]/max1 -1) 
    nnbreach[i]= -np.sum( np.sign(np.minimum(0, dd - trigger) ))
        
  smin1 = pd.Series(nnbreach, name = 'nbreachigh_' + str(n))  
  df = df.join(smin1)  
  return df
  
  
def nbday_high(df, n):
  '''nb of days from 1 year low '''
  close= df['close'].values
  ndaylow= np.zeros(len(close)); distlow=np.zeros(len(close))
  for i in range(n, len(close)) :
    kid, min1= np_find_maxpos(close[i-n:i])
    ndaylow[i]= n-kid
    distlow[i]= close[i] - min1
   
  smin1 = pd.Series(ndaylow, name = 'ndayhigh_' + str(n))  
  smin2 = pd.Series(distlow, name = 'ndisthigh_' + str(n))  
  df = df.join(smin1)  
  df = df.join(smin2)  
  return df


def distance_day(df, tk, tkname):
 tk= datetime.date(tk)
 date1= df['date'].values
 dist= np.zeros(len(date1)); 
 for i in range(0, len(date1)) : 
     dist[i]= (date1[i] - tk).days
 dist = pd.Series(dist, name = 'days_' + tkname)  
 df = df.join(dist)  
 return df


def distance(df, ind) :
  df2= pd.Series(100*(df['close'] / df[ind] -1),  name = ind + '_dist' )
  df = df.join(df2)  
  return df


#Moving Average  
def MA(df, n):  
    MA = pd.Series(pd.rolling_mean(df['close'], n), name = 'MA_' + str(n))  
    df = df.join(MA)  
    return df

#Exponential Moving Average  
def EMA(df, n):  
    EMA = pd.Series(pd.ewma(df['close'], span = n, min_periods = n - 1), name = 'EMA_' + str(n))  
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['close'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['close'].diff(n - 1)  
    N = df['close'].shift(n - 1)  
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)  
    return df

#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))  
    df = df.join(ATR)  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(pd.rolling_mean(df['close'], n))  
    MSD = pd.Series(pd.rolling_std(df['close'], n))  
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)  
    b2 = (df['close'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))  
    df = df.join(B2)  
    return df


#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['high'] + df['low'] + df['close']) / 3)  
    R1 = pd.Series(2 * PP - df['low'])  
    S1 = pd.Series(2 * PP - df['high'])  
    R2 = pd.Series(PP + df['high'] - df['low'])  
    S2 = pd.Series(PP - df['high'] + df['low'])  
    R3 = pd.Series(df['high'] + 2 * (PP - df['low']))  
    S3 = pd.Series(df['low'] - 2 * (df['high'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)  
    df = df.join(PSR)  
    return df

#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')  
    df = df.join(SOk)  
    return df

#Stochastic oscillator %D  
def STO(df, n):  
    SOk = pd.Series((df['close'] - df['low']) / (df['high'] - df['low']), name = 'SO%k')  
    SOd = pd.Series(pd.ewma(SOk, span = n, min_periods = n - 1), name = 'SO%d_' + str(n))  
    df = df.join(SOd)  
    return df

#Trix  
def TRIX(df, n):  
    EX1 = pd.ewma(df['close'], span = n, min_periods = n - 1)  
    EX2 = pd.ewma(EX1, span = n, min_periods = n - 1)  
    EX3 = pd.ewma(EX2, span = n, min_periods = n - 1)  
    i = 0  
    ROC_l = [0]  
    while i + 1 <= df.index[-1]:  
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]  
        ROC_l.append(ROC)  
        i = i + 1  
    Trix = pd.Series(ROC_l, name = 'Trix_' + str(n))  
    df = df.join(Trix)  
    return df

#Average Directional Movement Index  
def ADX(df, n, n_ADX):  
    i = 0  
    UpI = []  
    DoI = []  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')  
        DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1) / ATR)  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1) / ATR)  
    ADX = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span = n_ADX, min_periods = n_ADX - 1), name = 'ADX_' + str(n) + '_' + str(n_ADX))  
    df = df.join(ADX)  
    return df

#MACD, MACD Signal and MACD difference  
def MACD(df, n_fast, n_slow):  
    EMAfast = pd.Series(pd.ewma(df['close'], span = n_fast, min_periods = n_slow - 1))  
    EMAslow = pd.Series(pd.ewma(df['close'], span = n_slow, min_periods = n_slow - 1))  
    MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))  
    MACDsign = pd.Series(pd.ewma(MACD, span = 9, min_periods = 8), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))  
    MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))  
    df = df.join(MACD)  
    df = df.join(MACDsign)  
    df = df.join(MACDdiff)  
    return df

#Mass Index  
def MassI(df):  
    Range = df['high'] - df['low']  
    EX1 = pd.ewma(Range, span = 9, min_periods = 8)  
    EX2 = pd.ewma(EX1, span = 9, min_periods = 8)  
    Mass = EX1 / EX2  
    MassI = pd.Series(pd.rolling_sum(Mass, 25), name = 'Mass Index')  
    df = df.join(MassI)  
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF  
def Vortex(df, n):  
    i = 0  
    TR = [0]  
    while i < df.index[-1]:  
        Range = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR.append(Range)  
        i = i + 1  
    i = 0  
    VM = [0]  
    while i < df.index[-1]:  
        Range = abs(df.get_value(i + 1, 'high') - df.get_value(i, 'low')) - abs(df.get_value(i + 1, 'low') - df.get_value(i, 'high'))  
        VM.append(Range)  
        i = i + 1  
    VI = pd.Series(pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n), name = 'Vortex_' + str(n))  
    df = df.join(VI)  
    return df

#KST Oscillator  
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):  
    M = df['close'].diff(r1 - 1)  
    N = df['close'].shift(r1 - 1)  
    ROC1 = M / N  
    M = df['close'].diff(r2 - 1)  
    N = df['close'].shift(r2 - 1)  
    ROC2 = M / N  
    M = df['close'].diff(r3 - 1)  
    N = df['close'].shift(r3 - 1)  
    ROC3 = M / N  
    M = df['close'].diff(r4 - 1)  
    N = df['close'].shift(r4 - 1)  
    ROC4 = M / N  
    KST = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2, n2) * 2 + pd.rolling_sum(ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))  
    df = df.join(KST)  
    return df

#Relative Strength Index  
def RSI(df, n=14):  
  # If the RSI rises above 30, buy signal, RSI falls under 70, a sell signal occurs.
    i = 0  
    UpI = [0]  
    DoI = [0]  
    while i + 1 <= df.index[-1]:  
        UpMove = df.get_value(i + 1, 'high') - df.get_value(i, 'high')  
        DoMove = df.get_value(i, 'low') - df.get_value(i + 1, 'low')  
        if UpMove > DoMove and UpMove > 0:  
            UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:  
            DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI)  
    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))  
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))  
    df = df.join(RSI)  
    return df


#Relative Momentum Index 
def RMI(df, n=14, m=10):  
    #http://www.csidata.com/?page_id=797  , FinVIZ RMI 10
    i = m  
    UpI = list(np.zeros(m))  # Switch by m values
    DoI = list(np.zeros(m))  
     
    while i   <= df.index[-1]:  
        UpMove = df.get_value(i, 'high') - df.get_value(i-m, 'high')  
        DoMove = df.get_value(i-m, 'low') - df.get_value(i, 'low')  
        if UpMove > DoMove and UpMove > 0:    UpD = UpMove  
        else: UpD = 0  
        UpI.append(UpD)  
        if DoMove > UpMove and DoMove > 0:    DoD = DoMove  
        else: DoD = 0  
        DoI.append(DoD)  
        i = i + 1  
    UpI = pd.Series(UpI)  
    DoI = pd.Series(DoI) 

    PosDI = pd.Series(pd.ewma(UpI, span = n, min_periods = n - 1))  
    NegDI = pd.Series(pd.ewma(DoI, span = n, min_periods = n - 1))  
  
    RSI = pd.Series(100*PosDI / (PosDI + NegDI), name = 'RMI_' + str(n)+'_'+str(m))  
    df = df.join(RSI)  
    return df




#True Strength Index  
def TSI(df, r, s):  
    M = pd.Series(df['close'].diff(1))  
    aM = abs(M)  
    EMA1 = pd.Series(pd.ewma(M, span = r, min_periods = r - 1))  
    aEMA1 = pd.Series(pd.ewma(aM, span = r, min_periods = r - 1))  
    EMA2 = pd.Series(pd.ewma(EMA1, span = s, min_periods = s - 1))  
    aEMA2 = pd.Series(pd.ewma(aEMA1, span = s, min_periods = s - 1))  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    df = df.join(TSI)  
    return df

#Accumulation/Distribution  
def ACCDIST(df, n):  
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']  
    M = ad.diff(n - 1)  
    N = ad.shift(n - 1)  
    ROC = M / N  
    AD = pd.Series(ROC, name = 'Acc/Dist_ROC_' + str(n))  
    df = df.join(AD)  
    return df

#Chaikin Oscillator  
def Chaikin(df):  
    ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']  
    Chaikin = pd.Series(pd.ewma(ad, span = 3, min_periods = 2) - pd.ewma(ad, span = 10, min_periods = 9), name = 'Chaikin')  
    df = df.join(Chaikin)  
    return df

#Money Flow Index and Ratio  
def MFI(df, n):  
    PP = (df['high'] + df['low'] + df['close']) / 3  
    i = 0  
    PosMF = [0]  
    while i < df.index[-1]:  
        if PP[i + 1] > PP[i]:  
            PosMF.append(PP[i + 1] * df.get_value(i + 1, 'volume'))  
        else:  
            PosMF.append(0)  
        i = i + 1  
    PosMF = pd.Series(PosMF)  
    TotMF = PP * df['volume']  
    MFR = pd.Series(PosMF / TotMF)  
    MFI = pd.Series(pd.rolling_mean(MFR, n), name = 'MFI_' + str(n))  
    df = df.join(MFI)  
    return df

#On-balance Volume  
def OBV(df, n):  
    i = 0  
    OBV = [0]  
    while i < df.index[-1]:  
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') > 0:  
            OBV.append(df.get_value(i + 1, 'volume'))  
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') == 0:  
            OBV.append(0)  
        if df.get_value(i + 1, 'close') - df.get_value(i, 'close') < 0:  
            OBV.append(-df.get_value(i + 1, 'volume'))  
        i = i + 1  
    OBV = pd.Series(OBV)  
    OBV_ma = pd.Series(pd.rolling_mean(OBV, n), name = 'OBV_' + str(n))  
    df = df.join(OBV_ma)  
    return df

#Force Index  
def FORCE(df, n):  
    F = pd.Series(df['close'].diff(n) * df['volume'].diff(n), name = 'Force_' + str(n))  
    df = df.join(F)  
    return df

#Ease of Movement  
def EOM(df, n):  
    EoM = (df['high'].diff(1) + df['low'].diff(1)) * (df['high'] - df['low']) / (2 * df['volume'])  
    Eom_ma = pd.Series(pd.rolling_mean(EoM, n), name = 'EoM_' + str(n))  
    df = df.join(Eom_ma)  
    return df

#Commodity Channel Index  
def CCI(df, n):  
    PP = (df['high'] + df['low'] + df['close']) / 3  
    CCI = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n), name = 'CCI_' + str(n))  
    df = df.join(CCI)  
    return df

#Coppock Curve  
def COPP(df, n):  
    M = df['close'].diff(int(n * 11 / 10) - 1)  
    N = df['close'].shift(int(n * 11 / 10) - 1)  
    ROC1 = M / N  
    M = df['close'].diff(int(n * 14 / 10) - 1)  
    N = df['close'].shift(int(n * 14 / 10) - 1)  
    ROC2 = M / N  
    Copp = pd.Series(pd.ewma(ROC1 + ROC2, span = n, min_periods = n), name = 'Copp_' + str(n))  
    df = df.join(Copp)  
    return df

#Keltner Channel  
def KELCH(df, n):  
    KelChM = pd.Series(pd.rolling_mean((df['high'] + df['low'] + df['close']) / 3, n), name = 'KelChM_' + str(n))  
    KelChU = pd.Series(pd.rolling_mean((4 * df['high'] - 2 * df['low'] + df['close']) / 3, n), name = 'KelChU_' + str(n))  
    KelChD = pd.Series(pd.rolling_mean((-2 * df['high'] + 4 * df['low'] + df['close']) / 3, n), name = 'KelChD_' + str(n))  
    df = df.join(KelChM)  
    df = df.join(KelChU)  
    df = df.join(KelChD)  
    return df

#Ultimate Oscillator  
def ULTOSC(df):  
    i = 0  
    TR_l = [0]  
    BP_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'high'), df.get_value(i, 'close')) - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        TR_l.append(TR)  
        BP = df.get_value(i + 1, 'close') - min(df.get_value(i + 1, 'low'), df.get_value(i, 'close'))  
        BP_l.append(BP)  
        i = i + 1  
    UltO = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(pd.Series(TR_l), 7)) + (2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(pd.Series(TR_l), 14)) + (pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')  
    df = df.join(UltO)  
    return df

#Donchian Channel  
def DONCH(df, n):  
    i = 0  
    DC_l = []  
    while i < n - 1:  
        DC_l.append(0)  
        i = i + 1  
    i = 0  
    while i + n - 1 < df.index[-1]:  
        DC = max(df['high'].ix[i:i + n - 1]) - min(df['low'].ix[i:i + n - 1])  
        DC_l.append(DC)  
        i = i + 1  
    DonCh = pd.Series(DC_l, name = 'Donchian_' + str(n))  
    DonCh = DonCh.shift(n - 1)  
    df = df.join(DonCh)  
    return df

#Standard Deviation  
def STDDEV(df, n):  
    df = df.join(pd.Series(pd.rolling_std(df['close'], n), name = 'STD_' + str(n)))  
    return df  


def RWI(df,nn ,nATR) :
 return 0
 ''' 
 First we compute RWI for maxima:
 RWImax = [(day‘s high - (day's low * number of days))] / [(Average True Range * number of days * square root of number of days)]

 Similarly, we compute the RWI for minima:
 RWImin = [(day's high *number of days - (day's low))] / [(Average True Range * number of days * square root of number of days)]

 True range = higher of (day's high, previous close ) - lower of (day's low, previous close)
 '''



def nbday_low(df, n):
  '''nb of days from 1 year low '''
  close= df['close'].values
  ndaylow= np.zeros(len(close)); distlow=np.zeros(len(close))
  for i in range(n, len(close)) :
    kid, min1= np_find_minpos(close[i-n:i])
    ndaylow[i]= n-kid
    distlow[i]= close[i] - min1
   
  smin1 = pd.Series(ndaylow, name = 'ndaylow_' + str(n))  
  smin2 = pd.Series(distlow, name = 'ndistlow_' + str(n))  
  df = df.join(smin1)  
  df = df.join(smin2)  
  return df


def nbday_high(df, n):
  '''nb of days from 1 year low '''
  close= df['close'].values
  ndaylow= np.zeros(len(close)); distlow=np.zeros(len(close))
  for i in range(n, len(close)) :
    kid, min1= np_find_maxpos(close[i-n:i])
    ndaylow[i]= n-kid
    distlow[i]= close[i] - min1
   
  smin1 = pd.Series(ndaylow, name = 'ndayhigh_' + str(n))  
  smin2 = pd.Series(distlow, name = 'ndisthigh_' + str(n))  
  df = df.join(smin1)  
  df = df.join(smin2)  
  return df





'''
26 responses
 171bfddace1cb03c836e2f6054f1b9c8  Lionel  Mar 17, 2015
This is great , thank you for sharing this :)

Lionel.

 A31bd5accd3432cbf5a8e3dc84a04928  Ambooj Mittal  Mar 22, 2015
this seems great

 B95a5c9a32a4c951f474cc75181e5297  Andrea D'Amore  Mar 25, 2015
There's little need to see an indicator implementation, the whole point of using external libraries is delegating implementation details to someone else, and ta-lib is well tested in that regard. Moreover by using a python implementation you're possibly not using acceleration on numpy or panda's side.

It is IMHO better to understand and indicator's rationale and proper usage by reading the author's note or some specific article about it than looking at the code.

Also those are just methods from the pandas implemented module of pyTaLib with packages import rather than wildcards, I'd put the author's name back in the body (due to copyright) and remove your name from it since the file is essentially unmodified.

 B7dd86784deedc047d5ad0fb30378bdd  Peter Bakker  Mar 25, 2015
Your opinion granted. For me it actually helps to look at what happens and as I don't think I'm unique so that's why I put the code up.

The author Bruno is mentioned so I don't think that's an issue

The code is changed here and there and I added a few functions but I omitted to record what I added. As in my humble opinion it's useful I'll keep the post live.

 B7dd86784deedc047d5ad0fb30378bdd  Peter Bakker  Mar 25, 2015
In addition to that. Above code has quite a few indicators that talib does not have so only for that reason it's useful to have it around.

 D4b55852454c34c05c0d55b91c8575b1  Robby F  Mar 25, 2015
This is sick. I have spent way too much time trying to python talib functions for custom oscillators. Thanks.

edit: What is the good/standard way to build a dataframe with high/low/open/close/volume? I typically have/use individual history dataframes for each but I would like to know how to use the code as it is in the original post.

Thanks again

 409a3be6af196fecbaf1329aef6b50c2  Tarik Fehmi  Apr 24, 2015
Thank you!

 0c2b51aac4540dbc179f7583f7a231ea  Ethan Adair  Apr 24, 2015
Robby, here is some code from one of my algos I use to build an OHLC dataframe.. It works but I don't like the method. If anyone has a more streamlined approach I would be grateful to see it.
EDIT: you may want to modify it, I made it in such a way that it appends securities in the same data columns, this was useful for the way I was using the securities data

#Define Window  
    trail = 200  
    #store OHLCV data  
    open_hist = history(trail,'1d','open_price',ffill=False)  
    close_hist = history(trail,'1d','close_price',ffill=False)  
    high_hist = history(trail,'1d','high',ffill=False)  
    low_hist = history(trail,'1d','low',ffill=False)  
    volume_hist = history(trail,'1d','volume',ffill=False)  
    opencol = []  
    closecol = []  
    highcol = []  
    lowcol = []  
    volumecol = []  
    #trinsmit OHLCV to continuous numpy arrays  
    for sec in context.secs:  
        opencol = np.concatenate((opencol, open_hist[sec][:]))  
        closecol = np.concatenate((closecol, close_hist[sec][:]))  
        highcol = np.concatenate((highcol, high_hist[sec][:]))  
        lowcol = np.concatenate((lowcol, low_hist[sec][:]))  
        volumecol = np.concatenate((volumecol, volume_hist[sec][:]))  
    print("putting in pandas")  
    #Combine arrays into dataframe  
    df = pd.DataFrame({'O': opencol,  
                      'H': highcol,  
                      'L': lowcol,  
                      'C': closecol,  
                      'V': volumecol})  






'''




