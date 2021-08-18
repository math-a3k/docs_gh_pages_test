# -*- coding: utf-8 -*-
import scipy as sp;import numpy as np; import numexpr as ne
import pandas as pd; import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit, vectorize, guvectorize, float64, float32, int32, boolean
from timeit import default_timer as timer
import time 

import multiprocessing as mp; from functools import partial; import ctypes
#import multiprocessfunc as mpf #put the multi process function loop there


import global01 as global01


global ONE_SQRT_2PI, PI, TWOPI, ONE_2PI  #Global to this file
ONE_SQRT_2PI =  0.3989422804014326779399460599343818684758586311649346,
ONE_2PI =  0.159154943091895335768883763372514362034459645740456448747,
PI= 3.1415926535897932384626433832795028841971693993751058,
TWOPI= 6.2831853071795864769252867665590057683943387987502115,


global DAYS_PER_YEAR      #Global to this file
DAYS_PER_YEAR= 252




#-------Save Brownian generation on file------for MC simulation---------------------
def savebrownian(nbasset, step, nbsimul) :
 ss1 = "----------IID brownians  nbassets, time step, nb simulation"
 iidbm= np.random.normal(0, 1, (nbasset, step, nbsimul))   #independant brownian
 np.savez('F:/pythoncache/brownian_'+str(nbasset)+'asset_'+str(step)+'step_'+str(nbsimul)+'simul.npz',  iidbm=iidbm, details=ss1)

def loadbrownian(nbasset, step, nbsimul) :
 data = np.load('F:/pythoncache/brownian_'+str(nbasset)+'asset_'+str(step)+'step_'+str(nbsimul)+'simul.npz')
 return data['iidbm']

# savebrownian(2, 1, 5) 
# data1= loadbrownian(4, 2, 100000)




#------------------Black Scholes Formulae ---'''--------------------
# pdf gaussian # return np.exp(-0.5 * x * x )*ONE_SQRT_2PI 
def dN(d):   return    np.exp(-0.5*d*d)*ONE_SQRT_2PI   # ss.norm.pdf(d)   

def dN2d(x,y):   return  np.exp(-0.5*x*x-0.5*y*y)*ONE_2PI   # ss.norm.pdf(d)   


 
# c from scipy.integrate import quad, return quad(lambda x: dN(x), -20, d, limit=50)[0]
def N(d):    return  ss.norm.cdf(d)   

def d1f(St, K, t, T, r,d, vol):
    return  (np.log(St / K) + (r -d + 0.5 * vol*vol  )  * (T - t)) / (vol * np.sqrt(T - t))

def d2f(St, K, t, T, r,d, vol):
    return  (np.log(St / K) + (r - d - 0.5 * vol*vol )  * (T - t)) / (vol * np.sqrt(T - t))


def bsbinarycall(S0,K,t,T, r,d,vol):
  d2= d2f(S0, K, t, T, r,d, vol);   price1= np.exp(-r*T)*N(-d2)
  return price1  
  

def bscall(S0,K,t,T, r,d,vol):
  d1 =   d1f(S0, K, t, T, r,d, vol);    d2= d1 - vol*np.sqrt(T-t)
  price1= S0*np.exp(-d*(T-t))*N(d1) - K*np.exp(-r*(T-t))*N(d2)
  return price1


def bsput(S0,K,t,T, r,d,vol):
  d1 =  d1f(S0, K, t, T, r,d, vol);    d2= d1 - vol*np.sqrt(T-t)
  price1= -S0*np.exp(-d*(T-t))*N(-d1) + K*np.exp(-r*(T-t))*N(-d2)
  return price1


def bs(S0,K,t,T, r,d,vol,cp): #cp=1 call, cp=-1 put
  d1 =  d1f(S0, K, t, T, r,d, vol);   d2= d1 - vol*np.sqrt(T-t)
  price1= cp* (S0*np.exp(-d*(T-t))*N(cp*d1) - K*np.exp(-r*(T-t))*N(cp*d2))
  return price1


def bsdelta(St, K, t, T, r,d, vol, cp1): #be careful of equality for boolean
    d1 = d1f(St, K, t, T, r,d, vol)
    if cp1 == 1 :      aux= np.exp(-d*(T-t)) * N(d1)
    else:    aux= np.exp(-d*(T-t)) * (N(d1)-1)
    return aux

    
def bsgamma(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    return np.exp(-d*(T-t))*dN(d1) / (St * vol * np.sqrt(T - t))  #gamma is same between call put


def bsstrikedelta(s0,K,t,T, r,d,vol, cp1):  #discounted risk neutral probability
  d2= d2f(s0, K, t, T, r,d, vol)  
  if cp1 == 1 :       aux= - np.exp(-r*T)*N(d2) 
  else:         aux= np.exp(-r*T)*N(-d2) 
  return aux  


def bsstrikegamma(s0,K,t,T, r,d,vol): 
  d1 = d1f(s0, K, t, T, r,d, vol)
  return np.exp(-d*(T-t))*dN(d1) / (s0 * vol * np.sqrt(T - t))  #gamma is same between call put


def bstheta(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol);    ff= St*np.exp(-d*(T-t))
    return   -( ff *dN(d1) * vol / (2 * np.sqrt(T - t))    + cp*d * ff * N(cp*d1))
      

def bsrho(St, K, t, T, r,d, vol,cp):
    d2 = d2f(St, K, t, T, r,d, vol)
    return  cp*K * (T - t) * np.exp(-r * (T - t)) * N(cp*d2) *0.01


def bsvega(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    return   St*np.exp(-d*(T-t)) * np.sqrt(T - t) * dN(d1) *0.01


def bsdvd(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)  #    d2 = d1 - vol * np.sqrt(T - t)
    return  -cp*St * (T - t) * np.exp(-d * (T - t)) * N(cp*d1) *0.01


def bsvanna(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol);    d2 = d1 - vol * np.sqrt(T - t)
    return   -np.exp(-d*(T-t)) * dN(d1) *d2/vol


def bsvolga(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol);    d2 = d1 - vol * np.sqrt(T - t)
    return     St*np.exp(-d*(T-t)) * np.sqrt(T - t) * dN(d1) *d1*d2/vol


def bsgammaspot(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    return np.exp(-d*(T-t))*dN(d1) / (St*St * vol*vol*(T - t))*(d1+vol*np.sqrt(T-t))  #gamma is same between call put
 




#-----Generic Greeks by finite difference --------------------
def gdelta(St, K, t, T, r,d, vol, pv):
    return (pv(St*1.01, K, t, T, r,d, vol) - pv(St*0.99, K, t, T, r,d, vol) ) * 200

def ggamma(St, K, t, T, r, d, vol, pv):
    return (pv(St*1.01, K, t, T, r,d, vol) + pv(St*0.99, K, t, T, r,d, vol) - 2*pv(St, K, t, T, r,d, vol)  ) * 10000

def gvega(St, K, t, T, r, d,vol, pv):
    return (pv(St, K, t, T, r,d, vol*1.01) - pv(St, K, t, T, r,d, vol)  ) * 100

def gtheta(St, K, t, T, r, d,vol, pv):
    return (pv(St, K, t+0.0039682539, T, r,d, vol) - pv(St, K, t, T, r,d, vol)  ) * 100



#-----------Data generation function-------------------------
# with gg generator
@jit
def genmatrix(ni,nj,gg):
  mm= np.zeros((ni,nj))
  for i in range(0,ni):
    for j in range(0,nj):
      mm[i,j]= gg(i,j)
  return mm

# Same correl
@jit
def gensymmatrix(ni,nj,pp):
  mm= np.zeros((ni,nj))
  for i in range(0,ni):
    for j in range(0,nj):
      mm[i,j]= 1 if i == j else  pp
  return mm


'''  
def matgen1(i,j):  return 1 if i == j else  0.5
correlm= genmatrix(3,3,matgen1)
correlm2= gensymmatrix(3,3, 0.6)
'''


#------------Time Grid generation---------------------------------------
def timegrid(timestep ,maturityyears):
 nstep = np.int(maturityyears/ timestep)+1; timegrid1= np.zeros(nstep)
 for i in range(0,nstep):  timegrid1[i]= timestep*i
 return timegrid1 
 
 









#---------Multi Processing - Paralell version------------------------------------------
def generateall_multigbm1(process,ww,s0,mu, vol, corrmatrix, timegrid, nbsimul, nproc=-1, type1=-1, strike=0.0, cp=1):
 nbasset = len(vol);  n = len(timegrid)-1
 #allprocess = np.zeros((nbsimul, nbasset, n+1))  #nb of simulation in the process n path * k step

 corrmatrix= np.asmatrix(corrmatrix)
 upper_cholesky = sp.linalg.cholesky(corrmatrix, lower=False)  #Tranpose of Cholesky multi asset
 
 dt = (np.diff(timegrid)).reshape(1,n); 
 #dt = np.multiply( (np.ones(nbasset)).reshape(nbasset,1), dt)    #Multi Asset Time Grid 
 dt=  np.tile(dt, (nbasset, 1))  #Asset * Time Grid
  
 vol2 = np.multiply(vol,vol)
 volx= np.zeros((nbasset,n))
 for i in range(0,n):  volx[:,i] = vol  #time step of volatility
 sdt = (np.sqrt(dt)).reshape((nbasset,n))  #Needto reshape n=1 case, degenerate
 voldt=  np.multiply(volx, sdt)    # vol*sqrt(t)
 
 drift = (np.multiply(((mu - 0.5 * vol2)).reshape(nbasset,1), dt)).reshape((nbasset,n))  #multi vector drift
 price = np.zeros((nbasset,n+1),dtype=np.float); price[:,0] = s0 #Vector of prices

 if nproc==-1:  #No Pool, sequential computation
   res= mpf.multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky,  nbasset, n, price, type1, strike=strike, cp=cp)
   return res


 nbsimulproc= int(nbsimul/nproc)
 if type1 == 0: #-----Paralell Get Sum aggregate result--------------------------
   pool = mp.Pool(processes=nproc)
   res1 = pool.map(partial(mpf.multigbm_paralell_func, ww=ww, voldt=voldt, drift=drift, upper_cholesky=upper_cholesky,  nbasset=nbasset, n=n, price=price, strike=strike, cp=cp) ,  [nproc]*nbsimulproc ) #very important keep,list
   res= sum(res1) / float(nbsimulproc)


 if type1>0:  #No paralell results, TYoo Slow
   res= mpf.multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky,  nbasset, n, price, type1, strike=strike, cp=cp)
   return res

'''
 if type1 == 1: #----Get Vector Basket Final Value: No Multi Process TOO SLOW---------------------
#   res = [pool.apply_async(mpf.multigbm_processfast5, args=(nbsimul, s0, voldt, drift, upper_cholesky,  nbasset, n, price) for nbsimul in [nproc]*nbsimulproc]
 #  res = [p.get() for p in res]  
   global01.res_shared = shared_array([nbsimul])
   global01.lock= mp.Lock()  #concurrency management for modification   
   
   pool = mp.Pool(processes=nproc, initializer= mpf.init_global1, initargs=(global01.lock, global01.res_shared ))   
   res1 = pool.map(partial(mpf.multigbm_paralell_func, ww=ww, voldt=voldt, drift=drift, upper_cholesky=upper_cholesky,  nbasset=nbasset, n=n, price=price) ,  [nproc]*nbsimulproc ) #very important keep,list     
  
   return global01.res_shared
   pool.close();   pool.join();  

 if type1 == 2: #----Get Vector n Asset Final Value over nbsimul: No Multi Process TOO SLOW---------------------
 if type1 == 3: #----Get Vector n Asset, time t, nbsimul: No Multi Process TOO SLOW ---------------------

 pool.close();   pool.join();  # Need to terminate pool print(pool.is_alive()) 
 return res
 ''' 
  

'''
#--------------------simulatiion of Multi Asset Pricer --------------------------------
tt= 1.0 ;   timegrid1 = dx.timegrid(1/12.0,tt)
r1= 0.01;discount1 = np.exp(-r1*tt)

ww= [0.3, 0.2, 0.5];
s02 = [100.0, 200.0, 50.0];    mu2= [0.00,0.00, 0.00]
vol2 = [0.1,0.1, 0.1]
corrmatrix1 = [[1.0,0.125,0],[0.125,1.0,0],[0.0,0.0,1.0]]

start_time = time.time()
nprocs=4 #-1: No Multi Pass,  
nbsimul= 2 * 10**6
nchunk= 1000  #Chunk Based computation for CPU cache
type1= 0  #0: sum of all, 1:Bsk last value, 2: # Last value, 3:#ALl time step   

res= generateall_multigbm1(dx.multigbm_process,ww,s02,mu2, vol2, corrmatrix1, timegrid1, nbsimul,nprocs, 0)

print(str(nprocs)+"procs,"+ str(iters)+" iter,"+str(len(s02))+" assets,--- %s sec ---" % (time.time() - start_time))
'''


'''
4procs,20000 iter,3 assets,--- 2.8361618518829346 sec ---
-1procs,20000 iter,3 assets,--- 0.6770379543304443 sec ---
-1procs,200000 iter,3 assets,--- 6.708384037017822 sec ---
4procs,200000 iter,3 assets,--- 6.329361915588379 sec ---
4procs,2000000 iter,3 assets,--- 39.55426287651062 sec ---
'''




















#------------------------------------------------------------------------------
#-----Generate GBM process ----------------------------------------------------
def logret_to_ret(log_returns):   return np.exp(log_returns)

def logret_to_price(s0, log_ret):
    n = len(log_ret)
    ret = np.exp(log_ret)
    price = np.zeros(n+1)
    price[0] = s0
    for i in range(1, n+1):
        price[i]= price[i - 1] * ret[i - 1]          # Add the price at t-1 * return at t
    return np.array(price)
    
    
def brownian_logret(mu, vol, timegrid):
    n = len(timegrid) -1
    dt = np.diff(timegrid)  
    nn= np.random.normal(0, vol, n)
#    return  np.random.normal(0, 1, n) * vol * np.sqrt(dt)
    return nx.evaluate("nn * sqrt(dt)  ")
    
def brownian_process(s0, vol, timegrid):
    return logret_to_price(s0, brownian_logret(vol, timegrid))
    

def gbm_logret(mu, vol, timegrid):
    n = len(timegrid) -1
    dt = np.diff(timegrid)  
    nn= np.random.normal(0, vol, n)    #    drift = (mu - 0.5 *vol*vol) * dt
    gbm =  nx.evaluate("nn * sqrt(dt)  + (mu - 0.5 *vol*vol) * dt ")
    return gbm


# Old version slow
#def gbm_logret2(mu, vol, timegrid):          
#    dt = np.diff(timegrid)       
#    wiener_process = np.array(brownian_logret(0,vol, timegrid))
#    drift = (mu - 0.5 *vol*vol) * dt
#    return wiener_process + drift



def gbm_process(s0, mu, vol, timegrid):
    n = len(timegrid) -1
    dt = np.diff(timegrid)  
    nn= np.random.normal(0, vol, n)

    ret =  nx.evaluate("exp(nn * sqrt(dt)  + (mu - 0.5 *vol*vol) * dt )")

    price = np.zeros(n+1)
    price[0] = s0
    for i in range(1, n+1):
        price[i]= price[i - 1] * ret[i - 1]          # Add the price at t-1 * return at t
    return np.array(price)


def gbm_process_euro(s0, mu, vol, timegrid):  # can simplify the process
    n = len(timegrid) -1;    dt = np.diff(timegrid)  
    nn= np.random.normal(0, vol, n)

    price = np.zeros(n+1)
    price[1:]=  nx.evaluate("exp(nn * sqrt(dt)  + (mu - 0.5 *vol*vol) * dt )")
    price[0] = s0
    price = np.cumprod(price, axis = 0)  #exponen product st = st-1 *st    
       
#    for i in range(1, n+1):
#        price[i]= price[i - 1] * ret[i - 1]          # Add the price at t-1 * return at t

    return price
  

def gbm_process2(s0, mu, vol, timegrid):
    return logret_to_price(s0, gbm_logret(mu, vol, timegrid))
    


#--------------Single Asset, path dependant, Mutiple simulation, ---------------------------------------------
def generateallprocess(process, params01, timegrid1, nbsimul):
 allprocess = np.zeros((nbsimul, len(timegrid1)))  #number of simulation in the process n path * k step
 for k in range(0, nbsimul):
   allprocess[k,:] = process(timegrid=timegrid1, *params01 )  #one can pass param by a list

 return allprocess


#--------------Single Asset, european, Mutiple simulation  2 step 0 and T, European---------------------------------------------
def generateallprocess_gbmeuro(process, params01, timegrid1, nbsimul):  #Only at T
 s0= params01[0];  mu= params01[1];  vol= params01[2];  
 tt= timegrid1[-1] #final step
 st= s0*exp((mu - 0.5 *vol*vol) * tt )
 nnvoldt= np.random.normal(0, vol*sqrt(tt), nbsimul)
 allprocess =  nx.evaluate("st*exp(nnvoldt)")   

 return allprocess




#--------------Get PV from simulation, allpriceprocess is n path x k step ------------------------------------------------
@jit
def getpv(discount, payoff, allpriceprocess):
 size1= np.shape(allpriceprocess)     #number of simulation in the process n path * k step  
 n= size1[0]     #number of simulation in the process n path * k step
 payout = np.zeros(n)
 
 if allpriceprocess.ndim ==1 :  #Single step, Final Value Only, European
   payout = discount * payoff(allpriceprocess)   
 else:                          # multi step, Path dependance
   for k in range(0, n):
    payout[k] = discount * payoff(allpriceprocess[k,:])

 pricemean1= np.mean(payout)
 pricestd1= np.std(payout) /np.sqrt(n)
 
 return (pricemean1, pricestd1)



'''
#--- Path  depedance payoff 
strike1=100
def payoff1(pricepath):
  st= pricepath[-1];   return np.maximum(st - strike1,0)

#---------at time T
strike1=100
def payoffeuro1(st):
  return np.maximum(st - strike1,0)
'''





#--------------Multi Asset GBM--------------------------------------------------------
def multigbm_processfast(s0, voldt, drift, upper_cholesky,  nbasset, n,kk):
    iidbm= np.asmatrix(np.random.normal(0, 1, (nbasset,n)))   #independant brownian
    corrbm = np.dot(upper_cholesky, iidbm)    # correlated brownian    
    wiener_process= np.multiply(corrbm,voldt)  #multiply element by elt

    ret= np.exp(wiener_process + drift)

    price = np.asmatrix(np.zeros((nbasset,n+1)))
    price[:,0] = s0
    for i in range(1, n+1):
        price[:,i]= np.multiply( price[:,i - 1] , ret[:,i - 1] )    # Add the price at t-1 * return at t
    return price





#--------------Multi Asset GBM, using store data in browniandata--------------------------------------------------------
global Browniandata   #If declare global here, only to this file.....

def getbrowniandata(nbasset, step, simulk) : 
 return  Browniandata[0:nbasset,0:step, simulk ]
 
def multigbm_processfast2(s0, voldt, drift, upper_cholesky,  nbasset, n,kk):
    iidbm= np.asmatrix( getbrowniandata(nbasset, n,kk ))   #independant brownian
    corrbm = np.dot(upper_cholesky, iidbm)    # correlated brownian    
    wiener_process= np.multiply(corrbm,voldt)  #multiply element by elt

    ret= np.exp(wiener_process + drift)

    price = np.asmatrix(np.zeros((nbasset,n+1)))
    price[:,0] = s0
    for i in range(1, n+1):
        price[:,i]= np.multiply( price[:,i - 1] , ret[:,i - 1] )    # Add the price at t-1 * return at t
    return price



 
 

#-------------- Multi Asset Fast generation - N path simulation--------------------------

#process,s0,mu, vol, corrmatrix, timegrid, nbsimul, type1

@jit
def generateallmultigbmfast(process,s0,mu, vol, corrmatrix, timegrid, nbsimul, type1):
 nbasset = len(vol)
 allprocess = np.zeros((nbsimul, nbasset, len(timegrid)))  #nb of simulation in the process n path * k step

 mu = np.asmatrix(mu); vol= np.asmatrix(vol); corrmatrix= np.asmatrix(corrmatrix)
 dt = np.asmatrix(np.diff(timegrid))
 dt = np.dot( (np.asmatrix(np.ones(nbasset))).T, dt)    #Multi Asset Time Grid 
 nn = len(timegrid)-1
 s0= (np.asmatrix(s0)).T
 vol2 = np.multiply(vol,vol)

 volx= np.zeros((nbasset,nn))
 for i in range(0,nn):  volx[:,i] = vol  #time step of volatility
 sdt = np.asmatrix(np.sqrt(dt))
 voldt=  np.multiply(volx, sdt)    # vol*sqrt(t)
 
 drift = np.multiply(((mu - 0.5 * vol2).T), dt)  #multi vector drift
 upper_cholesky = (sp.linalg.cholesky(corrmatrix, lower=False)).T  #Tranpose of Cholesky
    
 if type1 == 1:
  for k in range(0, nbsimul):  #generate every time
    allprocess[k,:] = multigbm_processfast(s0, voldt,drift, upper_cholesky, nbasset, nn,k) 
 else :
  for k in range(0, nbsimul):   #use Stored data
    allprocess[k,:] = multigbm_processfast2(s0, voldt,drift, upper_cholesky, nbasset, nn,k)     
   
 return allprocess



@jit
def multigbm_processfast3(s0, voldt, drift, upper_cholesky,  nbasset, n,kk):
    iidbm= np.random.normal(0, 1, (nbasset,n))   #independant brownian
    corrbm = np.dot(upper_cholesky, iidbm)    # correlated brownian    
    wiener_process= np.multiply(corrbm,voldt)  #multiply element by elt

    ret= np.exp(wiener_process + drift)
    price = np.zeros((nbasset,n+1))
    price[:,0] = s0 #Vector of prices
    for i in range(1, n+1):
        price[:,i]= np.multiply( price[:,i - 1] , ret[:,i - 1] )    # Add the price at t-1 * return at t
    return price



'''
#--------------------positive correlation----------
s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.13333333333333333,0.13333333333333333]
corrmatrix1 = [[1.0,0.125],[0.125,1.0]]
np.random.seed(1234)
pp3= generateallmultiprocessfast(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)

ww= 	[0.5,0.5];    dx.getpv(discount1, payoff2, pp3)
'''






#-------------- Multi Asset Fast generation - N path simulation--------------------------
def generateallmultigbmfast2(process,s0,mu, vol, corrmatrix, timegrid, nbsimul, type1):
 nbasset = len(vol)
 allprocess = np.zeros((nbsimul, nbasset, len(timegrid)))  #nb of simulation in the process n path * k step

 mu = mu; vol= vol; corrmatrix= np.asmatrix(corrmatrix)
 dt = np.diff(timegrid)
 dt = np.dot( (np.ones(nbasset)).T, dt)    #Multi Asset Time Grid 
 nn = len(timegrid)-1
 s0= s0.T
 vol2 = np.multiply(vol,vol)

 volx= np.zeros((nbasset,nn))
 for i in range(0,nn):  volx[:,i] = vol  #time step of volatility
 sdt = np.sqrt(dt)
 voldt=  np.multiply(volx, sdt)    # vol*sqrt(t)
 
 drift = np.multiply(((mu - 0.5 * vol2).T), dt)  #multi vector drift
 upper_cholesky = sp.linalg.cholesky(corrmatrix, lower=False)  #Tranpose of Cholesky
    
 if type1 == 1:
  for k in range(0, nbsimul):  #generate the random
    allprocess[k,:] = multigbm_processfast(s0, voldt,drift, upper_cholesky, nbasset, nn,k) 
 else :
  for k in range(0, nbsimul):   #use Stored data
    allprocess[k,:] = multigbm_processfast2(s0, voldt,drift, upper_cholesky, nbasset, nn,k)     
   
 return allprocess














'''
#--------------------------------- Multi Asset : positive correlation----------
tt= 1.0 ;   timegrid1 = dx.timegrid(12/12.0,tt)
r1= 0.01
discount1 = np.exp(-r1*tt)

strike1=100.0
def payoff2(pricepath):
  size1 = np.shape(pricepath);  tt= size1[1]; nbasset=size1[0]
  baskett= sum( ww * pricepath[:,tt-1] )
  return np.maximum(baskett - strike1,0)

s02 = [100.0,100.0]
mu2= [0.01,0.01]
vol2 = [0.13333333333333333,0.13333333333333333]
corrmatrix1 = [[1.0,0.125],[0.125,1.0]]

np.random.seed(1234)
mm3= dx.generateallmultigbmfast(dx.multigbm_process,s02,mu2, vol2, corrmatrix1, timegrid1, 4)

ww= 	[0.5,0.5]
dx.getpv(discount1, payoff2, mm3)

'''




























# ----- Be Careful of Indexation and table setup----------------------------------
#--------------  Multi Asset of brownians    ------------------------------------
def multibrownian_logret(mu, vol, corrmatrix, timegrid):
    mu = np.asmatrix(mu); vol= np.asmatrix(vol); corrmatrix= np.asmatrix(corrmatrix)
    nbasset= np.shape(vol)[1] #nb of assets with individual vol. 
    tt = len(timegrid) -1
    
    vol2= np.zeros((nbasset,tt))
    for i in range(0,tt):  vol2[:,i] = vol  #time step of volatility
    
    sdt = np.asmatrix(np.sqrt(np.diff(timegrid)))
    voldt=  np.multiply(vol2, sdt)    # vol*sqrt(t)
    
    upper_cholesky = (sp.linalg.cholesky(corrmatrix, lower=False)).T  #Tranpose of Cholesky
    iidbm= np.asmatrix(np.random.normal(0, 1, (nbasset,tt)))   #independant brownian
    corrbm = np.dot(upper_cholesky, iidbm)    # correlated brownian
    
    corrbrownianstep= np.multiply(corrbm,voldt)  #multiply element by elt
    
    return corrbrownianstep



def multigbm_logret(mu, vol, corrmatrix, timegrid):
    mu = np.asmatrix(mu); vol= np.asmatrix(vol); corrmatrix= np.asmatrix(corrmatrix)
    nbasset= np.shape(vol)[1] #nb of assets with individual vol.
    
    dt = np.asmatrix(np.diff(timegrid))
    dt = np.dot( (np.asmatrix(np.ones(nbasset))).T, dt)    #Multi Asset Time Grid
    
    wiener_process = np.array(multibrownian_logret(mu, vol, corrmatrix, timegrid))
    vol2 = np.multiply(vol,vol)
    drift = np.multiply(((mu - 0.5 * vol2).T), dt)  #multi vector drift
  
    return wiener_process + drift



def multilogret_to_price(s0, log_ret):
    size1 = np.shape(log_ret);    nbasset = size1[0];    n= size1[1]+1
    ret = np.exp(log_ret)
    price = np.asmatrix(np.zeros((nbasset,n)))
    price[:,0] = (np.asmatrix(s0)).T
    for i in range(1, n):
        price[:,i]= np.multiply( price[:,i - 1] , ret[:,i - 1] )       # Add the price at t-1 * return at t
    return np.array(price)
    

def multigbm_process(s0, mu, vol, corrmatrix, timegrid):
    aux = multigbm_logret(mu, vol, corrmatrix, timegrid)
    return multilogret_to_price(s0, aux)
    


#--------------All path multi asset (slow version)-------------------------------------
def generateallmultiprocess(process,s0,mu, vol, corrmatrix, timegrid, nbsimul):
 nbasset = len(vol)
 allprocess = np.zeros((nbsimul, nbasset, len(timegrid1)))  #nb of simulation in the process n path * k step
 for k in range(0, nbsimul):
   aux = multigbm_process(s0,mu, vol, corrmatrix, timegrid) 
   allprocess[k,:] = aux
 return allprocess









#-----TO DO Later Generate One Path of Jump Geometric Brownian -NOT TESTED------------------------
def jump_process(lamda,jumps_mu, jumps_vol, timegrid):
    """:param param: the model parameters object :return: jump sizes for each point in time (mostly zeroes if jumps are infrequent) """
    dt = np.diff(timegrid)
    n = len(timegrid) -1
    jump_sizes = np.zeros(n)
   
    for j in range(0, n):
      nn= np.random.poisson(lamda*dt[j], 1)   #Nb of Jumer over ti+1 - ti  from poisson distribution
      aux=0
      for k in range(1,nn):  # Sum(LogJump(meanSize, StdJump),k=1.. Nt(=poisson)
         aux += np.random.normal(jumps_mu, jumps_vol)

      jump_sizes[j]=aux

    return jump_sizes


def gbmjump_logret(s0,mu, vol, lamda,jump_mu, jump_vol, timegrid):
    """returns a GBM process with jumps in it"""
    jump_diffusion = jump_process(lamda,jump_mu, jump_vol, timegrid)
    geometric_brownian_motion = gbm_logret(mu, vol, timegrid)
    return np.add(jump_diffusion, geometric_brownian_motion)


def gbmjump_process(s0,mu, vol, lamda,jump_mu, jump_vol, timegrid): #Price
    return logret_to_price(s0, gbmjump_logret(s0,mu, vol, lamda,jump_mu, jump_vol, timegrid))    
    




#-----Basket Option Moment Matching--------------------
#----Calculate moments
#fft = s02 * np.exp([tt*mu for mu in mu2])  #calculate forward


# calcultate exact value E(x**k), X= Sum of log normal
def lgnormalmoment1(ww, fft, vol, corr, tt) : 
  nasset = len(ww); EB=0
  for i in range(0,nasset) :  #kmax tuples, with 1..nasset
    EB= EB+ ww[i] *fft[i]
  return EB


# calcultate exact value E(x**k), X= Sum of log normal
def lgnormalmoment2(ww, fft, vol, corr, tt) : 
  nasset = len(ww); EB2=0
  for i,j in np.ndindex((nasset,nasset)) :  #kmax tuples, with 1..nasset
    EB2= EB2+ ww[i]*ww[j] *fft[i]*fft[j]*exp((corr[i,j]*vol[i]*vol[j])*tt)
  return EB2


def lgnormalmoment3(ww, fft, vol, corr, tt) : 
  nasset = len(ww);EB3=0
  for i,j,k in np.ndindex((nasset,nasset,nasset)) :  #kmax tuples, with 1..nasset
    EB3= EB3+ ww[i]*ww[j]*ww[k] *fft[i]*fft[j]*fft[k]   *  \
    exp((corr[i,j]*vol[i]*vol[j] + corr[i,k]*vol[i]*vol[k] + corr[j,k]*vol[j]*vol[k]   )*tt)
  return EB3
  
# calcultate exact value E(x**k), X= Sum of log normal  
def lgnormalmoment4(ww, fft, vol, corr, tt) : 
  nasset = len(ww); EB4=0
  for i,j,k,l in np.ndindex((nasset,nasset,nasset,nasset)) :  #kmax tuples, with 1..nasset
    ee= corr[i,j]*vol[i]*vol[j] + corr[i,k]*vol[i]*vol[k] + corr[i,l]*vol[i]*vol[l]
    ee= ee+ corr[j,k]*vol[j]*vol[k] + corr[j,l]*vol[j]*vol[l] 
    ee= ee+ corr[k,l]*vol[j]*vol[k]    
  
    EB4= EB4 + ww[i]*ww[j]*ww[k]*ww[l]  *fft[i]*fft[j]*fft[k]*fft[l] * exp(ee*tt)
   
  return EB4






def solve_momentmatch3(ww,b0, fft, vol, corr, tt) :
 m1= lgnormalmoment1(ww, fft, vol2, corr1, tt)
 m2= lgnormalmoment2(ww, fft, vol2, corr1, tt)
 m3= lgnormalmoment3(ww, fft, vol2, corr1, tt)
  
 bb1= -4 - (m1*(3*m2-2*m1*m1)-m3)**2 / ((m2-m1*m1)**3)
 x  = sy.symbols('x', real=True, Positive=True)
 ee= x**3 + 3*x**2+ bb1; 
 rs= sy.solve(ee,x)

 aa = rs[0]    # exp(volB**2.T) = aa
 ee = sy.sqrt((m2-m1*m1)/(aa-1))  #ForwardB= F0 * exp(u*T)
 kk= m1 -ee
 
 volb = sy.sqrt( sy.log(aa) / tt)
 ub=  sy.log(ee/b0)/tt
 
 aa= (ub,volb,kk,ee)
 return np.array(np.array(aa), np.float)






'''
 B(t) = B1(t) + kk   with B1 following log normal with (mub, volb)
 [B - strike ]+ = [B1 + kk - strike]+  and use log normal on B1



vol2 = [0.3,0.3]
corr1 = dx.gensymmatrix(2,2, 0.0)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)

vol=0.3, correl=0.0
(ub,volb,ee,kk)
(0.0254668211564405, 0.238077287194359, 37.0113014697911, 2.71495425727288)

'''



'''
vol2 = [0.1,0.1]
corr1 = dx.gensymmatrix(2,2, 0.0)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)
#ok (0.0377323829088803, 0.0719785742973266, 39.3086210088560, 0.417634718208006)


vol2 = [0.1,0.1]
corr1 = dx.gensymmatrix(2,2, -0.99)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)
# (-0.535997993729427, 0.274117665966603, 2.35039851827077, 37.3758572087932)


vol2 = [0.1,0.1]
corr1 = dx.gensymmatrix(2,2, -0.2)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)
# ok (0.0344917921947485, 0.0657586973174978, 38.6881536653952, 1.03810206166882)



vol2 = [0.1,0.1]
corr1 = dx.gensymmatrix(2,2, -0.6)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)
# ok (0.000737213163064528, 0.0564771365049024, 32.7796439392386, 6.94661178782538)



vol2 = [0.1,0.1]
corr1 = dx.gensymmatrix(2,2, -0.7)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)
#  (-0.0303810160456329, 0.0580734365252152, 28.1353201321121, 11.5909355949519)




vol2 = [0.3,0.3]
corr1 = dx.gensymmatrix(2,2, -0.99)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)
#  [ -0.19082151,   0.37310658,  12.79822959,  26.92802614]



vol2 = [0.3,0.3]
corr1 = dx.gensymmatrix(2,2, 0.0)
solve_momentmatch3(ww, b0,  fft, vol2, corr1, tt)
#  [ 2.54668212e-02,   2.38077287e-01,   3.70113015e+01,     2.71495426e+00






ub= 2.54668212e-02
volb=  2.38077287e-01
kk1=  2.71495426e+00
dx.bscall(b0, strike1-kk1,0,tt, ub,0,volb) /strike1*100
# 29.818967470432302



#------Using data-------------------------------
vol2 = [0.3,0.3]
corr1 = dx.gensymmatrix(2,2, 0.0)
mm3= dx.generateallmultigbmfast(dx.multigbm_process,s02,mu2, vol2, corr1, timegrid1, 50000, 2)
res= dx.getpv(discount1, payoff2, mm3) 

(res[0]/strike1, res[1]/strike1)
# (0.26361670272804849, 0.001916779118678702)








#-----Different Drift  0.01 and -0.01, same vol, different correl-----------------------
tt= 1.0;    dx.timegrid1 = dx.timegrid(12/12.0,tt)  
s01=100.0; strike1=100.0; mu1=0.01; r1 = 0.01;  sigma1=0.1; d1= 0.00;    #in years
discount1 = np.exp(-r1*tt)

s02 = [100.0,100.0]
mu2= [0.01,-0.01]
vol2 = [0.15,0.15]
corr1 = [[1.0,-0.11111111111111111],[-0.11111111111111111,1.0]]
ww= 	[0.5,0.5];   

allprocess1 = dx.generateallmultigbmfast(dx.gbm_process, s02,mu2, vol2, corr1,timegrid1 , 40000,2)
[dx.getpv(discount1, payoff2, allprocess1),
dx.bscall(s01,strike1,0,tt, 0,d1,sigma1) 
]
#[(3.9265661575471227, 0.030802722705509825), 3.987761167674492]



s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.13333333333333333,0.13333333333333333]
corrmatrix1 = [[1.0,0.125],[0.125,1.0]]
np.random.seed(1234)



Basket with average drift
mu2= [0.01,-0.01]
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 4000)
Out[39]: (4.0835447635662696, 0.096944585940324415)

mm2= dx.generateallmultiprocess(dx.multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
[dx.getpv(discount1, payoff2, mm2),
dx.bscall(s01,strike1,0,tt, 0,d1,sigma1) 
]


[(3.9861468221049261, 0.030769330410031107), 3.987761167674492]
Out[40]: (3.9921108519150379, 0.030847816177326366)


'''



























#-----Stochastic Interest Rates--------------------














#------Save Table/Data into Files--------------------------------
'''
ss1 = """----------Save Process data with 
  1,000,000 paths, montly 1/12 steps, 10 years, 
  s01=100.0;  mu1=0.01; r1 = 0.01;  vol1=0.2; d1= 0.00;    
"""
      
      
      
np.savez('F:/pythoncache/allprocesseuro_1D01.npz',  allprocesseuro=allprocesseuro, details=ss1)
data = np.load('F:/pythoncache/allprocesseuro_1D01.npz')
data['details']
allprocesseuro1= data['allprocesseuro']

'''



''' Independant Gaussian :
ss1 = """----------IID brownians
2 assets, 10 time step, 100000 simulation
"""
def savebrownian(nbasset, step, nbsimul) :
 ss1 = "----------IID brownians  nbassets, time step, nb simulation"
 iidbm= np.asmatrix(np.random.normal(0, 1, (nbasset, step, nbsimul)))   #independant brownian
 np.savez('F:/pythoncache/brownian_'+str(nbasset)+'_'+str(step)+'_'+str(nbsimul).npz',  iidbm=iidbm, details=ss1)

'''





#--------------------Plotting Tools ---------------------------------------------
def plot_greeks(function, greek):
    # Model Parameters
    St = 100.0  # index level
    K = 100.0  # option strike
    t = 0.0  # valuation date
    T = 1.0  # maturity date
    r = 0.05  # risk-less short rate
    vol = 0.2  # volatility

    # Greek Calculations
    tlist = np.linspace(0.01, 1, 25)
    klist = np.linspace(80, 120, 25)
    V = np.zeros((len(tlist), len(klist)), dtype=np.float)
    for j in range(len(klist)):
        for i in range(len(tlist)):
            V[i, j] = function(St, klist[j], t, tlist[i], r, vol)

    # 3D Plotting
    x, y = np.meshgrid(klist, tlist)
    fig = plt.figure(figsize=(9, 5))
    plot = p3.Axes3D(fig)
    plot.plot_wireframe(x, y, V)
    plot.set_xlabel('strike $K$')
    plot.set_ylabel('maturity $T$')
    plot.set_zlabel('%s(K, T)' % greek)




import mpl_toolkits.mplot3d.axes3d as p3
# Plotting the Greeks
def plot_greeks(function, greek):
    # Model Parameters
    St = 100.0  # index level
    K = 100.0  # option strike
    t = 0.0  # valuation date
    T = 1.0  # maturity date
    r = 0.05  # risk-less short rate
    vol = 0.2  # volatility

    # Greek Calculations
    tlist = np.linspace(0.01, 1, 25)
    klist = np.linspace(80, 120, 25)
    V = np.zeros((len(tlist), len(klist)), dtype=np.float)
    for j in range(len(klist)):
        for i in range(len(tlist)):
            V[i, j] = function(St, klist[j], t, tlist[i], r, vol)

    # 3D Plotting
    x, y = np.meshgrid(klist, tlist)
    fig = plt.figure(figsize=(9, 5))
    plot = p3.Axes3D(fig)
    plot.plot_wireframe(x, y, V)
    plot.set_xlabel('strike $K$')
    plot.set_ylabel('maturity $T$')
    plot.set_zlabel('%s(K, T)' % greek)




# Plotting European Option Values ''' Plots European option values for different parameters c.p. '''
def plot_values(function):  
    plt.figure(figsize=(10, 8.3))
    points = 100
    # Model Parameters
    St = 100.0  # index level
    K = 100.0  # option strike
    t = 0.0  # valuation date
    T = 1.0  # maturity date
    r = 0.05  # risk-less short rate
    vol = 0.2  # volatility

    # C(K) plot
    plt.subplot(221)
    klist = np.linspace(80, 120, points)
    vlist = [function(St, K, t, T, r, vol) for K in klist]
    plt.plot(klist, vlist)
    plt.grid()
    plt.xlabel('strike $K$')
    plt.ylabel('present value')

    # C(T) plot
    plt.subplot(222)
    tlist = np.linspace(0.0001, 1, points)
    vlist = [function(St, K, t, T, r, vol) for T in tlist]
    plt.plot(tlist, vlist)
    plt.grid(True)
    plt.xlabel('maturity $T$')

    # C(r) plot
    plt.subplot(223)
    rlist = np.linspace(0, 0.1, points)
    vlist = [function(St, K, t, T, r, vol) for r in rlist]
    plt.plot(tlist, vlist)
    plt.grid(True)
    plt.xlabel('short rate $r$')
    plt.ylabel('present value')
    plt.axis('tight')

    # C(vol) plot
    plt.subplot(224)
    slist = np.linspace(0.01, 0.5, points)
    vlist = [function(St, K, t, T, r, vol) for vol in slist]
    plt.plot(slist, vlist)
    plt.grid(True)
    plt.xlabel('volatility $\vol$')
    plt.tight_layout()


#--------------------------------------------------------------------------
#-------------------------------------------------------------------------




#--------------------- Cox-Ross-Rubinstein Binomial Model----------------------------
# Valuation Function
def CRR_option_value(S0, K, T, r, vol, otype, M=4):
    ''' Cox-Ross-Rubinstein European option valuation.
    otype : string  either 'call' or 'put'
    M : int  number of time intervals
    '''
    # Time Parameters
    dt = T / M  # length of time interval
    df = np.exp(-r * dt)  # discount per interval

    # Binomial Parameters
    u = np.exp(vol * np.sqrt(dt))  # up movement
    d = 1 / u  # down movement
    q = (np.exp(r * dt) - d) / (u - d)  # martingale branch probability

    # Array Initialization for Index Levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d ** md
    S = S0 * mu * md

    # Inner Values
    if otype == 'call':
        V = np.maximum(S - K, 0)  # inner values for European call option
    else:
        V = np.maximum(K - S, 0)  # inner values for European put option
    
    z = 0
    for t in range(M - 1, -1, -1):  # backwards iteration
        V[0:M - z, t] = (q * V[0:M - z, t + 1]
                        + (1 - q) * V[1:M - z + 1, t + 1]) * df
        z += 1
    return V[0, 0]










