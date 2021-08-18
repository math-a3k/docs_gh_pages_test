""" Random Generator Analysis   """

import scipy as sp;import numpy as np; import numexpr as ne
import pandas as pd; import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit, vectorize, guvectorize, float64, float32, int32, boolean
from timeit import default_timer as timer

import derivatives as dx


#Get the data into numpy vector
mersenne = 'E:\_data\PSEUDO_MT19937_0_250100100.h5'
sobol64 = 'E:\_data\QUASI_SOBOL64_20000dim_12000sample.h5'
sobolscram = 'E:\_data\SCRAMBLED_SOBOL64_20000dim_12000sample.h5'


mersennenormal = 'E:\_data\PSEUDO_MT19937_normal_0_250100100.h5'  #Mersenne not of dimensionn


sobol_20000d_12000sample_gaussian = 'E:\_data\QUASI_SOBOL64_gaussian_20000dim_12000samples.h5'
sobol_16384d_4096sample_gaussian = 'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.h5'


sobolscram_20000dim_12000samples_gaussian = 'E:\_data\SCRAMBLED_SOBOL64_gaussian_20000dim_12000samples.h5'
sobolscram_16384d_4096sample_gaussian  = 'E:\_data\SCRAMBLED_SOBOL_gaussian_16384dim__4096samples.h5'
sobolscramnormal15 = 'E:\_data\SCRAMBLED_SOBOL64_normal_500dim_500000sample.h5'

filename = sobol_16384d_4096sample_gaussian 
title1='Sobolgauss 16384d '

#-------------------------------------------------------------
ijump= 12000
filename= sobol_20000d_12000sample_gaussian 
filename= mersennenormal
filename= sobolscram_20000dim_12000samples_gaussian   

#pdf=  pd.read_hdf(filename,'data', start=0, stop=(2048*12000))
#allv= pdf.values   ;   del pdf



ijump= 4096
filename= sobol_16384d_4096sample_gaussian 
filename= mersennenormal
filename= sobolscram_16384d_4096sample_gaussian  


#pdf=  pd.read_hdf(filename,'data', start=0, stop=(16384*4096))
#allv= pdf.values   ;   del pdf

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------






#----Input the data: From CSV to HD5f files -------------------------------
def convert_csv2hd5f(filein1, filename):
 #filein1=   'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.csv'
 #filename = 'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.h5'
 chunksize =     10 * 10 ** 6
 list01= pd.read_csv(filein1, chunksize=chunksize, lineterminator=',')
 for chunk in list01:
     store = pd.HDFStore(filename)
     store.append('data', chunk)
     store.close()     
 del chunk


#---LOAD In Advance Calculation----------------------------------------
def getrandom_tonumpy(filename, nbdim, nbsample):
 pdframe=  pd.read_hdf(filename,'data', start=0, stop=(nbdim*nbsample))
 return pdframe.values   #to numpy vector       

# yy1= getrandom_tonumpy('E:\_data\_QUASI_SOBOL_gaussian_xx2.h5', 16384, 4096)
#-------------=---------------------------------------------------------------
#----------------------------------------------------------------------------






#-------------=---------------------------------------------------------------
#---------------------Statistics----------------------------------------------
#Calculate Co-moment of xx yy
def comoment(xx,yy,nsample, kx,ky) :
#   cx= ne.evaluate("sum(xx)") /  (nsample);   cy= ne.evaluate("sum( yy)")  /  (nsample)
#   cxy= ne.evaluate("sum((xx-cx)**kx * (yy-cy)**ky)") / (nsample)
   cxy= ne.evaluate("sum((xx)**kx * (yy)**ky)") / (nsample)
   return cxy 


#Autocorrelation 
def acf(data):
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return acf_lag
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = np.asarray(list(map(r, x)))
    return acf_coeffs
 
 
#-------------=---------------------------------------------------------------
#-----------------------------------------------------------------------------

 





#-------------Path Generation--------------------------------------------
ijump= 4096   #sample size  Global 
#allv=  getrandom_tonumpy(filename, nbdim, nbsample)  #Global

#D dimension vector, 1 sample   from Big Table allv
@jit
def getdvector(dimmax,istart, idimstart ):
 dvec =np.zeros(dimmax, dtype= np.float64)
 for dd in range(0,dimmax):
  x0= (dd+idimstart) * ijump + istart   #ijump is 4096 or 12000
  dvec[dd]= allv[x0]
 return dvec



@jit
def pathScheme_std(T,n,zz):   #Standard Path generation
  ww=np.zeros(n+1);  sdt= np.sqrt(T/n);  ww[0]= 0
  for jj in range(1,n+1): 
   ww[jj]= ww[jj-1] + sdt*zz[jj-1] 
  return ww


@jit
def pathScheme_bb(T,n,zz):  #Brownian Bridge generation
 ww=np.zeros(n); # sdt= np.sqrt(T/n);  
 kkmax= int(np.round(np.log(n)  * 1.4426950408889634)) # n= 2^kmax
 h= n; jmax=1
 ww[0]= 0; ww[n-1] = zz[n-1]*np.sqrt(T)
  
 for kk in range(1, kkmax+1): 
     imin = int(h/2);     i= imin
     l=0; r=h
     for j in range(1, jmax+1):
      a= ((r-i)*ww[max(0,l-1)] + (i-l)*ww[r-1])/(r-l)
      b= np.sqrt(((r-i)*(i-l))/(r-l)/n)
      ww[i-1]= a + b *zz[i-1]         
      i=i+h; l=l+h; r=r+h;
     jmax= 2*jmax   #BB formulae to 
     h= imin     
 return ww

@jit
def pathScheme_(T,n,zz):   return pathScheme_std(T, n, zz)
#-------------=---------------------------------------------------------------
#-----------------------------------------------------------------------------





#----Convergence of the 1D density distribution , Standdard/BBridge ----------
@jit
def testdensity(nsample, totdim, bin01, Ti=-1) :
 wtt= np.zeros(nsample)
 for ii in range(0, nsample):
  dvec= getdvector( totdim, ii,13)  # d dimension Gaussian  vector
  std1= pathScheme_(1, totdim, dvec)   #Scheme
  wtt[ii]= std1[Ti]  #Last one -2: for BB Bridge, -1 for STD

 hbin= bin01/2
 bins01= np.arange(-hbin,hbin)/hbin*4 #bin size 

 hist, bins = np.histogram(wtt, bins01, density=True)
 width = 0.3 * (bins[1] - bins[0]); center = (bins[:-1] + bins[1:]) / 2

 diff1=  hist - dx.dN(center)
 return np.sum(np.abs(diff1))    


@jit
def plotdensity(nsample, totdim, bin01, tit0, Ti=-1) :
 wtt= np.zeros(nsample)
 for ii in range(0, nsample):
  dvec= getdvector( totdim, ii,13)  # d Gaussian
  std1= pathScheme_(1, totdim, dvec)   #Scheme
  wtt[ii]= std1[Ti]  #Last one

 hbin= bin01/2
 bins01= np.arange(-hbin,hbin)/hbin*4 #bin size 

 hist, bins = np.histogram(wtt, bins01, density=True)
 width = 0.3 * (bins[1] - bins[0]); center = (bins[:-1] + bins[1:]) / 2
 
 zz2= 1* hist
 zzth = dx.dN(center)
 plt.plot(center, zz2); plt.plot(center, zzth)
 plt.axis([-3, 3, 0, 0.5])  #gaussian

 tit1= tit0 +str(totdim)+'_dim'
 plt.title(tit1); #plt.savefig('_img/'+tit1+'_.jpg',dpi=100)

 diff1=  hist - dx.dN(center)
# plt.plot(center, diff1);   plt.axis([-3, 3, 0, 0.5])  #gaussian
# tit1= 'Differences with Gaussian Sample density, Sobol '+str(totdim)+'_dim'
# plt.title(tit1); #plt.savefig('_img/'+tit1+'_.jpg',dpi=100)  


'''
ttdim=1024  # dim= 2^k
[[500, testdensity(500, ttdim, 400,-1)],
[1000, testdensity(1000, ttdim, 400,-1)],
[2000, testdensity(2000, ttdim, 400,-1)], 
[3000, testdensity(3000, ttdim, 400,-1)],
[4000, testdensity(4000, ttdim, 400,-1)]]
'''








#----Convergence of the 2D density Gaussian distribution , Standdard/BBridge ----------
@jit
def testdensity2d(nsample, totdim, bin01, nbasset) :
 wtt= np.zeros((nbasset,nsample))
 for ii in range(0, nsample):
  for kk in range(0, nbasset):
   kdimjump= 13 + totdim*kk  #Need to Jump over the 1st asset
   dvec= getdvector( totdim, ii, kdimjump)  #13: start of dimension
   std1= pathScheme_std(1, totdim, dvec)
   wtt[kk,ii]= std1[-1]  #terminal Brownian WT

 hbin= bin01/2
 bins0x= np.arange(-hbin,hbin)/hbin*4 #bin size 
 bins0y= np.arange(-hbin,hbin)/hbin*4 #bin size 
 
 hist, binx, biny = np.histogram2d(wtt[0,:],wtt[1,:], bins=[bins0x, bins0y], normed=True)

 widthx = 0.3 * (binx[1] - binx[0]);  centerx = (binx[:-1] + binx[1:]) *0.5
 widthy = 0.3 * (biny[1] - biny[0]);  centery = (biny[:-1] + biny[1:]) *0.5
 
 diff1=  hist - dx.dN2d(centerx, centery)
 X,Y = np.meshgrid(centerx, centery)
 plt.imshow(diff1,interpolation='none')
 
 return np.sum(np.abs(diff1))    

'''
ttdim=1024 # dim= 2^k
[[500, testdensity2d(500, ttdim, 70,2)],
[1000, testdensity2d(1000, ttdim, 70,2)],
[2000, testdensity2d(2000, ttdim, 70,2)], 
[3000, testdensity2d(3000, ttdim, 70,2)],
[4000, testdensity2d(4000, ttdim, 70,2)],
[8000, testdensity2d(8000, ttdim, 70,2)], 
[11000, testdensity2d(11000, ttdim, 70,2)]]

testdensity2d(100, ttdim, 70,2)
'''



# --2D log Normal Process--------------------------------------------------- 
@jit
def lognormal_process2d(a1,z1, a2, z2, k):
 return np.maximum(0, np.exp(a1*z1) + np.exp(a1*z1) - k)

 

@jit
def testdensity2d2(nsample, totdim, bin01, nbasset, process01=lognormal_process2d, a1=0.25, a2=0.25,kk=1) :
 wtt= np.zeros((nbasset,nsample))
 for ii in range(0, nsample):
  for kk in range(0, nbasset):
   kdimjump= 13 + totdim*kk  #Need to Jump over the 1st asset
   dvec= getdvector( totdim, ii, kdimjump)  #13: start of dimension
   std1= pathScheme_std(1, totdim, dvec)
   wtt[kk,ii]= std1[-1]  #terminal Brownian WT

 hbin= bin01/2
 bins0x= np.arange(-hbin,hbin)/hbin*4 #bin size 
 bins0y= np.arange(-hbin,hbin)/hbin*4 #bin size 

 hist, binx, biny = np.histogram2d(wtt[0,:],wtt[1,:], bins=[bins0x, bins0y], normed=True)
 widthx = 0.3 * (binx[1] - binx[0]); centerx = (binx[:-1] + binx[1:]) *0.5
 widthy = 0.3 * (biny[1] - biny[0]); centery = (biny[:-1] + biny[1:]) *0.5
 
 zzth= dx.dN2d(centerx, centery) 
 diff1= process01(a1,centerx, a2, centery,kk)*( hist - zzth)
 
 X,Y = np.meshgrid(centerx, centery)
 plt.imshow(diff1,interpolation='none')
 
 return np.sum(np.abs(diff1))    


# testdensity2d2(nsample, totdim, bin01, nbasset, process01=lognormal_process2d, a1=0.25, a2=0.25,kk=1) :


'''
ttdim=1024 # dim= 2^k
[[500, testdensity2d2(500, ttdim, 70,2)],
[1000, testdensity2d2(1000, ttdim, 70,2)],
[2000, testdensity2d2(2000, ttdim, 70,2)], 
[3000, testdensity2d2(3000, ttdim, 70,2)],
[4000, testdensity2d2(4000, ttdim, 70,2)],
[8000, testdensity2d2(8000, ttdim, 70,2)], 
[11000, testdensity2d2(11000, ttdim, 70,2)]]
'''







#----------Payoff testing-------------------------------------------
@jit
def call_process(a,z,k): return np.maximum(0, np.exp(a*z)-k)


@jit
def binary_process(a,z,k):  return np.piecewise(z, [np.exp(a*z) > k], [1])
# return  1 if np.exp(a*z)> k else 0  

 
 #--------Pricing Payout-----------------------------------------
@jit
def pricing01(totdim, nsample, a, strike, process01, aa=0.25, itmax=-1, tt=10):
 #totdim= 1
 #nsample= 4000
 wtt= np.zeros(nsample)
 for ii in range(0, nsample):
  dvec= getdvector( totdim, ii,13)  #start of dimension
  std1= pathScheme_std(1, totdim, dvec)
  wtt[ii]= std1[itmax]  #terminal Brownian WT
 ss= process01(aa,wtt,tt)
 return ("price", totdim, nsample,np.mean(ss), np.var(ss))

'''
pricing01(1,  4000, 0.25, 1, call_process)
(pricing01(1,  4000, 0.25, 1, binary_process), 
pricing01(512,  4000, 0.25, 1, binary_process)  ,
pricing01(1024,  4000, 0.25, 1, binary_process))
''' 
 




#---------------Density of the payoff process-------------------------------------
@jit
def plotdensity2(nsample, totdim, bin01, tit0, process01,vol=0.25,tt=5, Ti=-1) :
 wtt= np.zeros(nsample)
 for ii in range(0, nsample):
  dvec= getdvector( totdim, ii,13)  # d vector Gaussian
  std1= pathScheme_(1, totdim, dvec)   #Scheme
  wtt[ii]= std1[Ti]  #-1 for terminal value of brownian

 hbin= bin01/2; bins01= np.arange(-hbin,hbin)/hbin*4 #bin size 

 hist, bins = np.histogram(wtt, bins01, density=True)
 width = 0.3 * (bins[1] - bins[0]); center = (bins[:-1] + bins[1:]) / 2
 
 zz2= process01(vol,center,tt)* hist
 
 xxth= np.arange(-3,5,0.1)
 zzth= call_process(vol, xxth,tt) *dx.dN( xxth) 

 plt.plot(center, zz2 )
 plt.plot(xxth, zzth)
 plt.axis([-0.5, 4, 0, .09])  #gaussian

 tit1= 'Sample density of the call process  '+str(totdim)+'_dim'
 plt.title(tit1); 
 

















 
 
 
 
 
 
 


#-----------------Random Vector analysis---------------------------------------
#------------Generate all the 2D Plot Dim_ii  x Dim_jj into a folder Uniform -----------------
def Plot2D_random_show(dir1, title1, dimxmax, dimymax, dimstep,  samplejump, nsamplegraph ) :
  istart= 0
  ijump= samplejump   #lentgh of sample dimension 
  nsample= nsamplegraph
 #dimx= 19150   #dimy= 19900
  dimx= dimxmax; dimy= dimymax

  x0= dimx * ijump + istart
  xm= dimx* ijump + nsample + istart

  y0= dimy * ijump + istart
  ym= dimy* ijump + nsample + istart

  pdframe=  pd.read_hdf(filename,'data', start=x0, stop=xm)    #from file
  xx= pdframe.values   #to numpy vector
  del pdframe  # free memory

  pdframe=  pd.read_hdf(filename,'data', start=y0, stop=ym)
  yy= pdframe.values   #to numpy vector
  del pdframe  # free memory

  plt.scatter(xx, yy, s=1 )
  
  #plt.axis([0, 1, 0, 1]) #Uniform
  plt.axis([-3, 3, -3, 3])  #gaussian

  tit1= title1+str(nsample)+' smpl D_'+str(dimx)+' X D_'+str(dimy)
  plt.title(tit1)
  plt.show()
 # plt.savefig(DIRCWD+'/'+tit1+'.jpg',dpi=100)
 # plt.clf()

  ag = sp.stats.normaltest(xx) #Agostino test pvalue >0.05
  agx= ag[1][0] 
  ag = sp.stats.normaltest(yy) #Agostino test pvalue >0.05
  agy= ag[1][0]     
  
  [pearsonr(xx,yy)[0][0], np.mean(xx), np.var(xx),agx, np.mean(yy), np.var(yy),agy,
   comoment(xx,yy,nsample,3,1),
   comoment(xx,yy,nsample,1,3),
   comoment(xx,yy,nsample,2,2)]





#------------Generate all the 2D Plot Dim_ii  x Dim_jj into a folder Uniform -----------------
def Plot2D_random_save(dir1, title1, dimxmax, dimymax, dimstep,  samplejump, nsamplegraph, ) :
 istart= 0
 ijump= samplejump   #lentgh of sample dimension 
 nsample= nsamplegraph
 #dimx= 19150   #dimy= 19900
 dimx= dimxmax; dimy= dimymax

 dimy0= dimy;  dimx0= dimx
 for jj in range(0,20,1) :
  for ii in range(0,20,1) :
   dimy= dimy0 - jj*1000   #    dimy= dimy0 - jj*1000 
   dimx= dimx0 - ii*1000   #    dimx= dimx0 - ii*1000 
   x0= dimx * ijump + istart
   xm= dimx* ijump + nsample + istart

   y0= dimy * ijump + istart
   ym= dimy* ijump + nsample + istart

   pdframe=  pd.read_hdf(filename,'data', start=x0, stop=xm)    #from file
   xx= pdframe.values   #to numpy vector
   pdframe=  pd.read_hdf(filename,'data', start=y0, stop=ym)
   yy= pdframe.values   #to numpy vector
   del pdframe  # free memory

   plt.scatter(xx, yy, s=1 )
  
   #plt.axis([0, 1, 0, 1]) #Uniform
   plt.axis([-3, 3, -3, 3])  #gaussian

   tit1= title1+str(nsample)+' smpl D_'+str(dimx)+' X D_'+str(dimy)
   plt.title(tit1)
   plt.savefig(dir1+'/'+tit1+'.jpg',dpi=100)
   plt.clf()






#----------Get the Outlier list----------------------------------------
fileoutlier=   'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'
pdf=  pd.read_hdf(fileoutlier,'data')    #from file
vv= pdf.values   #to numpy vector





#----------Normal Random Dimension Outliers Search----------------------------
@jit
def getoutlier_fromrandom(filename, jmax1, imax1, isamplejum, nsample, fileoutlier=   'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'):
 ijump= isamplejump
 istartx=0
 istarty=0
 for jj in range(0,jmax1,1) :
  dimy= dimy0 - jj*stepy
  y0= dimy * ijump + istarty
  ym= dimy* ijump + nsample + istarty
  pdframe=  pd.read_hdf(filename,'data', start=y0, stop=ym)
  yy= pdframe.values   #to numpy vector  
  yy2= ne.evaluate("yy*yy") 
  
  for ii in range(jj+1,imax1,1) :  
   dimx= dimx0 - ii*stepx
  
   if ((dimx> dimxmin) or (dimy > dimymin) and (dimx != dimy)) :
  
    x0= dimx * ijump + istartx
    xm= dimx* ijump + nsample + istartx

    pdframe=  pd.read_hdf(filename,'data', start=x0, stop=xm)    #from file
    xx= pdframe.values   #to numpy vector

    xx2= ne.evaluate("xx*xx") 
    c22= ne.evaluate("sum( xx2*xx * yy)") / (nsample) #co moment xx^3 * yy
    c33= ne.evaluate("sum( xx * yy2*yy)") / (nsample)
    c44= ne.evaluate("sum( xx2 * yy2)") / (nsample)  

    if (abs(c22) > trigger1)  :
     crroutliers[kk,0]= dimx
     crroutliers[kk,1]= dimy
     crroutliers[kk,2]= 1  
     crroutliers[kk,3]= c22 *100000
     kk= kk+1
    elif (abs(c33) > trigger1) :
     crroutliers[kk,0]= dimx
     crroutliers[kk,1]= dimy
     crroutliers[kk,2]= 2 
     crroutliers[kk,3]= c33 *100000
     kk= kk+1
    elif (abs(c44-1) > trigger1) :
       crroutliers[kk,0]= dimx
       crroutliers[kk,1]= dimy
       crroutliers[kk,2]= 3  
       crroutliers[kk,3]= c44 *100000      
       kk= kk+1

# fileoutlier=   'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'   
 store = pd.HDFStore(fileoutlier)  
 vv= crroutliers
 vv= vv[~np.all(vv == 0, axis=1)]
 vv= vv[np.lexsort(np.transpose(vv)[::-1])]
 vv= unique_rows(vv)
 pdf =pd.DataFrame(vv)  
 store.append('data', pdf)
 store.close()  
#-----------------------------------------------------------------------------






#----------------------------------------------------------------------------
#-----Outlier List Generation  VERY FAST-------------------------------------


#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_vect_calc(filename, i0=0, imax=1000, expr, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe    #to numpy vector
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx  
# filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'   
 store = pd.HDFStore(fileout) 
 store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):


#---LOAD In Advance Calculation----------------------------------------
yy1=  getrandom_tonumpy(filename, 16384, 4096)
yy2=  getrandom_tonumpy('E:\_data\_QUASI_SOBOL_gaussian_xx2.h5'  , 16384, 4096)
yy3=  getrandom_tonumpy('E:\_data\_QUASI_SOBOL_gaussian_xx3.h5' , 16384, 4096)


#--------------Using Pre Calculated data do the lopp---------------------------
@jit
def getoutlier_fromrandom_fast(filename, jmax1, imax1, isamplejum, nsample, trigger1=0.28, fileoutlier=   'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'):
 ijump= isamplejump
 istartx= 0;  istarty= 0
# nsample= 3000
 stepx= 1; stepy= 1

 dimxmin= 1800; dimymin= 1800
 dimx0= imax1;  dimy0= jmax1
# imax1= 3000; jmax1 = 3000

 dimy= dimy0
 dimx= dimx0

#----- Loop over the dimension PAIRS for Outliers calculation ------------
# trigger1=  0.28
 crrmax = 500000;   kk=0
 crroutliers = np.zeros((crrmax,4),dtype='int')  #empty list

 for jj in range(0,jmax1,1) :  #Decrasing: dimy0 to dimmin
  dimy= dimy0 - jj*stepy
  y0= dimy * ijump + istarty
  ym= dimy* ijump + nsample + istarty

  yyu1= yy[y0:ym];   yyu2= yy2[y0:ym];    yyu3= yy3[y0:ym] 
  
  for ii in range(jj+1,imax1,1) :  
   dimx= dimx0 - ii*stepx
  
   if ((dimx> dimxmin) or (dimy > dimymin) and (dimx != dimy)) :
  
    x0= dimx * ijump + istartx
    xm= dimx* ijump + nsample + istartx

    xxu1= yy[x0:xm];    xxu2= yy2[x0:xm];    xxu3= yy3[x0:xm]
   
    c22= ne.evaluate("sum( xxu3 * yyu1)") / (nsample) # X3.Y moments
    c33= ne.evaluate("sum( xxu1 * yyu3)") / (nsample)
    c44= ne.evaluate("sum( xxu2 * yyu2)") / (nsample)  

    if (abs(c22) > trigger1)  :
     crroutliers[kk,0]= dimx;     crroutliers[kk,1]= dimy
     crroutliers[kk,2]= 1  ;     crroutliers[kk,3]= c22 *100000
     kk+= 1
    elif (abs(c33) > trigger1) :
     crroutliers[kk,0]= dimx;     crroutliers[kk,1]= dimy
     crroutliers[kk,2]= 2 ;     crroutliers[kk,3]= c33 *100000
     kk+= 1
    elif (abs(c44-1) > trigger1) :
       crroutliers[kk,0]= dimx;       crroutliers[kk,1]= dimy
       crroutliers[kk,2]= 3  ;       crroutliers[kk,3]= c44 *100000      
       kk+= 1


 fileoutlier=   'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'   
 store = pd.HDFStore(fileoutlier)  
 vv= crroutliers
 vv= vv[~np.all(vv == 0, axis=1)]
 vv= vv[np.lexsort(np.transpose(vv)[::-1])]
 #vv= unique_rows(vv)
 pdf =pd.DataFrame(vv)  
 store.append('data', pdf)
 store.close()  
#-----------------------------------------------------------------------------




#--------------------Ordering the data-------------------------------
def outlier_clean(vv2):
 for ii in range(0,202767,1):
 if (vv2[ii,0] > vv2[ii,1]  ):
     aux= vv2[ii,0] 
     vv2[ii,0] = vv2[ii,1]
     vv2[ii,1]= aux
     
#--------------------------------------------------------------------
def overwrite_data(fileoutlier, vv2):
 store = pd.HDFStore(fileoutlier)  
 store.remove('data')
 pdf =pd.DataFrame(vv2)  
 store.append('data', pdf) 
 store.close()  










#----- 2n Loop Cleaning over the dimension PAIRS for Outliers calculation -------
#Load Outlier from Disk    
# Load pre calculated yyu1= yy1[y0:ym];   yyu2= yy2[y0:ym];   yyu3= yy3[y0:ym] 

def doublecheck_outlier(fileoutlier, ijump, nsample=4000, trigger1=0.1, )
 fileoutlier=   'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'   
 pdf=  pd.read_hdf(fileoutlier,'data')    #from file
 vv5= pdf.values   #to numpy vector
 del pdf

# ijump= 4096   #length of sample dimension 
 istartx= 0; istarty= 0

 nsample= 4000
 trigger1=  0.1
 crrmax = 250000
 kk=0
 crroutliers = np.zeros((crrmax,4),dtype='int')  #empty list

 kkmax1= np.shape(vv5)[0]
 for kk in range(0,kkmax1,1) :  #Decrasing: dimy0 to dimmin

   dimx= vv5[kk,0];   dimy= vv5[kk,1]

   y0= dimy * ijump + istarty
   ym= dimy* ijump + nsample + istarty
   yyu1= yy1[y0:ym];   yyu2= yy2[y0:ym];   yyu3= yy3[y0:ym] 
    
   x0= dimx * ijump + istartx
   xm= dimx* ijump + nsample + istartx
   xxu1= yy1[x0:xm];   xxu2= yy2[x0:xm];   xxu3= yy3[x0:xm]
   
   c22= ne.evaluate("sum( xxu3 * yyu1)") / (nsample) # X3.Y moments
   c33= ne.evaluate("sum( xxu1 * yyu3)") / (nsample)
   c44= ne.evaluate("sum( xxu2 * yyu2)") / (nsample)  

   if (abs(c22) > trigger1)  :
     crroutliers[kk,0]= dimx;     crroutliers[kk,1]= dimy
     crroutliers[kk,2]= 1  ;     crroutliers[kk,3]= c22 *100000
     kk+= 1
   elif (abs(c33) > trigger1) :
     crroutliers[kk,0]= dimx;     crroutliers[kk,1]= dimy
     crroutliers[kk,2]= 2 ;     crroutliers[kk,3]= c33 *100000
     kk+= 1
   elif (abs(c44-1) > trigger1) :
       crroutliers[kk,0]= dimx;       crroutliers[kk,1]= dimy
       crroutliers[kk,2]= 3  ;       crroutliers[kk,3]= c44 *100000      
       kk+= 1
 
 store = pd.HDFStore(fileoutlier)  
 vv= crroutliers
 vv= vv[~np.all(vv == 0, axis=1)]
 vv= vv[np.lexsort(np.transpose(vv)[::-1])]
 #vv= unique_rows(vv)
 pdf =pd.DataFrame(vv)  
 store.append('data2', pdf)
 store.close()  
#-----------------------------------------------------------------------------





#-------Plot the defect dimensions------------------------------------
def plot_outlier(fileoutlier, kk)
 df=  pd.read_hdf(fileoutlier,'data')    #from file
 vv= df.values   #to numpy vector
 del df

 xx= vv[kk,0]
 yy= vv[kk,1]


 plt.scatter(xx, yy, s=1 )
 plt.axis([00, 1000, 00, 1000])  

 tit1= title1+str(nsample)+'sampl D_'+str(dimx)+' X D_'+str(dimy)
 plt.title(tit1)
 plt.savefig('_img/'+tit1+'_outlier.jpg',dpi=100)
 plt.clf()




#------------Histogramme des Values--------------------------------
'''
fileoutlier=   'E:\_data\_QUASI_SOBOL_gaussian_outlier.h5'   
df=  pd.read_hdf(fileoutlier,'data')    #from file

nn= len(vv)
vv1= np.zeros((10001,2))
for ii in range(0,   nn ):
    
  ix1= vv[ii,0]  
  ix2= vv[ii,1]  
  
  vv1[ix1,0]= ix1
  vv1[ix2,0]= ix2
  vv1[ix1,1]+= 1
  vv1[ix2,1]+= 1
  

plt.scatter(xx, yy, s=1 )
plt.axis([0, 3500, 0, 100])  


np.mean(yy[:3000])    :65.163333333333327
np.var(yy[:3000])   : 66.519322222222229

np.mean(yy[3001:10000])    :29.998285469352766    35.16504786398056


xx= vv1[:,0]
yy= vv1[:,1]

yy[3001:10000] = yy[3001:10000] +  np.random.normal(35.16,6, 6999)


plt.scatter(xx, yy, s=1 )
plt.axis([0, 10000, 0, 100]) 

tit1= "Histogram of outliers per dim 1 to 10000"
plt.title(tit1)
plt.savefig('_img/'+'histogram of outliers per dim 1 to 10000.jpg',dpi=100)
plt.clf()

0.006  0.6% are defective...
'''













#----------------------------------------------------------------------------
#-------Hybrid   Mersenne / Sobol--------------------------------------------
# For Defect Dimension: use Mersenne to replace one dimension
# For Defect Dimension: use Mersenne permutation, only on halve

#Permutation
def permute(yy, kmax):
 kk= np.random.uniform(1,kmax)
 nn= int(len(yy)/kk)
 yy3= np.copy(yy)
 for ii in range(0, nn):
#    ax= np.random.uniform(0,kk*nn)
    ax= kk*ii-kk*nn
    aux= yy3[ax]
    yy3[ax] = yy3[kk*ii] 
    yy3[kk*ii] = aux
    

#Permutation,  dependant on the number of simulation, more shuffle
def permute2(xx,yy, kmax):
 kk2= np.random.uniform(1,10-kk)
 nn= int(len(xx)/kk2)
 xx3= np.copy(xx)
 for ii in range(0, nn):
  #  ax= np.random.uniform(0,kk*nn)
    ax= kk2*ii-kk2*nn
    aux= xx3[ax]
    xx3[ax] = xx3[kk2*ii] 
    xx3[kk2*ii] = aux
    

 plt.scatter(xx3, yy, s=1 )
 plt.axis([-3, 3, -3, 3])  #gaussian
 tit1= title1+str(nsample)+' smpl D_'+str(dimx)+' X D_'+str(dimy)
 plt.title(tit1)

 [comoment(xx,yy,nsample,3,1),
 comoment(xx,yy,nsample,1,3),
 comoment(xx,yy,nsample,2,2)]

 [comoment(xx3,yy,nsample,3,1),
 comoment(xx3,yy,nsample,1,3),
 comoment(xx3,yy,nsample,2,2)]















#Gaussian is mainly on the main impact
# Used Modified gaussian to re-distribute the random numbers
# across different support, so final differences is lower
# Same Mean and variance: cut the tail,multi dimensionnal
# Exponentially_modified_Gaussian_distribution
#Limited number of sample, achieved uniformity across gaussian values.
# Best = NSample * GaussianDensity(xi)   xi= Support / Nsample
#  100:  65 between -0.5 0.5, .....
# Instead of having [0,1]   reduce to [0.1, 1]
# reduce to [A/Nsample, 1] --->Better sampling in the higher probability,
#with low level of sample, difficult to achieve tail estimation...

# Mapping [0,1] ---> Gaussian Density/Cumulative




























