#To test the library with pre computed data benchmarks
import numpy as np
import scipy as sp
import scipy.stats as ss
import scipy.optimize as sopt
import matplotlib.pyplot as plt


 #my library for pricing check
import derivatives as dx  

'''
Coding errors:

Table dimensions  returns vs price.
Tranpose of table --> use asymmetric dimentions 4x2
initialisation of variables :  use params, reinitialise properly.
  In Debug Mode, Save the parameter at each end of function process
  
Slow Code:  conversion done at every step, variable initialization.

convergence data.


Namespace: Global variable is only inside the local file

'''





#   Vanilla Call Put Black Scholes Formulae
#Be careful of write into floating  100.0   not  100 otherwise issues
s01=90; strike1=100.0; mu1=0.10; r1 = 0.10;  sigma1=1; d1= 0.04;  tt= 2.0  #in years

[dx.bscall(s01,strike1,0,tt, r1,d1,sigma1) ,
dx.bsdelta(s01,strike1,0,tt, r1,d1,sigma1,1),
dx.bsgamma(s01,strike1,0,tt, r1,d1,sigma1,1) ,
dx.bsvega(s01,strike1,0,tt, r1,d1,sigma1,1),
dx.bsrho(s01,strike1,0,tt, r1,d1,sigma1,1),
dx.bsdvd(s01,strike1,0,tt, r1,d1,sigma1,1),
dx.bsvolga(s01,strike1,0,tt, r1,d1,sigma1,1),
dx.bsvanna(s01,strike1,0,tt, r1,d1,sigma1,1)
]
#call benchmark
[43.5581 ,  0.7048, 0.0022, -0.0233, 0.3624, 0.3981]




#   Vanilla Put formulae   
[dx.bs(s01,strike1,0,tt, r1,d1,sigma1,-1) ,
dx.bsdelta(s01,strike1,0,tt, r1,d1,sigma1,-1),
dx.bsgamma(s01,strike1,0,tt, r1,d1,sigma1,-1) ,
dx.bsvega(s01,strike1,0,tt, r1,d1,sigma1,-1),
dx.bsrho(s01,strike1,0,tt, r1,d1,sigma1,-1),
dx.bsdvd(s01,strike1,0,tt, r1,d1,sigma1,-1),
dx.bsvolga(s01,strike1,0,tt, r1,d1,sigma1,-1),
dx.bsvanna(s01,strike1,0,tt, r1,d1,sigma1,-1)
]

#put benchmark
[]


'''  Vanilla Call Put 
43.5581   call            42.3374  put 
Delta 	      0.7048                     -0.2182  
Delta 100's 	   70.4849           -21.8166  
Gamma 	       0.0022             0.0022 
Theta 	     -0.0233            -0.0100     
Theta (7 days) 	         -0.1639          -0.0705 
Vega 	       0.3624             0.3624 

Rho 	   0.3981             -1.2411   

Psi 	-1.2705        0.3932  
Strike Sensitivity 	   -0.1988            0.6197    
Intrinsic Value 	  0.0000           10.0000       
Time Value 	  43.5581              32.3374      
Zero Volatility 	  1.2207             0.0000    
Market Option Price 	     20.04              8.51      
Implied Volatility (%) 	        42.35   
'''



#---------------------Gbm Process and Jump Process -----------------------------------------------------------
#-------------------------------------------------------------------------------------
s01=100.0; mu1=0.00; r1 = 0.00;  sigma1=0.25; d1= 0.0; tt= 1.0  #in years

dt1 = 12/12.0  #monthly step
nstep = np.int(tt/ dt1)+1; timegrid1= np.zeros(nstep)
for i in range(0,nstep):  timegrid1[i]= dt1*i
  
  
gbm_process(s01,mu1,sigma1,timegrid1)

lamda11= 0.2
jump_mu1=0.2
jump_sigma1=0.3
gbmjump_process(s01,mu1, sigma1, lamda11,jump_mu1, jump_sigma1, timegrid1)

#  Not Yet finished
#-------------------------------------------------------------------------------------  



'''
Forward Check
Call Put Parity
Vol zero check
Zero rate + Zero Vol
High Vol

'''


#-------------------------------------------------------------------------------------
#----------------------gbm monte carlo, Single asset ------------------------------------
tt= 1.0;    timegrid1 = dx.timegrid(12/12.0,tt)  
s01=100; strike1=100.0; mu1=0.0; r1 = 0.0;  sigma1=0.05; d1= 0.00; 
param11= [s01, mu1, sigma1]
discount1 = np.exp(-r1*tt)



#----- gbm with path dependance  
def payoff1(pricepath):
  st= pricepath[len(pricepath)-1]
  return np.maximum(st - strike1,0)

allprocess1 = dx.generateallprocess(dx.gbm_process, param11, timegrid1 , 10)

[dx.getpv(discount1, payoff1, allprocess1),
dx.bscall(s01,strike1,0,tt, r1,d1,sigma1) 
]


  
  
  
#----gbm with 1 single 0 -->T maturity, european payoff  -----------------------
def payoffeuro1(st):
  return np.maximum(st - strike1,0)

allprocesseuro = dx.generateallprocess_gbmeuro(dx.gbm_process, param11, timegrid1 , 15000)

[dx.getpv(discount1, payoffeuro1, allprocesseuro),
dx.bscall(s01,strike1,0,tt, r1,d1,sigma1) 
]

'''
 allprocess1 = dx.generateallprocess(dx.gbm_process, param11, timegrid1 , 10000000)
  [(1.9959660029595834, 0.00095141768895585577), 1.9945036390476076]
  [(1.994314116868233, 0.00095036926979524014), 1.9945036390476076]
  [(1.9932980642850344, 0.00095008633829582749), 1.9945036390476076]

allprocess2 = dx.generateallprocess_euro(dx.gbm_process, param11, timegrid1 , 15000000)
  [(1.9945671850924638, 0.00077617609369255778), 1.9945036390476076]


'''
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------







#-------------------------------------------------------------------------------------
#--------------------gbm Multi Asset ----------
tt= 1.0;    timegrid1 = dx.timegrid(12/12.0,tt)  
s01=100.0; strike1=100.0; mu1=0.01; r1 = 0.01;  sigma1=0.1; d1= 0.00;    #in years
discount1 = np.exp(-r1*tt)


def payoff2(pricepath):
  size1 = np.shape(pricepath);  tt= size1[1]; nbasset=size1[0]
  baskett= np.sum( ww * pricepath[:,tt-1] )
  return np.maximum(baskett - strike1,0)



#--------------------positive correlation----------
s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.13333333333333333,0.13333333333333333]
corrmatrix1 = [[1.0,0.125],[0.125,1.0]]
np.random.seed(1234)
pp3= generateallmultiprocessfast(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)

ww= 	[0.5,0.5];    dx.getpv(discount1, payoff2, pp3)


'''
np.random.seed(1234)
mm3= generateallmultiprocess3(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
ww= 	[0.5,0.5]
dx.getpv(discount1, payoff2, mm3)
Out[96]: (4.5205101976647004, 0.032731984909718598)
Same than benchmark

corrbm
4.714351637324930566e-01
-1.122705169176822171e+00

wiener
6.285802183099907514e-02
-1.496940225569096117e-01

same
6.285802183099907514e-02
-1.496940225569096117e-01

corrbm
1.432706968426097349e+00
-1.311113154435426809e-01


wiener
1.910275957901463040e-01
-1.748150872580568962e-02


same wiener
1.910275957901463040e-01
-1.748150872580568962e-02


return: new
1.066059492133584863e+00
8.619285404663706851e-01

1.211838595684223474e+00
9.837628691434218142e-01


mm3= generateallmultiprocess3(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 2)
getpv2(discount1, payoff2, mm3)
Out[89]: (4.8413799433426012, 3.4233725882380965)


11.54seconde  multi function
vs
3.6s  accelerate version

'''



#---------------------- positive correlation:  2 steps for check----------
s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.13333333333333333,0.13333333333333333]
corrmatrix1 = [[1.0,0.125],[0.125,1.0]]
np.random.seed(1234)
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 2)
ww= 	[0.5,0.5]
dx.getpv(discount1, payoff2, mm2)

'''
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 2)
ww= 	[0.5,0.5]
dx.getpv(discount1, payoff2, mm2)

Out[92]: (4.8413799433426012, 3.4233725882380965)

'''







#---------------- positive correlation, same drift----------
s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.13333333333333333,0.13333333333333333]
corrmatrix1 = [[1.0,0.125],[0.125,1.0]]
np.random.seed(1234)
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
ww= 	[0.5,0.5];     getpv2(discount1, payoff2, mm2)

'''
Volb = 0.1 -->  voli=    0.1/(0.5*np.sqrt(2+2*0.125)) =0.133333
ss=0.13333333333333333
np.sqrt(0.5*0.5*ss*ss*2 + 2*0.125*1/2*1/2*ss*ss)


mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
ww= 	[0.5,0.5]
getpv2(discount1, payoff2, mm2)
Out[32]: (4.5205101976647004, 0.032731984909718598)
'''




#--------------Negative correl test, same drift---------------------
s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.15,0.15]
corrmatrix1 = [[1.0,-0.11111111111111111],[-0.11111111111111111,1.0]]
np.random.seed(1234)
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)

ww= 	[0.5,0.5];    dx.getpv(discount1, payoff2, mm2)


'''
Negative correlation
0.1/(0.5*np.sqrt(2+2*-0.11111111111)) =0.15
ss=0.15
np.sqrt(0.5*0.5*ss*ss*2 + 2*-0.1111111111111*1/2*1/2*ss*ss)

mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
ww= 	[0.5,0.5]
getpv2(discount1, payoff2, mm2)
Out[36]: (4.5263177965998649, 0.032808009972488769)
'''



#-----Different Drift  0.01 and -0.01, same vol, different correl-----------------------
s02 = [100.0,100.0]
mu2= [0.01,-0.01]
sigma2 = [0.15,0.15]
corrmatrix1 = [[1.0,-0.11111111111111111],[-0.11111111111111111,1.0]]
np.random.seed(1234)
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)

ww= 	[0.5,0.5];   getpv2(discount1, payoff2, mm2)

np.random.seed(1234)
param12= [s01,0.0,0.1000]
allprocess1 = dx.generateallprocess(dx.gbm_process, param12, timegrid1 , 4000)
[dx.getpv(discount1, payoff1, allprocess1),
dx.bscall(s01,strike1,0,tt, 0,d1,sigma1) 
]


'''
Different Drift: opposite, total drift is around zeros

allprocess1 = dx.generateallprocess(dx.gbm_process, param12, timegrid1 , 4000)
[dx.getpv(discount1, payoff1, allprocess1),
dx.bscall(s01,strike1,0,tt, 0,d1,sigma1) 
Out[38]: [(3.979689078179415, 0.095912908156454188), 3.987761167674492]


Basket with average drift
mu2= [0.01,-0.01]
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 4000)
Out[39]: (4.0835447635662696, 0.096944585940324415)

mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
Out[40]: (3.9921108519150379, 0.030847816177326366)


Negative correlation
0.1/(0.5*np.sqrt(2+2*-0.11111111111)) =0.15
ss=0.15
np.sqrt(0.5*0.5*ss*ss*2 + 2*-0.1111111111111*1/2*1/2*ss*ss)

mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
ww= 	[0.5,0.5]
getpv2(discount1, payoff2, mm2)
Out[36]: (4.5263177965998649, 0.032808009972488769)


Different Drift, average is same.
'''



#----------- 2 perfectly correlated brownian----------
s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.1,0.1]
corrmatrix1 = [[1.0,0.999999],[0.999999,1.0]]

np.random.seed(1234)
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)

ww= 	[1,0.0];   dx.getpv(discount1, payoff2, mm2)

ww= 	[0.0,1.0];   dx.getpv(discount1, payoff2, mm2)

'''
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)
ww= 	[1,0.0]
getpv2(discount1, payoff2, mm2)
Out[7]: (4.5089623055063308, 0.032697110895897802)
'''


#----------gbm: 2 uncorrelated brownian----------
s02 = [100.0,100.0]
mu2= [0.01,0.01]
sigma2 = [0.1,0.1]
corrmatrix1 = [[1.0,0.0],[0.0,1.0]]

np.random.seed(1234)
mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 40000)

ww= 	[1,0.0]; dx.getpv(discount1, payoff2, mm2)
ww= 	[0.0,1.0] ;  dx.getpv(discount1, payoff2, mm2) 
 
 
 
'''
 getpv2(discount1, payoff2, mm2)   ww= 	[1,0.0]
 (4.8270140091144995, 0.42038206166281994)

mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 2)
Path 1
1.000000000000000000e+02	1.053526930028281470e+02
1.000000000000000000e+02	8.921709029928940993e+01

Path 2
1.000000000000000000e+02	1.159826815329504370e+02
1.000000000000000000e+02	9.740767403244210243e+01


mm2= generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 20000)
ww= 	[1,0.0]
 (4.4977706009288143, 0.045896450349368331)


ww= 	[0.0,1.0]
Out[24]: (4.496145729271416, 0.045922037298403891)


ww= 	[1,0.0]    40000
getpv2(discount1, payoff2, mm2)
Out[26]: (4.5089623055063308, 0.032697110895897802)


ww= 	[0.0,1.0]
getpv2(discount1, payoff2, mm2) 
Out[27]: (4.4908686314080324, 0.032581320712903102)

OK, matched with values
---'''
 
 
 
#--------------Local Test  ---------------------------
s02 = [100.0]
mu2= [0.01]
sigma2 = [0.1]
corrmatrix1 = [[1.0]]
strike1=100.0

ww= 	[1,0.0]
np.random.seed(1234)
mm2= dx.generateallmultiprocess(multigbm_process,s02,mu2, sigma2, corrmatrix1, timegrid1, 200)
getpv2(discount1, payoff2, mm2)


np.random.seed(1234)
allprocess1 = dx.generateallprocess(dx.gbm_process, param11, timegrid1 , 200)

[dx.getpv(discount1, payoff1, allprocess1),
dx.bscall(s01,strike1,0,tt, r1,d1,sigma1) 
]


'''------------------------------------------------------------
Benchmark asset

nn [ 0.04714352]

ret [ 1.05352693]

price  1.000000000000000000e+02
1.053526930028281470e+02

allprocess
1.000000000000000000e+02	1.053526930028281470e+02
1.000000000000000000e+02	8.921709029928940993e+01

allprocess1 = dx.generateallprocess(dx.gbm_process, param11, timegrid1 , 2)
[(2.6497164087801712, 1.8736324408697249), 4.485236409022086]


allprocess1 = dx.generateallprocess(dx.gbm_process, param11, timegrid1 , 200)
[(4.3455585414504228, 0.42033856617997645), 4.485236409022086]


multi asset  : Issues with Number of assets !!!!!!

np.array(multibrownian_logret(mu, sigma, corrmatrix, timegrid))
array([[ 0.1432707]])


Wiener
[[ 0.04714352]]
Price1 : 1.000000000000000000e+02	1.053526930028281470e+02

Wiener 2
-1.190975694706464527e-01
1.000000000000000000e+02	8.921709029928940993e+01

[[[ 100.         105.352693 ]]
 [[ 100.          89.2170903]]]


payout call :
5.352693002828146973e+00
0.000000000000000000e+00


getpv2(discount1, payoff2, mm2)
Out[13]: (2.6497164087801712, 1.8736324408697249)

getpv2(discount1, payoff2, mm2)
Out[17]: (4.3455585414504228, 0.42033856617997645)
  
  
-----------------------------------------------------------'''
#--------------------------------------------------------------------------------




#----------------------gbm European single asset Call Test----------------------
tt= 1.0;    timegrid1 = dx.timegrid(12/12.0,tt)  
s01=100.0; strike1=100.0; mu1=0.01; r1 = 0.01;  sigma1=0.1; d1= 0.00;   
param11= [s01, mu1, sigma1]
discount1 = np.exp(-r1*tt)


strike1=100
def payoffeuro1(st):
  return np.maximum(st - strike1,0)

np.random.seed(1234) 
allprocesseuro = dx.generateallprocess_gbmeuro(dx.gbm_process, param11, timegrid1 , 100000)
[dx.getpv(discount1, payoffeuro1, allprocesseuro), dx.bscall(s01,strike1,0,tt, r1,d1,sigma1) ]


np.random.seed(1234)
allprocess1 = dx.generateallprocess(dx.gbm_process, param11, timegrid1 , 2)
[dx.getpv(discount1, payoff1, allprocess1), dx.bscall(s01,strike1,0,tt, r1,d1,sigma1)]



'''
allprocesseuro = dx.generateallprocess_gbmeuro(dx.gbm_process, param11, timegrid1 , 40000)
[(4.4729029915015595, 0.032499221753412841), 0.0]


100000
[(4.4996879043671463, 0.020631737506477298), 4.485236409022086]

path dependant option
Out[38]: [(4.4793202991773553, 0.020582830967673955), 4.485236409022086]

np.random.seed(1234)
Out[81]: [(4.5083853687013846, 0.029159821255353019), 4.485236409022086]
'''








#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
'''
#-------------Testing of Heston model--------------------

Numerical Verification Of The Implemenation

To test the implementation of analytic solution for European call options, results from analytical solution were compared against those from Monte Carlo solutions. Following inputs were considered.
Spot: 1, r = 0.03%, time to maturity = 7 years, Initial variance = 0.1, Long term variance = 0.15, kappa = 3, sigma = 0.2, correlation coeffecient = -0.2, dividend yield = 0%, nSimulations for Monte Carlo = 150000 and nSteps = 150.
Results
Strike	Monte Carlo price	Analytic solution	Abs Error
0.25	0.808070	0.808179	0.000109
0.50	0.656430	0.654749	0.001681
0.75	0.541261	0.537977	0.003284
1.00	0.452703	0.454303	0.001600
1.25	0.383353	0.383914	0.000561
1.50	0.328098	0.332173	0.004075
1.75	0.283396	0.286217	0.002821

Weights For Observations

Equal weighted, vega weighted or inverse price spread (1/[Pbid-Pask]).
Calibration Results

The uploaded model has been pre loaded with sample data. The calibration run on the sample data produced the following results.
Results
Parameters	DE
Initial variance (v0)	0.190461
Long term variance (theta)	0.067924
Mean reversion speed (kappa)	6.237368
Vol of vol (sigma)	0.920504
Correlation (rho)	-0.755989
RMSE	3.596383
Time taken(min:sec)	68:56
Weighted sum of square errors	65.769942


'''






















































