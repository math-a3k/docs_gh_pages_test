# -*- coding: utf-8 -*-
#------------Symbolic useful Function -------------------------------
from sympy import symbols
from sympy.matrices import Matrix as mx
from sympy.core.symbol import Symbol as sy
from sympy.polys.polytools import poly
from sympy.solvers.solvers import solve	
from sympy import * 

import math as mth
import sympy.mpmath as mp


#from sympy.abc import a, x, y
init_printing(use_unicode=False, wrap_line=False, no_global=True)



#-----------Integral, equations-----------------
a,b,c,d, x, y, z, t,u,v,p = symbols('a b c d x y z t u v p', real=True)
i,j,k, m, n = symbols('i j k m n', integer=True) 
a2,b2 = symbols('a2 b2',real=True, positive=True)
a1,b1 = symbols('a2 b2',real=True, positive=True)

f, g, h = symbols('f g h', cls=Function)



#-----------Matrix calculation-----------------





# blackscholes
s,kk,k0, r,d, vol,t, tt, T  = symbols('s kk k0 r d vol t tt T', real=True, positive=True)
s1,k1,r1,d1, vol1, tt1, T1  = symbols('s1 k1 r1 d1 vol1 tt1 T1', real=True,  positive=True)
s2,k2,r2,d2, vol2, tt2, T2  = symbols('s2 k2 r2 d2 vol2 tt2 T2', real=True,  positive=True)
s3,k3,r3,d3, vol3, tt3, T3  = symbols('s3 k3 r3 d3 vol3 tt3 T3', real=True,  positive=True)

#Normalizee strike
xk, xk1, xk2, xk3 = symbols('xk xk1 xk2 xk3', Real=True)

#Constants
A2,B2,C2, D2, M2 = symbols('A2 B2 C2 D2 M2 ', real=True)
A1,B1,C1, D1, M1 = symbols('A1 B1 C1 D1 M1 ', real=True)


#Weights
w0,w1, w2, w3, w4 = symbols('w0 w1 w2 w3 w4', real=True)


#lagrangian
l0,l1, l2, l3, l4= symbols('l0 l1 l2 l3 l4', real=True)



#-----------Stochastic -----------------
uu, mu,aa = symbols('uu mu aa', real=True)



#--------------------------------------------------------------------------

#-----Print Setting--------------
def spp():	print'\n\n_________________________'

def print2(a0,a1='',a2='',a3='',a4='',a5='',a6='',a7='',a8=''  ):
 print a0; spp();
 if a1 !='' :    print a1; spp()	; 
 if a2 !='':  	 print a2; spp()	;
 if a3 !='':  	 print a3; spp()	;
 if a4 !='':  	 print a4; spp()	;
 if a5 !='':  	 print a5; spp()	; 
 if a6 !='':  	 print a6; spp()	;
 if a7 !='':  	 print a7; spp()	;
 if a8 !='':  	 print a8; spp()	;  
  	  	







#--------------Polynomial  -------------------------------------
def factorpoly(pp) :
 rr=roots(pp,x)
 ee=1
 for xx0,kk0 in rr.iteritems():
   ee = ee*(x-xx0)**kk0
 return ee


# factorpoly(a*x**2+b*x+c)





#--------------Expectation / Variance with Brownian -------------------------------------

# Calculate E[ff(Wt)]  and  VAR[ff(Wt)]   where wt is brownian
def EEvarbrownian(ff1d) :
  a2,b2 = symbols('a2 b2',real=True, positive=True)
  
  aa= factor(simplify(ff1d(sqrt(t)*x) *  exp((-x*x)/(2*b2)) ))
  ee= Integral( aa, (x, -oo, oo))
  ee= ee.doit()
  ee= simplify(ee)  
  ee= 1/sqrt(2*pi) * ee.subs({b2:1})  
  ee= simplify(ee)
  
  aa= factor(simplify(ff1d(sqrt(t)*x) **2  *  exp((-x*x)/(2*a2)) ))
  vv= Integral( aa , (x, -oo, oo))
  vv= vv.doit()
  vv= simplify(vv)  
  vv= 1/sqrt(2*pi) * vv.subs({a2:1})  
  vv= simplify(vv)
  
  return (ee, vv - ee*ee)


# Calculate E[ff(Wt)]  and  VAR[ff(Wt)]   where wt is brownian
# Need to simplify expression before integral, separate variable, reduce variable
def EEvarbrownian2d(ff) :
  a2,b2 = symbols('a2 b2',real=True, positive=True)
  
  aa= factor(simplify(ff(sqrt(t)*x,sqrt(t)*y ) *  exp((-x*x-y*y)/(2*b2)) ))
  ee= Integral(Integral( aa, (x, -oo, oo)), (y, -oo, oo))
  ee= ee.doit()
  ee= simplify(ee)  
  ee= 1/(2*pi*sqrt(1-p*p)) * ee.subs({b2:(1-p*p)})  
  ee= simplify(ee)
  
  aa= factor(simplify(ff(sqrt(t)*x,sqrt(t)*y ) **2  *  exp((-x*x-y*y)/(2*a2)) ))
  vv= Integral(Integral( aa , (x, -oo, oo)), (y, -oo, oo))
  vv= vv.doit()
  vv= simplify(vv)  
  vv= 1/(2*pi*sqrt(1-p*p)) * vv.subs({a2:(1-p*p)})  
  vv= simplify(vv)
  
  return (ee, vv - ee*ee)




'''
def ff1(Wt) :
  return exp((mu-a*a/2)*t + a*Wt)
  
EEvarbrownian(ff1)  

def ff2(Wt, Zt) :
  return exp((mu-a*a-b*b)*t + a*Wt + b*Zt)

EEvarbrownian2(ff2)  

'''







#-----lagrangian of function  minization algorithm--------------
def lagrangian2d(ll) :
 dllx= simplify(Derivative(ll(x,y,k0,k1),x).doit())
 dlly= simplify(Derivative(ll(x,y,k0,k1),y).doit())
 dllk0 = simplify(Derivative(ll(x,y,k0,k1),k0).doit())
 dllk1 = simplify(Derivative(ll(x,y,k0,k1),k1).doit())

 print2('Partial Derivatives Lagrangian D[l]=0', dllx, dlly, dllk0, dllk1)

 res2= solvers.solve((dllk0, dllk1), (x,y), rationnal=None)
 print2('partial Solutions:(x,y) ',res2)

 res= solvers.solve((dllx, dlly, dllk0, dllk1), (x,y,k0,k1), rationnal=None)
 print2('Full Solutions:(x,y,k0,k1) ',res)

 print2( 'ff Value')
 for kk in range(0, res.__len__()) :
  aux= simplify(ff(*((res[kk])[:2])))
  print2( 'ff_'+str(kk)+':   '+   str(  aux))







#-----------Matrix calculation----------------------------  
#Decompose Correlation matrix into cholesky and eigenVector, SVD
def decomposecorrel(m1) :
 d3= factor(m1.berkowitz_det())
 pol1= factor(simplify(m1.charpoly(t)),t)
 print2('Matrix,Determinant,  Polynom for Eigen', m1,d3, pol1 )
 
 eival1= m1.eigenvals()
 eivec1= m1.eigenvects()
 print2('EigenVal, EigenVec', eival1, eivec1) 
 

 u1 = simplify(factor(m1.cholesky()))
 print2('Cholesky', u1 )

 l1, d1 = m1.LDLdecomposition()
 print2('LDL Decomp, M=LDL', l1,d1 ),
 
 U, S, V = simplify(factor(svd_r(m1)))
 print2('SVD Decomp, M=USV:  U, S, V', U, S, V )



 
'''
SVD decomposition

Given A, two orthogonal (A real) or unitary (A complex) matrices U and V are calculated such that
A=USV,UtU=1,VVt=1
where S is a suitable shaped matrix whose off-diagonal elements are zero. Here  denotes the hermitian transpose (i.e. transposition and complex conjugation). The diagonal elements of S are the singular values of A, i.e. 
the square roots of the eigenvalues of AtA or AAt.
U, S, V = mp.svd_r(A)
print mp.chop(A - U * mp.diag(S) * V)

'''




#-----------New Functions----------------------------  	  	
def nn(x):
 return 1/sqrt(2*pi)	*Integral(exp(-t*t/2), (t, -oo, x))
		
def nn2(x,y,p):
 if abs(p) >= 1 : return 'Error Correl > 1' ;
 return 1/(2*pi*sqrt(1-p*p)) * Integral(Integral(exp((-t*t-v*v)/(2*(1-p*p))), (t, -oo, x)), (v, -oo, y) )


def dnn2(x,y,p):
 return 1/(2*pi*sqrt(1-p*p)) * exp((-t*t-v*v)/(2*(1-p*p)))

		
def dnn(x): 
 return  1/sqrt(2*pi)	*exp(-x*x)

#taylor for integral function
def taylor2(ff, x0, n ) :
 sum1= simplify(ff(0).doit())                                    
 for k in range(1,n) :
  dffk= Derivative(ff(x),x,k)
  dffk1= simplify( dffk.doit())
  dffx0=  simplify(Subs(dffk1, (x), (x0)).doit())
  sum1= sum1 +  dffx0 * 1/(mth.factorial(k)) * x**k
 return sum1


# differential for integral function
def diffn(ff,x0,kk) :     
 dffk= Derivative(ff(x),x,kk)
 dffk1= simplify( dffk.doit())
 dffx0=  simplify(Subs(dffk1, (x), (x0)).doit())
 return dffx0
 



#------------------Black Scholes Formulae ---'''-------------------------------
def dN(x):  
 return  1/sqrt(2*pi)	*exp(-x*x) 

def N(x):    
 return  1/sqrt(2*pi)	*Integral(exp(-t*t/2), (t, -oo, x))   

def d1f(St, K, t, T, r,d, vol):
#    return  (log(St / K) + (r -d + 1/2 * vol*vol  )  * (T - t)) / (vol * sqrt(T - t))
 return  log(St*exp((r -d)*(T - t)) / K) / (vol * sqrt(T - t)) + ( 1/2 * vol * sqrt(T - t)  )  


def d2f(St, K, t, T, r,d, vol):
#    return  (log(St / K) + (r - d - 1/2 * vol*vol )  * (T - t)) / (vol * sqrt(T - t))
    return  log(St*exp((r -d)*(T - t)) / K) / (vol * sqrt(T - t)) - ( 1/2 * vol * sqrt(T - t)  )  

 
def d1xf(St, K, t, T, r,d, vol):
    return  xk + ( 1/2 * vol * sqrt(T - t)  )  

def d2xf(St, K, t, T, r,d, vol):
    return  xk - ( 1/2 * vol * sqrt(T - t)  )  
    
    
def bsbinarycall(s0,K,t,T, r,d,vol):
  d2= d2f(s0, K, t, T, r,d, vol)
  price1= exp(-r*T)*N(-d2)
  return price1  
  

def bscall(s0,K,t,T, r,d,vol):
  d1 =   d1f(s0, K, t, T, r,d, vol)
  d2= d1 - vol*sqrt(T-t)
  price1= s0*exp(-d*(T-t))*N(d1) - K*exp(-r*(T-t))*N(d2)
  return price1


def bsput(s0,K,t,T, r,d,vol):
  d1 =  d1f(s0, K, t, T, r,d, vol)
  d2= d1 - vol*sqrt(T-t)
  price1= -s0*exp(-d*(T-t))*N(-d1) + K*exp(-r*(T-t))*N(-d2)
  return price1


def bs(s0,K,t,T, r,d,vol,cp): #cp=1 call, cp=-1 put
  d1 =  d1f(s0, K, t, T, r,d, vol)
  d2= d1 - vol*sqrt(T-t)
  price1= cp* (s0*exp(-d*(T-t))*N(cp*d1) - K*exp(-r*(T-t))*N(cp*d2))
  return price1


def bsdelta(St, K, t, T, r,d, vol, cp1):
    d1 = d1f(St, K, t, T, r,d, vol)
    if cp1 == 1 :     #be careful of equality for boolean
      aux= exp(-d*(T-t)) * N(d1)
    else:      
      aux= exp(-d*(T-t)) * (N(d1)-1)
    return aux

def bsstrikedelta(s0,K,t,T, r,d,vol, cp1):
  d2= d2f(s0, K, t, T, r,d, vol)
  if cp1 == 1 :     
    aux= - exp(-r*T)*N(d2) #discounted risk neutral probability
  else:      
    aux= exp(-r*T)*N(-d2) #discounted risk neutral probability
  return aux  


def bsstrikegamma(s0,K,t,T, r,d,vol):
    d1 = d1f(s0, K, t, T, r,d, vol)
    return exp(-d*(T-t))*dN(d1) / (s0 * vol * sqrt(T - t))  #gamma is same between call put
  
def bsgamma(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    return exp(-d*(T-t))*dN(d1) / (St * vol * sqrt(T - t))  #gamma is same between call put
 
def bstheta(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    ff= St*exp(-d*(T-t))
    return   -( ff *dN(d1) * vol / (2 * sqrt(T - t))    + cp*d * ff * N(cp*d1))
      

def bsrho(St, K, t, T, r,d, vol,cp):
    d2 = d1 - vol * sqrt(T - t)
    return  cp*K * (T - t) * exp(-r * (T - t)) * N(cp*d2) *1/100


def bsvega(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    return   St*exp(-d*(T-t)) * sqrt(T - t) * dN(d1) *1/100


def bsdvd(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    return  -cp*St * (T - t) * exp(-d * (T - t)) * N(cp*d1) *1/100


def bsvanna(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    d2 = d1 - vol * sqrt(T - t)
    return   -exp(-d*(T-t)) * dN(d1) *d2/vol


def bsvolga(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    d2 = d1 - vol * sqrt(T - t)
    return     St*exp(-d*(T-t)) * sqrt(T - t) * dN(d1) *d1*d2/vol


def bsgammaspot(St, K, t, T, r,d, vol,cp):
    d1 = d1f(St, K, t, T, r,d, vol)
    return exp(-d*(T-t))*dN(d1) / (St*St * vol*vol*(T - t))*(d1+vol*sqrt(T-t))  #gamma is same between call put
 






















print2('\n\n---------------Symbolic Math Loaded -----------------------------------------')













