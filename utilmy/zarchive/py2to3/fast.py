# -*- coding: utf-8 -*-
#utilities for Fast Computation
import  numpy as np, math as mm,  numba, numexpr as ne
from numba import jit, njit, autojit, int32, int64, float32, float64,  double
from math import exp, sqrt, cos, sin, log1p



'''
#------  High Speed Computation utilities
https://bitbucket.org/arita237/fast_utilities/overview

https://github.com/barbagroup/numba_tutorial_scipy2016/blob/master/notebooks/09.Tips.and.FAQ.ipynb

####################### Tweaking Numba #################################################################################
Set envvar NUMBA_DISABLE_JIT=1 to disable numba compilation (for debugging)
Install the "Hide Traceback" extension if you're prototyping in a notebook.

Unification errors
Thanks to Graham Markhall for the idea for these examples: http://gmarkall.github.io/tutorials/pycon-uk-2015
When Numba needs to declare the type of the output(s). If it can't do that in a consistent way, it gets upset

Globals are treated as compile-time constants by Numba: Don't USE Globals

you can instruct Numba to write the result of function compilation
into a file-based cache. This is done by passing cache=True:
@jit(cache=True)


####################### Signature Type #################################################################################
Type name(s)	Shorthand	Comments
boolean	b1	represented as a byte
uint8, byte	u1	8-bit unsigned byte
uint16	u2	16-bit unsigned integer
uint32	u4	32-bit unsigned integer
uint64	u8	64-bit unsigned integer
int8, char	i1	8-bit signed byte
int16	i2	16-bit signed integer
int32	i4	32-bit signed integer
int64	i8	64-bit signed integer
intc	–	C int-sized integer
uintc	–	C int-sized unsigned integer
intp	–	pointer-sized integer
uintp	–	pointer-sized unsigned integer
float32	f4	single-precision floating-point number
float64, double	f8	double-precision floating-point number
complex64	c8	single-precision complex number
complex128	c16	double-precision complex number
'V'	raw data (void)

x= np.array(np.random.randn(10000), dtype='f2')
f2  -0.10767
f4  -0.84343278
f8  -0.057429548276262309

@njit([  'f4(f4[:])', float32(float32[:]), float64(float64[:]) ],  cache=True, nogil=True, target='cpu')

Tuple signature
@nb.jit(nb.typeof((1.0,1.0))(nb.double),nopython=True)
def f(a): return a,a

@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:,:]))(nb.float64[:], nb.float64[:,:]),nopython=True)
def f(a, b) :
    return a, b

Vectorize, Parallel, cuda : Works very Badly

'''


'''
https://aboutsimon.com/blog/2016/08/04/datetime-vs-Arrow-vs-Pendulum-vs-Delorean-vs-udatetime.html

Decode a date-time string
Encode (serialize) a date-time string
Instantiate object with current time in UTC
Instantiate object with current time in local timezone
Instantiate object from timestamp in UTC
Instantiate object from timestamp in local timezone



'''


'''

######################################################################################
# http://toolz.readthedocs.io/en/latest/api.html#dicttoolz
# Fast Dictionnary iterator in cython ------------------------------------------------
import cytoolz

#Merge Dictionnary
cytoolz.dicttoolz.merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})


cytoolz.dicttoolz.assoc({'x': 1}, 'y', 3)


d= ALLDB['cokeon']['table_uri']

%timeit cytoolz.dicttoolz.assoc(d, 'new', 5) # : 754 ns per loop


def adddict(d, x, vx): 
  d[x]=vx
  return d

%timeit adddict(d, 'new', 5)    # 10000000 loops, best of 3: 194 ns per loop


############# Test Cache
from fastcache import clru_cache  #Not fast

%timeit weekday('1980-04-20-15:15') # 1000000 loops, best of 3: 274 ns per loop

%timeit weekday2('1980-04-20-15:15')  # 1000000 loops, best of 3: 480 ns per loop

def weekday2(s):
  s2= s[0:10]
  return _weekday(s2)

@clru_cache(maxsize=1000, typed=False)
def _weekday(s2):
  return arrow.get(s2, 'YYYY-MM-DD').weekday()
  

'''

def day(s):    return int(s[8:10])
def month(s):  return int(s[5:7])
def year(s):   return int(s[0:4])
def hour(s):   return int(s[11:13])
# def weekday(s):  return arrow.get(s, 'YYYY-MM-DD HH:mm:ss').weekday()


###Super Fast because of caching
cache_weekday= {}
def weekday(s):
  s2= s[0:10]
  try :
    return  cache_weekday[s2]
  except KeyError:
    wd= arrow.get(s2, 'YYYY-MM-DD').weekday()
    cache_weekday[s2]= wd
  return wd

def season(d):
  m=  int(d[5:7])
  if m > 3 and m  < 10: return 1
  else: return 0 


def daytime(d):
  h= int(d[11:13])
  if   h < 11 :   return 0
  elif h < 14 : return 1    #lunch
  elif h < 18 : return 2    # afternoon
  elif h < 21 : return 3    # dinner
  else :        return 4   #night



def fastStrptime(val, format) :
    l = len(val)
    if format == '%Y%m%d-%H:%M:%S.%f' and (l == 21 or l == 24):
        us = int(val[18:24])
        # If only milliseconds are given we need to convert to microseconds.
        if l == 21:
            us *= 1000
        return datetime.datetime(
            int(val[0:4]), # %Y
            int(val[4:6]), # %m
            int(val[6:8]), # %d
            int(val[9:11]), # %H
            int(val[12:14]), # %M
            int(val[15:17]), # %s
            us, # %f
        )

    # Default to the native strptime for other formats.
    return datetime.datetime.strptime(val, format)
    
    
    



@njit([ float32(float32[:]), float64(float64[:]) ],  cache=True, nogil=True, target='cpu')
def drawdown_calc_fast(price):
    n= len(price);
    maxprice = np.zeros(n); ddowndur2 = np.zeros(n);
    dd = np.zeros(n); ddowndur = np.zeros(n); ddstart = np.zeros(n);
    ddmin_date = np.zeros(n);
    maxstartdate=0
    minlevel= 0
    for t in range(1, n):
        if maxprice[t-1] < price[t]:
          maxprice[t]= price[t]  #update
          maxstartdate= t        #update the start date
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

    return minlevel




###################################  Statistical #######################################################################
@njit([ float32(float32[:]), float64(float64[:]) ],  cache=True, nogil=True, target='cpu')
def std(x):
    """Std Deviation 1D array"""
    n = x.shape[0]
    m = x.sum() / n
    return sqrt((( (x - m)**2 ).sum() / (n - 1)))

# bsk= np.array(bsk, dtype=np.float64)
# %timeit std(bsk)


@njit([ float32(float32[:]), float64(float64[:]) ],  cache=True, nogil=True, target='cpu')
def mean(x):
    """Mean  """
    return x.sum() / x.shape[0]


@njit([float64(float64,float64)], cache=True, nogil=True, target='cpu')
def log_exp_sum2 (a, b):
    if a >= b: return a + log1p(exp (-(a-b)))
    else:      return b + log1p(exp (-(b-a)))
    ## return max (a, b) + log1p (exp (-abs (a - b)))


@njit('Tuple( (int32, int32, int32) )(int32[:], int32[:])', cache=True, nogil=True, target='cpu')
def _compute_overlaps(u, v):
    a,b,c = 0,0,0
    m = u.shape[0]
    for idx in xrange(m):
        a += u[idx] & v[idx]
        b += u[idx] & ~v[idx]
        c += ~u[idx] & v[idx]
    return a, b, c
 
 
@njit(float32(int32[:], int32[:]), cache=True, nogil=True, target='cpu')
def distance_jaccard2(u, v):
    a, b, c = _compute_overlaps(u, v)
    return 1.0 - (a / float(a + b + c))


@njit([float32(int32[:], int32[:]), float32(float32[:], float32[:]), float64(float64[:], float64[:])  ],  cache=True, nogil=True, target='cpu')
def distance_jaccard(u, v):
    a = 0;    b = 0;    c = 0
    m = u.shape[0]
    for idx in xrange(m):
        a += u[idx] & v[idx]
        b += u[idx] & ~v[idx]
        c += ~u[idx] & v[idx] 
    return 1.0 - (a / float(a + b + c))


@njit(float32[:,:](int32[:,:]),  cache=True, nogil=True, target='cpu' )
def distance_jaccard_X(X):
 n= X.shape[0]    
 dist= np.zeros((n,n), dtype=np.float32)
 for i in xrange(0,n):
    for j in xrange(i+1,n):
      dist[i,j]=  distance_jaccard(X[i,:], X[j,:])
      dist[j,i]=  dist[i,j]
 return dist
 
     
# x= np.ones(1000); y= np.zeros(1000)
# %timeit jaccard(x, y)


@njit('float64(float64[:], float64[:])', cache=True, nogil=True, target='cpu')
def cosine(u, v):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue
             
        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]
 
    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)
     
    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    return ratio
    
# %timeit cosine(x, y)

# bsk= np.array(bsk, dtype=np.float64)
# %timeit mean(bsk)


@njit( cache=True, nogil=True, target='cpu')
def rmse(y, yhat):
    """ Calculate and return Root Mean Squared Error (RMSE)
    Returns: float: Root Mean Squared Error
    """
    return ((y - yhat) ** 2).mean() ** 0.5


@njit(cache=True, nogil=True, target='cpu')
def cross(vec1, vec2):
    """ Calculate the dot product of two 3d vectors. """
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result = np.zeros(3)
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


@njit(cache=True, nogil=True, target='cpu')
def norm(vec):
    """ Calculate the norm of a 3d vector. """
    return sqrt(vec[0]*vec[0] + vec[1]**2 + vec[2]**2)







'''
################################# Paralell Code / Cuda acceleration  ###################################################
from numba import vectorize, guvectorize
from numba import jit, njit,  autojit, int32, float32, float64, int64, double

import math
@vectorize(['float32(float32, float32, float32)', 'float64(float64, float64, float64)'],
            target='cuda')
def cu_discriminant(a, b, c):
    return math.sqrt(b ** 2 - 4 * a * c)

N = 15000000
dtype = np.float32

# prepare the input
A = np.array(np.random.sample(N), dtype=dtype)
B = np.array(np.random.sample(N) + 10, dtype=dtype)
C = np.array(np.random.sample(N), dtype=dtype)

%timeit cu_discriminant(A, B, C)


# 10E7 elements
10 loops, best of 3: 90.6 ms per loop   #   target='cuda'
10 loops, best of 3: 47.2 ms per loop   #  Paralell Super Slow
10 loops, best of 3: 57.4 ms per loop   #  CPU

#50e7 elements
10 loops, best of 3: 172 ms per loop #paralell
1 loop, best of 3: 287 ms per loop #cpu





#################################Precompiling Numba modules¶   ##########################################################
Ahead-of-Time compilation

While Numba's main use is in JIT compiling, they do provide tools for doing AOT compilation.
This pre-compiled module does not rely on Numba, only on NumPy. (If you are working with collaborators who don't have NumPy installed, I can't help you).
We need to import numpy, of course, and also numba.pycc.CC

import numpy
from math import sqrt
from numba.pycc import CC

#Name the module fast_num
cc = CC('fast_num')
cc.verbose = True


@cc.export('pressure_poisson',  'f8[:,:](f8[:,:], f8[:,:], f8)')
def pressure_poisson(p, b, l2_target):
    I, J = b.shape
    iter_diff = l2_target + 1
    n = 0
    while iter_diff > l2_target and n <= 500:
        pn = p.copy()
        for i in range(1, I - 1):
            for j in range(1, J - 1):
                p[i, j] = 5
        n += 1
    return p


Note: Each function in the module can be compiled with one type signature only. You can specify multiple types, each with its own function name, e.g.
@cc.export('pressure_poisson_single',  'f4[:,:](f4[:,:], f4[:,:], f4)')
@cc.export('pressure_poisson_double',  'f8[:,:](f8[:,:], f8[:,:], f8)')
@cc.export('pressure_poisson_quad',    'f16[:,:](f16[:,:], f16[:,:], f16)')
def pressure_poisson(p, b, l2_target=1e-4):

cc.compile()

%ls

Installing AoT compiled modules with setup.py
%load ../ppe_compile_module/main.py

%load ../ppe_compile_module/setup.py


'''


###################################  JIT Class ######################################################################
'''
@njit
def test():
    return np.zeros(10, dtype=np.int32)
test()


from numba import *
from numba.vectorize import vectorize
from math import exp, log1p


@vectorize([f8(f8,f8)])
def log_exp_sum2 (a, b):
    if a >= b: return a + log1p (exp (-(a-b)))
    else:      return b + log1p (exp (-(b-a)))
    ## return max (a, b) + log1p (exp (-abs (a - b)))


@jit(int32(int32, int32))
numba.float32[:,:,:]
numba.from_dtype(dtype)¶
Create a Numba type corresponding to the given Numpy dtype:

struct_dtype = np.dtype([('row', np.float64), ('col', np.float64)])
numba.from_dtype(struct_dtype)
Record([('row', '<f8'), ('col', '<f8')])
class numba.types.NPDatetime(unit)
Create a Numba type for Numpy datetimes of the given unit. unit should be a string amongst
the codes recognized by Numpy (e.g. Y, M, D, etc.).

class numba.types.NPTimedelta(unit)

@nb.jit(nb.typeof((1.0,1.0))(nb.double),nopython=True)
def f(a):
  return a,a

spec = [
    ('A', nb.float64[:,::1]),
    ('B', nb.float64[::1])
]


@nb.jitclass(spec)
class Data(object):
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def funcd(self):
        A = self.A
        B = self.B
        m,n = A.shape
        X = np.zeros(m)

        for i in range(m):
            for j in range(n):
                X[i] += A[i,j] * B[j]

        return X


@nb.jit(nopython=True)
def func1(A, B):
    m,n = A.shape
    X = np.zeros(m)

    for i in range(m):
        for j in range(n):
            X[i] += A[i,j] * B[j]

    return X

'''

###################################################################################################



'''
from timeit import timeit
L = 1000
N = 100
u = np.tile (np.log (np.ones (L)/L), (N, 1))
#v = log_exp_sum (u)
from timeit import timeit
print timeit ('log_exp_sum(u)', 'from __main__ import u, log_exp_sum', number=50)    
    


@jit(float64[:](float64[:]), nopython=True, nogil=True)
def sqrt(x):
    """Sqrt np.sqrt is 2x Faster   """
    return np.sqrt(x)



from numba import *
from numba.vectorize import vectorize
from math import exp, log1p
 
 
@vectorize([f8(f8,f8)])
def log_exp_sum2 (a, b):
    if a >= b: return a + log1p (exp (-(a-b)))
    else:      return b + log1p (exp (-(b-a)))
    ## return max (a, b) + log1p (exp (-abs (a - b)))
 
#@autojit
@jit(f8[:,:] (f8[:,:]))
def log_exp_sum (u):
    s = u.shape[-1]
    if s == 1:   return u[...,0]
    elif s == 2: return log_exp_sum2 (u[...,0], u[...,1])
    else:
        return log_exp_sum2 (
            log_exp_sum (u[...,:s/2]),
            log_exp_sum (u[...,s/2:]))



from math import sin, sqrt

import numpy as np
import numba as nb

@nb.jit(nopython=True)
def ex1(a, b):
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have same size")
    #if not (np.float64 == a.dtype or np.float64 == b.dtype):
    #    raise ValueError("a and b must have float64 type")
    size = a.shape[0]
    c = np.zeros(size, dtype=np.float64)
    for i in range(size):
        c[i] = sin(2.2 * a[i] - 3.3 * b[i]) / sqrt(4.4 * a[i] + 5.5 * b[i])
    return c



'''

