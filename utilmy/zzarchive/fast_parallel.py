# -*- coding: utf-8 -*-
#utilities for  Paralell and ( Fast Computation)
import  numpy as np, math as mm,  numba, numexpr as ne
from numba import jit, njit,  autojit, int32, float32, float64, int64, double
from math import exp, sqrt, cos, sin, log1p

import ipyparallel as ipp

from ipyparallel import Client




from ipyparallel import error, AsyncHubResult, DirectView as dv, Reference
##################################################################################################
######################### Usage of IPyrallel #####################################################
'''
import IPython,os,sys
DIRCWD=  'D:/_devs/Python01/project27/'
os.chdir('D:/_devs/Python01/project27/'); sys.path.append('D:/_devs/Python01/project27/aapackage/')
import util, numpy as np, ipyparallel as ipp, fast_parallel as pa


# Issue with Installing with CONDA,    pip install IParallel   is better


#Launch in Command line
# ipcluster start -n <put desired number of engines to run here>
# ipcluster nbextension enable


import ipyparallel as ipp,  time
t0 = time.time()
def sleep_here(t):
    time.sleep(t)
    return id,t

# create client & view
rclient = ipp.Client()
dview = rc[:]
# scatter 'id', so id=0,1,2 on engines 0,1,2
dview.scatter('idworker', rclient.ids, flatten=True)
print("Engine IDs: ", dv['idworker'])


print("Engine IDs (2n way): ", rc [:]['id'])
lview = rclient.load_balanced_view()    # Create Load Balancer for the ec2/worker


print("running with one call per task")
task_map = lview.map(sleep_here, [.01*t for t in range(10)])
for i, taski in enumerate(task_map):
    print('Task', i, 'Worker', taski[0],'Result',  taski)
#---------------------------------------------------------------------------------


# Remote Client/worker List
list_client = ipp.Client()
print('Worker_id', list_client.ids)
# Create Load Balancer for the ec2/worker
lview = list_client.load_balanced_view()
list_client[:].apply_sync(lambda : "Hello, World")

'''

####################################  Parellel run of task  ##################################################################################################
'''  http://ipyparallel.readthedocs.io/en/latest/task.html?highlight=tasks
Using IPyparalell :
  Can include Ipython run, work, Don't use Main memory
  Scheduler is busy waiting task is finished, need to Launch in Sub-Process
  No task failure handle

'''


def task_summary(tasks):
  print('\n--------------- Summary ------------------------')
  for k, t in enumerate(tasks) :
    if t.ready() :
       print('Task '+str(k)+' : ', t.get(), 'Wall_Time:', t.wall_time, )


def task_progress(tasks):
  ''' Monitor progress '''
  ss=0.0;  ss0= -0.1
  while ss < 0.9999999 :
    ss= np.mean([task.ready() for task in tasks])
    if ss!= ss0 :
      print("Tasks completion: {0}%".format(100 * ss))
      ss0= ss
  print("Tasks Finished")


def task_find_best(tasks, n_top=5):
    """Compute the best score of tasks_folder"""
    scores = [t.get() for t in tasks if t.ready()]
    return sorted(scores, reverse=True)[:n_top]


#Define Task Process Job  ----------------------------------------------------------------
def task_parallel_job_01(name, param, datadict) :
   ''' Sample task run in Parallel '''
   import os, sys
   p= param
   dirpackage,dircwd, diroutput, dirscript= p['packagedir'],  p['cwdir'], p['outputdir'], p['dirscript']
   os.chdir(dircwd); sys.path.append(dirpackage)
   import util

   #Load Dictionnary in memory
   # for k, v in datadict[0].items():  setattr(sys.modules['__main__'], k, v)

   #Computation using same variable name than datadict ------------------------------
   id1= param['id']
   # time.sleep(np.random.randint(1,2))

   # util.a_run_ipython("run -i " + dircwd + dirpackage )  # 'aapackage/allmodule.py'

   util.a_run_ipython("run -i " + dircwd + dirscript ) # '/sk_cluster/script/run_elvis_send_email_task_noemail.ipy'


   #---------------------------------------------------------------------------------
   res= name+'__'+ str(id1) + '__' + dircwd
   return res



'''
Enable Cluster
conda install -c conda-forge notebook ipyparallel
If you are using conda but not the conda-forge packages (e.g. Anaconda), or you otherwise don't see the IPython Clusters tab, you can still run the command above to enable the extension:

ipcluster nbextension enable
Or disable it with:

ipcluster nbextension disable


'''


'''
import IPython,os,sys
DIRCWD=  'D:/_devs/Python01/project27/'
os.chdir('D:/_devs/Python01/project27/'); sys.path.append('D:/_devs/Python01/project27/aapackage/')
import util, numpy as np
import ipyparallel as ipp

#Generate List of (param1, filedata) ------------------------------------------------
nbtask= 3
paramlist=[]
for k in xrange(0, nbtask) :
  param={}
  param['outputdir']= 'D:/_devs/Python01/project27/'
  param['cwdir']= 'D:/_devs/Python01/project27/'
  param['packagedir']= 'D:/_devs/Python01/project27/aapackage'

  param['id']= k+1000
  param['sym']= 555
  param['date']= (25,25)

  param['dfactor']= 25
  param['optimtype']= 'sharpe'

  paramlist.append(param)


#Datadict: Common Data, READ ONLY
DIRCWD= 'D:/_devs/Python01/project27/'
fpath= DIRCWD+'/aaserialize/session/sessiondata_daily_marketdata_kevin_55810.spydata'
datadict= util.session_load(fpath, towhere='main')


#Windows Command Line in Sub Window of Python Script
ipcluster start -n 3

#   ipcluster stop

# Remote Client/worker List
list_client = ipp.Client()
list_client.ids

# Create Load Balancer for the ec2/worker
lview = list_client.load_balanced_view()
list_client[:].apply_sync(lambda : "Hello, World")


#Asynchrone execution Tasks   -------------------------------------------------------
# Pb of Concurrency to access the data ---> Need to transmit the Dictionnary in Params
# data translitted to the ec2 are READ ONLY
# http://ipyparallel.readthedocs.io/en/latest/multiengine.html#calling-python-functions

tasks= []
for k in xrange(0, nbtask) :
    tasks.append(lview.apply_async(task_parallel_job_01, 'Optim_' + str(k), paramlist[k], datadict))
print('Nb of tasks_folder :', len(tasks))

task_progress(tasks)

task_summary(tasks)




#----- Small Test -----------------------------------------------------------------------------------
task_parallel_job_01('test', param, datadict)
for k, t in enumerate(tasks) :
    if t.ready() :
       print('Task '+str(k)+' : ', t.serial_time , t.wall_time, t.get(),  )



'''


'''  GOOD TUTORIAL
http://people.duke.edu/~ccc14/sta-663-2016/19C_IPyParallel.html

#Synchronous Job
.map_sync(lambda x, y, z: x + y + z, range(10), range(10), range(10))


#Asynchronous
.map_async(lambda x, y, z: x + y + z, range(10), range(10), range(10))


#Launch task on each of the worker with %%px

%%px --target [1,3] --noblock
%matplotlib inline
import seaborn as sns
x = np.random.normal(np.random.randint(-10, 10), 1, 100)
sns.kdeplot(x);




'''



'''
# Working with compiled code on Workers

Using numba.jit is straightforward.

In [53]:
with dv.sync_imports():
    import numba
importing numba on engine(s)


In [54]:
@numba.jit
def f_numba(x):
    return np.sum(x)


In [55]:
dv.map(f_numba, np.random.random((6, 4)))

'''


##################################################################################################
##################################################################################################


'''
########################## IPYparalell and with Numba            #######################################################
https://github.com/barbagroup/numba_tutorial_scipy2016/blob/master/notebooks/10.optional.Numba.and.ipyparallel.ipynb
# Add the dv.parallel decorator to enable it in parallel. (We're also enabling blocking here, for simplicity)
Function is compute in Paralell

@dv.parallel(block=True)
@jit(nopython=True)
def mandel_par_numba(x, y):
    max_iters = 20
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return i
    return 255

import numpy
In [ ]:
x = numpy.arange(-2, 1, 0.005)
y = numpy.arange(-1, 1, 0.005)
X, Y = numpy.meshgrid(x, y)

%%time
im_par_numba = numpy.reshape(mandel_par_numba.map(X.ravel(), Y.ravel()), (len(y), len(x)))


from matplotlib import pyplot, cm
%matplotlib inline
fig, axes = pyplot.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(im, cmap=cm.viridis)
axes[1].imshow(im_par, cmap=cm.viridis)
axes[2].imshow(im_par_numba, cmap=cm.viridis)


https://github.com/mmckerns/tuthpc


'''


'''
#------  High Speed Computation utilities   ----------------------------------------------------------------------------
# https://bitbucket.org/arita237/fast_utilities/overview

https://github.com/barbagroup/numba_tutorial_scipy2016/blob/master/notebooks/09.Tips.and.FAQ.ipynb

####################### Signature Type #################################################################################
Explicit @jit signatures can use a number of types. Here are some common ones:
  void is the return type of functions returning nothing (which actually return None when called from Python)
  intp and uintp are pointer-sized integers (signed and unsigned, respectively)
  intc and uintc are equivalent to C int and unsigned int integer types
  int8, uint8, int16, uint16, int32, uint32, int64, uint64 are fixed-width integers of the corresponding bit width (signed and unsigned)
  float32 and float64 are single- and double-precision floating-point numbers, respectively

  array types can be specified by indexing any numeric type, e.g. float32[:] for a one-dimensional
  single-precision array or int8[:,:] for a two-dimensional array of 8-bit integers.

The first character specifies the kind of data and the remaining characters specify the number of bytes per item,
except for Unicode, where it is interpreted as the number of characters.
'b'	boolean
'i'	(signed) integer
'u'	unsigned integer
'f'	floating-point
'c'	complex-floating point
'O'	(Python) objects
'S', 'a'	(byte-)string
'U'	Unicode
'V'	raw data (void)

f2: 16bits, f4: 32 bits,  f8: 64bits
dt = np.dtype('i4')   # 32-bit signed integer
dt = np.dtype('f8')   # 64-bit floating-point number
np.dtype('c16')  # 128-bit complex floating-point number
np.dtype('a25')  # 25-character string
'''

'''
###################################  Statistical #######################################################################
@dv.parallel(block=True)
def np_std_par(x):
    return np_std(x)

# bsk= np.array(bsk, dtype=np.float64)
# %timeit std(bsk)


@njit(float64(float64[:]),  cache=True, nogil=True, target='cpu')
def np_mean(x):
    """Mean  """
    return x.sum() / x.shape[0]


@njit([float64(float64,float64)], cache=True, nogil=True, target='cpu')
def np_log_exp_sum2 (a, b):
    if a >= b: return a + log1p(exp (-(a-b)))
    else:      return b + log1p(exp (-(b-a)))
    ## return max (a, b) + log1p (exp (-abs (a - b)))


@njit('Tuple( (int32, int32, int32) )(int32[:], int32[:])', cache=True, nogil=True, target='cpu')
def _compute_overlaps(u, v):
    a = 0
    b = 0
    c = 0
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


@njit(float32(int32[:], int32[:]),  cache=True, nogil=True, target='cpu')
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
 
     
x= np.ones(1000); y= np.zeros(1000)
 #%timeit jaccard(x, y)


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


@njit(nogil=True)
def cross(vec1, vec2):
    """ Calculate the dot product of two 3d vectors. """
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result = np.zeros(3)
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


@jit(nopython=True)
def norm(vec):
    """ Calculate the norm of a 3d vector. """
    return math.sqrt(vec[0]*vec[0] + vec[1]**2 + vec[2]**2)


@jit(nopython=True)
def normalize(vec):
    """ Calculate the normalized vector (norm: one). """
    return vec / norm(vec)
'''




'''
########################## IPYparalell and with Numba            #######################################################
https://github.com/barbagroup/numba_tutorial_scipy2016/blob/master/notebooks/10.optional.Numba.and.ipyparallel.ipynb
# Add the dv.parallel decorator to enable it in parallel. (We're also enabling blocking here, for simplicity)
Function is compute in Paralell

@dv.parallel(block=True)
@jit(nopython=True)
def mandel_par_numba(x, y):
    max_iters = 20
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if z.real * z.real + z.imag * z.imag >= 4:
            return i
    return 255


import numpy
In [ ]:
x = numpy.arange(-2, 1, 0.005)
y = numpy.arange(-1, 1, 0.005)
X, Y = numpy.meshgrid(x, y)

%%time
im_par_numba = numpy.reshape(mandel_par_numba.map(X.ravel(), Y.ravel()), (len(y), len(x)))


from matplotlib import pyplot, cm
%matplotlib inline
fig, axes = pyplot.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(im, cmap=cm.viridis)
axes[1].imshow(im_par, cmap=cm.viridis)
axes[2].imshow(im_par_numba, cmap=cm.viridis)



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
#benchmark Test  ----------------------------------------------------------------------------------------------------
import numpy as np
X = np.random.random((100000, 2))



#Simple Python    --------------------------------------------------------------------------------------------------
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

%timeit pairwise_python(X)
1 loops, best of 3: 13.4 s per loop




#Numpy Test -----------------------------------------------------------------
def pairwise_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))

%timeit pairwise_numpy(X)
10 loops, best of 3: 111 ms per loop




#--------------Cython  ----------------------------------------------------------------------------------------------
%load_ext cythonmagic

%%cython
import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def pairwise_cython(double[:, ::1] X):
    cdef int M = X.shape[0]
    cdef int N = X.shape[1]
    cdef double tmp, d
    cdef double[:, ::1] D = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = sqrt(d)
    return np.asarray(D)


%timeit pairwise_cython(X)
100 loops, best of 3: 9.87 ms per loop




# NUMBA Computation
@jit('float64(float64[:,:])', nopython=True, nogil=True, target='cpu')
def numba_pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

%timeit numba_pairwise_python(X)
100 loops, best of 3: 8.82 ms per loop





Numba
100 loops, best of 3: 8.82 ms per loop





100x faster than Python

10x faster than Numpy

Can have Same speed than C  / C++



'''


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


#  Not tested 
#------------------------------------------------------------------------------------

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


############################################################################
#---------------------             --------------------







############################################################################











############################################################################
#---------------------             --------------------




############################################################################





















############################################################################
#---------------------             --------------------






























