#-----Put all multi process function here-----------------------
import scipy as sp;import numpy as np; import numexpr as ne
import pandas as pd; import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit, vectorize, guvectorize, float64, float32, int32, boolean
from timeit import default_timer as timer

import global01 as global01 #as global varaibles   global01.varname

import derivatives as dx



#---Multi Process does not work with BIG Array   ------------------------------------
#Very Important make Shared Array as Global Resources-------------------------------


#-----------Return Partial Result the Monte Carlo simulation---------------
#----Returning Big array of price is very slow-----------------------------
def multigbm_paralell_func(nbsimul, ww, voldt, drift, upper_cholesky,  nbasset, n, price, type1=0, strike=0, cp=1):
  if type1==0: sum1=0.0    #Agregate sum
  elif type1==1: allprocess = np.zeros(nbsimul) # Bsk last value      
  elif type1==2: allprocess = np.zeros((nbsimul,nbasset)) # Last value
  elif type1==3: allprocess = np.zeros((nbsimul,nbasset,n)) #ALl time step


# Process MC calc 
# np.random.seed(1234)
  iidbm1 = np.random.normal(0, 1, (nbasset,n,nbsimul))  

  for k in range(0, nbsimul):  #generate the random

    corrbm = np.dot(upper_cholesky, iidbm1[:,:,k])    # correlated brownian    
    bm_process= np.multiply(corrbm,voldt)  #multiply element by elt
    price[:,1:]= np.exp(bm_process + drift)  
    price = np.cumprod(price, axis = 1)  #exponen product st = st-1 *st
     
    #Update data   
    if type1==0: 
        if cp==1 : sum1+=  np.maximum(0,sum(ww*price[:,-1])-strike1) 
        else:      sum1+=  np.maximum(0,strike1-sum(ww*price[:,-1]))  #Agregate sum but NO Strike ! Only Forward
   
    elif type1==1: allprocess[:]= sum(ww*price[:,-1])
       # lock.acquire()
       # res_shared[k] = 1 # float(sum(ww*price[:,-1]))  # Bsk last value      
       # lock.release()
    elif type1==2: allprocess[k,:] = price[:,-1]  # Last value
    elif type1==3: allprocess[k,:] = price #ALl time step
    
  #Return data
  if type1==0: return sum1/float(nbsimul)    #Agregate sum/nbsimul
  else: return allprocess          #Array of Values

















#-------------------------Test Function Multi Processing---------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def func(val, lock):
    for i in range(50):
        time.sleep(0.01)
        with lock:
            val.value += 1

'''
   Function that operates on shared memory.

    
    # Make sure your not modifying data when someone else is.
        
    
    array[i, :] = i
    
    # Always release the lock!
    lock.release()
'''


'''
#Can be generalized to 3D Vector + Make Chunk of NbSimul, work on vectorize
def multigbm_processfast6(nbsimul, s0, voldt, drift, upper_cholesky,  nbasset, n, price):
 # allprocess = np.zeros((nbsimul, nbasset, n))
  chunksize= 5000
  nchunk= nbsimul / chunksize  
  
  np.random.seed(1234)
  iidbm1 = np.random.normal(0, 1, (nbasset,n,nbsimul))
  corrbm1 = np.tensordot(upper_cholesky, iidbm1,axes=([-1],[0]))    # correlated brownian   
  sum1=0.0
  
  for k in range(0, nchunk) : #NbAsset * TimeStep * NbSimulChunk; Bloc1
 # for k in range(0, nbsimul):  #generate the random

    corrbm = corrbm1[:,:,k]     # correlated brownian  

    bm_process= np.multiply(corrbm,voldt)  #multiply elt by elt
    price[:,1:]= np.exp(bm_process + drift)
    price = np.cumprod(price, axis = 1)  #expo product st = st-1 *st
    sum1+= sum(price[:,-1]) 
  
  return sum1/float(nbsimul)


#Architecture of Paralellization through Tensor and Tensor operations
'''





       
'''
The issue is that np.dot(a,b) for multidimensional arrays makes the dot product of the last dimension of a with the second dimension of b:

np.dot(a,b) == np.tensordot(a, b, axes=([-1],[2]))
As you see, it does not work as a matrix multiplication for multidimensional arrays. Using np.tensordot() allows you to control in which axes from each input you want to perform the dot product. For example, to get the same result in c_mat1 you can do:

c_mat1 = np.tensordot(Q, a1, axes=([-1],[0]))
Which is forcing a matrix multiplication-like behavior.

http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.tensordot.html



iim= a = np.arange(4).reshape(2,2)
iidbm1 = np.random.randint(1000, 10000, (2,5, 10))

iiv=np.tensordot(iim, iidbm1, axes=([-1],[0]))
 
'''


 


def multigbm_processfast7(nbsimul, s0, voldt, drift, upper_cholesky,  nbasset, n, price):
 # allprocess = np.zeros((nbsimul, nbasset, n))
  np.random.seed(1234)
  iidbm1 = np.random.normal(0, 1, (nbasset,n,nbsimul))
  sum1=0.0
  for k in range(0, nbsimul):  #generate the random
    iidbm= iidbm1[:,:,k]
    corrbm = np.dot(upper_cholesky, iidbm)    # correlated brownian
    
 #   wtt= np.multiply(corrbm,voldt)  #multiply element by elt
    wtt=  bm_generator(corrbm, dt,n,1)

   #ret= np.exp(wtt+ drift)
    vol=0
    for t in range(1, n+1):
        for ii in range(0,nbasset):
            voli= vol[ii]
            price[ii,t]  = price[ii,t-1] * np.exp((drift[ii] - voli*voli**0.5)*dt + voli*wtt[ii,t] )
    
    sum1+= sum(price[:,-1]) 
  
  return sum1/float(nbsimul)

    



def bm_generator(bm,dt,n,type1):
 tt= dt*n
 if type1==1:
     return bm
 else:
     return bm






# variables and set-up
M = 1000 	# Number of paths
N = 50	 	# Number of time steps
T = 1.0 	# Simulation time horizon

sigma = 0.3 	# annual volatlity 
mu = 0.05	# annual drift rate

dt = T/N	# simulation time step 
S0 = 100	# assset price at t=0

S = np.zeros((M,N+1))
S[:,0] = S0

# full vectorisation 
#eps = np.random.normal(0, 1, (M,N))
#S[:,1:] = np.exp((mu-0.5*sigma**2)*dt + eps*sigma*np.sqrt(dt));
#S = np.cumprod(S, axis = 1);




def merge(d2):
    time.sleep(1) # some time consuming stuffs
    for key in d2.keys():
        if key in d1:
            d1[key] += d2[key]
        else:
            d1[key] = d2[key]




def integratenp2(its, nchunk):  
#    nchunk = 10000
    chunk_size = its / nchunk   
    np.random.seed() 
    sum = 0.0
    for i in range(0,nchunk):  # do a vectorised Monte Carlo calculation
        u = np.random.uniform(size= chunk_size)
        sum += ne.evaluate("sum(exp(-u * u))")  # Do the Monte Carlo 

    return sum / float(its)


def integratenp(its):  
    nchunk= 5000 
    
    chunk_size = its / nchunk   # I totally cheated and tweaked the number of chunks to get the fastest result
    np.random.seed()  # Each process needs a different seed!
    sum = 0.0
    for i in range(0,nchunk):  # .do a vectorised Monte Carlo calculation
        u = np.random.uniform(size= chunk_size)
        sum += np.sum(np.exp(-u * u))  

    return sum / float(its)


def integratene(its):  
    nchunk= 2500
    chunk_size = its / nchunk   
    np.random.seed()  # Each process needs a different seed!
    sum = 0.0
    for i in range(0,nchunk):  # For each chunk......do a vectorised Monte Carlo calculation
        u = np.random.uniform(size= chunk_size)
        sum += ne.evaluate("sum(exp(-u * u))")  # Do the Monte Carlo

    return sum / float(its)




def list_append(count, id, out_list):
	"""Creates an empty list and then appends a random number to the list 'count' number of times. A CPU-heavy operation!"""
	for i in range(count):
		out_list.append(random.random())



def parzen_estimation(x_samples, point_x, h):
    k_n = 0
    for row in x_samples:
        x_i = (point_x - row[:,np.newaxis]) / (h)
        for row in x_i:
            if np.abs(row) > (1/2):
                break
            else: # "completion-else"*
               k_n += 1
    return (h, (k_n / len(x_samples)) / (h**point_x.shape[1]))




def init2(d):
    global d1
    d1 = d
    
def init_global1(l, r):
   global01.lock=l
   global01.res_shared=r

#into multiprocess function
def np_sin(value):
    return np.sin(value)
    
def ne_sin(x):
      return ne.evaluate("sin(x)")
    
def res_shared2():
 return res_shared    
    





'''

from sys import stdin
from multiprocessing import Pool, Array, Process
import mymodule

def count_it( key ):
  count = 0
  for c in mymodule.toShare:
    if c == key:
      count += 1
  return count

def initProcess(share):
  mymodule.toShare = share

if __name__ == '__main__':
  # allocate shared array - want lock=False in this case since we 
  # aren't writing to it and want to allow multiple processes to access
  # at the same time - I think with lock=True there would be little or 
  # no speedup
  maxLength = 50
  toShare = Array('c', maxLength, lock=False)

  # fork
  pool = Pool(initializer=initProcess,initargs=(toShare,))

  # can set data after fork
  testData = "abcabcs bsdfsdf gdfg dffdgdfg sdfsdfsd sdfdsfsdf"
  if len(testData) > maxLength:
      raise ValueError, "Shared array too small to hold data"
  toShare[:len(testData)] = testData

  print pool.map( count_it, ["a", "b", "s", "d"] )
  
  
  


  price[:,1:]= np.exp(bm_process + drift)
  price = np.cumprod(price, axis = 1)  #expo product st = st-1 *st

  for i in range(1, n+1):
     price[:,i]  = np.multiply( price[:,i-1] , ret[:,i-1] )    # Add the price at t-1 * return at t





Put the defintion of rand_string in a separate file, called test2.
Import test2 as module into my test.py script

import test2 as test2

modify the following line to access the test2 module

processes = [mp.Process(target=test2.rand_string, args=(5, output)) for x in range(4)]
Run test.py

Call myFunction()

Be Happy :)

The solution is based on this multiprocessing tutorial that suggests to import the target function from another script. This solution bypasses the safe self import by the if __name__-wrapper to get access to the target function.    
    
    
    
    
    
    
    
The key difference between imap and map/map_async is in how they consume the iterable 
you pass to them. map will convert the iterable to a list (assuming it isn't a list already),
 break it into chunks, and send those chunks to the worker processes in the Pool.
 Breaking the iterable into chunks allows performance faster than it would be if each item
 in the iterable was sent between processes one item at a time - particularly if the iterable
 is large. However, turning the iterable into a list in order to chunk it can have 
 a very high memory cost, since the entire list will need to be kept in memory. 
 map_async (and map, which is actually implemented by simply calling map_async(...).get())
 returns a list containing the result for every item in the iterable. 
 There's no way to get partial results; either you have the entire result list, or you have nothing.


imap doesn't turn the iterable you give it into a list, nor does break it into chunks (by default).
 It will iterate over the iterable one element at a time, and send them each to a worker process.
 This means you don't take the memory hit of converting the whole iterable to a list, 
 but it also means the performance is slower for large iterables, because of the lack of chunking.
 This can be mitigated by passing a chunksize argument larger than default of 1, however. 
 The other major advantage of imap, and imap_unordered, is that you can start receiving results
 from workers as soon as they're ready, rather than having to wait for all of them to be finished. 
 With imap, the results will be returned as soon as they're ready, 
 while still preserving the ordering of the input iterable. 
 With imap_unordered, results will be yielded as soon as they're ready, 
 regardless of the order of the input iterable. So, say you have this:
    
    
    
    
#include <math.h>
#include <mpi.h>
#include <gsl/gsl_rng.h>
#include "gsl-sprng.h"
 
int main(int argc,char *argv[])
{
  int i,k,N; long Iters; double u,ksum,Nsum; gsl_rng *r;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&N);
  MPI_Comm_rank(MPI_COMM_WORLD,&k);
  Iters=1e9;
  r=gsl_rng_alloc(gsl_rng_sprng20);
  for (i=0;i<(Iters/N);i++) {
    u = gsl_rng_uniform(r);
    ksum += exp(-u*u);
  }
  MPI_Reduce(&ksum,&Nsum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  if (k == 0) {
    printf("Monte carlo estimate is %f\n", Nsum/Iters );
  }
  MPI_Finalize();
  exit(EXIT_SUCCESS);
}

Intel I7 on c++

NProcesseir	T1	T2	T3  in secondes1billio
1	62.046	62.042	61.900
2	35.652	34.737	36.116
3	29.048	28.238	28.567
4	23.273	24.184	22.207
5	24.418	24.735	24.580
6	21.279	21.184	22.379
7	20.072	19.758	19.836
8	17.858	17.836	18.330
9	20.392	21.290	21.279
10	22.342	19.685	19.309




https://docs.python.org/2/library/multiprocessing.html

#!/usr/bin/env python3
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support

def func(a, b):
    return a + b

def main():
    a_args = [1,2,3]
    second_arg = 1
    with Pool() as pool:
        L = pool.starmap(func, [(1, 1), (2, 1), (3, 1)])
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        N = pool.map(partial(func, b=second_arg), a_args)
        assert L == M == N

if __name__=="__main__":
    freeze_support()
    main()
    
    

def integrate(its, nchunks):  
    chunk_size = its / nchunks   # I totally cheated and tweaked the number of chunks to get the fastest result
    np.random.seed()  # Each process needs a different seed!
    sum = 0.0
    for i in range(0,chunks):  # For each chunk......do a vectorised Monte Carlo calculation
        u = np.random.uniform(size=its/chunks)
        sum += np.sum(np.exp(-u * u))  # Do the Monte Carlo

    return sum / float(its)


def mc01(nprocs, nmax, chunksize):
    n_perproc = nmax / nprocs  # Each process gets a share of the iterations
    pool = mp.Pool(processes=nprocs)

    result = pool.map(mpf.integrate, [nmax], chunksize   )
    # Async is faster element 'i' being the return value of 'integrate' fromprocess 'i'

    pool.terminate();   pool.join();

    print(sum(result) / float(n_perproc))
    
    
start_time = time.time()

nprocs=4
iters= 10000000
nchunk=10000
mc01(nprocs,iters, nchunk)

print(str(nprocs)+"procs,"+ str(iters)+" iter,"+str(nchunk)+" chunks,--- %s sec ---" % (time.time() - start_time))



    
    
import random
import multiprocessing


def list_append(count, id, out_list):
	"""
	Creates an empty list and then appends a 
	random number to the list 'count' number
	of times. A CPU-heavy operation!
	"""
	for i in range(count):
		out_list.append(random.random())

if __name__ == "__main__":
	size = 10000000   # Number of random numbers to add
	procs = 2   # Number of processes to create

	# Create a list of jobs and then iterate through
	# the number of processes appending each process to
	# the job list 
	jobs = []
	for i in range(0, procs):
		out_list = list()
		process = multiprocessing.Process(target=list_append, 
			                              args=(size, i, out_list))
		jobs.append(process)

	# Start the processes (i.e. calculate the random number lists)		
	for j in jobs:
		j.start()

	# Ensure all of the processes have finished
	for j in jobs:
		j.join()

	print "List processing complete."


You can use the shared memory stuff from multiprocessing together with Numpy fairly easily:
    
import multiprocessing
import ctypes
import numpy as np

shared_array_base = multiprocessing.Array(ctypes.c_double, 10*10)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(10, 10)

#-- edited 2015-05-01: the assert check below checks the wrong thing
#   with recent versions of Numpy/multiprocessing. That no copy is made
#   is indicated by the fact that the program prints the output shown below.
## No copy was made
##assert shared_array.base.base is shared_array_base.get_obj()

# Parallel processing
def my_func(i, def_param=shared_array):
    shared_array[i,:] = i

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(my_func, range(10))

    print shared_array






For those stuck using Windows, which does not support fork() (unless using CygWin), pv's answer does not work. Globals are not made available to child processes.
Instead, you must pass the shared memory during the initializer of the Pool/Process as such:
import time

from multiprocessing import Process, Queue, Array

def f(q,a):
    m = q.get()
    print m
    print a[0], a[1], a[2]
    m = q.get()
    print m
    print a[0], a[1], a[2]

if __name__ == '__main__':
    a = Array('B', (1, 2, 3), lock=False)
    q = Queue()
    p = Process(target=f, args=(q,a))
    p.start()
    q.put([1, 2, 3])
    time.sleep(1)
    a[0:3] = (4, 5, 6)
    q.put([4, 5, 6])
    p.join()
    






    
    
Explicit task definition
from multiprocessing import Pool
import numpy
if __name__ == '__main__':
 pool = Pool()
 results = [pool.apply_async(numpy.sqrt, (x,))
 for x in range(100)]
 roots = [r.get() for r in results]
 print roots
1. pool.apply_async returns
a proxy object immediately
2. proxy.get() waits for task
completion and returns the
Use for: result
launching different tasks_folder in parallel
launching tasks_folder with more than one argument
 better control of task distribution



Shared memory    It is possible to share blocks of memory between processes. This eliminates
the serialization overhead.
Multiprocessing can create shared memory blocks containing C variables
and C arrays. A NumPy extension adds shared NumPy arrays. It it not
possible to share arbitrary Python objects.
NumPy extension: http://bitbucket.org/cleemesser/numpy-sharedmem
you care about your mental sanity, don’t modify shared memory
contents in the slave processes. You will end up debugging race
conditions.
Use shared memory only to transfer data from the master to the slaves!


Shared memory
from multiprocessing import Pool
from parutils import distribute
import numpy
import sharedmem
def apply_sqrt(a, imin, imax):
 return numpy.sqrt(a[imin:imax])
if __name__ == '__main__':
 pool = Pool()
 data = sharedmem.empty((100,), numpy.float)
 data[:] = numpy.arange(len(data))
 slices = distribute(len(data))
 results = [pool.apply_async(apply_sqrt, (data, imin, imax))
 for (imin, imax) in slices]
 for r, (imin, imax) in zip(results, slices):
 data[imin:imax] = r.get()
 print data



The module monoprocessing contains a class Pool with the same methods
as multiprocessing.Pool, but all tasks_folder are executed immediately and in the
same process. This permits debugging with standard tools.
If your programs works with monoprocessing but not with multiprocessing,
explore the following possibilities:
• Serialization: some object cannot be serialized
• The code of a task refers to a global variable in the master process
• The code of a tasks_folder modifies data in shared memory
Get monoprocessing from: http://pypi.python.org/pypi/monoprocessing/0.1






Distributes a sequence equally (as much as possible) over the
available processors. Returns a list of index pairs (imin, imax)
that delimit the slice to give to one task
parutils.distribute
from multiprocessing import cpu_count
default_nprocs = cpu_count()
def distribute(nitems, nprocs=None):
 if nprocs is None:
 nprocs = default_nprocs
 nitems_per_proc = (nitems+nprocs-1)/nprocs
 return [(i, min(nitems, i+nitems_per_proc))
 for i in range(0, nitems, nitems_per_proc)]


Multiprocessing introduces subtle changes in information flow that make debugging hard unless you know some shortcuts. For instance, you might have a script that works fine when indexing through a dictionary in under many conditions, but infrequently fails with certain inputs.

Normally we get clues to the failure when the entire python process crashes; however, you don't get unsolicited crash tracebacks printed to the console if the multiprocessing function crashes. Tracking down unknown multiprocessing crashes is hard without a clue to what crashed the process.

The simplest way I have found to track down multiprocessing crash informaiton is to wrap the entire multiprocessing function in a try / except and use traceback.print_exc():

import traceback
def reader(args):
    try:
        # Insert stuff to be multiprocessed here
        return args[0]['that']
    except:
        print "FATAL: reader({0}) exited while multiprocessing".format(args) 
        traceback.print_exc()
Now, when you find a crash you see something like:

FATAL: reader([{'crash', 'this'}]) exited while multiprocessing
Traceback (most recent call last):
  File "foo.py", line 19, in __init__
    self.run(task_q, result_q)
  File "foo.py", line 46, in run
    raise ValueError
ValueError



If you need more than two points to communicate, use a Queue().

If you need absolute performance, a Pipe() is much faster because Queue() is built on top of Pipe().



Implementing MapReduce with multiprocessing¶
The Pool class can be used to create a simple single-server MapReduce implementation. Although it does not give the full benefits of distributed processing, it does illustrate how easy it is to break some problems down into distributable units of work.

SimpleMapReduce

In a MapReduce-based system, input data is broken down into chunks for processing by different worker instances. Each chunk of input data is mapped to an intermediate state using a simple transformation. The intermediate data is then collected together and partitioned based on a key value so that all of the related values are together. Finally, the partitioned data is reduced to a result set.

import collections
import itertools
import multiprocessing

class SimpleMapReduce(object):
    
    def __init__(self, map_func, reduce_func, num_workers=None):
        """
        map_func

          Function to map inputs to intermediate data. Takes as
          argument one input value and returns a tuple with the key
          and a value to be reduced.
        
        reduce_func

          Function to reduce partitioned version of intermediate data
          to final output. Takes as argument a key as produced by
          map_func and a sequence of the values associated with that
          key.
         
        num_workers

          The number of workers to create in the pool. Defaults to the
          number of CPUs available on the current host.
        """
        self.map_func = map_func
        self.reduce_func = reduce_func
        self.pool = multiprocessing.Pool(num_workers)
    
    def partition(self, mapped_values):
        """Organize the mapped values by their key.
        Returns an unsorted sequence of tuples with a key and a sequence of values.
        """
        partitioned_data = collections.defaultdict(list)
        for key, value in mapped_values:
            partitioned_data[key].append(value)
        return partitioned_data.items()
    
    def __call__(self, inputs, chunksize=1):
        """Process the inputs through the map and reduce functions given.
        
        inputs
          An iterable containing the input data to be processed.
        
        chunksize=1
          The portion of the input data to hand to each worker.  This
          can be used to tune performance during the mapping phase.
        """
        map_responses = self.pool.map(self.map_func, inputs, chunksize=chunksize)
        partitioned_data = self.partition(itertools.chain(*map_responses))
        reduced_values = self.pool.map(self.reduce_func, partitioned_data)
        return reduced_values
Counting Words in Files

The following example script uses SimpleMapReduce to counts the “words” in the reStructuredText source for this article, ignoring some of the markup.

import multiprocessing
import string

from multiprocessing_mapreduce import SimpleMapReduce

def file_to_words(filename):
    """Read a file and return a sequence of (word, occurances) values.
    """
    STOP_WORDS = set([
            'a', 'an', 'and', 'are', 'as', 'be', 'by', 'for', 'if', 'in', 
            'is', 'it', 'of', 'or', 'py', 'rst', 'that', 'the', 'to', 'with',
            ])
    TR = string.maketrans(string.punctuation, ' ' * len(string.punctuation))

    print multiprocessing.current_process().name, 'reading', filename
    output = []

    with open(filename, 'rt') as f:
        for line in f:
            if line.lstrip().startswith('..'): # Skip rst comment lines
                continue
            line = line.translate(TR) # Strip punctuation
            for word in line.split():
                word = word.lower()
                if word.isalpha() and word not in STOP_WORDS:
                    output.append( (word, 1) )
    return output


def count_words(item):
    """Convert the partitioned data for a word to a
    tuple containing the word and the number of occurances.
    """
    word, occurances = item
    return (word, sum(occurances))


if __name__ == '__main__':
    import operator
    import glob

    input_files = glob.glob('*.rst')
    
    mapper = SimpleMapReduce(file_to_words, count_words)
    word_counts = mapper(input_files)
    word_counts.sort(key=operator.itemgetter(1))
    word_counts.reverse()
    
    print '\nTOP 20 WORDS BY FREQUENCY\n'
    top20 = word_counts[:20]
    longest = max(len(word) for word, count in top20)
    for word, count in top20:
        print '%-*s: %5s' % (longest+1, word, count)
The file_to_words() function converts each input file to a sequence of tuples containing the word and the number 1 (representing a single occurrence) .The data is partitioned by partition() using the word as the key, so the partitioned data consists of a key and a sequence of 1 values representing each occurrence of the word. The partioned data is converted to a set of suples containing a word and the count for that word by count_words() during the reduction phase.

$ python multiprocessing_wordcount.py

PoolWorker-1 reading basics.rst
PoolWorker-3 reading index.rst
PoolWorker-4 reading mapreduce.rst
PoolWorker-2 reading communication.rst

TOP 20 WORDS BY FREQUENCY

process         :    80
starting        :    52
multiprocessing :    40
worker          :    37
after           :    33
poolworker      :    32
running         :    31
consumer        :    31
processes       :    30
start           :    28
exiting         :    28
python          :    28
class           :    27
literal         :    26
header          :    26
pymotw          :    26
end             :    26
daemon          :    22
now             :    21
func            :    20





'''











