#------ Multi processing testing----------------------------------------------
import scipy as sp;import numpy as np; import numexpr as ne
import pandas as pd; import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit, vectorize, guvectorize, float64, float32, int32, boolean
from timeit import default_timer as timer
import time 

import multiprocessing as mp; from functools import partial
import multiprocessfunc as mpf #put the multi process function loop there

import derivatives as dx




#Create partial function wirh extra param,MemoryError with big size
def mc01(nprocs, nmax, nchunk):  #numexpr
    n_perproc = nmax / nprocs  # Each process gets a share of the iterations
    pool = mp.Pool(processes=nprocs)
     #very important keep,list:   [n_perproc]: create a list of n_perproc]  elts
    result = pool.map(partial(mpf.integratenp2, nchunk= nchunk), [n_perproc]*nprocs   ) 

    print(sum(result) / float(nprocs))
    pool.terminate();   pool.join();


def mc02(nprocs, nmax, chunksize): #numpy
    n_perproc = nmax / nprocs  # Each process gets a share of the iterations
    pool = mp.Pool(processes=nprocs)

    result = pool.map(mpf.integratene,  [n_perproc]*nprocs  ) #very important keep,list

    print(sum(result) / float(nprocs))
    pool.terminate();   pool.join();    
    

    
#----------------Compute Time for Monte Carlo-------------------------------
start_time = time.time()
nprocs=4
iters= 200 * 1000 * 1000
nchunk= 1000

mc01(nprocs,iters, nchunk)

print(str(nprocs)+"procs,"+ str(iters)+" iter,"+str(nchunk)+" chunks,--- %s sec ---" % (time.time() - start_time))



4procs,200000000 iter,5000 chunks,--- 4.779273986816406 sec ---
4procs,200000000 iter,2500 chunks,--- 4.739270925521851 sec --- ne
4procs,200000000 iter,2500 chunks,--- 5.2202980518341064 sec --- numpy



#numexpr version
8procs,1000000000 iter,2500 chunks,--- 15.606893062591553 sec ---
8procs,1000000000 iter,10000chunks,--- 15.872907876968384 sec ---
4procs,1000000000 iter,2500 chunks,--- 13.003743886947632 sec ---


4procs,500000000 iter,2500 chunks,--- 7.475428104400635 sec ---
1procs,500000000 iter,2500 chunks,--- 13.429769039154053 sec ---

4procs,2000000000 iter,2500 chunks,--- 23.793360948562622 sec ---
1procs,2000000000 iter,2500 chunks,--- 49.759846210479736 sec ---



4procs,1000000000 iter,5000 chunks,--- 12.442711114883423 sec ---
4procs,1000000000 iter,2000 chunks,--- 12.416709899902344 sec ---
4procs,1000000000 iter,1000 chunks,--- 13.246757984161377 sec ---
4procs,1000000000 iter,2500 chunks,--- 12.205698013305664 sec ---



#numpy version
6procs,1000000000 iter,5000 chunks,--- 17.375993967056274 sec ---
2procs,1000000000 iter,5000 chunks,--- 20.572175979614258 sec ---
4procs,1000000000 iter,2000 chunks,--- 18.141037940979004 sec ---
4procs,1000000000 iter,5000 chunks,--- 16.357936143875122 sec ---
4procs,1000000000 iter,40000 chunks,--- 17.33899188041687 sec ---
4procs,1000000000 iter,20000 chunks,--- 16.61395001411438 sec ---
4procs,1000000000 iter,10000 chunks,--- 16.992971897125244 sec ---




#-----------------------------------------------------------


The key difference between imap and map/map_async is in how they consume the iterable 
you pass to them.

map will convert the iterable to a list,break it into chunks, and 
send those chunks to the worker processes in the Pool.

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

With imap, the results will be returned as soon as they're ready, while still preserving the ordering of the input iterable. 
With imap_unordered, results will be yielded as soon as they're ready, 
regardless of the order of the input iterable. So, say you have this:
    
    







#------ Multi processing testing----------------------------------------------
import scipy as sp;import numpy as np; import numexpr as ne
import pandas as pd; import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit, vectorize, guvectorize, float64, float32, int32, boolean
from timeit import default_timer as timer
import time 

import multiprocessing as mp; from functools import partial
import multiprocessfunc as mpf #put the multi process function loop there

import derivatives as dx




def serial(samples, x, widths):
    return [mpf.parzen_estimation(samples, x, w) for w in widths]

def multiprocess(processes, samples, x, widths):
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(mpf.parzen_estimation, args=(samples, x, w)) for w in widths]
    results = [p.get() for p in results]
#    results.sort() # to sort the results by input window width
    pool.terminate();   pool.join();  # Need to terminate pool print(pool.is_alive())
     
    return results



# Generate random 2D-patterns
nsize= 120000
mu_vec = np.array([0,0])
cov_mat = np.array([[1,0],[0,1]])
x_2Dgauss = np.random.multivariate_normal(mu_vec, cov_mat, nsize)

widths = np.arange(0.1, 1.3, 0.1)
point_x = np.array([[0],[0]])
results = []





start_time = time.time()

results = serial(x_2Dgauss, point_x, widths)

print(str(nsize)+" size--- %s sec ---" % (time.time() - start_time))





start_time = time.time()

results2 = multiprocess(4, x_2Dgauss, point_x, widths)

print("4 Procs---"+str(nsize)+" size--- %s sec ---" % (time.time() - start_time))





60000 size--- 8.841506004333496 sec ---
4Proc 60000 size--- 8.369478940963745 sec ---

120000 size--- 17.978028059005737 sec ---
4 Procs---120000 size--- 11.919681787490845 sec ---



#----------------------------------------------


  
def test01():
	size = 10000000   # Number of random numbers to add
	procs = 4   # Number of processes to create

	# Create a list of jobs and then iterate through the number of processes appending each process to the job list 
	jobs = []
	for i in range(0, procs):
		out_list = list()
		process = mp.Process(target=mpf.list_append,   args=(size, i, out_list))
		jobs.append(process)
	
	for j in jobs:  	# Start the processes (i.e. calculate the random number lists)	
		j.start()

	for j in jobs: 	# Ensure all of the processes have finished
		j.terminate()            
		j.join()

	print ("List processing complete.")



pool.terminate();   pool.join(); 


%time test01()



time python testmulti.py












def random_tree(Data):
    tree = calculation(Data)
    forest.append(tree)

forest = list()
for i in range(300):
    random_tree(Data)
    
    
    

from multiprocessing import Pool

def random_tree(Data):
    return calculation(Data)

pool = Pool(processes=4)
forest = pool.map(random_tree, (Data for i in range(300)))













'''

from __future__ import print_function

import multiprocessing
import ctypes
import numpy as np

def shared_array(shape):
    """
    Form a shared memory numpy array.
    
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing 
    """
    
    shared_array_base = multiprocessing.Array(ctypes.c_double, shape[0]*shape[1])
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)
    return shared_array


# Form a shared array and a lock, to protect access to shared memory.
array = shared_array((1000, 1000))
lock = multiprocessing.Lock()


def parallel_function(i, def_param=(lock, array)):
    """
    Function that operates on shared memory.
    """
    
    # Make sure your not modifying data when someone else is.
    lock.acquire()    
    
    array[i, :] = i
    
    # Always release the lock!
    lock.release()

if __name__ == '__main__':
    """
    The processing pool needs to be instantiated in the main 
    thread of execution. 
    """
        
    pool = multiprocessing.Pool(processes=4)
        
    # Call the parallel function with different inputs.
    args = [(0), 
            (1), 
            (2)]
    
    # Use map - blocks until all processes are done.
    pool.map(parallel_function, args )
    
    print(array)
    
    
    '''





    






    
    
    
    
    


def test01():
	size = 10000000   # Number of random numbers to add
	procs = 4   # Number of processes to create

	# Create a list of jobs and then iterate through
	# the number of processes appending each process to
	# the job list 
	jobs = []
	for i in range(0, procs):
		out_list = list()
		process = mp.Process(target=mpf.list_append,   args=(size, i, out_list))
		jobs.append(process)

	# Start the processes (i.e. calculate the random number lists)		
	for j in jobs:
		j.start()

	# Ensure all of the processes have finished
	for j in jobs:
		j.terminate()            
		j.join()

	print ("List processing complete.")






