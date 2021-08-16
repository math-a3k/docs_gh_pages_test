


    

    
def multithread_run(fun_async, input_list:list, n_pool=5, verbose=True):
    """  input is as list of tuples
    def fun_async(xlist):
      for x in xlist :   
            hdfs.upload(x[0], x[1])
    """
    #### Input xi #######################################    
    xi_list = [ []  for t in range(n_pool) ]     
    for i, xi in enumerate(input_list) :
        jj = i % n_pool 
        xi_list[jj].append( xi )

    #### Pool execute ###################################
    pool     = ThreadPool(processes=n_pool)
    job_list = []
    for i in range(n_pool):
         job_list.append( pool.apply_async(fun_async, (xi_list[i], )))
         if verbose : log(i, xi_list[i] )

    res_list = []            
    for i in range(n_pool):
        if i >= len(job_list): break
        res_list.append( job_list[ i].get() )
        log(i, 'job finished')

    pool.terminate() ; pool.join()  ; pool = None          
    log('n_processed', len(res_list) )    



    
    
    
