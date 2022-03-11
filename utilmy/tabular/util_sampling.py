HELP = """

Various samplers




"""

def test():
  """function test
  Args:
  Returns:
      
  """
  def matrix_source(): 
      for i in range(0, 1000, 10): 
          yield np.tile(np.arange(i, i + 10), (5, 1)).T

  print(next(src))
  print(next(src))

  hist = np.zeros(1000, dtype=int)
  for run in range(500): 
      res = reservoir_sampling(matrix_source(), 100)
      hist += np.bincount(res[:, 0], minlength=1000)
    
    

def reservoir_sampling(src, nsample, temp_fac=1.5, rs=None): 
    """  When having a inifinte stream of data
    samples nsample vectors from an iterator src that yields matrices
    nsample * temp_fac is the max size of the temporary buffer.
    rs is a RandomState object   
    """
    if rs is None: 
        rs = np.random
    maxsize = int(nsample * temp_fac)
    
    reservoir = []      # represented as a list of subsampled matrices
    nreservoir = 0      # size of the reservoir
    nseen = 0           # number of vectors seen so far 
    threshold = 1.0     # probability for a vector to be included in the reservoir
    
    for mat in src:
        n = len(mat)
        
        if nseen + n < maxsize: 
            # so far, no need to sub-sample
            reservoir.append(mat)
            nreservoir += n
        else: 
            # sample from the input matrix
            mask = rs.rand(n) < threshold
            mat_sampled = mat[mask]
            # add to reservoir
            nreservoir += len(mat_sampled)
            reservoir.append(mat_sampled)
            
            if nreservoir > maxsize: 
                # resamlpe reservoir to nsample
                reservoir = np.vstack(reservoir)
                idx = rs.choice(nreservoir, size=nsample, replace=False)
                reservoir = [reservoir[idx]]
                nreservoir = nsample
                # update threshold
                threshold = nsample / (nseen + n)
            
        nseen += n
    
    # do a last sample
    reservoir = np.vstack(reservoir)
    if nreservoir > nsample: 
        idx = rs.choice(nreservoir, size=nsample, replace=False)
        reservoir = reservoir[idx]
    return reservoir
  
  
  
  
  
