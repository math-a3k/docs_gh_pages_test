
##### Input Data

Dataframes
Xdf:
userid :  Int64
genre_list1  :  string:    "4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545,4343/4343/4545"
genre_list2  :  string :   "4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545,4343/4343/4545"

    
Ydf:
genre_list3  :  string:    "4343/4343/4545, 4343/4343/4545, 4343/4343/4545, 4343/4343/4545,4343/4343/4545"


To transform into :

X sparse of dimension:
      ncol: 5000 (hash size) *   (nlist1 + nlist2) 
      nrows:  n_userid
      
      
y sparse of dimension:
      ncol: 5000 (hash size) *   (nlist3) 
      nrows:  n_userid
      





import scipy, numpy as np
X=  scipy.sparse.coo_matrix(( np.random.random(120), (np.random.randint(0,100,120), np.random.randint(0,100,120) )), dtype='float32')
X = X.tocsr()
n1 = X.shape[1]


y=  scipy.sparse.coo_matrix(( np.random.randint(0,2, 120), (np.random.randint(0,100, 120), np.random.randint(0,100, 120) )), dtype='float32')
Y =y.tocsr()



