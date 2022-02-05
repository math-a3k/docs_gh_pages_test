
##### Input Data


def pandas_to_csr(Xdf,  ydf, hashsize=5000):
    """
    
        genre_list1 : Consider as list of music genre_id, listened by one person

            one genre_id : 234234/234234/42342 : Hiearchy of sub-genre

            Goal is to predict which music genre that will listen

            Using a sparse model


    
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


        ll = "4343/4343/4545".split("/")
        for t in ll :
           colid = hash(t) % 5000 ---> into 0---5000 bucket (ie one hot).
    
           genrehash = hash("genreid")  % 5000      (there are 100,000 genres, too big, so hash into 5000 buckets)  
    
    
    """
    
    
    
    
    
    return X, y





import scipy, numpy as np
X=  scipy.sparse.coo_matrix(( np.random.random(120), (np.random.randint(0,100,120), np.random.randint(0,100,120) )), dtype='float32')
X = X.tocsr()
n1 = X.shape[1]


y=  scipy.sparse.coo_matrix(( np.random.randint(0,2, 120), (np.random.randint(0,100, 120), np.random.randint(0,100, 120) )), dtype='float32')
Y =y.tocsr()



