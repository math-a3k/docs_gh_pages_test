"""

only Linux
Install

https://github.com/amzn/pecos

conda create env -n test   python=3.7.11
conda install   *.tar.gz

conda update --all
conda install python=3.7.11=h12debd9_0 libffi=3.3=he6710b0_2
pip install readline==7.0  sqlite==3.33.0



5


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
      



"""
import scipy
from scipy import sparse


import scipy, numpy as np
X=  scipy.sparse.coo_matrix(( np.random.random(120), (np.random.randint(0,100,120), np.random.randint(0,100,120) )), dtype='float32')
X = X.tocsr()
n1 = X.shape[1]


y=  scipy.sparse.coo_matrix(( np.random.randint(0,2, 120), (np.random.randint(0,100, 120), np.random.randint(0,100, 120) )), dtype='float32')
Y =y.tocsr()


from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory

# Build hierarchical label tree and train a XR-Linear model
label_feat = LabelEmbeddingFactory.create(Y, X)
cluster_chain = Indexer.gen(label_feat)
model = XLinearModel.train(X, Y, C=cluster_chain)
model.save("./save-models")


#After learning the model, we do prediction and evaluation
Xt=X; Yt=Y
from pecos.utils import smat_util
Yt_pred = model.predict(Xt)
# print precision and recall at k=10
print(smat_util.Metrics.generate(Yt, Yt_pred))


### PECOS also offers optimized C++ implementation for fast real-time inference

model = XLinearModel.load("./save-models", is_predict_only=True)

for i in range(X_tst.shape[0]):
  y_tst_pred = model.predict(X_tst[i], threads=1)




  
  
  
