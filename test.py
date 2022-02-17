
# -*- coding: utf-8 -*-
"""
python test.py   test_all
python test.py   test_viz_vizhtml
Rules to follow :
   Put import only inside the function.
   def  test_{pythonfilename.py}() :
       from utilmy import parallel as m
       m.test_all()
"""
import os, sys, time, datetime,inspect, random, pandas as pd, random, numpy as np, glob


#### NEVER IMPORT HERE  !!!!
# from utilmy import pd_random, pd_generate_data
# from tensorflow.python.ops.gen_array_ops import one_hot

#########################################################################################
def log(*s):
   print(*s, flush=True)

def import_module(mname:str='utilmy.oos'):
    import importlib
    m = importlib.import_module(mname)
    return m

   
def pd_random(ncols=7, nrows=100):
   import pandas as pd
   ll = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
   df = pd.DataFrame(ll, columns = [str(i) for i in range(0,ncols)])
   return df


def pd_generate_data(ncols=7, nrows=100):
    """ Generate sample data for function testing
    categorical features for anova test
    """
    import numpy as np, pandas as pd
    np.random.seed(444)
    numerical    = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
    df = pd.DataFrame(numerical, columns = [str(i) for i in range(0,ncols)])
    df['cat1']= np.random.choice(  a=[0, 1],  size=nrows,  p=[0.7, 0.3]  )
    df['cat2']= np.random.choice(  a=[4, 5, 6],  size=nrows,  p=[0.5, 0.3, 0.2]  )
    df['cat1']= np.where( df['cat1'] == 0,'low',np.where(df['cat1'] == 1, 'High','V.High'))
    return df   
   
   
   
#########################################################################################
def test_utilmy():
   from utilmy import utilmy as m
   m.test_all()
   
   
   #####  Bug of globals() in utilmy.py #################################################
   log("\n##### Session  ")
   sess = m.Session("ztmp/session")

   global mydf
   mydf = pd_generate_data()

   sess.save('mysess', glob=globals(), tag='01')
   os.system("ls ztmp/session")
   sess.show()

   import glob
   flist = glob.glob("ztmp/session/" + "/*")
   for f in flist:
       t = os.path.exists(os.path.abspath(f))
       assert  t == True, "session path not created "

       pickle_created = os.path.exists(os.path.abspath(f + "/mydf.pkl"))
       assert  pickle_created == True, "Pickle file not created"

   sess.load('mysess')
   sess.load('mysess', tag='01')

   

##########################################################################################
def test_ppandas():
    from utilmy import ppandas as m
    m.test_all()

   
#########################################################################################
def test_docs_cli():
    """  from utilmy.docs.generate_doc import run_markdown, run_table 
    """
    cmd = "doc-gen  --repo_dir utilmy/      --doc_dir docs/"
    os.system(cmd)
    os.system('ls docs/')
   
   
#########################################################################################
def test_adatasets():
    """ #### python test.py   test_adatasets
    """
    from utilmy import adatasets as m
    m.test_all()      


#########################################################################################
def test_nnumpy():
    """#### python test.py   test_nnumpy
    """
    from utilmy import nnumpy as m
    m.test_all()



#########################################################################################
def test_dates():
    #### python test.py   test_dates
    from utilmy import dates as m
    m.test_all()


#########################################################################################
def test_decorators():
    #### python test.py   test_decorators
    from utilmy import  decorators as m
    m.test_all()


   
#########################################################################################
def test_text():
    from utilmy.nlp import util_cluster as m
    m.test_all()  

    from utilmy.nlp import util_gensim as m
    m.test_all()  

   
#########################################################################################
def test_viz_vizhtml():
   from utilmy.viz import vizhtml as m
   log("Visualization ")
   log(" from utilmy.viz import vizhtml as vi     ")
   m.test_all()

   

#########################################################################################
def test_parallel():
   from utilmy import parallel as m
   log("from utilmy import parallel")
   m.test_pdreadfile()
   m.test0()
   # par.test2()
   

#########################################################################################
def test_distributed():
   from utilmy import distributed as m
   log("from utilmy import distributed as m ")
   m.test_all()

   
  
#######################################################################################
def test_utils():
    """ #### python test.py   test_utils
    """
    from utilmy import utils as m
    m.test_all() 
         

#######################################################################################
def test_oos():
   """#### python test.py   test_oos
   """
   from utilmy import oos as m
   m.test_all() 


#######################################################################################
def test_tabular():
   from utilmy.tabular import util_sparse as m
   m.test_all()

   from utilmy.tabular import util_explain as m
   m.test_all()


   
#########################################################################################
def test_deeplearning_keras():
    from utilmy.deeplearning.keras import  util_similarity as m
    m.test_tf_cdist()

   
#######################################################################################
def test_deeplearning_yolov5():
   from utilmy.deeplearning import util_yolo as m
   m.test_all()


#######################################################################################
def test_recsys_ab():
   from utilmy.recsys import ab as m
   log("from utilmy.recsys import ab")
   m.test_all()




import utilmy as  uu




#######################################################################################
def test_all():
    test_utilmy()
    test_decorators()
    test_ppandas()  
    test_text()
    test_docs_cli()


    ################
    # test_oos()
    test_tabular()
    test_adatasets()
    test_dates()
    test_utils()


    ################
    test_deeplearning_keras()
    test_deeplearning_yolov5()


    ###############
    test_recsys_ab()


      
#######################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire() 

   
   
