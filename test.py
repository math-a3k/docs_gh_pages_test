
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
import os, sys, time, datetime,inspect, random, pandas as pd, random, numpy as np



#### NEVER IMPORT HERE  !!!!
### from utilmy import pd_random, pd_generate_data
# from utilmy.tabular import pd_data_drift_detect
# from tensorflow.python.ops.gen_array_ops import one_hot

#########################################################################################
def log(*s):
   print(*s, flush=True)

def import_module(mname:str='utilmy.oos'):
    import importlib
    m = importlib.import_module(mname)
    return m

   
   
   
   
   
#########################################################################################
def test_utilmy():
   from utilmy import utilmy as m
   m.test_all()
   
   
   #####  Bug of test_all() ##############################################################################
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
   
   
########################################################################################################
def test_adatasets():
    """ #### python test.py   test_adatasets
    """
    from utilmy import adatasets as m
    m.test_all()      


########################################################################################################
def test_nnumpy():
    """#### python test.py   test_nnumpy
    """
    from utilmy import nnumpy as m
    m.test_all()



########################################################################################################
def test_dates():
    #### python test.py   test_dates
    from utilmy import dates as m
    m.test_all()


########################################################################################################
def test_decorators():
    #### python test.py   test_decorators
    from utilmy import  decorators as m
    m.test_all()


########################################################################################################
def test_deeplearning_keras():
    from utilmy.deeplearning.keras import  util_similarity as m
    m.test_tf_cdist()

   
#########################################################################################
def test_text():
    from utilmy import text as m
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
         

########################################################################################################
def test_oos():
   """#### python test.py   test_oos
   """
   from utilmy import oos as m
   m.test_all() 


########################################################################################################
def test_tabular():
   from utilmy import tabular as m
   m.test_all()




#########################################################################################
def test_all():
    test_utilmy()
    test_decorators()
    # test_tabular_test()
    test_ppandas()  
    test_text()
    test_docs_cli()


    ################
    # test_oos()
    test_tabular()
    test_adatasets()
    test_dates()
    test_decorators()
    test_utils()


    ################
    test_deeplearning_keras()



      
#########################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire() 
