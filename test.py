
# -*- coding: utf-8 -*-
"""
python test.py   test_all
python test.py   test_viz_vizhtml
Rules to follow :
   Put import only inside the function.
   def  test_{pythonfilename.py}() :
       from utilmy import parallel as m
       Put all the test  below
       n.myfun()
"""
import os, sys, time, datetime,inspect, random, pandas as pd, random, numpy as np

#### Only for testing
from utilmy import pd_random, pd_generate_data


#### NEVER IMPORT HERE  !!!!
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
    from utilmy import util_nnumpy as m
    m.test_all()



########################################################################################################
def test_dates():
    #### python test.py   test_dates
    from utilmy import util_dates as m
    m.test_all()


########################################################################################################
def test_decorators():
    #### python test.py   test_decorators
    from utilmy import  util_decorators as m
    m.test_tf_cdist()


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
   from utilmy.viz import vizhtml as vi
   log("Visualization ")
   log(" from utilmy.viz import vizhtml as vi     ")
   vi.test1()
   vi.test2()
   vi.test3()
   vi.test4()
   vi.test_scatter_and_histogram_matplot()
   vi.test_pd_plot_network()
   vi.test_cssname()
   

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
    from utilmy.utils import utils as m
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



