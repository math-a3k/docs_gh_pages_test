
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
from tensorflow.python.ops.gen_array_ops import one_hot
from utilmy import pd_random, pd_generate_data
from utilmy.tabular import pd_data_drift_detect


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
    from utilmy import text
    from difflib import SequenceMatcher
    from pandas._testing import assert_series_equal

    list1 = ['dog', 'cat']
    list2 = ['doggy', 'cat']

    cols = ['name','pet_name']
    sample_df = pd.DataFrame(zip(list1, list2), columns=cols)
    original_value = text.pd_text_similarity(sample_df, cols)['score']

    check_similarity = lambda *x: SequenceMatcher(None, *x[0]).ratio()
    
    output_value = pd.Series(sample_df.apply(lambda x: check_similarity(x[[*cols]]), axis=1), name="score")

    assert_series_equal(original_value, output_value, check_names=False)


    from utilmy import text as m
    log(m.pd_text_getcluster )
    m.test_lsh()
      



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
    """
    #### python test.py   test_utils
    """
    def test_logs(): 
        from utilmy.utils import log,log2, logw, loge, logger_setup
        print("testing logs utils........")
        logger_setup()
        log("simple log ")
        log2("debug log")
        logw("warning log")
        loge("error log")
    
    def config_load_test():
        from utilmy.utils import config_load
        config_load()
    
    def dataset_download_test():
        from utilmy.utils import dataset_donwload
        dataset_donwload("https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz", './testdata/tmp/test/dataset/')
    
    def os_extract_archive_test():
        from utilmy.utils import os_extract_archive
        os_extract_archive("./testdata/tmp/test/dataset/mnist_png.tar.gz","./testdata/tmp/test/dataset/archive/", archive_format = "auto")
    
    def to_file_test():
        from utilmy.utils import to_file
        to_file("to_file_test_str", "./testdata/tmp/test/to_file.txt")

    test_logs()
    config_load_test()
    dataset_download_test()
    os_extract_archive_test()
    to_file_test()



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



