
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
    """
    #### python test.py   test_oos
    """
    return 1  
    log("Testing oos.py............................")
    from utilmy import oos as m

    from utilmy.oos import os_makedirs, os_system, os_removedirs
    os_makedirs('ztmp/ztmp2/myfile.txt')
    os_makedirs('ztmp/ztmp3/ztmp4')
    os_makedirs('/tmp/one/two')
    os_makedirs('/tmp/myfile')
    os_makedirs('/tmp/one/../mydir/')
    os_makedirs('./tmp/test')
    os.system("ls ztmp")

    path = ["/tmp/", "ztmp/ztmp3/ztmp4", "/tmp/", "./tmp/test","/tmp/one/../mydir/"]
    for p in path:
       f = os.path.exists(os.path.abspath(p))
       assert  f == True, "path " + p

    rev_stat = os_removedirs("ztmp/ztmp2")
    assert not rev_stat == False, "cannot delete root folder"

    ############################################################
    res = os_system( f" ls . ",  doprint=True)
    log(res)
    res = os_system( f" ls . ",  doprint=False)

    from utilmy import os_platform_os
    assert os_platform_os() == sys.platform


    ############################################################
    def test_log():
        from utilmy.oos import log, log2, log5
        log("Testing logs ...")
        log2("log2")
        log5("log5")
    
    
    def int_float_test():
        log("Testing int/float ..")
        from utilmy.oos import is_float,to_float,is_int,to_int
        int_ = 1
        float_ = 1.1
        is_int(int_)
        is_float(float_)
        to_float(int_)
        to_int(float_)
    
    def os_path_size_test():
        log("Testing os_path_size() ..")
        from utilmy.oos import os_path_size
        size_ = os_path_size()
        log("total size", size_)
    
    def os_path_split_test():
        log("Testing os_path_split() ..")
        from utilmy.oos import os_path_split
        result_ = os_path_split("test/tmp/test.txt")
        log("result", result_)
    
    def os_file_replacestring_test():
        log("Testing os_file_replacestring() ..")
        
    def os_walk_test():
        log("Testing os_walk() ..")
        from utilmy.oos import os_walk
        import os
        cwd = os.getcwd()
        # log(os_walk(cwd))
    
    def os_copy_safe_test():
        log("Testing os_copy_safe() ..")
        from utilmy.oos import os_copy_safe
        os_copy_safe("./testdata/tmp/test", "./testdata/tmp/test_copy/")
    
    def z_os_search_fast_test():
        log("Testing z_os_search_fast() ..")
        from utilmy.oos import z_os_search_fast
        with open("./testdata/tmp/test/os_search_test.txt", 'a') as file:
            file.write("Dummy text to test fast search string")
        res = z_os_search_fast("./testdata/tmp/test/os_search_test.txt", ["Dummy"],mode="regex")
        print(res)
    
    def os_search_content_test():
        log("Testing os_search_content() ..")
        from utilmy.oos import os_search_content
        with open("./testdata/tmp/test/os_search_content_test.txt", 'a') as file:
            file.write("Dummy text to test fast search string")
        import os
        cwd = os.getcwd()
        '''TODO: for f in list_all["fullpath"]:
            KeyError: 'fullpath'
        res = os_search_content(srch_pattern= "Dummy text",dir1=os.path.join(cwd ,"tmp/test/"))
        log(res)
        '''
    
    def os_get_function_name_test():
        log("Testing os_get_function_name() ..")
        from utilmy.oos import os_get_function_name
        log(os_get_function_name())
    
    def os_variables_test():
        log("Testing os_variables_test ..")
        from utilmy.oos import os_variable_init, os_variable_check, os_variable_exist, os_import, os_clean_memory
        ll = ["test_var"]
        globs = {}
        os_variable_init(ll,globs)
        os_variable_exist("test_var",globs)
        os_variable_check("other_var",globs,do_terminate=False)
        os_import(mod_name="pandas", globs=globs)
        os_clean_memory(["test_var"], globs)
        log(os_variable_exist("test_var",globs))

    def os_system_list_test():
        log("Testing os_system_list() ..")
        from utilmy.oos import os_system_list
        cmd = ["pwd","whoami"]
        os_system_list(cmd, sleep_sec=0)
    
    def os_file_check_test():
        log("Testing os_file_check()")
        from utilmy.oos import os_to_file, os_file_check
        os_to_file(txt="test text to write to file",filename="./testdata/tmp/test/file_test.txt", mode="a")
        os_file_check("./testdata/tmp/test/file_test.txt")
    
    def os_utils_test():
        log("Testing os utils...")
        from utilmy.oos import os_platform_os, os_cpu, os_memory,os_getcwd, os_sleep_cpu,os_copy,\
             os_removedirs,os_sizeof, os_makedirs
        log(os_platform_os())
        log(os_cpu())
        log(os_memory())
        log(os_getcwd())
        os_sleep_cpu(cpu_min=30, sleep=1, interval=5, verbose=True)
        os_makedirs("./testdata/tmp/test")
        with open("./testdata/tmp/test/os_utils_test.txt", 'w') as file:
            file.write("Dummy file to test os utils")
            
        os_makedirs("./testdata/tmp/test/os_test")
        from utilmy.oos import os_file_replacestring
        with open("./testdata/tmp/test/os_test/os_file_test.txt", 'a') as file:
            file.write("Dummy text to test replace string")

        os_file_replacestring("text", "text_replace", "./testdata/tmp/test/os_test/")

        #os_copy(os.path.join(os_getcwd(), "tmp/test"), os.path.join(os_getcwd(), "tmp/test/os_test"))
        os_removedirs("./testdata/tmp/test/os_test")
        pd_df = pd_random()
        log(os_sizeof(pd_df, set()))

    def os_system_test():
        log("Testing os_system()...")
        from utilmy.oos import os_system
        os_system("whoami", doprint=True)


    test_log()
    int_float_test()
    #os_path_size_test()
    #os_path_split_test()
    #os_file_replacestring_test()
    # os_walk_test()
    os_copy_safe_test()
    #z_os_search_fast_test()
    #os_search_content_test()
    #os_get_function_name_test()
    #os_variables_test()
    #os_system_list_test()
    #os_file_check_test()
    #os_utils_test()
    #os_system_test()



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



