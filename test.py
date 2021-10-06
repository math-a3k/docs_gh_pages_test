
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
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect, random, pandas as pd, random, numpy as np
from scipy.stats.stats import mode

from utilmy.tabular import test_heteroscedacity, test_mutualinfo


def log(*s):
   print(*s, flush=True)


from utilmy import pd_random, pd_generate_data



#########################################################################################
#########################################################################################
def test_utilmy():

   ###################################################################################
   from utilmy import git_repo_root, git_current_hash
   print(git_repo_root())
   assert not git_repo_root() == None, "err git repo"


   ###################################################################################
   from utilmy import global_verbosity
   print('verbosity', global_verbosity(__file__, "config.json", 40,))
   print('verbosity', global_verbosity('../', "config.json", 40,))
   print('verbosity', global_verbosity(__file__))

   verbosity = 40
   gverbosity = global_verbosity(__file__)
   assert gverbosity == 5, "incorrect default verbosity"
   gverbosity =global_verbosity(__file__, "config.json", 40,)
   assert gverbosity == verbosity, "incorrect verbosity "


   ###################################################################################
   from utilmy import Session
   sess = Session("ztmp/session")

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




###########################################################################
## ppandas.py
def test_ppandas():
    from utilmy import ppandas as m
    from utilmy import os_makedirs
    os_makedirs("tmp/save_files/")


    df1 = pd_random(100)
    df2 = pd_random(100)
    df3 = pd.DataFrame({"a":[1,1,2,2,2]})
    df_str = pd.DataFrame({"a": ["A", "B", "B", "C", "C"],
                           "b": [1, 2, 3, 4, 5]})


    m.pd_plot_histogram(df1["a"],path_save="tmp/histogram")
   
    m.pd_merge(df1, df2, on="b")

    df = m.pd_filter(df3, filter_dict="a>1")
    assert df.shape[0] == 3, "not filtered properly"

    m.pd_to_file(df1, "tmp/file.csv")
    m.pd_sample_strat(df1, col="a", n=10)

    bins = m.pd_col_bins(df1, "a", 5)
    assert len(np.unique(bins)) == 5, "bins not formed"

    m.pd_dtype_reduce(df1)
    m.pd_dtype_count_unique(df1,col_continuous=['b'])

    df = m.pd_dtype_to_category(df_str, col_exclude=["b"], treshold=0.7)
    assert df.dtypes["a"] == "category", "Columns was not converted to category"

    m.pd_dtype_getcontinuous(df_str,cols_exclude=["a"])
    m.pd_add_noise(df1,level=0.01,cols_exclude=["a"])

    m.pd_cols_unique_count(df_str)
    m.pd_del(df_str,cols=["a"])

    # ax = m.pd_plot_multi(df1,plot_type='pair',cols_axe1=['a','b'])
    
    a = pd.DataFrame({"a":[1,2,3,4,5]})
    b = pd.DataFrame({"b":[1,2,3,4,5]})
    cartesian_df = m.pd_cartesian(a,b)

    m.pd_show(df_str)




#########################################################################################
#########################################################################################
def test_docs_cli():
    """  from utilmy.docs.generate_doc import run_markdown, run_table
    """
    cmd = "doc-gen  --repo_dir utilmy/      --doc_dir docs/"
    os.system(cmd)
    os.system('ls docs/')
   



        
#########################################################################################
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
   
   

def test_parallel():
   from utilmy import parallel as m
   log("from utilmy import parallel")
   m.test_pdreadfile()
   m.test0()
   # par.test2()
   
   
def test_distributed():
   from utilmy import distributed as m
   log("from utilmy import distributed as m ")
   m.test_all()

   
  
#######################################################################################
##     UTILS.py
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
        dataset_donwload("https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz", './tmp/test/dataset/')
    
    def os_extract_archive_test():
        from utilmy.utils import os_extract_archive
        os_extract_archive("./tmp/test/dataset/mnist_png.tar.gz","./tmp/test/dataset/archive/", archive_format = "auto")
    
    def to_file_test():
        from utilmy.utils import to_file
        to_file("to_file_test_str", "./tmp/test/to_file.txt")

    test_logs()
    config_load_test()
    dataset_download_test()
    os_extract_archive_test()
    to_file_test()



        


################################################################################################
## oos.py
def test_oos():
    """
    #### python test.py   test_oos
    """
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
    
    def profiler_test():
        log("Tesing profiler ....")
        from utilmy.oos import profiler_start, profiler_stop
        profiler_start()
        profiler_stop()
    
    def to_dict_test():
        log("Testing to_dict() ...")
        from utilmy.oos import to_dict
        params = "hello world"
        log("to_dict",to_dict(kw =params))
    
    def to_timeunix_test():
        log("Testing to_timeuinx() ..")
        from utilmy.oos import to_timeunix
        timeunix = to_timeunix()
        log("timeunix",timeunix)
    
    def to_datetime_test():
        log("Testing to_datetime() ..")
        from utilmy.oos import to_datetime
        datetime = to_datetime("10/05/2021")
        log("datetime",datetime)

    def np_list_intersection_test():
        log("Testing np_list_intersection() ..")
        from utilmy.oos import np_list_intersection
        l1 = [1,2,3]
        l2 = [3,4,1]
        result = np_list_intersection(l1,l2)
        log("np_list_intersection",result)
        
    def np_add_remove_test():
        log("Testing np_add_remove() ..")
        from utilmy.oos import np_add_remove
        set_ = {1,2,3,4,5}
        result = np_add_remove(set_,[1,2],6)
        log("np_add_remove",result)
    
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
        from utilmy.oos import os_file_replacestring
        with open("./tmp/test/os_file_test.txt", 'a') as file:
            file.write("Dummy text to test replace string")

        os_file_replacestring("text", "text_replace", "./tmp/test/")
        
    def os_walk_test():
        log("Testing os_walk() ..")
        from utilmy.oos import os_walk
        import os
        cwd = os.getcwd()
        log(os_walk(cwd))
    
    def os_copy_safe_test():
        log("Testing os_copy_safe() ..")
        from utilmy.oos import os_copy_safe
        os_copy_safe("./tmp/test", "./tmp/test_copy/")
    
    def z_os_search_fast_test():
        log("Testing z_os_search_fast() ..")
        from utilmy.oos import z_os_search_fast
        with open("./tmp/test/os_search_test.txt", 'a') as file:
            file.write("Dummy text to test fast search string")
        res = z_os_search_fast("./tmp/test/os_search_test.txt", ["Dummy"],mode="regex")
        print(res)
    
    def os_search_content_test():
        log("Testing os_search_content() ..")
        from utilmy.oos import os_search_content
        with open("./tmp/test/os_search_content_test.txt", 'a') as file:
            file.write("Dummy text to test fast search string")
        import os
        cwd = os.getcwd()
        res = os_search_content(srch_pattern= "Dummy text",dir1=os.path.join(cwd ,"tmp/test/"))
        log(res)
    
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
        os_to_file(txt="test text to write to file",filename="./tmp/test/file_test.txt", mode="a")
        os_file_check("./tmp/test/file_test.txt")
    
    def os_utils_test():
        log("Testing os utils...")
        from utilmy.oos import os_platform_os, os_cpu, os_memory,os_getcwd, os_sleep_cpu,os_copy,\
             os_removedirs,os_sizeof, os_makedirs
        log(os_platform_os())
        log(os_cpu())
        log(os_memory())
        log(os_getcwd())
        os_sleep_cpu(cpu_min=30, sleep=10, interval=5, verbose=True)
        os_makedirs("./tmp/test")
        with open("./tmp/test/os_utils_test.txt", 'w') as file:
            file.write("Dummy file to test os utils")
            
        os_makedirs("./tmp/test/os_test")

        os_copy(os.path.join(os_getcwd(), "tmp/test"), os.path.join(os_getcwd(), "tmp/test/os_test"))
        os_removedirs("./tmp/test/os_test")
        pd_df = pd_random()
        log(os_sizeof(pd_df, set()))

    def os_system_test():
        log("Testing os_system()...")
        from utilmy.oos import os_system
        os_system("sudo service nginx restart", doprint=True)


    test_log()
    profiler_test()
    to_dict_test()
    to_timeunix_test()
    to_datetime_test()
    np_list_intersection_test()
    np_add_remove_test()
    int_float_test()
    os_path_size_test()
    os_path_split_test()
    os_file_replacestring_test()
    os_walk_test()
    os_copy_safe_test()
    z_os_search_fast_test()
    os_search_content_test()
    os_get_function_name_test()
    os_variables_test()
    os_system_list_test()
    os_file_check_test()
    os_utils_test()
    os_system_test()

## tabular.py
########################################################################################################


def test_tabular():
    """
    #### python test.py   test_tabular
    """
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    model = DecisionTreeRegressor(random_state=1)

    df = pd.read_csv("./tmp/test/crop.data.csv")
    y = df.fertilizer
    X = df[["yield","density","block"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    def test():
        from utilmy.tabular import test_anova
        log("Testing anova ...")
        #test_anova(df,"yield","fertilizer")
        log("Testing normality...")
        from utilmy.tabular import test_normality2, test_normality
        test_normality2(df,"yield","Shapiro")
        log(test_normality(df["yield"]))
        
        log("Testing plot...")
        from utilmy.tabular import test_plot_qqplot
        test_plot_qqplot(df,"yield")

        log("Testing heteroscedacity...")
        from utilmy.tabular import test_heteroscedacity
        log(test_heteroscedacity(y_test,y_pred))
    
        log("Testing test_mutualinfo()...")
        from utilmy.tabular import test_mutualinfo
        #test_mutualinfo(y_pred,X_test,colname="yield")
    
        log("Testing hypothesis_test()...")
        from utilmy.tabular import test_hypothesis
        log(test_hypothesis(X_train, X_test,"chisquare"))

    def test_estimator():
        log("Testing estimators()...")
        from utilmy.tabular import estimator_std_normal,estimator_boostrap_bayes,estimator_bootstrap
        log(estimator_std_normal(y_pred))
        log(estimator_boostrap_bayes(y_pred))
        #log(estimator_bootstrap(y_pred))
    
    def test_pd_utils():
        log("Testing pd_utils ...")
        from utilmy.tabular import pd_train_test_split_time,pd_to_scipy_sparse_matrix,pd_stat_correl_pair,\
            pd_stat_pandas_profile,pd_stat_distribution_colnum,pd_stat_histogram,pd_stat_shift_trend_changes,\
            pd_stat_shift_trend_correlation,pd_stat_shift_changes
        pd_train_test_split_time(df, coltime="block")
        pd_to_scipy_sparse_matrix(df)
        pd_stat_correl_pair(df,coltarget=["fertilizer"],colname=["yield"])
        #pd_stat_pandas_profile(df,savefile="./tmp/test/report.html", title="Pandas profile")
        pd_stat_distribution_colnum(df, nrows=len(df))
        #pd_stat_histogram(df, bins=50, coltarget="yield")
        #pd_stat_shift_trend_changes(df,"density","block")
        #pd_stat_shift_trend_correlation(X_train, X_test,"yield","block")
        #pd_stat_shift_changes(df,"yield", features_list=["density","block"])
    
    def test_np_utils():
        log("Testing np_utils ...")
        from utilmy.tabular import np_col_extractname, np_conv_to_one_col, np_list_remove
        import numpy as np
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        np_col_extractname(["aa_","bb-","cc"])
        np_list_remove(arr,[1,2,3], mode="exact")
        np_conv_to_one_col(arr)


    test()
    test_estimator()
    test_pd_utils()
    test_np_utils()



#######################################################################################
#######################################################################################
def test_tabular2():
    """
    """
    from utilmy import tabular as m
    df = pd_generate_data(7, 100)
    m.test_anova(df, 'cat1', 'cat2')
    m.test_normality2(df, '0', "Shapiro")
    m.test_plot_qqplot(df, '1')





## adatasets.py
########################################################################################################

def test_adatasets():
    """
    #### python test.py   test_adatasets
    """
    def test():
        log("Testing  ...")
        from utilmy.adatasets import test_dataset_classification_fake, test_dataset_classification_petfinder, test_dataset_classifier_covtype,\
            test_dataset_regression_fake,dataset_classifier_pmlb
        test_dataset_regression_fake(nrows=500, n_features=17)
        test_dataset_classification_fake(nrows=10)
        test_dataset_classification_petfinder(nrows=10)
        test_dataset_classifier_covtype(nrows=10)
        dataset_classifier_pmlb(name="test")
    
    def test_pd_utils():
        import pandas as pd
        from utilmy.adatasets import pd_train_test_split,pd_train_test_split2, fetch_dataset
        fetch_dataset("https://github.com/arita37/dsa2_data/raw/main/input/titanic/train/features.zip",path_target="./tmp/test")
        df = pd.read_csv("./tmp/test/crop.data.csv")
        pd_train_test_split(df)
        pd_train_test_split2(df, "block")

    test()
    test_pd_utils()

## dates.py
########################################################################################################

def test_dates():
    """
    #### python test.py   test_dates
    """
    def test():
        log("Testing  ...")
        import pandas as pd
        from utilmy.dates import date_generate,date_weekyear_excel,date_weekday_excel,date_is_holiday,\
            date_now,pd_date_split
        date_ = date_generate(start='2021-01-01', ndays=100)
        date_weekyear_excel('20210317')
        date_weekday_excel('20210317')
        #date_is_holiday([ pd.to_datetime("2015/1/1") ] * 10)
        #date_now(fmt="%Y-%m-%d %H:%M:%S %Z%z", add_days=0, timezone='Asia/Tokyo')
        df = pd.DataFrame(columns=[ 'Gender', 'Birthdate'])
        df['Gender'] = random_genders(10)
        df['Birthdate'] = random_dates(start=pd.to_datetime('1940-01-01'), end=pd.to_datetime('2008-01-01'), size=10)
        #pd_date_split(df,coldate="Birthdate")
    
    def random_dates(start, end, size):
        # Unix timestamp is in nanoseconds by default, so divide it by
        # 24*60*60*10**9 to convert to days.
        divide_by = 24 * 60 * 60 * 10**9
        start_u = start.value // divide_by
        end_u = end.value // divide_by
        return pd.to_datetime(np.random.randint(start_u, end_u, size), unit="D")
    def random_genders(size, p=None):
        """Generate n-length ndarray of genders."""
        if not p:
            # default probabilities
            p = (0.49, 0.49, 0.01, 0.01)
        gender = ("M", "F", "O", "")
        return np.random.choice(gender, size=size, p=p)
    test()

## decorators.py
########################################################################################################

def test_decorators():
    """
    #### python test.py   test_decorators
    """
    from utilmy.decorators import thread_decorator, timeout_decorator, profiler_context,profiler_decorator, profiler_decorator_base

    @thread_decorator
    def thread_decorator_test():
        log("thread decorator")


    @profiler_decorator_base
    def profiler_decorator_base_test():
        log("profiler decorator")

    @timeout_decorator(10)
    def timeout_decorator_test():
        log("timeout decorator")

    @profiler_decorator
    def profiler_decorator_test():
        log("profiler  decorator")
    
    with profiler_context() as profiler:
        log("profiler context")
    
    profiler_decorator_test()
    profiler_decorator_base_test()
    timeout_decorator_test()
    thread_decorator_test()
    # profiler_context_test()


def test_decorators2(*args):
    from utilmy.decorators import profiler_decorator, profiler_context

    @profiler_decorator
    def profiled_sum():
       return sum(range(100000))

    profiled_sum()

    with profiler_context():
       x = sum(range(1000000))
       print(x)


    from utilmy import profiler_start, profiler_stop
    profiler_start()
    print(sum(range(1000000)))
    profiler_stop()


    ###################################################################################
    from utilmy.decorators import timer_decorator
    @timer_decorator
    def dummy_func():
       time.sleep(2)

    class DummyClass:
       @timer_decorator
       def method(self):
           time.sleep(3)

    dummy_func()
    a = DummyClass()
    a.method()



#########################################################################################
def test_all():
    test_utilmy()
    test_decorators()
    # test_tabular_test()
    test_ppandas()  
    test_text()
    test_docs_cli()

    ################
    test_utils()
    # test_oos()
    test_tabular()
    test_adatasets()
    test_dates()
    test_decorators()

      
#########################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire() 