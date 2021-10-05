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





#########################################################################################
def test_ppandas():
    from utilmy import ppandas as m
    from utilmy import os_makedirs
    os_makedirs("save_files")


    df1 = pd_random(100)
    df2 = pd_random(100)
    df3 = pd.DataFrame({"a":[1,1,2,2,2]})
    df_str = pd.DataFrame({"a": ["A", "B", "B", "C", "C"],
                        "b": [1, 2, 3, 4, 5]})


    m.pd_plot_histogram(df1["a"],path_save="save_files/histogram")

    f = os.path.exists(os.path.abspath("save_files/histogram.png"))
    assert f == True, "save_files/histogram.png"

    m.pd_merge(df1, df2, on="b")

    df = m.pd_filter(df3, filter_dict="a>1")
    assert df.shape[0] == 3, "not filtered properly"

    m.pd_to_file(df1, "save_files/file.csv")
    m.pd_sample_strat(df1, col="a", n=10)

    bins = m.pd_col_bins(df1, "a", 5)
    assert len(np.unique(bins)) == 5, "bins not formed"

    m.pd_dtype_reduce(df1)
    m.pd_dtype_count_unique(df1)

    df = m.pd_dtype_to_category(df_str, col_exclude=["b"], treshold=0.7)
    assert df.dtypes["a"] == "category", "Columns was not converted to category"

    m.pd_dtype_getcontinuous(df_str,cols_exclude=["a"])
    m.pd_add_noise(df1,level=0.01,cols_exclude=["a"])

    m.pd_cols_unique_count(df_str)
    m.pd_del(df_str,cols=["a"])

    # ax = m.pd_plot_multi(df1,plot_type='pair',cols_axe1=['a','b'])
    
    # a = pd.DataFrame({"a":[1,2,3,4,5]})
    # b = pd.DataFrame({"b":[1,2,3,4,5]})
    # cartesian_df = m.pd_cartesian(a,b)

    m.pd_show(df_str)





def test_oos():
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

    res = os_system( f" ls . ",  doprint=True)
    log(res)
    res = os_system( f" ls . ",  doprint=False)

    from utilmy import os_platform_os
    assert os_platform_os() == sys.platform


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
def test_decorators(*args):
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
      
      
   
#######################################################################################
#######################################################################################
def test_tabular():
    """
    ANOVA test
    """
    from utilmy import tabular as m
    df = pd_generate_data(7, 100)
    m.test_anova(df, 'cat1', 'cat2')
    m.test_normality2(df, '0', "Shapiro")
    m.test_plot_qqplot(df, '1')




        
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

   
   
def test_all():
    test_utilmy()
    test_decorators()
    # test_tabular_test()
    test_text()
    test_docs_cli()

      
#########################################################################################   
if __name__ == "__main__":
    import fire
    fire.Fire()    


