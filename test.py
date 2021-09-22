# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect

def log(*s):
   print(*s, flush=True)

#########################################################################################
def pd_random(ncols=7, nrows=100):
   import pandas as pd, random
   ll = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
   df = pd.DataFrame(ll, columns = [str(i) for i in range(0,ncols)])
   return df




#########################################################################################
#########################################################################################
def test_utilmy_plot():
    df = pd_random(7, 100)
    from utilmy import pd_plot_multi
    pd_plot_multi(df, cols_axe1=['0', '1'])


def test_utilmy_pd_os_session():
   from utilmy import (pd_show, git_current_hash, )

   ############################################################################
   from utilmy import pd_read_file
   import pandas as pd, random
   ncols = 7
   nrows = 100
   ll = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
   # Required for it to be detected in Session's globals()
   global df
   df = pd.DataFrame(ll, columns = [str(i) for i in range(0,ncols)])
   n0 = len(df)
   s0 = df.values.sum()
   os.makedirs("data/parquet/", exist_ok= True)

   ##### m_job , n_pool tests  ##############################
   ncopy = 20
   for i in range(0, ncopy) :
      df.to_csv( f"data/parquet/ppf_{i}.csv.gz",  compression='gzip' , index=False)

   df1 = pd_read_file("data/parquet/ppf*.gz", verbose=1, n_pool= 7 )

   assert len(df1) == ncopy * n0,         f"df1 {len(df1) }, original {n0}"
   assert round(df1.values.sum(), 5) == round(ncopy * s0,5), f"df1 {df1.values.sum()}, original {ncopy*s0}"


   ####################################################
   df.to_csv( "data/parquet/fa0b2.csv.gz",   compression='gzip' , index=False)
   df.to_csv( "data/parquet/fab03.csv.gz",   compression='gzip' , index=False)
   df.to_csv( "data/parquet/fabc04.csv.gz",  compression='gzip' , index=False)
   df.to_csv( "data/parquet/fa0bc05.csv.gz", compression='gzip' , index=False)

   df1 = pd_read_file("data/parquet/fab*.*", verbose=1)
   assert len(df1) == 2 * n0, f"df1 {len(df1) }, original {n0}"


   ##### Stresss n_pool
   df2 = pd_read_file("data/parquet/fab*.*", n_pool=1000 )
   assert len(df2) == 2 * n0, f"df1 {len(df2) }, original {n0}"

   
   ###################################################################################
   ###################################################################################
   from utilmy import git_repo_root
   print(git_repo_root())
   assert not git_repo_root() == None, "err git repo"



   ###################################################################################
   ###################################################################################
   from utilmy import os_makedirs, os_system, os_removedirs
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
   print(res)
   res = os_system( f" ls . ",  doprint=False)

   from utilmy import os_platform_os
   assert os_platform_os() == sys.platform
   

   ###################################################################################
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

   
   
   
def test_utilmy_session():   
   from utilmy import Session
   sess = Session("ztmp/session")
   sess.save('mysess', globals(), '01')
   os.system("ls ztmp/session")

   sess.save('mysess', globals(), '02')
   sess.show()

   import glob
   flist = glob.glob("ztmp/session/" + "/*")
   for f in flist:
       t = os.path.exists(os.path.abspath(f))
       assert  t == True, "session path not created "

       pickle_created = os.path.exists(os.path.abspath(f + "/df.pkl"))
       assert  pickle_created == True, "Pickle file not created"

   sess.load('mysess')
   sess.load('mysess', None, '02')

   


def test_docs_cli():
    """
      from utilmy.docs.generate_doc import run_markdown, run_table
    """
    cmd = "doc-gen  --repo_dir utilmy/      --doc_dir docs/"
    os.system(cmd)
    os.system('ls docs/')
   
   

#########################################################################################
#########################################################################################
def test_decorators_os(*args):
    from utilmy.parallel import multithread_run_list
    def test_print(*args):
        print(args[0]*args[0])
        return args[0]*args[0]

    assert multithread_run_list(function1=(test_print, (5,)),
                          function2=(test_print, (4,)),
                          function3=(test_print, (2,)))

      
    ###################################################################################
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
#######################################################################################
def pd_generate_data(ncols=7, nrows=100):
    """
    Generate sample data for function testing
    categorical features for anova test
    """
    import pandas as pd, random
    import numpy as np 
    np.random.seed(444) 
    numerical = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
    categorical2 =  data = np.random.choice(  a=[0, 1],  size=100,  p=[0.7, 0.3]  ) 
    categorical1 =  data = np.random.choice(  a=[4, 5, 6],  size=100,  p=[0.5, 0.3, 0.2]  ) 
    df = pd.DataFrame(numerical, columns = [str(i) for i in range(0,ncols)])
    df['cat1']=categorical1
    df['cat2']=categorical2
    df['cat1']= np.where( df['cat1'] == 4,'low',np.where(df['cat1'] == 5, 'High','V.High'))
    return df


def test_tabular_test():
        """
        ANOVA test
        """
        from utilmy.tabular import test_anova, test_normality2,test_plot_qqplot
        df = pd_generate_data(7, 100)
        test_anova(df, 'cat1', 'cat2')
        test_normality2(df, '0', "Shapiro")
        test_plot_qqplot(df, '1')



         
#########################################################################################
#########################################################################################
def test_text_similarity():
    from utilmy import text
    from difflib import SequenceMatcher
    from pandas._testing import assert_series_equal

    import pandas as pd
    list1 = ['dog', 'cat']
    list2 = ['doggy', 'cat']

    cols = ['name','pet_name']
    sample_df = pd.DataFrame(zip(list1, list2), columns=cols)
    original_value = text.pd_text_similarity(sample_df, cols)['score']

    check_similarity = lambda *x: SequenceMatcher(None, *x[0]).ratio()
    
    output_value = pd.Series(sample_df.apply(lambda x: check_similarity(x[[*cols]]), axis=1), name="score")

    assert_series_equal(original_value, output_value, check_names=False)
      

def test_text_pdcluster():
    from utilmy.text import text as txt
    log(txt.pd_text_getcluster )
    txt.test_lsh()

   
   
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
   log("from utilmy import parallel as par ")
   # par.test1()
   # par.test2()
   
   
def test_distributed():
   from utilmy import distributed as m
   log("from utilmy import distributed as m ")
   m.test_all()

   
   
def test_all():
    test_utilmy_pd_os_session()
    test_decorators_os()
    # test_tabular_test()
    test_text_similarity()
    test_docs_cli()

      
#########################################################################################   
if __name__ == "__main__":
    import fire
    fire.Fire()    
    #### python test.py   test_all
    #### python test.py   test_viz_vizhtml      
      
"""   
    test_utilmy_pd_os_session()
    test_decorators_os()
    # test_tabular_test()
    test_text_similarity()
    test_docs_cli()
      
    test_viz_vizhtml()   
"""

