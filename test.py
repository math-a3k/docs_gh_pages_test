# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
import os, sys, time, datetime,inspect


#########################################################################################
def test1():
   from utilmy import (Session,
                       global_verbosity,


                       os_makedirs,
                       os_system ,
                       os_removedirs,


                       pd_read_file,
                       pd_show,


                       git_repo_root,
                       git_current_hash,


                      )


   #################################################################################### 
   import pandas as pd, random

   ncols = 7
   nrows = 100
   ll = [[ random.random() for i in range(0, ncols)] for j in range(0, nrows) ]
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
   assert df1.values.sum() == ncopy * s0, f"df1 {df.values.sum()}, original {ncopy*s0}"

   
   ########################################################### 
   df.to_csv( "data/parquet/fa0b2.csv.gz",   compression='gzip' , index=False)
   df.to_csv( "data/parquet/fab03.csv.gz",   compression='gzip' , index=False)
   df.to_csv( "data/parquet/fabc04.csv.gz",  compression='gzip' , index=False)
   df.to_csv( "data/parquet/fa0bc05.csv.gz", compression='gzip' , index=False)
   df.to_csv( "data/parquet/fa0bc05.csv")
   df.to_csv( "data/parquet/fa0bc05.txt")

   df1 = pd_read_file("data/parquet/fab*.*", verbose=1)
   b = df1.mean()
   a = df.mean()
   assert len(df1) == 2 * n0, f"df1 {len(df1) }, original {n0}"
   assert len(a) == len(b), f"df2 mean {len(df1)}, original {ncols}"
   
   df1 = pd_read_file("data/parquet/f*.gz", verbose=1, n_pool=3)
   print('pd_read_file gzip ', df1)
   n1  = len(df1)
   n0 = len(df)
   assert round(5*n0,5) == round(n1,5), f"df1 {n1}, original {n0}"

   df2 = pd_read_file("data/parquet/fabc04.csv.gz", verbose=1, n_pool=1)
   b = df2.mean()
   a = df.mean()
   assert len(df2.columns) == ncols, f"df2 {len(df2.columns)}, original {ncols}"
   assert len(a) == len(b), f"df2 mean {len(df2)}, original {ncols}"

   df2 = pd_read_file("data/parquet/fabc06.zip", verbose=1, n_pool=1)
   b = df2.mean()
   a = df.mean()
   assert len(df2.columns) == ncols, f"df2 {len(df2.columns)}, original {ncols}"
   assert len(a) == len(b), f"df2 mean {len(df2)}, original {ncols}"

   df2 = pd_read_file("data/parquet/fabc05.txt", verbose=1, n_pool=1)
   b = df2.mean()
   a = df.mean()
   assert len(df2.columns) == ncols, f"df2 {len(df2.columns)}, original {ncols}"
   assert len(a) == len(b), f"df2 mean {len(df2)}, original {ncols}"

   df2 = pd_read_file("data/parquet/fabc05.cvs", verbose=1, n_pool=1)
   b = df2.mean()
   a = df.mean()
   assert len(df2.columns) == ncols, f"df2 {len(df2.columns)}, original {ncols}"
   assert len(a) == len(b), f"df2 mean {len(df2)}, original {ncols}"


   ##### Stresss n_pool
   df2 = pd_read_file("data/parquet/fab*.*", n_pool=1000 )
   assert len(df2) == 2 * n0, f"df1 {len(df2) }, original {n0}"



   ###################################################################################
   ###################################################################################
   print(git_repo_root())








   ###################################################################################
   ###################################################################################
   os_makedirs('ztmp/ztmp2/myfile.txt')
   os_makedirs('ztmp/ztmp3/ztmp4')
   os_makedirs('/tmp/')
   os_makedirs('/tmp/one/two')
   os_makedirs('/tmp/myfile')
   os_makedirs('/tmp/one/../mydir/')
   os_makedirs('./tmp/test')
    
   os.system("ls ztmp")


   os_removedirs("ztmp/ztmp2")

   res = os_system( f" ls . ",  doprint=True)
   print(res)
   res = os_system( f" ls . ",  doprint=False) 







   ###################################################################################
   ###################################################################################   
   print('verbosity', global_verbosity(__file__, "config.json", 40,))
   print('verbosity', global_verbosity('../', "config.json", 40,))
   print('verbosity', global_verbosity(__file__))













   ###################################################################################
   ###################################################################################
   sess = Session("ztmp/session")
   sess.save('mysess', globals(), '01')
   os.system("ls ztmp/session")

   sess.save('mysess', globals(), '02')
   sess.show()

   sess.load('mysess')
   sess.load('mysess', None, '02')





if __name__ == "__main__":
    test1()





