import os, sys, time, datetime,inspect


##################################################################################################
def test1():
   from myutil import (os_makedirs, Session, verbosity_get, os_system  
                       
                      )
 
   os_makedirs('ztmp/ztmp2/myfile.txt')
   os_makedirs('ztmp/ztmp3/ztmp4')
   os.system("ls ztmp")


   print('verbo', verbosity_get(__file__, "../confi.json", 40,))
   print("success")

   sess = Session("ztmp/session")
   sess.save('mysess', globals(), '01')
   os.system("ls ztmp/session")

   res = os_system( f" ls . ",  doprint=True) 
   print(res)

if __name__ == "__main__":
    import fire
    fire.Fire()





