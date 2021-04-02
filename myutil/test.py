import os, sys, time, datetime,inspect

from myutil import os_makedirs, Session, verbosity_get

##################################################################################################
def test():
   os_makedirs('ztmp/ztmp2/myfile.txt')
   os_makedirs('ztmp/ztmp3/ztmp4')


   print("success")



if __name__ == "__main__":
    import fire
    fire.Fire(test)






      
