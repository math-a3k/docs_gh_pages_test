# -*- coding: utf-8 -*-
#---------Various Utilities function for Python 2  and 3--------------------------------------
from __future__ import print_function
import os, sys, gc, numpy as np


#####################################################################################
import argparse, os, sys,  platform, random, arrow
CFG   = {'plat': sys.platform[:3]+"-"+os.path.expanduser('~').split("\\")[-1].split("/")[-1], "ver": sys.version_info.major}
DIRCWD= { 'win-asus1': 'D:/_devs/Python01/project27/', 'win-unerry': 'G:/_devs/project27/' , 'lin-noel': '/home/noel/project27/', 'lin-ubuntu': '/home/ubuntu/project27/' }[CFG['plat']]
os.chdir(DIRCWD)

__path__= DIRCWD +'/aapackage/'
EC2CWD=   '/home/ubuntu/notebook/'
####################################################################################################




#################### Batch processing  ###############################################################

########### WAIT BEFORE LAUNCH #########################################################################
def os_wait_cpu(priority=300, cpu_min=50) :
 from time import sleep; import psutil, arrow
 aux= psutil.cpu_percent()
 while aux > cpu_min :
    print("CPU:", aux, arrow.utcnow().to('Japan').format())
    sleep( priority );  aux=  psutil.cpu_percent(); sleep(10); aux= 0.5*(aux + psutil.cpu_percent() )
 print("Starting script:", aux, arrow.utcnow().to('Japan').format())
#######################################################################################################












#####################################################################################
 #-------- Python General---------------------------------------------------------
#  Import File
# runfile('D:/_devs/Python01/project27/stockMarket/google_intraday.py', wdir='D:/_devs/Python01/project27/stockMarket')

def isexist(a) :
 try:
   a; return True
 except Exception as e:
  print(e);  return False

def isfloat(x):
  try:
   v=   False if x == np.inf else True
   float(x)
   return v
  except :    return False

def isint(x): return isinstance(x, ( int, np.int8, np.int16, np.int, np.int64, np.int32 ) )

def a_isanaconda():
 import sys;
 txt= sys.version
 if txt.find('Continuum') > 0 : return True
 else: return False



###################### OS- ######################################################################
def os_zip_checkintegrity(filezip1):
  import zipfile
  zip_file = zipfile.ZipFile(filezip1)
  try :
    ret= zip_file.testzip()
    if ret is not None:
        print("First bad file in zip: %s" % ret)
        return  False
    else: return True
  except Exception as e : return False

def os_zipfile(folderin, folderzipname, iscompress=True):
  import os, zipfile
  compress = zipfile.ZIP_DEFLATED if iscompress else  zipfile.ZIP_STORED
  zf= zipfile.ZipFile(folderzipname, "w", compress, allowZip64=True)

  for dirname, subdirs, files in os.walk(folderin):
        zf.write(dirname)
        for filename in files:  zf.write(os.path.join(dirname, filename))
  zf.close()

  r= os_zip_checkintegrity(folderzipname)
  if r : return  folderin, folderzipname
  else :
     print('Corrupt File')
     return  folderin, False

def os_zipfolder(dir_tozip='/zdisks3/output', zipname='/zdisk3/output.zip', dir_prefix=None, iscompress=True):
 '''
 shutil.make_archive('/zdisks3/results/output', 'zip',
                     root_dir=/zdisks3/results/',
                     base_dir='output')

 os_zipfolder('zdisk/test/aapackage', 'zdisk/test/aapackage.zip', 'zdisk/test')'''

 dir_tozip= dir_tozip if dir_tozip[-1] != '/' else dir_tozip[:-1]
 # dir_prefix= dir_prefix if dir_prefix[-1] != '/' else dir_prefix[:-1]

 if dir_prefix is None :
   dir_tozip, dir_prefix= '/'.join(dir_tozip.split('/')[:-1]), dir_tozip.split('/')[-1]

 import shutil
 shutil.make_archive(zipname.replace('.zip',''), 'zip', dir_tozip, base_dir=dir_prefix)


 '''
 shutil.make_archive(base_name, format[, root_dir[, base_dir[, verbose[, dry_run[, owner[, group[, logger]]]]]]])
Create an archive file (eg. zip or tar) and returns its name.

base_name is the name of the file to create, including the path, minus any format-specific extension. format is the archive format: one of “zip” (if the zlib module or external zip executable is available), “tar”, “gztar” (if the zlib module is available), or “bztar” (if the bz2 module is available).

root_dir is a directory that will be the root directory of the archive; ie. we typically chdir into root_dir before creating the archive.

base_dir is the directory where we start archiving from; ie. base_dir will be the common prefix of all files and directories in the archive.

root_dir and base_dir both default to the current directory.

owner and group are used when creating a tar archive. By default, uses the current owner and group.

logger must be an object compatible with PEP 282, usually an instance of logging.Logger.
  '''

 '''
 import zipfile
 compress = zipfile.ZIP_DEFLATED if iscompress else  zipfile.ZIP_STORED
 zf= zipfile.ZipFile(zipname, "w", compress, allowZip64=True)
 for dirname, subdirs, files in os.walk(dir_tozip):
    zf.write(dirname)
    for filename in files:  zf.write(os.path.join(dirname, filename))
 zf.close()
 '''
 r= os_zip_checkintegrity(zipname)
 if r : return  zipname
 else :
    print('Corrupt File'); return False

def os_zipextractall(filezip_or_dir="folder1/*.zip", tofolderextract='zdisk/test', isprint=1):
   '''os_zipextractall( 'aapackage.zip','zdisk/test/'      )  '''
   import zipfile

   if filezip_or_dir.find("*") > - 1 :    #  Many Zip
      ziplist1= os_file_listall(filezip_or_dir[:filezip_or_dir.find("*")],'*.zip')
      fileziplist_full=   ziplist1[2]

   else :                                 # Only 1
    filename=    os_file_getname(filezip_or_dir)
    fileziplist_full= [filezip_or_dir]


   # if os.path.exists( foldernew2  ) :      #Either File or Folder exists
   #   print('Renaming Folder ' + foldernew + ' with _')
   #   # lastfolder, beforelast= tofolderextract.split('/')[]
   #   os_folder_copy(foldernew2,  foldernew2+'_' )


   for filezip in fileziplist_full :
     filezip_name=    os_file_getname(filezip)
     zip_ref = zipfile.ZipFile(filezip, 'r')
     zip_ref.extractall(tofolderextract)   #Will create the path
     zip_ref.close()
     isok= os.path.exists( tofolderextract  )
     if not isok : print('Error: ' + filezip_name)

   if isok : return  tofolderextract
   else :    return -1

def os_folder_copy(src, dst, symlinks=False, pattern1="*.py", fun_file_toignore=None):
   '''
       callable(src, names) -> ignored_names
       'src' parameter, which is the directory being visited by copytree(), and
       'names' which is the list of `src` contents, as returned by os.listdir():

    Since copytree() is called recursively, the callable will be called once for each directory that is copied.
    It returns a  list of names relative to the `src` directory that should not be copied.
   '''
   import shutil, errno, fnmatch
   def fun_file_toignore(src, names) :
       pattern= '!' + pattern1
       file_toignore= fnmatch.filter(names, pattern)
       return file_toignore

   try:
        shutil.copytree(src, dst, symlinks=False, ignore=fun_file_toignore)
   except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst,symlinks=False, ignore=fun_file_toignore)
        else: raise

def os_folder_create(directory) :
   DIR0= os.getcwd()
   if not os.path.exists(directory): os.makedirs(directory)
   os.chdir(DIR0)

def os_folder_robocopy(from_folder='', to_folder='', my_log='H:/robocopy_log.txt'):
    """
    Copy files to working directory
    robocopy <Source> <Destination> [<File>[ ...]] [<Options>]
    We want to copy the files to a fast SSD drive
    """
    if os.path.isdir(from_folder) &  os.path.isdir(to_folder):
        os.subprocess.call(["robocopy", from_folder, to_folder, "/LOG:%s" % my_log])
    else:
        print("Paths not entered correctly")

def os_file_replacestring1(findStr, repStr, filePath):
    "replaces all findStr by repStr in file filePath"
    import fileinput
    file1= fileinput.FileInput(filePath, inplace=True, backup='.bak')
    for line in file1:
       line= line.replace(findStr,  repStr)
       sys.stdout.write(line)
    file1.close()

    print("OK: "+format(filePath))

def os_file_replacestring2(findstr, replacestr, some_dir, pattern="*.*", dirlevel=1  ):
  ''' #fil_replacestring_files("logo.png", "logonew.png", r"D:/__Alpaca__details/aiportfolio",    pattern="*.html", dirlevel=5  )
  '''
  list_file= os_file_listall(some_dir, pattern=pattern, dirlevel=dirlevel)
  list_file= list_file[2]
  for file1 in list_file : os_file_replacestring1(findstr, replacestr, file1)

def os_file_getname(path):
    import ntpath
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def os_file_getpath(path):
    import ntpath
    head, tail = ntpath.split(path)
    return head

def os_file_gettext(file1):
 with open(file1, 'r',encoding='UTF-8',) as f:
  return f.read()
 #def os_file_listall(some_dir, pattern="*.*", dirlevel=1):
 #  return listallfile(some_dir, pattern=pattern, dirlevel=dirlevel)

def os_file_listall(dir1, pattern="*.*", dirlevel=1, onlyfolder=0):
  '''
   # DIRCWD=r"D:\_devs\Python01\project"
   # aa= listallfile(DIRCWD, "*.*", 2)
   # aa[0][30];   aa[2][30]

   :param dir1:
   :param pattern:
   :param dirlevel:
   :param onlyfolder:
   :return:
  '''
  import fnmatch; import os; import numpy as np;  matches = []
  dir1 = dir1.rstrip(os.path.sep)
  num_sep = dir1.count(os.path.sep)

  if onlyfolder :
   for root, dirs, files in os.walk(dir1):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([])  # Filename, DirName
    for dirs in fnmatch.filter(dirs, pattern):
      matches[0].append(os.path.splitext(dirs)[0])
      matches[1].append(os.path.splitext(root)[0])
      matches[2].append(os.path.join(root, dirs))
   return np.array(matches)

  for root, dirs, files in os.walk(dir1):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([])  # Filename, DirName
    for files in fnmatch.filter(files, pattern):
      matches[0].append(os.path.splitext(files)[0])
      matches[1].append(os.path.splitext(files)[1])
      matches[2].append(os.path.join(root, files))
  return np.array(matches)

def os_file_rename(some_dir, pattern="*.*", pattern2="", dirlevel=1):
  import fnmatch; import os; import numpy as np; import re;  matches = []
  some_dir = some_dir.rstrip(os.path.sep)
  assert os.path.exists(some_dir)
  num_sep = some_dir.count(os.path.sep)
  for root, dirs, files in os.walk(some_dir):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([])  # Filename, DirName
    for files in fnmatch.filter(files, pattern):

     #replace pattern by pattern2
      nfile= re.sub(pattern, pattern2, files)
      os.path.abspath(root)
      os.rename(files, nfile)

      matches[0].append(os.path.splitext(nfile)[0])
      matches[1].append(os.path.splitext(nfile)[1])
      matches[2].append(os.path.join(root, nfile))
  return np.array(matches).T


def os_print_tofile(vv, file1,  mode1='a'):  # print into a file='a
    '''
    Here is a list of the different modes of opening a file:
r
Opens a file for reading only. The file pointer is placed at the beginning of the file. This is the default mode.

rb

Opens a file for reading only in binary format. The file pointer is placed at the beginning of the file. This is the default mode.
r+

Opens a file for both reading and writing. The file pointer will be at the beginning of the file.
rb+

Opens a file for both reading and writing in binary format. The file pointer will be at the beginning of the file.
w

Opens a file for writing only. Overwrites the file if the file exists. If the file does not exist, creates a new file for writing.
wb

Opens a file for writing only in binary format. Overwrites the file if the file exists. If the file does not exist, creates a new file for writing.
w+

Opens a file for both writing and reading. Overwrites the existing file if the file exists. If the file does not exist, creates a new file for reading and writing.
wb+

Opens a file for both writing and reading in binary format. Overwrites the existing file if the file exists. If the file does not exist, creates a new file for reading and writing.
a

Opens a file for appending. The file pointer is at the end of the file if the file exists. That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.
ab

Opens a file for appending in binary format. The file pointer is at the end of the file if the file exists. That is, the file is in the append mode. If the file does not exist, it creates a new file for writing.
a+

Opens a file for both appending and reading. The file pointer is at the end of the file if the file exists. The file opens in the append mode. If the file does not exist, it creates a new file for reading and writing.
ab+

Opens a file for both appending and reading in binary format. The file pointer is at the end of the file if the file exists. The file opens in the append mode. If the file does not exist, it creates a new file for reading and writing.
To open a text file, use:
fh = open("hello.txt", "r")

To read a text file, use:
print fh.read()

To read one line at a time, use:
print fh.readline()

To read a list of lines use:
print fh.readlines()

To write to a file, use:
fh = open("hello.txt", "w")
lines_of_text = ["a line of text", "another line of text", "a third line"]
fh.writelines(lines_of_text)
fh.close()

To append to file, use:
fh = open("Hello.txt", "a")
fh.close()

    '''
    with open(file1, mode1) as text_file:  text_file.write(str(vv))


def a_get_pythonversion() :
   return 3

def os_path_norm(pth): #Normalize path for Python directory
    ''' #r"D:\_devs\Python01\project\03-Connect_Java_CPP_Excel\PyBindGen\examples" '''
    if a_get_pythonversion()==2:
     ind = pth.find(":")
     if ind > -1 :
       a, b = pth[:ind], pth[ind + 1:].encode("string-escape").replace("\\x", "/")
       return "{}://{}".format(a, b.lstrip("\\//").replace("\\\\", "/"))
     else: return pth
    else:
      pth = pth.encode("unicode-escape").replace(b"\\x", b"/")
      return pth.replace(b"\\\\", b"/").decode("utf-8")

def os_path_change(path1): path1= os_path_norm(path1); os.chdir(path1)    #Change Working directory path

def os_path_current(): return DIRCWD

def os_file_exist(file1): return os.path.exists(file1)

def os_file_size(file1): return os.path.getsize(file1)

def os_file_read(file1):
 fh = open(file1,"r")
 return fh.read()

def os_file_mergeall(nfile, dir1, pattern1, deepness=2):
 ll= os_file_listall(dir1,pattern1,deepness)
 with open(nfile, mode='a', encoding='UTF-8') as nfile1:
  txt=''; ii=0
  for l in ll[2]:
   txt= '\n\n\n\n' +  os_file_gettext( l)
   nfile1.write(txt)
 nfile1.close()



def os_path_append(p1, p2=None, p3=None, p4=None):
 sys.path.append(p1)
 if p2 is not None:  sys.path.append(p2)
 if p3 is not None:  sys.path.append(p3)
 if p4 is not None:  sys.path.append(p4)



 #-----Get Documentation of the module
def os_split_dir_file(dirfile) :
    lkey= dirfile.split('/')
    if len(lkey)== 1 :dir1=""
    else :            dir1= '/'.join(lkey[:-1]); dirfile= lkey[-1]
    return dir1, dirfile



#######################  Python Interpreter ###################################################
def py_importfromfile(modulename, dir1):
 #Import module from file:  (Although this has been deprecated in Python 3.4.)
 vv= a_get_pythonversion()
 if vv==3:
  from importlib.machinery import SourceFileLoader
  foo = SourceFileLoader("module.name", "/path/to/file.py").load_module()
  foo.MyClass()
 elif vv==2 :
  import imp
  foo = imp.load_source('module.name', '/path/to/file.py')
  foo.MyClass()

def py_memorysize(o, ids, hint=" deep_getsizeof(df_pd, set()) "):
    """ deep_getsizeof(df_pd, set())
    Find the memory footprint of a Python object
    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    """
    from collections import Mapping, Container
    from sys import getsizeof

    d = py_memorysize
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        r= r

    if isinstance(o, Mapping):
        r= r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        r=  r + sum(d(x, ids) for x in o)

    return r * 0.0000001


def save(obj, folder='/folder1/keyname', isabsolutpath=0 ) :
 return py_save_obj(obj, folder=folder, isabsolutpath=isabsolutpath)

def load(folder='/folder1/keyname', isabsolutpath=0 ) :
 return py_load_obj(folder=folder, isabsolutpath=isabsolutpath)

def save_test(folder='/folder1/keyname', isabsolutpath=0  ) :
 ztest=  py_load_obj(folder=folder, isabsolutpath=isabsolutpath)
 print('Load object Type:' + str(type(ztest)) )
 del ztest; gc.collect()


def py_save_obj(obj, folder='/folder1/keyname', isabsolutpath=0):
    import pickle
    if isabsolutpath==0  and folder.find('.pkl') == -1 : #Local Path
       dir0, keyname= z_key_splitinto_dir_name(folder)
       os_folder_create(DIRCWD+'/aaserialize/' + dir0)
       dir1= DIRCWD+'/aaserialize/' + dir0 + '/'+ keyname + '.pkl'
    else :
       dir1= folder

    with open( dir1, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    return dir1

def py_load_obj(folder='/folder1/keyname', isabsolutpath=0, encoding1='utf-8'):
    '''def load_obj(name, encoding1='utf-8' ):
         with open('D:/_devs/Python01/aaserialize/' + name + '.pkl', 'rb') as f:
            return pickle.load(f, encoding=encoding1)
    '''
    import pickle
    if isabsolutpath==0 and folder.find('.pkl') == -1 :
       dir0, keyname= z_key_splitinto_dir_name(folder)
       os_folder_create(DIRCWD+'/aaserialize/' + dir0)
       dir1= DIRCWD+'/aaserialize/' + dir0 + '/'+ keyname + '.pkl'
    else :
       dir1= folder

    with open(dir1, 'rb') as f:
        return pickle.load(f)

def z_key_splitinto_dir_name(keyname) :
    lkey= keyname.split('/')
    if len(lkey)== 1 :dir1=""
    else :            dir1= '/'.join(lkey[:-1]); keyname= lkey[-1]
    return dir1, keyname





