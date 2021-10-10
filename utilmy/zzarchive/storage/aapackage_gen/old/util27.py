# -*- coding: utf-8 -*-
#---------Various Utilities function for Python----------------------------
import os;
import sys ;

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np;
import pandas as pd;

import global01 as global01 #as global varaibles   global01.varname


#--------------------Utilities---------------------------
# http://win32com.goermezer.de/content/blogsection/7/284/


#####################################################################################
#-------- Python-------------------------------------------------------------------
def getmodule_doc(module1, fileout=''):
  from automaton import codeanalysis as ca
  ca.getmodule_doc(module1, fileout)
  
#  getmodule_doc("jedi")

''' Convert Python 2 to Python 3
import lib2to3

!2to3 D:\_devs\Python01\project\aapackage\codeanalysis.py


D:\_devs\Python01\project\zjavajar

'''


def help2():
 str= """ installfromgit:    
  !pip install https://github.com/pymc-devs/pymc3/archive/master.zip 
   
  !pip install  https://github.com/tcalmant/jpype-py3 /zipball/master  
                                                      /tarball/master   

  def pip1(name1):   !pip install {name1}  #install package
    
  """
 print( str)


def pythonversion():    return sys.version_info[0]

# os.system('cd D:\_app\visualstudio13\Common7\IDE') #Execute any command    
# os.path.abspath('memo.txt') #get absolute path
# os.path.exists('memo.txt')
# os.path.isdir('memo.txt')
#  os.getenv('PATH')    #Get environ variable  


    
#Import module from file:  (Although this has been deprecated in Python 3.4.)
def importfromfile(modulename, dir1): 
 vv= pythonsversion()
 if vv==3:
  from importlib.machinery import SourceFileLoader
  foo = SourceFileLoader("module.name", "/path/to/file.py").load_module()
  foo.MyClass()
 elif vv==2 :
  import imp
  foo = imp.load_source('module.name', '/path/to/file.py')
  foo.MyClass()


# runfile('D:/_devs/Python01/project27/stockMarket/google_intraday.py', wdir='D:/_devs/Python01/project27/stockMarket')


def isfloat(value):
  try:    float(value);    return True
  except :    return False


#####################################################################################
#-------- FILE-------------------------------------------------------------------

def listallfile(some_dir, pattern="*.*", dirlevel=1):
  import fnmatch; import os; import numpy as np;  matches = []
  some_dir = some_dir.rstrip(os.path.sep)
  assert os.path.isdir(some_dir)
  num_sep = some_dir.count(os.path.sep)
  for root, dirs, files in os.walk(some_dir):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([]);   # Filename, DirName
    for files in fnmatch.filter(files, pattern):
      matches[0].append(os.path.splitext(files)[0])   
      matches[1].append(os.path.splitext(files)[1])  
      matches[2].append(os.path.join(root, files))   
  return np.array(matches).T              
     
# DIRCWD=r"D:\_devs\Python01\project"
# aa= listallfile(DIRCWD, "*.*", 2)
# aa[0][30];   aa[1][30]


def renamefile(some_dir, pattern="*.*", pattern2="", dirlevel=1):
  import fnmatch; import os; import numpy as np; import re;  matches = []
  some_dir = some_dir.rstrip(os.path.sep)
  assert os.path.isdir(some_dir)
  num_sep = some_dir.count(os.path.sep)
  for root, dirs, files in os.walk(some_dir):
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    matches.append([]); matches.append([]); matches.append([]);   # Filename, DirName
    for files in fnmatch.filter(files, pattern):
        
     #replace pattern by pattern2        
      nfile= re.sub(pattern, pattern2, files) 
      os.path.abspath(root)
      os.rename(files, nfile)
 
      matches[0].append(os.path.splitext(nfile)[0])   
      matches[1].append(os.path.splitext(nfile)[1])  
      matches[2].append(os.path.join(root, nfile))   
  return np.array(matches).T     


def printinfile(vv, file1):   #print into a file
 with open(file1, "w") as text_file:  text_file.write(str(vv))   
    
def normpath(pth): #Normalize path for Python directory
    if pythonversion()==2: 
     ind = pth.index(":")
     a, b = pth[:ind], pth[ind + 1:].encode("string-escape").replace("\\x", "/")
     return "{}://{}".format(a, b.lstrip("\\//").replace("\\\\", "/"))        
    else: 
      pth = pth.encode("unicode-escape").replace(b"\\x", b"/")
      return pth.replace(b"\\\\", b"/").decode("utf-8")
#r"D:\_devs\Python01\project\03-Connect_Java_CPP_Excel\PyBindGen\examples"
   
     
def changepath(path1): path1= normpath(path1); os.chdir(path1)    #Change Working directory path

def currentpath(): return os.getcwd()   

def isfileexist(file1): return os.path.exists(file1)

def filesize(file1): return os.path.getsize(file1)





#####################################################################################
#-------- XML / HTML processing ------------------------------------------------------
'''
https://pypi.python.org/pypi/RapidXml/
http://pugixml.org/benchmark.html


'''







#####################################################################################
#--------CSV processing ------------------------------------------------------
#Put Excel and CSV into Database / Extract CSV from database








#####################################################################################
#-------- STRING--------------------------------------------------------------------
def empty_string_array(size):
 return ["" for x in range(size)]



def reindent(s, numSpaces): #change indentation of multine string
    s = string.split(s, '\n')
    s = [(numSpaces * ' ') + string.lstrip(line) for line in s]
    s = string.join(s, '\n')
    return s

'''
   if args:
       aux= name1+'.'+obj.__name__ +'('+ str(args) +')  \n' + str(inspect.getdoc(obj))
       aux= aux.replace('\n', '\n       ') 
       aux= aux.rstrip()
       aux= aux + ' \n'
       wi( aux)
'''


'''
http://www.ibm.com/developerworks/library/l-pyint/


>>> dir('this is a string')
['__add__', '__class__', '__contains__', '__delattr__', '__doc__', '__eq__',
'__ge__', '__getattribute__', '__getitem__', '__getslice__', '__gt__',
'__hash__', '__init__', '__le__', '__len__', '__lt__', '__mul__', '__ne__',
'__new__', '__reduce__', '__repr__', '__rmul__', '__setattr__', '__str__',
'capitalize', 'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'replace', 'rfind',
'rindex', 'rjust', 'rstrip', 'split', 'splitlines', 'startswith', 'strip',
'swapcase', 'title', 'translate', 'upper', 'zfill']


spam = spam.upper()
spam = spam.lower()
spam.upper().lower()
spam.startswith('Hello')
spam.endswith('world!')

spam.strip()
spam.lstrip()
spam.rstrip()


	
capitalize()   Capitalizes first letter of string

center(width, fillchar)  Returns space-padded string with string centered  w

count(str, beg= 0,end=len(string))
Counts how many times str occurs in string or in a substring of



decode(encoding='UTF-8',errors='strict')
Decodes the string using the codec registered for encoding. encoding defaults to the default string encoding.
	
encode(encoding='UTF-8',errors='strict')
Returns encoded string version of string; on error, default is to raise a ValueError unless errors is given with 'ignore' or 'replace'.


endswith(suffix, beg=0, end=len(string))
Determines if string or a substring of string (if starting index beg and ending index end are given) ends with suffix; returns true if so and false otherwise.


expandtabs(tabsize=8)
Expands tabs in string to multiple spaces; defaults to 8 spaces per tab if tabsize not provided.
8	
find(str, beg=0 end=len(string))
Determine if str occurs in string or in a substring of string if starting index beg and ending index end are given returns index if found and -1 otherwise.


9	
index(str, beg=0, end=len(string))
Same as find(), but raises an exception if str not found.

10	
isalnum()
Returns true if string has at least 1 character and all characters are alphanumeric and false otherwise.
11	
salpha()


Returns true if string has at least 1 character and all characters are alphabetic and false otherwise.

	
isdigit()
Returns true if string contains only digits and false otherwise.

islower()
Returns true if string has at least 1 cased character and all cased characters are in lowercase and false otherwise.
14	
isnumeric()


Returns true if a unicode string contains only numeric characters and false otherwise.
15	
isspace()


Returns true if string contains only whitespace characters and false otherwise.
16	
istitle()


Returns true if string is properly "titlecased" and false otherwise.
17	
isupper()


Returns true if string has at least one cased character and all cased characters are in uppercase and false otherwise.
18	
join(seq)


Merges (concatenates) the string representations of elements in sequence seq into a string, with separator string.
19	
len(string)


Returns the length of the string
20	
ljust(width[, fillchar])


Returns a space-padded string with the original string left-justified to a total of width columns.
21	
lower()


Converts all uppercase letters in string to lowercase.
22	
lstrip()


Removes all leading whitespace in string.
23	
maketrans()


Returns a translation table to be used in translate function.
24	
max(str)
Returns the max alphabetical character from the string str.

25	
min(str)
Returns the min alphabetical character from the string str.

26	
replace(old, new [, max])
Replaces all occurrences of old in string with new or at most max occurrences if max given.
27	
rfind(str, beg=0,end=len(string))


Same as find(), but search backwards in string.
28	
rindex( str, beg=0, end=len(string))
Same as index(), but search backwards in string.

29	
rjust(width,[, fillchar])


Returns a space-padded string with the original string right-justified to a total of width columns.




31	
split(str="", num=string.count(str))


Splits string according to delimiter str (space if not provided) and returns list of substrings; split into at most num substrings if given.
32	
splitlines( num=string.count('\n'))


Splits string at all (or num) NEWLINEs and returns a list of each line with NEWLINEs removed.
33	
startswith(str, beg=0,end=len(string))


Determines if string or a substring of string (if starting index beg and ending index end are given) starts with substring str; returns true if so and false otherwise.
34	
strip([chars])


Performs both lstrip() and rstrip() on string
35	
swapcase()


Inverts case for all letters in string.
36	
title()


Returns "titlecased" version of string, that is, all words begin with uppercase and the rest are lowercase.
37	
translate(table, deletechars="")


Translates string according to translation table str(256 chars), removing those in the del string.
38	
upper()


Converts lowercase letters in string to uppercase.
39	
zfill (width)


Returns original string leftpadded with zeros to a total of width characters; intended for numbers, zfill() retains any sign given (less one zero).
40	
isdecimal()


Returns true if a unicode string contains only decimal characters and false otherwise.

'''







#####################################################################################
#-------- PANDA UTIL----------------------------------------------------------------
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def remove_zeros(vv, axis1=1):
   return vv[~np.all(vv == 0, axis=axis1)]

def sort_array(vv): 
 return vv[np.lexsort(np.transpose(vv)[::-1])]      #Sort the array by different column


def save_topanda(vv, filenameh5):  # 'E:\_data\_data_outlier.h5'   
 store = pd.HDFStore(filenameh5)  
 pdf =pd.DataFrame(vv); store.append('data', pdf); store.close()  


def load_frompanda(filenameh5):  # 'E:\_data\_data_outlier.h5'
 pdf=  pd.read_hdf(fileoutlier,'data')    #from file
 return pdf.values   #to numpy vector
















#####################################################################################
###########################-Utilities for Numerical Calc---------------------------


#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_vect_calc(filename, expr, i0=0, imax=1000, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe    #to numpy vector
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx  
# filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'   
 store = pd.HDFStore(fileout) 
 store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):


#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_topanda(filename, expr,  i0=0, imax=1000, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe   
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx    # filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5' 
 store = pd.HDFStore(fileout);  store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):


def textvect_topanda(vv, fileout=""):
 pd= pd.DataFrame(vv);  st= pd.HDFStore(fileout);  st.append('data', pd); del pd    




#----Input the data: From CSV to Panda files -------------------------------
def csv_topanda(filein1, filename, tablen='data'):
 #filein1=   'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.csv'
 #filename = 'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.h5'
 chunksize =     10 * 10 ** 6
 list01= pd.read_csv(filein1, chunksize=chunksize, lineterminator=',')
 for chunk in list01:
     store = pd.HDFStore(filename);     
     store.append(tablen, chunk);     store.close()     
 del chunk


#---LOAD Panda Vector----------------------------------------
def getpanda_tonumpy(filename, nsize, tablen='data'):
 pdframe=  pd.read_hdf(filename, tablen, start=0, stop=(nsize))
 return pdframe.values   #to numpy vector       


def getrandom_tonumpy(filename, nbdim, nbsample, tablen='data'):
 pdframe=  pd.read_hdf(filename,tablen, start=0, stop=(nbdim*nbsample))
 return pdframe.values   #to numpy vector       



# yy1= getrandom_tonumpy('E:\_data\_QUASI_SOBOL_gaussian_xx2.h5', 16384, 4096)
#-------------=---------------------------------------------------------------
#----------------------------------------------------------------------------









#-------------=---------------------------------------------------------------
#---------------------Statistics----------------------------------------------
#Calculate Co-moment of xx yy
def comoment(xx,yy,nsample, kx,ky) :
#   cx= ne.evaluate("sum(xx)") /  (nsample);   cy= ne.evaluate("sum( yy)")  /  (nsample)
#   cxy= ne.evaluate("sum((xx-cx)**kx * (yy-cy)**ky)") / (nsample)
   cxy= ne.evaluate("sum((xx)**kx * (yy)**ky)") / (nsample)
   return cxy 


#Autocorrelation 
def acf(data):
    n = len(data)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return acf_lag
    x = np.arange(n) # Avoiding lag 0 calculation
    acf_coeffs = np.asarray(list(map(r, x)))
    return acf_coeffs
 
 
#-------------=---------------------------------------------------------------
#-----------------------------------------------------------------------------

 





#-----------Plot Save-------------------------------------------------------
def plotsave(xx,yy,title1=""):
  plt.scatter(xx, yy, s=1 )
  plt.autoscale(enable=True, axis='both', tight=None)
#  plt.axis([-3, 3, -3, 3])  #gaussian

  tit1= title1+str(nsample)+' smpl D_'+str(dimx)+' X D_'+str(dimy)
  plt.title(tit1)
  plt.savefig('_img/'+tit1+'.jpg',dpi=100)
  plt.clf()


def plotshow(xx,yy,title1=""):
  plt.scatter(xx, yy, s=1 )
  plt.autoscale(enable=True, axis='both', tight=None)
#  plt.axis([-3, 3, -3, 3])  #gaussian

  tit1= title1+str(nsample)+' smpl D_'+str(dimx)+' X D_'+str(dimy)
  plt.title(tit1)
  plt.show()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------












'''
#--------------Installation procedure------------------------------
!pip install pipwin

!pipwin install pycuda



#--------------From Git package----------------------------------
!pip install  https://github.com/tcalmant/jpype-py3/zipball/master
                                                   /tarball/master





Install from Github:

http://stackoverflow.com/questions/8247605/configuring-so-that-pip-install-can-work-from-github/8382819#8382819


git clone http://github.com/facebook/python-sdk.git

PYLEARN:
!pip install https://github.com/lisa-lab/pylearn2/zipball/master

GIT in Command Line:
import subprocess subprocess.call(["git", "pull"]) 
subprocess.call(["make"]) 
subprocess.call(["make", "test"])


Command Line in Python:
    sts = Popen("mycmd" + " myarg", shell=True).wait()
    subprocess.Popen() is strict superset of os.system()




'''










'''
changepath(DIRCWD)


DIRCWD = "D:\_devs\Python01\project\\03-Connect_Java_CPP_Excel\PyBindGen\examples"
dirs = os.listdir(DIRCWD)

    
os.path.abspath(DIRCWD)

os.fsdecode(DIRCWD)

os.getcwd()   #current folder
    


from win32com.ec2.gencache import EnsureDispatch
 
for w in EnsureDispatch("Shell.Application").Windows():
    print(w.LocationName + "=" + w.LocationURL)


We can place additional strings after the Python file 
and access those command line arguments in our Python program.
 Here is a simple program that demonstrates reading arguments from the command line:

import sys
print 'Count:', len(sys.argv)
print 'Type:', type(sys.argv)
for arg in sys.argv:
   print 'Argument:', arg
   
   
   
'''
   
   
'''
How do I have to configure so that I don't have to type python script.py but simply script.py in CMD on Windows?

I added my python directory to %PATH% that contains python.exe but still scripts are not run correctly.
C:\> assoc .py=Python
C:\> ftype Python="C:\python27\python.exe %1 %*"
Or whatever the relevant path is - you can also set command line args using ftype.

In order to make a command recognized without having to give the suffix (.py), similar to how it works for .exe files, add .py to the semi-colon separated list of the (global) PATHEXT variable.





#------------Command Line in Python Script------------------------------
#https://ipython.org/ipython-doc/3/interactive/magics.html

import os

#------!Shell_command_exe : execute the Shell in Ipython-----------------
!pip  install    packageName  #install package


directory_name= ""
!cd {python_var_name}    #change folder


cd ..  #jumpe one folder up
cd ..\  #jumpe one folder down

os.chdir(DIRCWD)


DIRCWD = r" D:\_devs\Python01\project\03-Connect_Java_CPP_Excel\PyBindGen\examples"

dir2= os.path.abspath(DIRCWD)

os.chdir(DIRCWD)


dir2= os.fsdecode(DIRCWD)
!cd {dir2}


from os.path import expanduser
home = expanduser("~")
os.fsdecode(DIRCWD)

for i in range(3):
       !echo {i+1}

%run script.py  #Run the Script
%timeit [x*x for x in range(1000)]  #Benchmark time



'''








###############################################################################
########################### COMPILING UTILITIES ################################
def compileVSsolution(dir1, flags1="", type1="devenv", compilerdir=""):
  if type=="devenv":  #devenv
    if compilerdir1=="" :  compilerdir1= global01.VS_DEVENV
    cmd= compilerdir1 +" "+ dir1 + " " + flags1
    os.system(cmd)
    
  if type=="msbuild":  #msbuild
    if compilerdir1=="" :  compilerdir1= global01.VS_MSBUILD
    cmd= compilerdir1 +" "+ dir1 + " " + flags1


dirsol= r"D:\_devs\_github\cudasamples\a7_CUDALibraries\_full_rand_mersenne_sobol\rand_mt_sobol_VS2013.sln"

dirsol= r"D:\_devs\_github\cudasamples\7_CUDALibraries\_full_rand_mersenne_sobol\rand_mt_sobol_VS2013.sln"

dir1= normpath(dirsol)
compilerdir1= global01.VS_DEVENV
compilerdir1= "devenv "

flags1= " /p:Configuration=Debug"
cmd1= compilerdir1 +" "+ dir1 + " " + flags1

os.system(cmd1)



compileVSsolution(dirsol, "Release")

# Sample code snippet.. https://heejune.me/2015/12/08/writing-a-python-build-script-for-your-visual-c-project/
def VS_start(self, version):

        msbuild = os.getenv('MSBUILD_PATH', r"C:\Program Files\MSBuild\12.0\Bin\MSBuild.exe")
        project_output_dir = os.getenv('PROJECT_OUTPUT_DIR', r'c:\Build_distribute\\')

        if not os.path.exists(msbuild):   raise Exception('not found ' + msbuild)

        projects = [r"..\yoursolution.sln", r"..\yourproject\yourproject.vcxproj"]
        win32_targets = '/t:ProjectA:rebuild;ProjectB:rebuild;ProjectC:rebuild'
        x64_targets = '/t:ProjectA:rebuild;ProjectB:rebuild;ProjectC:rebuild'

        rebuild = '/t:Rebuild'
        debug = '/p:Configuration=Debug'
        release = '/p:Configuration=Release'
        x64 = '/p:Platform=x64'
        win32 = '/p:Platform=Win32'
        xp_toolset = '/p:PlatformToolset=v110/v100/v90'

        #msbuild %s.vcxproj /t:rebuild /p:configuration=VC90Release,platform=%s
        #msbuild myproject.vcxproj /p:PlatformToolset=v90 /t:rebuild
        #msbuild myproject.vcxproj /p:PlatformToolset=v110_xp /t:rebuild

        # making command line to run
        default = [msbuild]
        default.append('/m:1')  # https://msdn.microsoft.com/en-us/library/ms164311.aspx

        libs = default [:]
        libs.append(projects[0])    # append a project/solution name to build command-line

        if self.build_arch_target == 'x86':
            default.append(win32)
            # win32 targets
            default.append(win32_targets)


def VS_build(self, lib_to_build):
        build_result = False
        print ('****** Build Start, Target Platform: ' + self.build_arch_target)
        print('configuration: ' + self.build_type)

        process = subprocess.Popen(args = lib_to_build, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        while True:
            nextline = process.stdout.readline()
            if nextline == b'' and process.poll() != None: break
            sys.stdout.write(nextline.decode('cp949'))      # adjust the codepage for your console
            sys.stdout.flush()

        output = process.communicate()[0]
        exitCode = process.returncode

        if (exitCode == 0):
            build_result = True
            pass    #return output
        else:
            build_result = False
            #raise Exception(command, exitCode, output)
        print('**build finished %d ' % process.returncode)
        return build_result

# VS_build(DIRCWD)



# source taken from
# http://stackoverflow.com/questions/4003725/modifying-rc-file-with-python-regexp-involved
#How will you specify the appropriate version for your *.rc version resource file?
# target_version =  "2,3,4,5"

def set_rc_version(rcfile, target_version):
    with open(rcfile, "r+") as f:
        rc_content = f.read()

        # first part
        #FILEVERSION 6,0,20,163         
        #PRODUCTVERSION 6,0,20,163
        #...

        # second part
        #VALUE "FileVersion", "6, 0, 20, 163"
        #VALUE "ProductVersion", "6, 0, 20, 163"

        # first part
        regex_1 = re.compile(r"\b(FILEVERSION|FileVersion|PRODUCTVERSION|ProductVersion) \d+,\d+,\d+,\d+\b", re.MULTILINE)

        # second part
        regex_2 = re.compile(r"\b(VALUE\s*\"FileVersion\",\s*\"|VALUE\s*\"ProductVersion\",\s*\").*?(\")", re.MULTILINE)
        
        version = r"\1 " + target_version
        #modified_str = re.sub(regex, version, rc_content)

        pass_1 = re.sub(regex_1, version, rc_content)
        version = re.sub(",", ", ", version) #replacing "x,y,v,z" with "x, y, v, z"
        pass_2 = re.sub(regex_2, r"\g<1>" + target_version + r"\2", pass_1)

        # overwrite
        f.seek(0); f.write(pass_2); f.truncate()


#set_rc_version(r"C:/repo/projectA/resources/win/projectA.rc", "3,4,5,6")





















































































































