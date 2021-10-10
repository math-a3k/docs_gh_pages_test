# -*- coding: utf-8 -*-
#---------Various Utilities function for Python----------------------------
import datetime
import dill
import sys ;

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np;
import pandas as pd;
from bs4 import BeautifulSoup
from numba import jit, float32

import global01 as global01 #as global varaibles   global01.varname

'''
@jit(int32(int32, int32))
numba.float32[:,:,:]
numba.from_dtype(dtype)¶
'''

 #--------------------Utilities---------------------------
# http://win32com.goermezer.de/content/blogsection/7/284/
'''
WhenPackage is destroyed resintall and copy paste the data.
from saved package

!pip install theano_lstm
https://pypi.python.org/pypi/theano-lstm

http://docs.python-guide.org/en/latest/writing/structure/


Packages
Python provides a very straightforward packaging system, which is simply an extension 
of the module mechanism to a directory.

Any directory with an __init__.py file is considered a Python package. 
The different modules in the package are imported in a similar manner as plain modules, 
but with a special behavior for the __init__.py file, which is used to gather all package-wide definitions.

A file modu.py in the directory pack/ is imported with the statement import pack.modu. This statement will look for an __init__.py file in pack, execute all of its top-level statements. Then it will look for a file named pack/modu.py and execute all of its top-level statements. After these operations, any variable, function, or class defined in modu.py is available in the pack.modu namespace.

A commonly seen issue is to add too much code to __init__.py files. When the project complexity grows, there may be sub-packages and sub-sub-packages in a deep directory structure. In this case, importing a single item from a sub-sub-package will require executing all __init__.py files met while traversing the tree.

Leaving an __init__.py file empty is considered normal and even a good practice, if the package’s modules and sub-packages do not need to share any code.

Lastly, a convenient syntax is available for importing deeply nested packages: import very.deep.module as mod. This allows you to use mod in place of the verbose repetition of very.deep.module.


How to use a package:

Now we should be able to use our module once we have it on our Python path. 
You can copy the folder into your Python’s site-packages folder to do this. 
On Windows it’s in the following general location: C:\Python26\Lib\site-packages. 
Alternatively, you can edit the path on the fly in your test code. Let’s see how that’s done:

import sys
 
sys.path.append('C:\Users\mdriscoll\Documents')
import mymath
 
print mymath.add(4,5)
print mymath.division(4, 2)
print mymath.multiply(10, 5)
print mymath.fibonacci(8)
print mymath.squareroot(48)


https://pythonconquerstheuniverse.wordpress.com/2009/10/15/python-packages/



'''

'''
https://service.wi2.ne.jp/wi2net/SbjLogin/2/


I agree with @phyrox, that dill can be used to persist your live objects to disk so you can restart later.  dill can serialize numpy arrays with dump(), and the entire interpreter session with dump_session().

However, it sounds like you are really asking about some form of caching… so I'd have to say that the comment from @Alfe is probably a bit closer to what you want. If you want seamless caching and archiving of arrays to memory… then you want joblib or klepto.

klepto is built on top of dill, and can cache function inputs and outputs to memory (so that calculations don't need to be run twice), and it can seamlessly persist objects in the cache to disk or to a database.

The versions on github are the ones you want. https://github.com/uqfoundation/klepto or https://github.com/joblib/joblib. Klepto is newer, but has a much broader set of caching and archiving solutions than joblib. Joblib has been in production use longer, so it's better tested -- especially for parallel computing.

Here's an example of typical klepto workflow: https://github.com/uqfoundation/klepto/blob/master/tests/test_workflow.py

Here's another that has some numpy in it: https://github.com/uqfoundation/klepto/blob/master/tests/test_cache.py

'''

import os
PATH1 = os.getcwd()

#Serialize Python Session
# https://github.com/uqfoundation/dill

def load_session(name='test_20160815'):
 n1= 'D:/_devs/Python01/aaserialize/session_'+ name + '.pkl',
 dill.load_session(n1)
 print n1


def save_session(name=''):
 t1= date_now()
 n1= 'D:/_devs/Python01/aaserialize/session_'+ name + '_'+t1+'.pkl',
 dill.dump_session(n1)
 print n1




###################################################################################
 #-------- Python General---------------------------------------------------------
#  Import File
# runfile('D:/_devs/Python01/project27/stockMarket/google_intraday.py', wdir='D:/_devs/Python01/project27/stockMarket')


def isfloat(value):
  try:    
   float(value)
   if value == np.inf : return False
   else : return True
  except :    return False      
      
      
def isint(x): return isinstance(x, ( int, long, np.int, np.int64, np.int32 ) )


def aa_isanaconda():
 import sys; 
 txt= sys.version
 if txt.find('Continuum') > 0 : return True
 else: return False
 

if aa_isanaconda() :  #Import the Packages
  pass
  
  
  


 #-----Get Documentation of the module
def aa_cleanmemory():
  import gc;  gc.collect() 

def aa_getmodule_doc(module1, fileout=''):
  from automaton import codeanalysis as ca
  ca.getmodule_doc(module1, fileout)
  
#  getmodule_doc("jedi", r"D:\_devs\Python01\aapackage\doc.txt")

''' Convert Python 2 to Python 3
import lib2to3

!2to3 D:\_devs\Python01\project\aapackage\codeanalysis.py

D:\_devs\Python01\project\zjavajar

'''

#  %load_ext autoreload
#  %autoreload 2

def aa_getlistofpackage() : aa_helpackage()

def aa_cmd(cmd1):aa_subprocess_output_and_error_code(cmd1, shell=True)
# cmd("ipconfig")

def a_help():
 str= """ 
 1) %load_ext autoreload     #Reload the packages
    %autoreload 2

 2) Install PIP:
     If Permission access,Exit Spyder,  Use CMD  
       D:\WinPython-64-2710\scripts>pip install scikit-learn --upgrade -
 
     !pip install NAME --upgrade --no-deps  #NO dependencies installed.     
     
     !pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps
     !pip install https://github.com//pymc3/archive/master.zip  
     !pip install  https://github.com/jpype-py3 /zipball/master  or  /tarball/master   

     !pip install c:\this_is_here\numpy-1.10.4+mkl-cp27-cp27m-win_amd64.whl
     !pip freeze   to get the list of package isntall

     !pip install -h    Help on pip  

 3) Windows CMD: 
    a_run_cmd("ipconfig")

    A: [enter] Changes the default drive from C to A.
   cd \furniture\chairs.  Moves you to the directory  'FURNITURE'
   cd .. Moves you up one level in the path.
   cd \  Takes you back to the root directory (c: in this case).


 
 4) import sys  
    reload(sys)  
    sys.setdefaultencoding('utf8')
    sys.getdefaultencoding()

 5) 
    !pip install packagename
    The ! prefix is a short-hand for the %sc command to run a shell command.

    You can also use the !! prefix which is a short-hand for the %sx command to execute a shell command 
    and capture its output (saved into the _ variable by default).

  6) Compile a modified package
    1) Put into a folder
    2) Add the folder to the Python Path (Spyder Python path)
 
 
    3) Compile in Spyder using, (full directory)
       !!python D:\\_devs\\Python01\\project27\\scikit_learn\\sklearn\\setup.py install

    4) Project is built here:
      D:\_devs\Python01\project27\build\lib.win-amd64-2.7
      http://scikit-learn.org/dev/developers/contributing.html#git-repo
python setup.py develop



  """
 print( str)

def aa_infosystem():
 import platform; print(platform.platform() + '\n')
 import sys; print("Python", sys.version)
 aa_helpackage()

def aa_helpackage() :
 import pip
 installed_packages = pip.get_installed_distributions()
 installed_packages_list = sorted(["%s==%s" % (i.key, i.version) for i in installed_packages])
 for p in  installed_packages_list : print(p)


def aa_pythonversion():    return sys.version_info[0]

# os.system('cd D:\_app\visualstudio13\Common7\IDE') #Execute any command    
# os.path.abspath('memo.txt') #get absolute path
# os.path.exists('memo.txt')
# os.path.isdir('memo.txt')
# os.getenv('PATH')    #Get environ variable  

# sitepackage= r'D:\_devs\Python01\WinPython-64-2710\python-2.7.10.amd64\Lib\site-packages/'





def aa_zipfolder(folderin, folderzipname ):
 import os, zipfile
 zf = zipfile.ZipFile(folderzipname, "w")
 for dirname, subdirs, files in os.walk(folderin):
    zf.write(dirname)
    for filename in files:
        zf.write(os.path.join(dirname, filename))
 zf.close()
 return  folderin, folderzipname
 
    
#Import module from file:  (Although this has been deprecated in Python 3.4.)
def aa_importfromfile(modulename, dir1): 
 vv= aa_pythonversion()
 if vv==3:
  from importlib.machinery import SourceFileLoader
  foo = SourceFileLoader("module.name", "/path/to/file.py").load_module()
  foo.MyClass()
 elif vv==2 :
  import imp
  foo = imp.load_source('module.name', '/path/to/file.py')
  foo.MyClass()



def aa_subprocess_output_and_error_code(cmd, shell=True):
    import subprocess
    cmd= normpath(cmd)
    PIPE=subprocess.PIPE
    STDOUT=subprocess.STDOUT
    proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=shell)
    stdout, stderr = proc.communicate()
    err_code = proc.returncode
    print("Console Msg: \n")
    print(str(stdout)) #,"utf-8"))
    print("\nConsole Error: \n"+ str(stderr) )
#    return stdout, stderr, int(err_code)




'''
"""
Ex: Dialog (2-way) with a Popen()
"""

p = subprocess.Popen('Your Command Here',
                 stdout=subprocess.PIPE,
                 stderr=subprocess.STDOUT,
                 stdin=PIPE,
                 shell=True,
                 bufsize=0)
p.stdin.write('START\n')
out = p.stdout.readline()
while out:
  line = out
  line = line.rstrip("\n")

  if "WHATEVER1" in line:
      pr = 1
      p.stdin.write('DO 1\n')
      out = p.stdout.readline()
      continue

  if "WHATEVER2" in line:
      pr = 2
      p.stdin.write('DO 2\n')
      out = p.stdout.readline()
      continue
"""
..........
"""

out = p.stdout.readline()

p.wait()
'''

###############################################################################
#------------------Printers----------------------------------------------------
#Print Object Table
def print_object(vv, txt='')  :
 print ("\n\n" + txt +"\n")
 sh= np.shape(vv)
 kkmax, iimax= sh[0], sh[1]
 for k in range(0, kkmax):
   aux= ""
   for i in range(0,iimax) :
     if vv[k,0] != None :
      aux+=  str(vv[k,i])    + "," 
      
   if vv[k,0] != None :   
       print (aux )
       
#Print to file Object   Table       
def print_object_tofile(vv, txt, file1='d:/regression_output.py')  :
 with open(file1, mode='a')  as file1 :
  file1.write("\n\n" + txt +"\n" )  
  sh= np.shape(vv)
  kkmax, iimax= sh[0], sh[1]
  for k in range(0, kkmax):
    aux= ""
    for i in range(0,iimax) :
     if vv[k,0] != None :
      aux+=  str(vv[k,i])    + "," 
      
    if vv[k,0] != None :   
       #print (aux )
       file1.write(aux + 	"\n")
    




##############################################################################
#--------------------FILE-------------------------------------------------------------------


def fil_replacestring_onefile(findStr, repStr, filePath):
    "replaces all findStr by repStr in file filePath"
    import fileinput  
    file1= fileinput.FileInput(filePath, inplace=True, backup='.bak')
    for line in file1:
       line= line.replace(findStr,  repStr)
       sys.stdout.write(line)
    file1.close()

    print("OK: "+format(filePath))


def fil_replacestring_files(findstr, replacestr, some_dir, pattern="*.*", dirlevel=1  ):
  list_file= listallfile(some_dir, pattern=pattern, dirlevel=dirlevel)
  list_file= list_file[2]
  for file1 in list_file : fil_replacestring_onefile(findstr, replacestr, file1)

  
#fil_replacestring_files("logo.png", "logonew.png", r"D:/__Alpaca__details/aiportfolio", 
#                         pattern="*.html", dirlevel=5  )

  




def gettext_fromfile(file1):
 with open(file1, 'r',encoding='UTF-8',) as f:      
  return f.read()

def fil_listallfile(some_dir, pattern="*.*", dirlevel=1):
  return listallfile(some_dir, pattern=pattern, dirlevel=dirlevel)

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
# aa[0][30];   aa[2][30]


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


def savetext_tofile(vv, file1):   #print into a file
 with open(file1, "w") as text_file:  text_file.write(str(vv))   
    
def normpath(pth): #Normalize path for Python directory
    if aa_pythonversion()==2: 
     ind = pth.find(":")
     if ind > -1 :
       a, b = pth[:ind], pth[ind + 1:].encode("string-escape").replace("\\x", "/")
       return "{}://{}".format(a, b.lstrip("\\//").replace("\\\\", "/"))        
     else: return pth
    else: 
      pth = pth.encode("unicode-escape").replace(b"\\x", b"/")
      return pth.replace(b"\\\\", b"/").decode("utf-8")
#r"D:\_devs\Python01\project\03-Connect_Java_CPP_Excel\PyBindGen\examples"
   
     
def changepath(path1): path1= normpath(path1); os.chdir(path1)    #Change Working directory path

def currentpath(): return os.getcwd()   

def isfileexist(file1): return os.path.exists(file1)

def filesize(file1): return os.path.getsize(file1)

def writeText(text, filename) :
 text_file = open(filename, "a"); text_file.write(text); text_file.close()

def readfile(file1):
 fh = open(file1,"r", encoding='UTF-8')
 return fh.read()
 
 
def merge_allfile(nfile, dir1, pattern1, deepness=2):
 ll= util.listallfile(dir1,pattern1,deepness)
 with open(nfile, mode='a', encoding='UTF-8') as nfile1:
  txt=''; ii=0
  for l in ll[2]:
   txt= '\n\n\n\n' + util.gettext_fromfile(l)
   nfile1.write(txt) 
 nfile1.close()


#Extract text from html
def extracttext_allfile(nfile, dir1, pattern1="*.html", htmltag='p', deepness=2):
 ll= util.listallfile(dir1, pattern1,5)
 with open(nfile, mode='a', encoding='UTF-8') as nfile1:
  txt=''; ii=0
  for l in ll[2]:
   page= util.gettext_fromfile(l)
   soup = BeautifulSoup(page, "lxml")
   txt2 = ' \n\n'.join(map(lambda p: p.text, soup.find_all(htmltag)))
   
   txt= '\n\n\n\n' + txt2.strip()
   newfile1.write(txt) 
 nfile1.close()



    
def save_obj(obj, name ):
    import pickle
    name2= PATH1+'/aaserialize/'+ name + '.pkl'
    with open(name2, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    return name2
    
def load_obj(name ):
    import pickle
    name2= PATH1+'/aaserialize/'+ name + '.pkl'    
    with open(name2, 'rb') as f:
        return pickle.load(f)

'''def load_obj(name, encoding1='utf-8' ):
    import pickle
    with open('D:/_devs/Python01/aaserialize/' + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding=encoding1)
'''



'''
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


#####################################################################################
#-------- XML / HTML processing ------------------------------------------------------
'''
https://pypi.python.org/pypi/RapidXml/
http://pugixml.org/benchmark.html

'''



#####################################################################################
#-------- PDF processing ------------------------------------------------------------
def print_topdf() :
 import datetime
 import numpy as np
 from matplotlib.backends.backend_pdf import PdfPages
 import matplotlib.pyplot as plt

 # Create the PdfPages object to which we will save the pages:
 # The with statement makes sure that the PdfPages object is closed properly at
 # the end of the block, even if an Exception occurs.
 with PdfPages('multipage_pdf.pdf') as pdf:
    plt.figure(figsize=(3, 3))
    plt.plot(range(7), [3, 1, 4, 1, 5, 9, 2], 'r-o')
    plt.title('Page One')
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    plt.rc('text', usetex=True)
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 5, 0.1)
    plt.plot(x, np.sin(x), 'b-')
    plt.title('Page Two')
    pdf.savefig()
    plt.close()

    plt.rc('text', usetex=False)
    fig = plt.figure(figsize=(4, 5))
    plt.plot(x, x*x, 'ko')
    plt.title('Page Three')
    pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
    plt.close()

    # We can also set the file's metadata via the PdfPages object:
    d = pdf.infodict()
    d['Title'] = 'Multipage PDF Example'
    d['Author'] = u'Jouni K. Sepp\xe4nen'
    d['Subject'] = 'How to create a multipage pdf file and set its metadata'
    d['Keywords'] = 'PdfPages multipage keywords author title subject'
    d['CreationDate'] = datetime.datetime(2009, 11, 13)
    d['ModDate'] = datetime.datetime.today()




#####################################################################################
#--------CSV processing -------------------------------------------------------------
#Put Excel and CSV into Database / Extract CSV from database
'''
import csv
with open(file2, 'r',encoding='UTF-8',) as f:      
  reader = csv.reader(f,  delimiter='/' )
  kanjidict=  dict(reader) 
  
 return kanjidict  



'''


#####################################################################################
#-------- STRING--------------------------------------------------------------------
def str_make_unicode(input, errors='replace'):
    ttype= type(input) 
    if ttype != unicode  : 
        input =  input.decode('utf-8', errors=errors);  return input
    else:  return input


def str_empty_string_array(x,y=1):
 if y==1:  return ["" for x in range(x)]
 else: return [["" for row in range(0,x)] for col in range(0,y)]


def str_empty_string_array_numpy(nx, ny=1):
 arr = np.empty((nx, ny), dtype=object)
 arr[:, :] = ''
 return arr

def str_isfloat(value):
  try:    float(value);    return True
  except :    return False

def str_is_azchar(x):
  try:    float(x);    return True
  except :    return False  
    
def str_is_az09char(x):
  try:    float(x);    return True
  except :    return False  
    

def str_reindent(s, numSpaces): #change indentation of multine string
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

def str_split2(delimiters, string, maxsplit=0):  #Split into Sub-Sentence
    import re
    regexPattern = '|'.join(map(re.escape, delimiters))
    return re.split(regexPattern, string, maxsplit)

def str_split_pattern(sep2, ll, maxsplit=0):  #Find Sentence Pattern
 regexPat = '|'.join(sep2) 
 regex3 = re.compile( '(' + regexPat + r')|(?:(?!'+ regexPat +').)*', re.S)
 #re.compile(r'(word1|word2|word3)|(?:(?!word1|word2|word3).)*', re.S)
 ll= regex3.sub(lambda m: m.group(1) if m.group(1) else "P", ll)
 return ll
 





'''
http://www.ibm.com/developerworks/library/l-pyint/

spam.upper().lower()

spam.strip()
spam.lstrip()
spam.rstrip()  Performs both lstrip() and rstrip() on string

len(string)
count(str, beg= 0,end=len(string))  Counts how many times str occurs in string or in a substring of
find(str, beg=0 end=len(string))  str occurs in string or in a substring of strindex if found  -1 otherwise.
replace(old, new [, max])   Replaces all occurrences of old in string with new or at most max occurrences if max given.


isdecimal()   Returns true if a unicode string contains only decimal characters and false otherwise.
isalnum()  Returns true if string has at least 1 character and all characters are alphanumeric and false otherwise.
salpha()   Returns true if string has at least 1 character and all characters are alphabetic and false otherwise.
isdigit()  Returns true if string contains only digits and false otherwise.
islower()  Returns true if string has at least 1 cased character and all cased characters are in lowercase and false otherwise.
isnumeric() Returns true if a unicode string contains only numeric characters and false otherwise.
isspace()  Returns true if string contains only whitespace characters and false otherwise.
istitle()  Returns true if string is properly "titlecased" and false otherwise.

lower()   Converts all uppercase letters in string to lowercase.
capitalize()   Capitalizes first letter of string
center(width, fillchar)  Returns space-padded string with string centered  w
title()  Returns "titlecased" version of string, that is, all words begin with uppercase and the rest are lowercase.
upper()   Converts lowercase letters in string to uppercase.

join(seq)  Merges (concatenates) the string representations of elements in sequence seq into a string, with separator string.
	
split(str="", num=string.count(str))   Splits string according to delimiter str  and returns list of substrings;
splitlines( num=string.count('\n'))  Splits string at all (or num) NEWLINEs and returns a list of each line with NEWLINEs removed.

ljust(width[, fillchar])   Returns a space-padded string with the original string left-justified to a total of width columns.

maketrans()  Returns a translation table to be used in translate function.

index(str, beg=0, end=len(string))   Same as find(), but raises an exception if str not found.
rfind(str, beg=0,end=len(string))   Same as find(), but search backwards in string.

rindex( str, beg=0, end=len(string))   Same as index(), but search backwards in string.

rjust(width,[, fillchar])  Returns a space-padded string with string right-jus to a total of width columns.

startswith(str, beg=0,end=len(string))
Determines if string or a substring of string (if starting index beg and ending index end are given) starts with substring str; returns true if so and false otherwise.

decode(encoding='UTF-8',errors='strict')
Decodes the string using the codec registered for encoding. encoding defaults to the default string encoding.
	
encode(encoding='UTF-8',errors='strict')
Returns encoded string version of string; on error, default is to raise a ValueError

endswith(suffix, beg=0, end=len(string))
Determines if string or a substring of string  ends with suffix; returns true if so and false otherwise.

expandtabs(tabsize=8)
Expands tabs in string to multiple spaces; defaults to 8 spaces per tab if tabsize not provided.
	
zfill (width)
Returns original string leftpadded with zeros to a total of width characters; intended for numbers, zfill() retains any sign given (less one zero).



'''
################################################################################
#-------- Japanese Utilitie--------------------------------------------------------
import re
# Regular expression unicode blocks collected from 
# http://www.localizingjapan.com/blog/2012/01/20/regular-expressions-for-japanese-text/
hiragana_full = r'[ぁ-ゟ]'
katakana_full = r'[゠-ヿ]'
kanji = r'[㐀-䶵一-鿋豈-頻]'
radicals = r'[⺀-⿕]'
katakana_half_width = r'[｟-ﾟ]'
alphanum_full = r'[！-～]'
symbols_punct = r'[、-〿]'
misc_symbols = r'[ㇰ-ㇿ㈠-㉃㊀-㋾㌀-㍿]'
ascii_char = r'[ -~]'
'''
hiragana_full = ur'[\\u3041-\\u3096]'
katakana_full = ur'[\u30A0-\u30FF]'
kanji = ur'[\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A]'
radicals = ur'[\u2E80-\u2FD5]'
half_width = ur'[\uFF5F-\uFF9F]'
alphanum_full = ur'[\uFF01-\uFF5E]'
symbols_punct = ur'[\x3000-\x303F]'
misc_symbols = ur'[\x31F0-\x31FF\x3220-\x3243\x3280-\x337F]'
'''

def ja_extract_unicode_block(unicode_block, string):
    ''' extracts and returns all texts from a unicode block from string argument. '''
    return re.findall( unicode_block, string)

def ja_remove_unicode_block(unicode_block, string):
    ''' removes all cha from a unicode block and return remaining texts from string  '''
    return re.sub( unicode_block, ' ', string)

def ja_getkanji(vv):
 vv= remove_unicode_block(hiragana_full, vv)
 vv= remove_unicode_block(katakana_full, vv)
 vv= remove_unicode_block(radicals, vv)
 vv= remove_unicode_block(katakana_half_width, vv)
 vv= remove_unicode_block(symbols_punct, vv)
 vv= remove_unicode_block(misc_symbols, vv)  
 vv= remove_unicode_block(alphanum_full, vv)  
 ff= vv.split(' '); vv=''
 for aa in ff:
    if not aa=='': vv+= ' '+ aa    
 return vv

# text = '初めての駅 自、りると、ママは、トットちゃん㈠㉃㊀㋾㌀㍿'
#  remove_unicode_block(kanji, text))
# ''.join(extract_unicode_block(hiragana_full, text)))


# Mode 1 : Get all the prununciation sentence
def ja_getpronunciation_txten(txt): 
 import java, romkan
 ll= java.japanese_tokenizer_kuromoji(txt,  parsermode="NORMAL")
 vv= ''
 for tt in ll:
   if tt[8] !='' :
    vv= (vv + ' '+ (romkan.to_roma(tt[8]))).strip()
   if tt[0]=='。' : vv= vv + '\n\n'
 return vv



# Mode 2 : Get all the prununciation for each Kanji
def ja_getpronunciation_kanji(txt, parsermode="SEARCH"):
 import java, romkan   
 txt= remove_unicode_block(symbols_punct, txt)
 txt= remove_unicode_block(misc_symbols, txt)  
 txt= remove_unicode_block(alphanum_full, txt) 
 ll2= java.japanese_tokenizer_kuromoji(txt,  parsermode=parsermode)
 vv= ''
 for tt in ll2:
  if "".join(extract_unicode_block(kanji, tt[0])) != '' :
   if (tt[7] !=''   and ( tt[1]=='動詞'  or tt[1]=='名詞' )):
    vv= vv + ' '+ tt[0] +  ' '+ romkan.to_roma(tt[8])  + '\n'
 return vv 


def ja_importKanjiDict(file2):
 import csv
 with open(file2, 'r',encoding='UTF-8',) as f:      
  reader = csv.reader(f,  delimiter='/' )
  kanjidict=  dict(reader)   
 return kanjidict  

#kanjidict= importKanjiDict(r"E:/_data/japanese/edictnewF.txt")
#kanjidict['拳固']

kanjidictfile= r"E:/_data/japanese/edictnewF.py"

# Mode 2 : Get all the prununciation for each Kanji
def ja_getpronunciation_kanji3(txt, parsermode="SEARCH"):
 import java, romkan   
 txt= remove_unicode_block(symbols_punct, txt)
 txt= remove_unicode_block(misc_symbols, txt)  
 txt= remove_unicode_block(alphanum_full, txt) 
 ll2= java.japanese_tokenizer_kuromoji(txt,  parsermode=parsermode)
 vv= ''
 kanjidict= importKanjiDict(kanjidictfile)
 for tt in ll2:
  if "".join(extract_unicode_block(kanji, tt[0])) != '' :
   if (tt[7] !=''  ):
    name= tt[0]   
    try :
     vv= vv + ' '+ name +' : '+ romkan.to_roma(tt[8])+ " : " + kanjidict[name]+'\n'
    except: 
     pass
 return vv 


# Mode 3 : Get all the prununciation sentence
def ja_gettranslation_textenja(txt): 
 import java
 ll= java.japanese_tokenizer_kuromoji(txt,  parsermode="SEARCH")
 kanjidict= importKanjiDict(kanjidictfile)
 vv= ''; vv2=''; xx=''
 for tt in ll:
   name= tt[0]
   if tt[8] !='' :
    vv2= (vv2 + '  '+ (name)).strip()
    if (tt[7] !=''   and ( tt[1]=='動詞'  or tt[1]=='名詞' )):
     try:
      vv= (vv + ' '+ (kanjidict[name])).strip() + ' /'
     except KeyError:
      pass
   if name=='。' : 
       xx= xx + vv2 + '\n' + vv + '\n\n'
       vv= ''; vv2=''
 return xx






# Mode 3 : Get all the prununciation sentence
def ja_getpronunciation_textenja(txt): 
 import java, romkan
 ll= java.japanese_tokenizer_kuromoji(txt,  parsermode="NORMAL")
 vv= ''; vv2=''; xx=''
 for tt in ll:
   if tt[8] !='' :
    vv2= (vv2 + '  '+ (tt[0])).strip()
    vv= (vv + ' '+ (romkan.to_roma(tt[8]))).strip()
   if tt[0]=='。' : 
       xx= xx + vv2 + '\n' + vv + '\n\n'
       vv= ''; vv2=''
 return xx


# Send Text Pronunciation by email
def ja_send_textpronunciation(url1, email1):
 aa= web_gettext_fromurl(url1)   
 kk= ja_getpronunciation_kanji3(aa)
 mm= ja_getpronunciation_textenja(aa)
 mm2= ja_gettranslation_textenja(aa) 
 mm= mm + '\n\n\n' + kk + '\n\n\n' + mm2
 send_email("Kevin", email1, "JapaneseText:"+mm[0:20] ,mm  )    

    
# Send Text Pronunciation by email
def ja_sendjp(url1): ja_send_textpronunciation(url1)

    



###############################################################################
#-------- Internet data connect--------------------------------------------
'''
https://moz.com/devblog/benchmarking-python-content-extraction-algorithms-dragnet-readability-goose-and-eatiht/
pip install numpy
pip install --upgrade cython
!pip install lxml
!pip install libxml2
!pip install dragnet
https://pypi.python.org/pypi/dragnet


Only Python 2.7
!pip install goose-extractor

https://github.com/grangier/python-goose

from goose import Goose
url = 'http://edition.cnn.com/2012/02/22/world/europe/uk-occupy-london/index.html?hpt=ieu_c2'
g = Goose()
article = g.extract(url=url)
 
article.title

article.meta_description

article.cleaned_text[:150]


'''

 # return the title and the text of the article at the specified url
def web_gettext_fromurl(url, htmltag='p'):
 http = urllib3.connection_from_url(url) 
 page = http.urlopen('GET',url).data.decode('utf8')
 
 soup = BeautifulSoup(page, "lxml")
 text = ' \n\n'.join(map(lambda p: p.text, soup.find_all('p')))
 return soup.title.text + "\n\n" + text


def web_gettext_fromhtml(file1, htmltag='p'):
 with open(file1, 'r',encoding='UTF-8',) as f:      
   page=f.read()
 
 soup = BeautifulSoup(page, "lxml")
 text = ' \n\n'.join(map(lambda p: p.text, soup.find_all(htmltag)))
 return soup.title.text + "\n\n" + text




'''

I know its been said already, 
but I'd highly recommend the Requests python package
: http://docs.python-requests.org/en/latest/index.html

If you've used languages other than python, 
you're probably thinking urllib and urllib2 are easy to use, 
not much code, and highly capable, that's how I used to think. 
But the Requests package is so unbelievably useful and 
short that everyone should be using it.

First, it supports a fully restful API, and is as easy as:

import requests
...

resp = requests.get('http://www.mywebsite.com/user')
resp = requests.post('http://www.mywebsite.com/user')
resp = requests.put('http://www.mywebsite.com/user/put')
resp = requests.delete('http://www.mywebsite.com/user/delete')
Regardless of whether GET/POST you never have to encode parameters again, it simply takes a dictionary as an argument and is good to go.

userdata = {"firstname": "John", "lastname": "Doe", "password": "jdoe123"}
resp = requests.post('http://www.mywebsite.com/user', params=userdata)
Plus it even has a built in json decoder (again, i know json.loads() isn't a lot more to write, but this sure is convenient):

resp.json()
Or if your response data is just text, use:

resp.text
This is just the tip of the iceberg. This is the list of features from the requests site:

International Domains and URLs
Keep-Alive & Connection Pooling
Sessions with Cookie Persistence
Browser-style SSL Verification
Basic/Digest Authentication
Elegant Key/Value Cookies
Automatic Decompression
Unicode Response Bodies
Multipart File Uploads
Connection Timeouts
.netrc support
List item
Python 2.6—3.4
Thread-safe.
'''





def web_getlink_fromurl(url):
 http = urllib3.connection_from_url(url) 
 page = http.urlopen('GET',url).data.decode('utf8')
 soup = BeautifulSoup(page, "lxml")    
 soup.prettify()
 links=[]
 for anchor in soup.findAll('a', href=True):
    lnk= anchor['href']
    links.append(  anchor['href'])
    
 return set(links)


def web_send_email(FROM, recipient, subject, body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
#    TO = recipient if type(recipient) is list else [recipient]
    TO= recipient
    msg = MIMEMultipart("alternative");    msg.set_charset("utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM
    msg["To"] = TO
    part2 = MIMEText(body, "plain", "utf-8")
    msg.attach(part2)
    
    try:   # SMTP_SSL Example
        server_ssl = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server_ssl.ehlo() # optional, called by login()
        server_ssl.login("mizenjapan@gmail.com", "sophieelise237")  
        server_ssl.sendmail(FROM, [TO], msg.as_string())
        server_ssl.close();        print ('successfully sent the mail'  )              
    except:
        print( "failed to send mail")
# send_email("Kevin", "brookm291@gmail.com", "JapaneseText:" , "txt")



# Send Text by email
def web_sendurl(url1):
 mm= web_gettext_fromurl(url1)   
 send_email("Python", "brookm291@gmail.com", mm[0:30] , url1 + '\n\n'+ mm )    




###############################################################################
#-------- LIST UTIL / Array ---------------------------------------------------
def np_remove_NA_INF_2d(X):
 im, jm= np.shape(X)
 for i in range(0, im) :
   for j in range(0,jm) :
    if np.isnan(X[i,j]) or np.isinf(X[i,j]): X[i,j]= X[i-1,j]
 return X
  
def np_addcolumn(arr, nbcol):
  sh= np.shape(arr);  
  vv= np.zeros((sh[0], nbcol))
  arr2= np.column_stack((arr, vv))
  return arr2

def np_addrow(arr, nbrow):
  sh= np.shape(arr);
  if len(sh)  > 1 :  
   vv= np.zeros((nbrow, sh[1]))
   arr2= np.row_stack((arr, vv))
   return arr2
  else:
    return None
    


# used to flatten a list or tupel [1,2[3,4],[5,[6,7]]] -> [1,2,3,4,5,6,7]
def np_list_flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):  l.append(elt2)
        else:   l.append(elt)
    return l


def np_dict_tolist(dict1):
 dictlist= []
 for key, value in dict1.items():
    temp = [key,value];    dictlist.append(temp)
 return dictlist   

def np_removelist(x0, xremove=[]) :
  xnew=[]
  for x in x0 :
    if np_findfirst(x, xremove) < 0 : xnew.append(x)
  return xnew


def np_transform2d_int_1d(m2d, onlyhalf=False) :
  imax, jmax= np.shape(m2d)
  v1d= np.zeros((imax*jmax, 4))  ; k=0
  for i  in range(0,imax):
    for j in range(i+1, jmax):
       v1d[k,0]= i; v1d[k,1]=j
       v1d[k,2]= m2d[i,j]
       v1d[k,3]= np.abs(m2d[i,j])
       k+=1
  v1d= v1d[v1d[:,2] !=0]
  return np_sortbycol(v1d,3, asc=False)
  



def np_mergelist(x0, x1) :
  xnew= list(x0)
  for x in x1 :
    xnew.append(x)
  return list(xnew)


def np_enumerate2(vec_1d):
 v2= np.empty((len(vec_1d), 2))
 for k,x in enumerate(vec_1d): v2[k,0]= k; v2[k,1]= x 
 return v2


#Pivot Table from List data    
def np_pivottable_count(mylist):
    mydict={}.fromkeys(mylist,0)
    for e in mylist:
        mydict[e]= mydict[e] + 1   # Map Reduce function 
    ll2= np_dict_tolist(mydict)
    ll2= sorted(ll2, key = lambda x: int(x[1]), reverse=True)
    return ll2 


def __np_nan_helper(y):
    """ Input:  - y, 1d numpy array with possible NaNs
        Output - nans, logical indices of NaNs - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def np_interpolate_nan(y):
 nans, x= __np_nan_helper(y)
 y[nans]= np.interp(x(nans), x(~nans), y[~nans])
 return y


def and1(x,y, x3=None, x4=None, x5=None, x6=None, x7=None, x8=None): 
  if not x8 is None :  return np.logical_and.reduce((x8,x7, x6, x5, x4,x3,x,y))
  if not x7 is None :  return np.logical_and.reduce((x7, x6, x5, x4,x3,x,y))
  if not x6 is None:  return np.logical_and.reduce((x6, x5, x4,x3,x,y))
  if not x5 is None:  return np.logical_and.reduce((x5, x4,x3,x,y))
  if not x4 is None :  return np.logical_and.reduce((x4,x3,x,y))
  if not x3 is None :  return np.logical_and.reduce((x3,x,y))



def sortcol(arr, colid, asc=1): 
 """ df.sort(['A', 'B'], ascending=[1, 0])  """
 df = pd.DataFrame(arr)
 arr= df.sort_values(colid, ascending=asc)   
 return arr.values

def sort(arr, colid, asc=1): 
 """ df.sort(['A', 'B'], ascending=[1, 0])  """
 df = pd.DataFrame(arr)
 arr= df.sort_values(colid, ascending=asc)   
 return arr.values


def np_ma(vv, n):
  '''Moving average '''
  return np.convolve(vv, np.ones((n,))/n)[(n-1):]


@jit(float32[:,:](float32[:,:]))
def np_cleanmatrix(m):
  m= np.nan_to_num(m)
  imax, jmax= np.shape(m)
  for i in range(0,imax):
    for j in range(0,jmax):
       if abs(m[i,j]) > 300000.0 : m[i,j]= 0.0
  return m


 
 
def np_sortbycolumn(arr, colid, asc=True): 
 df = pd.DataFrame(arr)
 arr= df.sort_values(colid, ascending=asc)   
 return arr.values

def np_sortbycol(arr, colid, asc=True): 
 if len(np.shape(arr)) > 1 :
  df = pd.DataFrame(arr)
  arr= df.sort_values(colid, ascending=asc)   
  return arr.values
 else :
  return np.reshape(arr, (1,len(arr)))  
    
# if colid==0 : return arr[np.argsort(arr[:])]
# else : return arr[np.argsort(arr[:, colid])]

def np_findfirst(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in xrange(len(vec)):
        if item == vec[i]: return i
    return -1

def np_find(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in xrange(len(vec)):
        if item == vec[i]: return i
    return -1

def find(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in xrange(len(vec)):
        if item == vec[i]: return i
    return -1


def findx(item, vec):
    """return the index of the first occurence of item in vec"""
    try :    
      if type(vec)== list : i2= vec.index(item)
      else :   
        i=np.where(vec==item)[0]
        i2= i[0] if len(i) > 0 else -1
    except: 
       i2= -1
    return i2
    
def finds(itemlist, vec):
  """return the index of the first occurence of item in vec"""
  idlist= []  
  for x in itemlist :
    ix=-1    
    for i in xrange(len(vec)):
        if x == vec[i]:  idlist.append(i); ix=i
    if ix==-1: idlist.append(-1)       
  if idlist== [] : return -1 
  else : return idlist


def findhigher(x, vec):
    """return the index of the first occurence of item in vec"""
    for i in xrange(len(vec)):
        if vec[i] > x : return i
    return -1


def findlower(x, vec):
    """return the index of the first occurence of item in vec"""
    for i in xrange(len(vec)):
        if vec[i] < x: return i
    return -1



  
import operator
def np_find_minpos(values):
 min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
 return min_index, min_value
 
def np_find_maxpos(values):
 max_index, max_value = max(enumerate(values), key=operator.itemgetter(1))
 return max_index, max_value
 

def np_find_maxpos_2nd(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for i,x in enumerate(numbers):
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x; i2= i
    return i2, m2 if count >= 2 else None



def np_findlocalmax2(v, trig):
 n=len(v)  
 v2= np.zeros((n,8))
 tmax,_= np_find_maxpos(v)
 if n < 3:
   max_index, max_value= np_find_maxpos(v)
   v2=  [[max_index, max_value]] 
   return v2
 else:
  for i,x in enumerate(v):
    if i < n-1 and i > 0 :
     if x > v[i-1] and x > v[i+1] : 
           v2[i,0]= i;  v2[i,1]=x
  v2 =np_sortbycolumn(v2,1,asc=False)
 
  #Specify the max :
 for k in range(0, len(v2)) :
  kmax= v2[k,0]
  kmaxl=  findhigher(v2[k,1], v[:kmax][::-1])  #Find same level of max
  kmaxr=  findhigher(v2[k,1], v[kmax+1:])

  kmaxl= 0 if kmaxl==-1 else kmax-kmaxl
  kmaxr= n if kmaxr==-1 else kmaxr+kmax
  
  v2[k,2]= np.abs(kmaxr-kmaxl)   #Range 
  v2[k,3]= np.abs(kmax-tmax)   #Range of the Max After
  v2[k,4]= 0  #Range of the Max Before
  v2[k,5]= kmax-kmaxl
  v2[k,6]= kmaxr-kmax

 v2= v2[np.logical_and( v2[:,5] > trig , v2[:,6] > trig ) ]
 v2 =np_sortbycolumn(v2,0,asc=1)
 return v2
 

def np_findlocalmin2(v, trig):
 n=len(v)  
 v2= np.zeros((n,8))
 tmin,_= np_find_minpos(v)
 if n < 3:
   max_index, max_value= np_find_minpos(v)
   v2= [[max_index, max_value]]
   return v2
 else:
  for i,x in enumerate(v):
    if i < n-1 and i > 0 :
     if x < v[i-1] and x < v[i+1] :      
           v2[i,0]= i;  v2[i,1]=x
  v2 =np_sortbycolumn(v2,1,asc=False)
 
 #Classification of the Local min
  for k in range(0, len(v2)) :
   if v2[k,1] != 0.0 :
    kmin= v2[k,0]
    kminl=  findlower(v2[k,1], v[:kmin][::-1])  #Find same level of min
    kminr=  findlower(v2[k,1], v[kmin+1:])

    kminl= 0 if kminl==-1 else kmin-kminl
    kminr= n if kminr==-1 else kminr+kmin

    v2[k,2]= np.abs(kminr-kminl)   #Range 
    v2[k,3]= np.abs(kmin-tmin)   #Range of the min After
    v2[k,4]= 0  #Range of the min After
    v2[k,5]= kmin-kminl
    v2[k,6]= kminr-kmin
 v2= v2[np.logical_and( v2[:,5] > trig , v2[:,6] > trig ) ]
 v2 =np_sortbycolumn(v2,0,asc=1)
 return v2
 



def np_findlocalmax(v):
 n=len(v)   
 v2= np.zeros((n,2))
 if n > 2:
  for i,x in enumerate(v):
    if i < n-1 and i > 0 :
     if x > v[i-1] and x > v[i+1] : 
           v2[i,0]= i;  v2[i,1]=x
  v2 =np_sortbycolumn(v2,1,asc=False)
  return v2
 else : 
   max_index, max_value= np_find_maxpos(v)
   return  [[max_index, max_value]]


def np_findlocalmin(v):
 n=len(v)   
 v2= np.zeros((n,2))
 if n > 2:
  for i,x in enumerate(v):
    if i < n-1 and i > 0 :
     if x < v[i-1] and x < v[i+1] :      
           v2[i,0]= i;  v2[i,1]=x
 
  v2 =np_sortbycolumn(v2[:i2],0,asc=True)
  return v2
 else : 
   
   max_index, max_value= np_find_minpos(v)
   return  [[max_index, max_value]]






def np_stack(v1, v2=None, v3=None, v4=None, v5=None):
   sh= np.shape(v1)
   if sh[0] < sh[1] :    
     v1= np.row_stack((v1,v2))
     if v3 != None : v1= np.row_stack((v1,v3))
     if v4 != None : v1= np.row_stack((v1,v4))
     if v5 != None : v1= np.row_stack((v1,v5))
   else :
     v1= np.column_stack((v1,v2))
     if v3 != None : v1= np.column_stack((v1,v3))
     if v4 != None : v1= np.column_stack((v1,v4))
     if v5 != None : v1= np.column_stack((v1,v5))       
       
   return v1

def np_uniquerows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def np_remove_zeros(vv, axis1=1):
   return vv[~np.all(vv == 0, axis=axis1)]

def np_sort(vv): 
 return vv[np.lexsort(np.transpose(vv)[::-1])]      #Sort the array by different column

def np_memory_array_adress(x):
    # This function returns the memory block address of an array.
    return x.__array_interface__['data'][0]
# b = a.copy(); id(b) == aid
    


###############################################################################
#-------- SK Learn TREE UTIL----------------------------------------------------------------
def sk_featureimportance(clfrf, feature_name) :
 importances = clfrf.feature_importances_
 indices = np.argsort(importances)[::-1]
 for f in range(0, len(feature_name)):  
    if importances[indices[f]] > 0.0001 :
      print str(f + 1), str(indices[f]),  feature_name[indices[f]], str(importances[indices[f]])


def sk_showconfusion(clfrf, X_train,Y_train, isprint=True):
  Y_pred = clfrf.predict(X_train) ;
  cm = sk.metrics.confusion_matrix(Y_train, Y_pred); cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  if isprint: print( cm_norm[0,0] + cm_norm[1,1]); print(cm_norm); print(cm)
  return cm, cm_norm, cm_norm[0,0] + cm_norm[1,1]



def sk_tree(Xtrain,Ytrain, nbtree, maxdepth, print1):
  # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)
  clfrf= sk.ensemble.RandomForestClassifier( n_estimators=nbtree, max_depth=maxdepth, max_features="sqrt",
                             criterion="entropy", min_samples_split=2, min_samples_leaf=2, class_weight= "balanced")
  clfrf.fit(Xtrain, Ytrain)  
  Y_pred = clfrf.predict(Xtrain) 
  
  cm = sk.metrics.confusion_matrix(Ytrain, Y_pred); cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  if print1: print( cm_norm[0,0] + cm_norm[1,1]); print(cm_norm); print(cm)
  return clfrf, cm, cm_norm



def sk_gen_ensemble_weight(vv, acclevel, maxlevel=0.88):
 imax= min(acclevel,len(vv))
 estlist= np.empty(imax, dtype= np.object) ; estww=[]
 for i in range(0, imax ) :
  #if vv[i,3]> acclevel:
   estlist[i]= vv[i,1] ;  estww.append( vv[i,3] )
   #print 5
  #Log Proba Weighted + Impact of recent False discovery
 estww= np.log( 1/(maxlevel - np.array(estww)/2.0) )  
 # estww= estww/np.sum(estww)
# return np.array(estlist), np.array(estww)
 return estlist, np.array(estww)


def sk_votingpredict(estimators, voting, ww, X_test) :
  ww= ww/np.sum(ww)
  Yproba0= np.zeros((len(X_test),2))
  Y1= np.zeros((len(X_test)))

  for k,clf in enumerate(estimators) :
     Yproba= clf.predict_proba(X_test)
     Yproba0= Yproba0 + ww[k]*Yproba
  
  for k in range(0,  len(X_test)) :
     if  Yproba0[k,0] > Yproba0[k,1]:      Y1[k]=-1
     else : Y1[k]=1
  return  Y1, Yproba0
    




def sk_tree_get_ifthen(tree, feature_names, target_names, spacer_base=" "):
    """Produce psuedo-code for decision tree.
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (output) names.
    spacer_base -- used for spacing code (default: "    ").
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if " + features[node] + " <= " + str(threshold[node]) + " :")
#            print(spacer + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) :")
            if left[node] != -1:
                    recurse(left, right, threshold, features, left[node], depth+1)
            print( "" + spacer +"else :")
            if right[node] != -1:
                    recurse(left, right, threshold, features, right[node], depth+1)
       #     print(spacer + "")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],  target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) +  " ( " + str(target_count) + ' examples )"')

    recurse(left, right, threshold, features, 0, 0)




###############################################################################
#-------- PANDA UTIL----------------------------------------------------------------
def pd_array_todataframe(price, symbols=None, date1=None, dotranspose=False):
   sh= np.shape(price)
   if len(sh) > 1 :
     if sh[0] < sh[1] and dotranspose :   #  masset x time , need Transpose
         return pd.DataFrame(data= price.T, index= date1,  columns=symbols)  
     else :
         return pd.DataFrame(data= price, index= date1,  columns=symbols)  
   else :
     return pd.DataFrame(data= price, index= date1,  columns=symbols)



def pd_date_intersection(qlist) :
 date0= set(qlist[0]['date'].values)
 for qi in qlist :
   qs= set(qi['date'].values)
   date0 = set.intersection(date0, qs)
 date0= list(date0); date0= sorted(date0)
 return date0




def pd_resetindex(df): 
  df.index = list(np.arange(0,len(df.index)))
  return df



def pd_create_colmap_nametoid(df) :
  ''' 'close' ---> 5    '''
  col= df.columns.values; dict1={}
  for k,x in enumerate(col) :
    dict1[x]= k
  return dict1
 
 
def pd_dataframe_toarray(df):
  date1= df.index
  array1= (df.reset_index().values)[1:,:]
  column_name= df.columns
  return column_name, date1, array1
   
   
def pd_changeencoding(data, cols):
    #  Western European: 'cp1252'
    for col in cols:
        data[col] = data[col].str.decode('iso-8859-1').str.encode('utf-8')
    return data 
  
  
def pd_createdf(val1, col1=None, idx1=None ) :
  return  pd.DataFrame(data=val1, index=idx1, columns=col1)
   


def pd_insertcolumn(df, colname,  vec):
 ''' Vec and Colname must be aligned '''
 ncol= len(df.columns.values)
 sh= np.shape(vec)
 if len(sh) > 1 :
    imax, jmax= sh[0], sh[1]
    for j in range(0,jmax) : 
      df.insert(ncol, colname[j], vec[:,j])
      
 else :
    df.insert(ncol, colname, vec)
 return df


def pd_insertrows(df, rowval, index1=None):
 sh= np.shape(rowval)
 istart= df.index.values[-1]+1
 if index1 == None : index1= np.arange(istart, istart + sh[0]) 
 df2 = pd.DataFrame(index= index1, columns= df.columns.values )

 for i in range(0, sh[0]): 
  df2.loc[i+istart]= rowval[i,:]
  
 df= df.append(df2)
 return df


def pd_replacevalues(df,  matrix):
 ''' Matrix replaces df.values  '''
 imax, jmax= np.shape(vec)
 colname= df.columns.values
 for j in jmax :
   df.loc[colname[j]] = matrix[:,j]

 return df
 

def pd_storeadddf(df, dfname, dbfile='F:\temp_pandas.h5') :
  store = pd.HDFStore(dbfile)
  if find(dfname, store.keys()) > 0 :
     dfname= dfname + '_1'    
  store.append(dfname, df); store.close() 



def pd_storedumpinfo(dbfile='E:\_data\stock\intraday_google.h5'):
  store = pd.HDFStore(dbfile)
  extract=[]; errsym=[]
  for symbol in store.keys(): 
     try: 
       df= pd.read_hdf(dbfile, symbol)       
       extract.append([symbol[1:], df.shape[1], 
                       df.shape[0], datetime_tostring(df.index.values[0]),
                        datetime_tostring(df.index.values[-1]) ]) 
            
     except: errsym.append(symbol)  
  return np.array(extract), errsym

  
def pd_remove_row(df, row_list_index=[23,45]) : 
 return df.drop(row_list_index)

def pd_extract_col_idx_val(df):
 return df.columns.values,  df.index.values, df.values   


def pd_split_col_idx_val(df):
 return df.columns.values,  df.index.values, df.values   
    

def pd_addcolumn(df1, name1='new'):
 tmax= len(df1.index)
 if type(name1)==str : 
    df1.loc[:, name1] = pd.Series(np.zeros(tmax), index=df1.index)
 else :
   for name0 in name1 :
     df1.loc[:, name0] = pd.Series(np.zeros(tmax), index=df1.index)
 return df1

def pd_removecolumn(df1, name1):
 return df1.drop(name1, 1)


def pd_save_vectopanda(vv, filenameh5):  # 'E:\_data\_data_outlier.h5'   
 store = pd.HDFStore(filenameh5)  
 pdf =pd.DataFrame(vv); store.append('data', pdf); store.close()  


def pd_load_panda2vec(filenameh5, store_id='data'):  # 'E:\_data\_data_outlier.h5'
 pdf=  pd.read_hdf(filenameh5, store_id)    #from file
 return pdf.values   #to numpy vector


def pd_csv_topanda(filein1, filename, tablen='data'):
 #filein1=   'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.csv'
 #filename = 'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.h5'
 chunksize =     10 * 10 ** 6
 list01= pd.read_csv(filein1, chunksize=chunksize, lineterminator=',')
 for chunk in list01:
     store = pd.HDFStore(filename);     
     store.append(tablen, chunk);     store.close()     
 del chunk


 #---LOAD Panda Vector----------------------------------------
def pd_getpanda_tonumpy(filename, nsize, tablen='data'):
 pdframe=  pd.read_hdf(filename, tablen, start=0, stop=(nsize))
 return pdframe.values   #to numpy vector       


def pd_getrandom_tonumpy(filename, nbdim, nbsample, tablen='data'):
 pdframe=  pd.read_hdf(filename,tablen, start=0, stop=(nbdim*nbsample))
 return pdframe.values   #to numpy vector       




'''
#List unique values in a DataFrame column
pd.unique(df.column_name.ravel())

#Convert Series datatype to numeric, getting rid of any non-numeric values
df['col'] = df['col'].astype(str).convert_objects(convert_numeric=True)

#Grab DataFrame rows where column has certain values
valuelist = ['value1', 'value2', 'value3']
df = df[df.column.isin(value_list)]

#Grab DataFrame rows where column doesn't have certain values
valuelist = ['value1', 'value2', 'value3']
df = df[~df.column.isin(value_list)]

#Delete column from DataFrame
del df['column']

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['column_one']>2004) & (df['column_two']==9)]

#Rename several DataFrame columns
df = df.rename(columns = {
    'col1 old name':'col1 new name',
    'col2 old name':'col2 new name',
})

#lower-case all DataFrame column names
df.columns = map(str.lower, df.columns)

#even more fancy DataFrame column re-naming
#lower-case all DataFrame column names (for example)
df.rename(columns=lambda x: x.split('.')[-1], inplace=True)

#Loop through rows in a DataFrame
#(if you must)
for index, row in df.iterrows():
    print index, row['some column']  

#Next few examples show how to work with text data in Pandas.
#Full list of .str functions: http://pandas.pydata.org/pandas-docs/stable/text.html

#Slice values in a DataFrame column (aka Series)
df.column.str[0:2]

#Lower-case everything in a DataFrame column
df.column_name = df.column_name.str.lower()

#Get length of data in a DataFrame column
df.column_name.str.len()

#Sort dataframe by multiple columns
df = df.sort(['col1','col2','col3'],ascending=[1,1,0])

#get top n for each group of columns in a sorted dataframe
#(make sure dataframe is sorted first)
top5 = df.groupby(['groupingcol1', 'groupingcol2']).head(5)

#Grab DataFrame rows where specific column is null/notnull
newdf = df[df['column'].isnull()]

#select from DataFrame using multiple keys of a hierarchical index
df.xs(('index level 1 value','index level 2 value'), level=('level 1','level 2'))

#Change all NaNs to None (useful before
#loading to a db)
df = df.where((pd.notnull(df)), None)

#Get quick count of rows in a DataFrame
len(df.index)

#Pivot data (with flexibility about what what
#becomes a column and what stays a row).
#Syntax works on Pandas >= .14
pd.pivot_table(
  df,values='cell_value',
  index=['col1', 'col2', 'col3'], #these stay as columns
  columns=['col4']) #data values in this column become their own column

#change data type of DataFrame column
df.column_name = df.column_name.astype(np.int64)

# Get rid of non-numeric values throughout a DataFrame:
for col in refunds.columns.values:
  refunds[col] = refunds[col].replace('[^0-9]+.-', '', regex=True)

#Set DataFrame column values based on other column values
df['column_to_change'][(df['column1'] == some_value) & (df['column2'] == some_other_value)] = new_value

#Clean up missing values in multiple DataFrame columns
df = df.fillna({
    'col1': 'missing',
    'col4': 'missing',
    'col6': '99'
})

#Concatenate two DataFrame columns into a new, single column
#(useful when dealing with composite keys, for example)
df['newcol'] = df['col1'].map(str) + df['col2'].map(str)

#Doing calculations with DataFrame columns that have missing values
#In example below, swap in 0 for df['col1'] cells that contain null
df['new_col'] = np.where(pd.isnull(df['col1']),0,df['col1']) + df['col2']

# Split delimited values in a DataFrame column into two new columns
df['new_col1'], df['new_col2'] = zip(*df['original_col'].apply(lambda x: x.split(': ', 1)))

# Collapse hierarchical column indexes
df.columns = df.columns.get_level_values(0)

#Convert Django queryset to DataFrame
qs = DjangoModelName.objects.all()
q = qs.values()
df = pd.DataFrame.from_records(q)

#Create a DataFrame from a Python dictionary
df = pd.DataFrame(list(a_dictionary.items()), columns = ['column1', 'column2'])
'''



#####################################################################################
#-----------Scikit Learn----------------------------------------
import sklearn as sk


def sk_cluster_kmeans(x, nbcluster=5, isplot=True) :
  stdev=  np.std(x, axis=0)
  x= (x-np.mean(x, axis=0)) / stdev

  kmeans = sk.cluster.KMeans(n_clusters= nbcluster)
  kmeans.fit(np.reshape(x,(len(x),1)))
  centroids, labels= kmeans.cluster_centers_,  kmeans.labels_

  if isplot :
   import matplotlib.pyplot as plt
   colors = ["g.","r.","y.","b.", "k."]
   for i in range(0,len(x),5): plt.plot(x[i], colors[labels[i]], markersize = 2)
   plt.show()

  return labels, centroids, stdev




#####################################################################################
#------------Date Manipulation-------------------------------------------------------
from dateutil import parser


def datetime_tostring(datelist1):
 if isinstance(datelist1, datetime.date) :      return datelist1.strftime("%Y%m%d")
 if isinstance(datelist1, np.datetime64 ) :
   t= pd.to_datetime(str(datelist1)) ;  return t.strftime("%Y%m%d") 
  
 date2 = []
 for t in datelist1:
     date2.append(t.strftime("%Y%m%d"))
 return date2


def date_remove_bdays(from_date, add_days):
    isint1= isint(from_date)  
    if isint1 :  from_date= dateint_todatetime(from_date)
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add < 0:
        current_date += datetime.timedelta(days=-1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add += 1
    if isint1 : return datetime_toint(current_date) 
    else :      return   current_date
      
      
      
    

def date_add_bdays(from_date, add_days):
    isint1= isint(from_date)
    if isint1 :  from_date= dateint_todatetime(from_date)
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += datetime.timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    if isint1 : return datetime_toint(current_date) 
    else :      return   current_date
    

def datestring_todatetime(datelist1, format1= "%Y%m%d") :
 if isinstance(datelist1, str)  :  return parser.parse(datelist1)
 date2 = []
 for s in datelist1:
     date2.append(parser.parse(s))
     #date2.append(datetime.datetime.strptime(s, format1))
 return date2    


def datetime_toint(datelist1):
 if isinstance(datelist1, datetime.date) :      return int(datelist1.strftime("%Y%m%d"))
 date2 = []
 for t in datelist1:
     date2.append(int(t.strftime("%Y%m%d")))
 return date2
  
  
def dateint_todatetime(datelist1) :
 if isinstance(datelist1, int)  :  return parser.parse(str(datelist1))
 date2 = []
 for s in datelist1:
     date2.append(parser.parse(str(s)))
     #date2.append(datetime.datetime.strptime(s, format1))
 return date2    


def date_diffindays(intdate1, intdate2):
  dt= dateint_todatetime(intdate2) - dateint_todatetime(intdate1)
  return dt.days


def date_finddateid(date1, dateref) :
  i= util.np_findfirst(date1, dateref)
  if i==-1 : i= util.np_findfirst(date1+1, dateref) 
  if i==-1 : i= util.np_findfirst(date1-1, dateref)     
  if i==-1 : i= util.np_findfirst(date1+2, dateref)    
  if i==-1 : i= util.np_findfirst(date1-2, dateref) 
  if i==-1 : i= util.np_findfirst(date1+3, dateref)    
  if i==-1 : i= util.np_findfirst(date1-3, dateref) 
  if i==-1 : i= util.np_findfirst(date1+5, dateref)    
  if i==-1 : i= util.np_findfirst(date1-5, dateref)  
  if i==-1 : i= util.np_findfirst(date1+7, dateref)    
  if i==-1 : i= util.np_findfirst(date1-7, dateref)  
  return i 
   
def datestring_toint(datelist1) :
 if isinstance(datelist1, str) :      return int(datelist1)    
 date2 = []
 for s in datelist1:   date2.append(int(s))
 return date2   


def date_now(i=0):
 from datetime import datetime
 d= datetime.now()
 if i > 0 : d= date_add_bdays(d,i)
 else:      d= date_remove_bdays(d,i)
 return str(datetime_toint(d))


def date_as_float(dt):
    size_of_day = 1. / 366.
    size_of_second = size_of_day / (24. * 60. * 60.)
    days_from_jan1 = dt - datetime.datetime(dt.year, 1, 1)
    if not isleap(dt.year) and days_from_jan1.days >= 31+28:
        days_from_jan1 += timedelta(1)
    return dt.year + days_from_jan1.days * size_of_day + days_from_jan1.seconds * size_of_second

# start_date = datetime(2010,4,28,12,33)
# end_date = datetime(2010,5,5,23,14)
# difference_in_years = date_as_float(end_time) - date_as_float(start_time)

def datediff_inyear(startdate, enddate):
 return date_as_float(startdate) - date_as_float(enddate)
   
    
def date_generatedatetime(start="20100101", nbday=10, end=""):
  from dateutil.rrule import DAILY, rrule, MO, TU, WE, TH, FR
  start= datestring_todatetime(start)
  if end=="" :  end = date_add_bdays(start,nbday-1) #  + datetime.timedelta(days=nbday) 
  date_list= list(rrule(DAILY, dtstart=start, until=end, byweekday=(MO,TU,WE,TH,FR)))
  
  return np.array(date_list)



def datetime_tostring(datelist1):
 if isinstance(datelist1, datetime.date) :      return datelist1.strftime("%Y%m%d")
 date2 = []
 for t in datelist1:
     date2.append(t.strftime("%Y%m%d"))
 return date2


def datestring_todatetime(datelist1, format1= "%Y%m%d") :
 if isinstance(datelist1, str)  :  return parser.parse(s)
 date2 = []
 for s in datelist1:
     date2.append(parser.parse(s))
     #date2.append(datetime.datetime.strptime(s, format1))
 return date2    


def datestring_toint(datelist1) :
 if isinstance(datelist1, str) :      return int(datelist1)    
 date2 = []
 for s in datelist1:   date2.append(int(s))
 return date2
 




#####################################################################################
###########################-Utilities for Numerical Calc----------------------------


#----------Advance Calculation   ----------------------------------------------------
# New= xx*xx  over very large series
def numexpr_vect_calc(filename, expr, i0=0, imax=1000, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe    #to numpy vector
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx  
# filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'   
 store = pd.HDFStore(fileout) 
 store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):


def numexpr_topanda(filename, expr,  i0=0, imax=1000, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe   
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx    # filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5' 
 store = pd.HDFStore(fileout);  store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):


def textvect_topanda(vv, fileout=""):
 pd= pd.DataFrame(vv);  st= pd.HDFStore(fileout);  st.append('data', pd); del pd    






# yy1= getrandom_tonumpy('E:\_data\_QUASI_SOBOL_gaussian_xx2.h5', 16384, 4096)
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------









#----------------------------------------------------------------------------
#--------------------Statistics------------------------------------------------------
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
 
 
#-----------------------------------------------------------------------------

 





#####################################################################################
#-----------------Plot Utilities-----------------------------------------------------
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

  plt.title(title1)
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





'''

#List unique values in a DataFrame column
pd.unique(df.column_name.ravel())

#Convert Series datatype to numeric, getting rid of any non-numeric values
df['col'] = df['col'].astype(str).convert_objects(convert_numeric=True)

#Grab DataFrame rows where column has certain values
valuelist = ['value1', 'value2', 'value3']
df = df[df.column.isin(value_list)]

#Grab DataFrame rows where column doesn't have certain values
valuelist = ['value1', 'value2', 'value3']
df = df[~df.column.isin(value_list)]

#Delete column from DataFrame
del df['column']

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['column_one']>2004) & (df['column_two']==9)]

#Rename several DataFrame columns
df = df.rename(columns = {
    'col1 old name':'col1 new name',
    'col2 old name':'col2 new name',
    'col3 old name':'col3 new name',
})

#lower-case all DataFrame column names
df.columns = map(str.lower, df.columns)

#even more fancy DataFrame column re-naming
#lower-case all DataFrame column names (for example)
df.rename(columns=lambda x: x.split('.')[-1], inplace=True)

#Loop through rows in a DataFrame
#(if you must)
for index, row in df.iterrows():
    print index, row['some column']  

#Next few examples show how to work with text data in Pandas.
#Full list of .str functions: http://pandas.pydata.org/pandas-docs/stable/text.html

#Slice values in a DataFrame column (aka Series)
df.column.str[0:2]

#Lower-case everything in a DataFrame column
df.column_name = df.column_name.str.lower()

#Get length of data in a DataFrame column
df.column_name.str.len()

#Sort dataframe by multiple columns
df = df.sort(['col1','col2','col3'],ascending=[1,1,0])

#get top n for each group of columns in a sorted dataframe
#(make sure dataframe is sorted first)
top5 = df.groupby(['groupingcol1', 'groupingcol2']).head(5)

#Grab DataFrame rows where specific column is null/notnull
newdf = df[df['column'].isnull()]

#select from DataFrame using multiple keys of a hierarchical index
df.xs(('index level 1 value','index level 2 value'), level=('level 1','level 2'))

#Change all NaNs to None (useful before
#loading to a db)
df = df.where((pd.notnull(df)), None)

#Get quick count of rows in a DataFrame
len(df.index)

#Pivot data (with flexibility about what what
#becomes a column and what stays a row).
#Syntax works on Pandas >= .14
pd.pivot_table(
  df,values='cell_value',
  index=['col1', 'col2', 'col3'], #these stay as columns
  columns=['col4']) #data values in this column become their own column

#change data type of DataFrame column
df.column_name = df.column_name.astype(np.int64)

# Get rid of non-numeric values throughout a DataFrame:
for col in refunds.columns.values:
  refunds[col] = refunds[col].replace('[^0-9]+.-', '', regex=True)

#Set DataFrame column values based on other column values
df['column_to_change'][(df['column1'] == some_value) & (df['column2'] == some_other_value)] = new_value

#Clean up missing values in multiple DataFrame columns
df = df.fillna({
    'col1': 'missing',
    'col2': '99.999',
    'col3': '999',
    'col4': 'missing',
    'col5': 'missing',
    'col6': '99'
})

#Concatenate two DataFrame columns into a new, single column
#(useful when dealing with composite keys, for example)
df['newcol'] = df['col1'].map(str) + df['col2'].map(str)

#Doing calculations with DataFrame columns that have missing values
#In example below, swap in 0 for df['col1'] cells that contain null
df['new_col'] = np.where(pd.isnull(df['col1']),0,df['col1']) + df['col2']

# Split delimited values in a DataFrame column into two new columns
df['new_col1'], df['new_col2'] = zip(*df['original_col'].apply(lambda x: x.split(': ', 1)))

# Collapse hierarchical column indexes
df.columns = df.columns.get_level_values(0)

#Convert Django queryset to DataFrame
qs = DjangoModelName.objects.all()
q = qs.values()
df = pd.DataFrame.from_records(q)

#Create a DataFrame from a Python dictionary
df = pd.DataFrame(list(a_dictionary.items()), columns = ['column1', 'column2'])



#List unique values in a DataFrame column
pd.unique(df.column_name.ravel())

#Convert Series datatype to numeric, getting rid of any non-numeric values
df['col'] = df['col'].astype(str).convert_objects(convert_numeric=True)

#Grab DataFrame rows where column has certain values
valuelist = ['value1', 'value2', 'value3']
df = df[df.column.isin(value_list)]

#Grab DataFrame rows where column doesn't have certain values
valuelist = ['value1', 'value2', 'value3']
df = df[~df.column.isin(value_list)]

#Delete column from DataFrame
del df['column']

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['column_one']>2004) & (df['column_two']==9)]

#Rename several DataFrame columns
df = df.rename(columns = {
    'col1 old name':'col1 new name',
    'col2 old name':'col2 new name',
    'col3 old name':'col3 new name',
})

#lower-case all DataFrame column names
df.columns = map(str.lower, df.columns)

#even more fancy DataFrame column re-naming
#lower-case all DataFrame column names (for example)
df.rename(columns=lambda x: x.split('.')[-1], inplace=True)

#Loop through rows in a DataFrame
#(if you must)
for index, row in df.iterrows():
    print index, row['some column']  

#Next few examples show how to work with text data in Pandas.
#Full list of .str functions: http://pandas.pydata.org/pandas-docs/stable/text.html

#Slice values in a DataFrame column (aka Series)
df.column.str[0:2]

#Lower-case everything in a DataFrame column
df.column_name = df.column_name.str.lower()

#Get length of data in a DataFrame column
df.column_name.str.len()

#Sort dataframe by multiple columns
df = df.sort(['col1','col2','col3'],ascending=[1,1,0])

#get top n for each group of columns in a sorted dataframe
#(make sure dataframe is sorted first)
top5 = df.groupby(['groupingcol1', 'groupingcol2']).head(5)

#Grab DataFrame rows where specific column is null/notnull
newdf = df[df['column'].isnull()]

#select from DataFrame using multiple keys of a hierarchical index
df.xs(('index level 1 value','index level 2 value'), level=('level 1','level 2'))

#Change all NaNs to None (useful before
#loading to a db)
df = df.where((pd.notnull(df)), None)

#Get quick count of rows in a DataFrame
len(df.index)

#Pivot data (with flexibility about what what
#becomes a column and what stays a row).
#Syntax works on Pandas >= .14
pd.pivot_table(
  df,values='cell_value',
  index=['col1', 'col2', 'col3'], #these stay as columns
  columns=['col4']) #data values in this column become their own column

#change data type of DataFrame column
df.column_name = df.column_name.astype(np.int64)

# Get rid of non-numeric values throughout a DataFrame:
for col in refunds.columns.values:
  refunds[col] = refunds[col].replace('[^0-9]+.-', '', regex=True)

#Set DataFrame column values based on other column values
df['column_to_change'][(df['column1'] == some_value) & (df['column2'] == some_other_value)] = new_value

#Clean up missing values in multiple DataFrame columns
df = df.fillna({
    'col1': 'missing',
    'col2': '99.999',
    'col3': '999',
    'col4': 'missing',
    'col5': 'missing',
    'col6': '99'
})

#Concatenate two DataFrame columns into a new, single column
#(useful when dealing with composite keys, for example)
df['newcol'] = df['col1'].map(str) + df['col2'].map(str)

#Doing calculations with DataFrame columns that have missing values
#In example below, swap in 0 for df['col1'] cells that contain null
df['new_col'] = np.where(pd.isnull(df['col1']),0,df['col1']) + df['col2']

# Split delimited values in a DataFrame column into two new columns
df['new_col1'], df['new_col2'] = zip(*df['original_col'].apply(lambda x: x.split(': ', 1)))

# Collapse hierarchical column indexes
df.columns = df.columns.get_level_values(0)

#Convert Django queryset to DataFrame
qs = DjangoModelName.objects.all()
q = qs.values()
df = pd.DataFrame.from_records(q)

#Create a DataFrame from a Python dictionary
df = pd.DataFrame(list(a_dictionary.items()), columns = ['column1', 'column2'])

'''

















###############################################################################
#-------------------- COMPILING UTILITIES ################################
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





















































































































