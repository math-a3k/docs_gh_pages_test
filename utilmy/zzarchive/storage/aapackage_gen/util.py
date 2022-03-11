# -*- coding: utf-8 -*-
#---------Various Utilities function for Python----------------------------
import scipy as sp;import numpy as np; import numexpr as ne
import pandas as pd; import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit, vectorize, guvectorize, float64, float32, int32, boolean
from timeit import default_timer as timer
import global01 as global01 #as global varaibles   global01.varname
import os; import sys ; import glob
import urllib3; from bs4 import BeautifulSoup


#--------------------Utilities---------------------------
# http://win32com.goermezer.de/content/blogsection/7/284/
'''
WhenPackage is destroyed resintall and copy paste the data.

!pip install theano_lstm
https://pypi.python.org/pypi/theano-lstm

let suppose you have placed them at c:\this_is_here

pip install c:\this_is_here\numpy-1.10.4+mkl-cp27-cp27m-win_amd64.whl
pip install c:\this_is_here\scipy-0.17.0-cp27-none-win_amd64.whl
'''



#####################################################################################
#-------- Python General-------------------------------------------------------------------
#  Import File
# runfile('D:/_devs/Python01/project27/stockMarket/google_intraday.py', wdir='D:/_devs/Python01/project27/stockMarket')

#-----Get Documentation of the module
def getmodule_doc(module1, fileout=''):
  """function getmodule_doc
  Args:
      module1:   
      fileout:   
  Returns:
      
  """
  import codeanalysis as ca
  ca.getmodule_doc(module1, fileout)
  
#  getmodule_doc("jedi", r"D:\_devs\Python01\aapackage\doc.txt")

''' Convert Python 2 to Python 3
import lib2to3

!2to3 D:\_devs\Python01\project\aapackage\codeanalysis.py

D:\_devs\Python01\project\zjavajar

'''

def cmd(cmd1): subprocess_output_and_error_code(cmd1, shell=True)
# cmd("ipconfig")

def help2():
 str= """ installfromgit:    
  !pip install https://github.com/pymc-devs/pymc3/archive/master.zip 
   
  !pip install  https://github.com/tcalmant/jpype-py3 /zipball/master  or    /tarball/master   

  def pip1(name1):   !pip install {name1}  #install package
  cmd("ipconfig")
  pip install c:\this_is_here\numpy-1.10.4+mkl-cp27-cp27m-win_amd64.whl

  !pip freeze   to get the isntall
  
  """
 print( str)


def pythonversion():    return sys.version_info[0]

# os.system('cd D:\_app\visualstudio13\Common7\IDE') #Execute any command    
# os.path.abspath('memo.txt') #get absolute path
# os.path.exists('memo.txt')
# os.path.isdir('memo.txt')
# os.getenv('PATH')    #Get environ variable  


    
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



def subprocess_output_and_error_code(cmd, shell=True):
    import subprocess
    cmd= normpath(cmd)
    PIPE=subprocess.PIPE
    STDOUT=subprocess.STDOUT
    proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=shell)
    stdout, stderr = proc.communicate()
    err_code = proc.returncode
    print("Console Msg: \n")
    print(str(stdout,"utf-8"))
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




##############################################################################
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


def savetext_tofile(vv, file1):   #print into a file
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

def writeText(text, filename) :
 text_file = open(filename, "a"); text_file.write(text); text_file.close()

def readfile(file1):
 fh = open(file1,"r", encoding='UTF-8')
 return fh.read()
 
 


    
def save_obj(obj, name ):
    import pickle
    with open(''+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    import pickle
    with open('' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


'''
To open a text file, use:
fh = open("hello.txt", "r")

To read a text file, use:
fh = open("hello.txt","r")
print fh.read()

To read one line at a time, use:
fh = open("hello".txt", "r")
print fh.readline()

To read a list of lines use:
fh = open("hello.txt.", "r")
print fh.readlines()

To write to a file, use:
fh = open("hello.txt","w")
write("Hello World")
fh.close()

To write to a file, use:
fh = open("hello.txt", "w")
lines_of_text = ["a line of text", "another line of text", "a third line"]
fh.writelines(lines_of_text)
fh.close()

To append to file, use:
fh = open("Hello.txt", "a")
write("Hello World again")
fh.close

To close a file, use
fh = open("hello.txt", "r")
print fh.read()
fh.close()

'''


#####################################################################################
#-------- XML / HTML processing ------------------------------------------------------
'''
https://pypi.python.org/pypi/RapidXml/
http://pugixml.org/benchmark.html


'''



#####################################################################################
#-------- SFRAME Large Dataset in Python:  
'''
!pip install sframe
https://dato.com/products/create/docs/generated/graphlab.SFrame.html
sf[sf.apply(lambda x: math.log(x['id']) <= 1)]


'''









#####################################################################################
#--------CSV processing ------------------------------------------------------
#Put Excel and CSV into Database / Extract CSV from database








#####################################################################################
#-------- STRING--------------------------------------------------------------------
def empty_string_array(x,y=1):
 if y==1:  return ["" for x in range(x)]
 else: return [["" for row in range(0,x)] for col in range(0,y)]


def empty_string_array_numpy(nx, ny=1):
 arr = np.empty((nx, ny), dtype=object)
 arr[:, :] = ''
 return arr

def isfloat(value):
  try:    float(value);    return True
  except :    return False


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

dir('this is a string')
['__add__', '__class__', '__contains__', '__delattr__', '__doc__', '__eq__',
'__ge__', ]

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

rindex( str, beg=0, end=len(string))    Same as index(), but search backwards in string.

rjust(width,[, fillchar])  Returns a space-padded string with the string right-justified to a total of width columns.

startswith(str, beg=0,end=len(string))
Determines if string or a substring of string (if starting index beg and ending index end are given) starts with substring str; returns true if so and false otherwise.

decode(encoding='UTF-8',errors='strict')
Decodes the string using the codec registered for encoding. encoding defaults to the default string encoding.
	
encode(encoding='UTF-8',errors='strict')
Returns encoded string version of string; on error, default is to raise a ValueError unless errors is given with 'ignore' or 'replace'.

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

def extract_unicode_block(unicode_block, string):
    ''' extracts and returns all texts from a unicode block from string argument. '''
    return re.findall( unicode_block, string)

def remove_unicode_block(unicode_block, string):
    ''' removes all cha from a unicode block and return remaining texts from string  '''
    return re.sub( unicode_block, ' ', string)

def getkanji(vv):
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
def getpronunciation_txten(txt): 
 import java, romkan
 ll= java.japanese_tokenizer_kuromoji(txt,  parsermode="NORMAL")
 vv= ''
 for tt in ll:
   if tt[8] !='' :
    vv= (vv + ' '+ (romkan.to_roma(tt[8]))).strip()
   if tt[0]=='。' : vv= vv + '\n\n'
 return vv



# Mode 2 : Get all the prununciation for each Kanji
def getpronunciation_kanji(txt, parsermode="NORMAL"):
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



# Mode 3 : Get all the prununciation sentence
def getpronunciation_textenja(txt): 
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
def send_textpronunciation(url1):
 aa= gettext_fromurl(url1)   
 kk= getpronunciation_kanji(aa)
 mm= getpronunciation_textenja(aa)
 send_email("Kevin", "brookm291@gmail.com", "JapaneseText:"+mm[0:20] , mm + '\n\n\n' + kk)    

    
# Send Text Pronunciation by email
def sendjp(url1): send_textpronunciation(url1)

    





###############################################################################
#-------- Internet data connect--------------------------------------------
 # return the title and the text of the article at the specified url
def gettext_fromurl(url):
 http = urllib3.connection_from_url(url) 
 page = http.urlopen('GET',url).data.decode('utf8')
 
 soup = BeautifulSoup(page, "lxml")
 text = ' \n\n'.join(map(lambda p: p.text, soup.find_all('p')))
 return soup.title.text + "\n\n" + text



def getlink_fromurl(url):
 http = urllib3.connection_from_url(url) 
 page = http.urlopen('GET',url).data.decode('utf8')
 soup = BeautifulSoup(page, "lxml")    
 soup.prettify()
 links=[]
 for anchor in soup.findAll('a', href=True):
    lnk= anchor['href']
    links.append(  anchor['href'])
    
 return set(links)


def send_email(FROM, recipient, subject, body):
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
def sendurl(url1):
 mm= gettext_fromurl(url1)   
 send_email("Python", "brookm291@gmail.com", mm[0:30] , url1 + '\n\n'+ mm )    




###############################################################################
#-------- LIST UTIL----------------------------------------------------
# used to flatten a list or tupel [1,2[3,4],[5,[6,7]]] -> [1,2,3,4,5,6,7]
def flatten(seq):
    l = []
    for elt in seq:
        t = type(elt)
        if t is tuple or t is list:
            for elt2 in flatten(elt):  l.append(elt2)
        else:
            l.append(elt)
    return l





###############################################################################
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
'''














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

 





#####################################################################################
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





















































































































