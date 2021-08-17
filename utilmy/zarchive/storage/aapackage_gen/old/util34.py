# -*- coding: utf-8 -*-
#---------Various Utilities function for Python----------------------------
import os;
import sys ;

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np;
import pandas as pd;

import global01 as global01 #as global varaibles   global01.varname


#--------------------Utilities for Command Line---------------------------
# http://win32com.goermezer.de/content/blogsection/7/284/


def getmodule_doc(module1, fileout=''):
  from automaton import codeanalysis as ca
  ca.getmodule_doc(module1, fileout)
  
#  getmodule_doc("jedi")





''' Convert Python 2 to Python 3
import lib2to3

!2to3 D:\_devs\Python01\project\aapackage\codeanalysis.py

'''


def help2():
 str= """ installfromgit:    
  !pip install https://github.com/pymc-devs/pymc3/archive/master.zip 
   
  !pip install  https://github.com/tcalmant/jpype-py3 /zipball/master  
                                                      /tarball/master   

  def pip1(name1):   !pip install {name1}  #install package
    
  """
 print( str)

#def pip1(name1):   
#    !pip install {name1}  #install package

def pythonversion():    return sys.version_info[0]

def normpath(pth): #Normalize path for Python directory
    if pythonversion()==2: 
     ind = pth.index(":")
     a, b = pth[:ind], pth[ind + 1:].encode("string-escape").replace("\\x", "/")
     return "{}://{}".format(a, b.lstrip("\\//").replace("\\\\", "/"))        
    else: 
      pth = pth.encode("unicode-escape").replace(b"\\x", b"/")
      return pth.replace(b"\\\\", b"/").decode("utf-8")
#r"D:\_devs\Python01\project\03-Connect_Java_CPP_Excel\PyBindGen\examples"
   
     
def changepath(path1):import os; path1= normpath(path1); os.chdir(path1); return path1    #Change Working directory path

def currentpath():import os; return os.getcwd()   

def isfileexist(file1):import os; return os.path.exists(file1)

# os.system('cd D:\_app\visualstudio13\Common7\IDE') #Execute any command    
# os.path.abspath('memo.txt') #get absolute path
# os.path.exists('memo.txt')
# os.path.isdir('memo.txt')
#  os.getenv('PATH')    #Get environ variable  


def listallfile(some_dir, pattern="*.*", dirlevel=1):
  import fnmatch; import os;  matches = []
  some_dir = some_dir.rstrip(os.path.sep)
  assert os.path.isdir(some_dir)
  num_sep = some_dir.count(os.path.sep)
  for root, dirs, files in os.walk(some_dir):
 #   yield root, dirs, files
    num_sep_this = root.count(os.path.sep)
    if num_sep + dirlevel <= num_sep_this: del dirs[:]
    for files in fnmatch.filter(files, pattern):
      matches.append(os.path.join(root, files))     
  return matches            
     
# DIRCWD=r"D:\_devs\Python01\project"
# listallfile(DIRCWD, "*.*", 2)



#Import module from file:  (Although this has been deprecated in Python 3.4.)
def importfromfile(modulename, dir1): 
 vv=pythonsversion()
 if vv==3:
  from importlib.machinery import SourceFileLoader
  foo = SourceFileLoader("module.name", "/path/to/file.py").load_module()
  foo.MyClass()
 elif vv==2:
  import imp
  foo = imp.load_source('module.name', '/path/to/file.py')
  foo.MyClass()



def empty_string_array(size):
 return ["" for x in range(size)]




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
    


#List of subdirectory
import glob
DIRCWD=r"D:\_devs\Python01\project"
vv= glob.glob(DIRCWD+"\*\*.py")




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

#VS_build(self, DIRCWD)



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






#--------------------Utilities for Numerical Calc---------------------------


#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_vect_calc(filename, i0=0, imax=1000, expr="", fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe    #to numpy vector
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx  
# filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'   
 store = pd.HDFStore(fileout) 
 store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):


#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_topanda(filename, i0=0, imax=1000, expr="", fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
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

 

#----Data utils ----------------------------------------------------

#--------Clean array------------------------------------------------
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


























































































