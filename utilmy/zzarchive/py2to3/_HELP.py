

DB  --> Tables, Schema, Data

DB <---> Object model, ORM  ---> Object relation

Functionnal Action --> Functionnal Object

Put Object relation INTO Functionnal Object
   through CLASS DEFINITION







Naming Convention in Object

Object_action_SubObject_Sub_action

Action_Object_SubAction_SubObject


Reflect the data structure / Usage






#------------------------------ PANDA-----------------------------------------------------------------------------------
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





#-------------------------------- SQL ALCHEMY --------------------------------------------------------------------------

#Schema creation in SQL Alchemy
from datetime import datetime
from sqlalchemy import DateTime
users = Table('users', metadata,
 Column('user_id', Integer(), primary_key=True),
 Column('username', String(15), nullable=False, unique=True),
 Column('email_address', String(255), nullable=False),
 Column('phone', String(20), nullable=False),
 Column('password', String(25), nullable=False),
 Column('created_on', DateTime(), default=datetime.now),
 Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now)
)

ins = insert(users).values(
 username="cookiemon",
 email_address="damon@cookie.com",
 phone="111-111-1111",
 password="password"
)
result = connection.execute(ins)

'''
If datetime is not created, put it in AUTO Creation in the database

1) Auto update dateime in Firefox SQLITE, change table creation
,"date_update" DATETIME DEFAULT (CURRENT_TIMESTAMP)

2)

ptable['date_create'] = ptable['date_create'].astype(datetime.datetime)

'''


Metadata
Metadata is used to tie together the database structure so it can be quickly accessed
inside SQLAlchemy. It’s often useful to think of metadata as a kind of catalog of Table
objects with optional information about the engine and the connection.



import urllib
import pyodbc
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd

params = urllib.quote_plus('DRIVER={SQL Server};SERVER=<nameofserver>;DATABASE=<nameofdb>;UID=<userid>;PWD=<pwd>')
engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
Base = declarative_base()

class LookupTable(Base):
    """ LookupTable is the Python object that represents the Lookup Table used in the database. """
    __tablename__ = 'myTableName'

    myid = Column(String(50), primary_key=True)
    value = Column(String(512))

if __name__ == '__main__':

    # Setup
    Base.metadata.create_all(engine)
    pdsql = pd.io.sql.SQLDatabase(engine)

    # Query Database Table and return a DataFrame
    df_a = pd.read_sql("SELECT * FROM myTableName", con=engine)

    # Do DataFrame Stuff; here its just a simple join
    df_b = pd.DataFrame({'myid': [1, 2, 3, 4],
                         'value': ['Hello', 'World', 'Liu', 'Summers']})
    df_c = pd.merge(df_a, df_b, on='myid', how='inner')

    # Write new DataFrame back to Database Table
    df_c.to_sql(name='myTableName', if_exists='append',con=engine)




#   Reflecting a Database with Automap
In order to reflect a database, instead of using the declarative_base we’ve been
using with the ORM so far, we’re going to use the automap_base. Let’s start by creating
a Base object to work with, as shown in Example 10-


#Create Automap
from sqlalchemy.ext.automap import automap_base
Base = automap_base()

from sqlalchemy import create_engine
engine = create_engine('sqlite:///Chinook_Sqlite.sqlite')

#Mapping the whole database
Base.prepare(engine, reflect=True)


Base.classes.keys()
Artist = Base.classes.Artist
Album = Base.classes.Album


from sqlalchemy.orm import Session
session = Session(engine)
for artist in session.query(Artist).limit(10):
print(artist.ArtistId, artist.Name)



artist = session.query(Artist).first()
for album in artist.album_collection:
print('{} - {}'.format(artist.Name, album.Title))








#  Iparalell
http://nbviewer.jupyter.org/github/rossant/ipython-minibook/blob/master/chapter5/501-parallel-computing.ipynb





















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
changepath(dir1)


dir1 = "D:\_devs\Python01\project\\03-Connect_Java_CPP_Excel\PyBindGen\examples"
dirs = os.listdir(dir1)


os.path.abspath(dir1)

os.fsdecode(dir1)

os.getcwd()   #current folder



from win32com.client.gencache import EnsureDispatch

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

os.chdir(dir1)


dir1 = r" D:\_devs\Python01\project\03-Connect_Java_CPP_Excel\PyBindGen\examples"

dir2= os.path.abspath(dir1)

os.chdir(dir1)


dir2= os.fsdecode(dir1)
!cd {dir2}


from os.path import expanduser
home = expanduser("~")
os.fsdecode(dir1)

for i in range(3):
       !echo {i+1}

%run script.py  #Run the Script
%timeit [x*x for x in range(1000)]  #Benchmark time





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





























































###############################################################################
# -------------------- COMPILING UTILITIES ################################
def os_compileVSsolution(dir1, flags1="", type1="devenv", compilerdir=""):
   if type=="devenv":  # devenv
      if compilerdir1=="" :  compilerdir1= global01.VS_DEVENV
      cmd= compilerdir1 +" " + dir1 + " " + flags1
      os.system(cmd)

   if type=="msbuild":  # msbuild
      if compilerdir1=="":  compilerdir1=global01.VS_MSBUILD
      cmd=compilerdir1 + " " + dir1 + " " + flags1


'''
dirsol= r"D:\_devs\_github\cudasamples\a7_CUDALibraries\_full_rand_mersenne_sobol\rand_mt_sobol_VS2013.sln"

dirsol= r"D:\_devs\_github\cudasamples\7_CUDALibraries\_full_rand_mersenne_sobol\rand_mt_sobol_VS2013.sln"

DIRCWD= normpath(dirsol)
compilerdir1= global01.VS_DEVENV
compilerdir1= "devenv "

flags1= " /p:Configuration=Debug"
cmd1= compilerdir1 +" "+ DIRCWD + " " + flags1

os.system(cmd1)

compileVSsolution(dirsol, "Release")
'''


def os_VS_build(self, lib_to_build):
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
# How will you specify the appropriate version for your *.rc version resource file?
# target_version =  "2,3,4,5"

def set_rc_version(rcfile, target_version):
   with open(rcfile, "r+") as f:
      rc_content=f.read()

      # first part
      # FILEVERSION 6,0,20,163
      # PRODUCTVERSION 6,0,20,163
      # ...

      # second part
      # VALUE "FileVersion", "6, 0, 20, 163"
      # VALUE "ProductVersion", "6, 0, 20, 163"

      # first part
      regex_1=re.compile(r"\b(FILEVERSION|FileVersion|PRODUCTVERSION|ProductVersion) \d+,\d+,\d+,\d+\b", re.MULTILINE)

      # second part
      regex_2=re.compile(r"\b(VALUE\s*\"FileVersion\",\s*\"|VALUE\s*\"ProductVersion\",\s*\").*?(\")", re.MULTILINE)

      version=r"\1 " + target_version
      # modified_str = re.sub(regex, version, rc_content)

      pass_1=re.sub(regex_1, version, rc_content)
      version=re.sub(",", ", ", version)  # replacing "x,y,v,z" with "x, y, v, z"
      pass_2=re.sub(regex_2, r"\g<1>" + target_version + r"\2", pass_1)

      # overwrite
      f.seek(0);
      f.write(pass_2);
      f.truncate()

# set_rc_version(r"C:/repo/projectA/resources/win/projectA.rc", "3,4,5,6")




# Sample code snippet.. https://heejune.me/2015/12/08/writing-a-python-build-script-for-your-visual-c-project/
def os_VS_start(self, version):

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







#####################################################################################
# -----------------------Cython  ---------- ------------------------------------------

def fun_cython(a):
 b=a
 for i in xrange(0, 100000):
    a += i + b
 #print a


def fun_python(a):
 b=a
 for i in xrange(0, 100000):
    a += i + b
 #print a







