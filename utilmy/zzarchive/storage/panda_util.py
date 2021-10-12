# -*- coding: utf-8 -*-
#------------------------utilities for Panda----------------------------------
import numpy as np; import pandas as pd



#From database into Panda and get the results
#Process the results and put it in database : 
# !pip install db.py, already installed





http://pandas.pydata.org/pandas-docs/stable/io.html#hierarchical-keys






#Provide Analytics for Data analysis, connected to server based:
  Java / Database / Excel CSV / C++ library /


Do data analytics
Load very amount of data
Process them:
   

Find Inference:



and put back in database
    

#----------------Details----------------------

How to use Python as a replacement of SQL developer ?


1) Connect to database details:
  ---> Size of the vector
  

2) create functions to visualize
    

3)


#--------------------------------------------
http://pandas.pydata.org/pandas-docs/stable/io.html#hierarchical-keys

http://pandas.pydata.org/pandas-docs/stable/cookbook.html#hdfstore



#-----Excel Sheet into Panda--------------------------------------
xls_file = pd.ExcelFile('/Users/chrisralbon/Dropbox (Personal)/Public/example.xls')
xls_file
xls_file.sheet_names

df = xls_file.parse('Sheet1')
df




#------Put XLS inside Pandas: Be careful of having <> names-----------------
def excel_topandas(filein, fileout): 
  filename= ""
  xls = pd.ExcelFile(filein)  ;  name1= xls.sheet_names  
  for name in names:
    pd_hiearchy= filename + "/" + name1
    df = xls_file.parse(name)
    df.to_hdf(fileout, pd_hiearchy, df)
  store= pd.HDFStore(fileout)
  return store     #store.keys() 
  


def panda_toexcel(dirin, fileout):
    
    
    
def panda_todabatase():
    
    
    
def database_topanda():



def sqlquery_topanda():
    
    
    
def folder_topanda( x   ,dirlevel=1):
    
    
    
def panda_tofolder():

    







In [302]: store.put('foo/bar/bah', df)
In [303]: store.append('food/orange', df)
In [304]: store.append('food/apple',  df)
In [305]: store
Out[305]: 
<class 'pandas.io.pytables.HDFStore'>
File path: store.h5
/df                     frame_table  (typ->appendable,nrows->8,ncols->3,indexers->[index])
/foo/bar/bah            frame        (shape->[8,3])                                       
/food/apple             frame_table  (typ->appendable,nrows->8,ncols->3,indexers->[index])
/food/orange            frame_table  (typ->appendable,nrows->8,ncols->3,indexers->[index])

# a list of keys are returned
In [306]: store.keys()
Out[306]: ['/df', '/food/apple', '/food/orange', '/foo/bar/bah']

# remove all nodes under this level
In [307]: store.remove('food')

In [308]: store
Out[308]: 
<class 'pandas.io.pytables.HDFStore'>
File path: store.h5
/df                     frame_table  (typ->appendable,nrows->8,ncols->3,indexers->[index])
/foo/bar/bah            frame        (shape->[8,3])       


You can just iterate thru your .csv and store/append them one by one. Something like:

for f in files:
  df = pd.read_csv(f)
  df.to_hdf('file.h5',f,df)

Would be one way (creating a separate node for each file)

the node name is f, store[f] = pd.read_csv(f) is equivalent to df.to_hdf`, 
but the df.to_hdf`` auto opens/closes the store for you. 

store['df'] = df creates/overwrites the node named 'df'. 

FYI not a good idea to keep doing this, see the section on deleting data. 

A node can only hold one object (e.g. a frame), but you can create a hierarchy of nodes if you want (e.g. node_a/df, node_a/sub_node/df effectively holds multiple frames in a single node – Jeff May 19 '13 at 18:14 








#------------Search A Pandas Column For A Value-----------------------
df['preTestScore'].where(df['postTestScore'] > 50)


'''




from scipy.interpolate import interp1d

f2 = interp1d(qq['Datetime'], qq['Open'],bounds_error=False)
open =    np.column_stack((open, f2(dater)))

f2(qq.Datetime)


open = qq[['Datetime','Close']]  #new dataframe

open = open.join(pd.DataFrame(data={'Close1':qq['Close']}), how='outer', rsuffix='_1')




f2= pd.DataFrame( qq['Close'], index=qq['Datetime'])

pd.concat([df1,df2,df3])

qq['Datetime','Close']




#------Generate Sequence of date-------------------
# Every 5 hours 10 minutes

dater= dater.values
dater= pd.date_range(start='12/24/2015', end='1/1/2016',   freq='0h05min').values
Indexing DatetimeIndex objects. (To get all data from December 2012 through the end of May 2013 data you could do df.ix['December 2012':May 2013'])



qq.ix[dater]



df.ix[datetime.date(year=2014,month=1,day=1):datetime.date(year=2014,month=2,day=1)]


#  Get values where Index is not in  
rpt[~rpt['STK_ID'].isin(stk_list)]

qq['Close'].where( qq['Datetime']= dater)


qq['Datetime' = dater]
qq.resample('5min')





# evenly spaced times
t1 = np.array([0,0.5,1.0,1.5,2.0])
y1 = t1

# unevenly spaced times
t2 = np.array([0,0.34,1.01,1.4,1.6,1.7,2.01])
y2 = 3*t2

df1 = pd.DataFrame(data={'y1':y1,'t':t1})
df2 = pd.DataFrame(data={'y2':y2,'t':t2})

f2 = interp1d(t2,y2,bounds_error=False)
df1['y2'] = f2(df1.t)

'''



#---------------------------------------------------------------
Breaking Up A String Into Columns Using Regex In Pandas

Repo: Python 3 code snippets for data science
Note: Originally based on this tutorial in nbviewer.
Import modules
In [24]:
import re
import pandas as pd
Create a dataframe of raw strings
In [25]:
# Create a dataframe with a single column of strings
data = {'raw': ['Arizona 1 2014-12-23       3242.0',
                'Iowa 1 2010-02-23       3453.7',
                'Oregon 0 2014-06-20       2123.0',
                'Maryland 0 2014-03-14       1123.6',
                'Florida 1 2013-01-15       2134.0',
                'Georgia 0 2012-07-14       2345.6']}
df = pd.DataFrame(data, columns = ['raw'])
df
Out[25]:
raw
0	Arizona 1 2014-12-23 3242.0
1	Iowa 1 2010-02-23 3453.7
2	Oregon 0 2014-06-20 2123.0
3	Maryland 0 2014-03-14 1123.6
4	Florida 1 2013-01-15 2134.0
5	Georgia 0 2012-07-14 2345.6
6 rows × 1 columns
Search a column of strings for a pattern
In [26]:
# Which rows of df['raw'] contain 'xxxx-xx-xx'?
df['raw'].str.contains('....-..-..', regex=True)
Out[26]:
0    True
1    True
2    True
3    True
4    True
5    True
Name: raw, dtype: bool
Extract the column of single digits
In [27]:
# In the column 'raw', extract single digit in the strings
df['female'] = df['raw'].str.extract('(\d)')
df['female']
Out[27]:


Name: female, dtype: object
Extract the column of dates
In [28]:
# In the column 'raw', extract xxxx-xx-xx in the strings
df['date'] = df['raw'].str.extract('(....-..-..)')
df['date']
Out[28]:
0    2014-12-23
1    2010-02-23
2    2014-06-20
3    2014-03-14
4    2013-01-15
5    2012-07-14
Name: date, dtype: object
Extract the column of thousands
In [29]:
# In the column 'raw', extract ####.## in the strings
df['score'] = df['raw'].str.extract('(\d\d\d\d\.\d)')
df['score']
Out[29]:
0    3242.0
1    3453.7
2    2123.0
3    1123.6
4    2134.0
5    2345.6
Name: score, dtype: object
Extract the column of words
In [31]:
# In the column 'raw', extract the word in the strings
df['state'] = df['raw'].str.extract('([A-Z]\w{0,})')
df['state']
Out[31]:
0     Arizona
1        Iowa
2      Oregon
3    Maryland
4     Florida
5     Georgia
Name: state, dtype: object
View the final dataframe
In [33]:
df
Out[33]:
raw	female	date	score	state
0	Arizona 1 2014-12-23 3242.0	1	2014-12-23	3242.0	Arizona
1	Iowa 1 2010-02-23 3453.7	1	2010-02-23	3453.7	Iowa
2	Oregon 0 2014-06-20 2123.0	0	2014-06-20	2123.0	Oregon
3	Maryland 0 2014-03-14 1123.6	0	2014-03-14	1123.6	Maryland
4	Florida 1 2013-01-15 2134.0	1	2013-01-15	2134.0	Florida
5	Georgia 0 2012-07-14 2345.6	0	2012-07-14	2345.6	Georgia
6 rows × 5 columns







#----------   xxx  To Panda  -------------------------------------------
def numpy_topanda(vv, fileout="", colname="data"):
 pd= pd.DataFrame(vv);  st= pd.HDFStore(fileout);  st.append(colname, pd)    

def panda_tonumpy(filename, nsize, tablen='data'):
 pdframe=  pd.read_hdf(filename, tablen, start=0, stop=(nsize))
 return pdframe.values   #to numpy vector       

def df_topanda(vv, filenameh5, colname='data'):  # 'E:\_data\_data_outlier.h5'   
 store = pd.HDFStore(filenameh5); pdf= pd.DataFrame(vv); store.append(colname, pdf); store.close()  

def load_frompanda(filenameh5, colname="data"):  # 'E:\_data\_data_outlier.h5'
 pdf=  pd.read_hdf(fileoutlier,colname); return pdf.values   #to numpy vector


def csv_topanda(filein1, filename, tablen='data', lineterminator=","): #Big CSV in Data
 #filein1=   'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.csv'
 #filename = 'E:\_data\_QUASI_SOBOL_gaussian_16384dim__4096samples.h5'
 chunksize =     10 * 10 ** 6
 list01= pd.read_csv(filein1, chunksize=chunksize, lineterminator=lineterminator)
 for chunk in list01:
     store = pd.HDFStore(filename);  store.append(tablen, chunk);   store.close()     


def getrandom_tonumpy(filename, nbdim, nbsample, tablen='data'):
 pdframe=  pd.read_hdf(filename,tablen, start=0, stop=(nbdim*nbsample))
 return pdframe.values   #to numpy vector       

# yy1= getrandom_tonumpy('E:\_data\_QUASI_SOBOL_gaussian_xx2.h5', 16384, 4096)


#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_topanda(filename, i0=0, imax=1000, expr, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe   
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx    # filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5' 
 store = pd.HDFStore(fileout);  store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):



# put Excel Sheet into panda-----
def excel_topanda(wk, r1,r2):
    
    

# put array/dataframe data into Excel
def array_toexcel(vv, wk, r1)    
    



'''

>>> %timeit df3.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
1000 loops, best of 3: 1.54 ms per loop

>>> %timeit df3.groupby(df3.index).first()
1000 loops, best of 3: 580 µs per loop

'''

#remove duplicate
df4 = df3.drop_duplicates(subset='rownum', take_last=True)



grouped = df3.groupby(level=0)
df4 = grouped.last()







#--------Clean array----------------------------------------------------------
def unique_rows(a):
    a = np.ascontiguousarray(a);  unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def remove_zeros(vv, axis1=1):    return vv[~np.all(vv == 0, axis=axis1)]

def sort_array(vv):  return vv[np.lexsort(np.transpose(vv)[::-1])]  #Sort the array by different column



#-------------=---------------------------------------------------------------






#------------------Pivot Table from dataframe---------------------------------
#http://pbpython.com/pandas-pivot-table-explained.html
table = pd.pivot_table(df,index=["Manager","Status"],   #Pivot agregation
               columns=["Product"],  #1 more layer
               values=["Quantity","Price"],  #Value to agregate
               aggfunc={"Quantity":len,"Price":[np.sum,np.mean,len]},fill_value=0)
               #agregate by sum, mean,Count    -    remove NA by zero  fill_value
table





#-------------=---------------------------------------------------------------



#----Count the strig with max size---------------
x = ['ab', 'bcd', 'dfe', 'efghik']
x = np.repeat(x, 1e7)
df = pd.DataFrame(x, columns=['col1'])

df.col1.map(len).max()








#-------------Text Processing---------------------------------------------
!pip install regex
https://bitbucket.org/mrabarnett/mrab-regex
https://pypi.python.org/pypi/regex
Unicode codepoint properties with the \p{} syntax.



#-----------------!pip install pyodbc ------------------------------------
!pip install pyodbc






















http://nbviewer.jupyter.org/github/pybokeh/ipython_notebooks/blob/master/pandas/PandasCheatSheet.ipynb





















