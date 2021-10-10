# -*- coding: utf-8 -*-
#---------Various Utilities function for Python----------------------------
import scipy as sp;import numpy as np; import numexpr as ne
import pandas as pd; import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit, vectorize, guvectorize, float64, float32, int32, boolean
from timeit import default_timer as timer

import global01 as global01 #as global varaibles   global01.varname



#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_vect_calc(filename, i0=0, imax=1000, expr, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe    #to numpy vector
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx  
# filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'   
 store = pd.HDFStore(fileout) 
 store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):


#---In Advance Calculation   New= xx*xx  over very large series
def numexpr_topanda(filename, i0=0, imax=1000, expr, fileout='E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):
 pdframe=  pd.read_hdf(filename,'data', start=i0, stop=imax)    #from file
 xx= pdframe.values;  del pdframe   
 xx= ne.evaluate(expr)  
 pdf =pd.DataFrame(xx); del xx    # filexx3=   'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5' 
 store = pd.HDFStore(fileout);  store.append('data', pdf); del pdf

#numexpr_vect_calc(filename, 0, imax=16384*4096, "xx*xx", 'E:\_data\_QUASI_SOBOL_gaussian_xx3.h5'  ):



#----Input the data: From CSV to Panda files -------------------------------
def convertcsv_topanda(filein1, filename, tablen='data'):
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
































































































































#############################################################
# Edouard TALLENT @ TaGoMa.Tech, March, 2014                #
# Parse a series of online PDF documents to  build          #
# historical time series                                    #
# QuantCorner @ https://quantcorner.wordpress.com            #
#############################################################

# Some sources
# http://www.unixuser.org/~euske/python/pdfminer/programming.html
# http://denis.papathanasiou.org/2010/08/04/extracting-text-images-from-pdf-files/
# http://stackoverflow.com/questions/25665/python-module-for-converting-pdf-to-text

# Required headers
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import  TextConverter # , XMLConverter, HTMLConverter
import urllib2
from urllib2 import Request
import datetime
import re

# Define a PDF parser function
def parsePDF(url):

    # Open the url provided as an argument to the function and read the content
    open = urllib2.urlopen(Request(url)).read()

    # Cast to StringIO object
    from StringIO import StringIO
    memory_file = StringIO(open)

    # Create a PDF parser object associated with the StringIO object
    parser = PDFParser(memory_file)

    # Create a PDF document object that stores the document structure
    document = PDFDocument(parser)

    # Define parameters to the PDF device objet 
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    codec = 'utf-8'

    # Create a PDF device object
    device = TextConverter(rsrcmgr, retstr, codec = codec, laparams = laparams)

    # Create a PDF interpreter object
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Process each page contained in the document
    for page in PDFPage.create_pages(document):
        interpreter.process_page(page)
        data =  retstr.getvalue()

    # Get values for Iowa B100 prices 
    reg = '(?<=\n---\n\n)\d.\d{2}-\d.\d{2}'
    matches = re.findall(reg, data)                 # Our data are contained in matches[0]

    # Compute the average
    # Extract value from previous regex 
    low = re.search('\d.\d{2}(?=-)', matches[0])
    high = re.search('(?<=-)\d.\d{2}', matches[0])

    # Cast string variables to float type
    low_val = float(low.group(0))
    high_val = float (high.group(0))

    # Calculate the average
    #import numpy
    #value = [high_val, low_val]
    #print value.mean
    ave = (high_val + low_val) /2

    # Search the date of the report 
    reg = '\w{3},\s\w{3}\s\d{2},\s\d{4}'
    match = re.search(reg, data)        # Result is contained in matches[0]
    dat = match.group(0)

    # Cast to date format
    #import datetime
    #form = datetime.datetime.strptime(match.group(0), '%a, %b %d, %Y')
    #print form

    # http://stackoverflow.com/questions/9752958/how-can-i-return-two-values-from-a-function-in-python
    return (dat, ave)

# The date of the latest weekly price bulletin
start_date = raw_input("Enter the date of the latest weekly price bulletin (dd/mm/yyyy): ")

# Convert start_date string to Python date format
dat = datetime.datetime.strptime(start_date,  '%d/%m/%Y')

# Time series length
back_weeks = raw_input("How many weeks back in time: ")

# A bit of order onto the screen
print '\n'
print 'Date as read in PDF' + '\t' + 'Formatted date' + '\t' + 'Value'

# Loop through the dates
for weeks in xrange(0, int(back_weeks)):

    # Basic exception handling mechamism
    try:           
        wk = datetime.timedelta(weeks = weeks)
        date_back = dat - wk

        # Construct the url
        url = 'http://search.ams.usda.gov/mndms/' + str(date_back.year) + \
              '/' + str(date_back.month).zfill(2) + '/LS' + str(date_back.year) + \
              str(date_back.month).zfill(2) + str(date_back.day).zfill(2) + \
              'WAGENERGY.PDF'

        # Call to function
        fun =  parsePDF(url)

        # Information we are after
        res = str(fun[0]) + '\t' +  str(date_back.day).zfill(2) + '/' + \
              str(date_back.month).zfill(2) +  '/' + str(date_back.year) + \
              '\t' + str(fun[1])

    except Exception:
        print 'NA\t\t\t' +  str(date_back.day).zfill(2) + '/' + \
              str(date_back.month).zfill(2) +  '/' + str(date_back.year) + \
              '\t' + 'NA'

    # Output onto the screen
    print res
