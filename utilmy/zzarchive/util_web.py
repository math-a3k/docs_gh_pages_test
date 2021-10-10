# coding=utf-8
from __future__ import division
from future import standard_library
standard_library.install_aliases()
from builtins import next
from builtins import map
from builtins import zip
from builtins import str
from builtins import range
from past.builtins import basestring
from past.utils import old_div
from builtins import object
# -*- coding: utf-8 -*-
#---------Various Utilities function for Python--------------------------------------
import os, sys
# if sys.platform.find('win') > -1 :
#  from guidata import qthelpers  #Otherwise Erro with Spyder Save

import datetime, time, arrow,  shutil,  IPython, gc
import matplotlib.pyplot as plt
import numexpr as ne, numpy as np, pandas as pd, scipy as sci
import urllib3
from bs4 import BeautifulSoup
from numba import jit, float32

import util as util




##############Internet data connect- #################################################################
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

def web_restapi_toresp(apiurl1):
 import requests
 resp = requests.get(apiurl1)
 if resp.status_code != 200:     # This means something went wrong.
    raise ApiError('GET /tasks_folder/ {}'.format(resp.status_code))
 return resp

def web_getrawhtml(url1) :
 import requests
 resp = requests.get(url1)
 if resp.status_code != 200:  # This means something went wrong.
    raise ApiError('GET /tasks_folder/ {}'.format(resp.status_code))
 else:
    return resp.text

def web_importio_todataframe(apiurl1, isurl=1):
 import requests
 if isurl :
   resp = requests.get(apiurl1)
   if resp.status_code != 200:     # This means something went wrong.
    raise ApiError('GET /tasks_folder/ {}'.format(resp.status_code))
 au= resp.json()
 txt= au['extractorData']['data'][0]['group']
 colname=[]; i=-1
 for row in txt :
   i+=1;
   if i==1: break;
   for key, value in list(row.items()):
     if i==0:  colname.append( str(key))
 colname= np.array(colname); colmax=len(colname)

 dictlist= np.empty((5000, colmax), dtype=np.object); i=-1
 for row in txt :
   j=0; i+=1
   for key, value in list(row.items()):
     dictlist[i,j]= str(value[0]['text'])
     j+=1

 dictlist= dictlist[0:i+1,:]
 df= pd_createdf(dictlist, col1=colname, idx1= np.arange(0, len(dictlist)))
 return df

def web_getjson_fromurl(url):
 import json
 http = urllib3.connection_from_url(url)
 jsonurl = http.urlopen('GET',url)

 # soup = BeautifulSoup(page)
 print(jsonurl)
 data = json.loads(jsonurl.read())

 return data



 # return the title and the text of the article at the specified url

def web_gettext_fromurl(url, htmltag='p'):
 http = urllib3.connection_from_url(url)
 page = http.urlopen('GET',url).data.decode('utf8')

 soup = BeautifulSoup(page, "lxml")
 text = ' \n\n'.join([p.text for p in soup.find_all('p')])
 return soup.title.text + "\n\n" + text

def web_gettext_fromhtml(file1, htmltag='p'):
 with open(file1, 'r',encoding='UTF-8',) as f:
   page=f.read()

 soup = BeautifulSoup(page, "lxml")
 text = ' \n\n'.join([p.text for p in soup.find_all(htmltag)])
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

def web_send_email(FROM, recipient, subject, body, login1="mizenjapan@gmail.com", pss1="sophieelise237", server1="smtp.gmail.com", port1=465):
    '''  # send_email("Kevin", "brookm291@gmail.com", "JapaneseText:" , "txt") '''
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
        server_ssl = smtplib.SMTP_SSL( server1, port1)
        server_ssl.ehlo() # optional, called by login()
        server_ssl.login(login1, pss1)
        server_ssl.sendmail(FROM, [TO], msg.as_string())
        server_ssl.close();        print ('successfully sent the mail'  )
        return 1
    except:
        print( "failed to send mail")
        return -1

def web_send_email_tls(FROM, recipient, subject, body, login1="mizenjapan@gmail.com", pss1="sophieelise237",
                   server1="smtp.gmail.com", port1=465):
    # send_email("Kevin", "brookm291@gmail.com", "JapaneseText:" , "txt")
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    #    TO = recipient if type(recipient) is list else [recipient]
    TO = recipient
    msg = MIMEMultipart("alternative");
    msg.set_charset("utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM
    msg["To"] = TO
    part2 = MIMEText(body, "plain", "utf-8")
    msg.attach(part2)

    try:  # SMTP_SSL Example
        mailserver = smtplib.SMTP(server1, port1)
        # identify ourselves to smtp gmail ec2
        mailserver.ehlo()
        # secure our email with tls encryption
        mailserver.starttls()
        # re-identify ourselves as an encrypted connection
        mailserver.ehlo()
        mailserver.login(login1, pss1)

        mailserver.sendmail(FROM, [TO], msg.as_string())
        mailserver.quit()

        print ('successfully sent the mail')
        return 1
    except:
        print("failed to send mail")
        return -1


def web_sendurl(url1):
 # Send Text by email
 mm= web_gettext_fromurl(url1)
 send_email("Python", "brookm291@gmail.com", mm[0:30] , url1 + '\n\n'+ mm )








