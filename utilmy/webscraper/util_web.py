# -*- coding: utf-8 -*-
from __future__ import division

# ---------Various Utilities function for Python--------------------------------------
# if sys.platform.find('win') > -1 :
#  from guidata import qthelpers  #Otherwise Erro with Spyder Save
import os
import sys
from builtins import str

import numpy as np
import requests
import urllib3
from bs4 import BeautifulSoup
from future import standard_library

# noinspection PyUnresolvedReferences
from attrdict import AttrDict as dict2

standard_library.install_aliases()

# CFG   = {'plat': sys.platform[:3]+"-"+os.path.expanduser('~').split("\\")[-1].split("/")[-1],
# "ver": sys.version_info.major}
# DIRCWD= {'win-asus1': 'D:/_devs/Python01/project27/', 'win-unerry': 'G:/_devs/project27/',
# 'lin-noel': '/home/noel/project27/', 'lin-ubuntu': '/home/ubuntu/project27/' }[CFG['plat']]
# DIRCWD= os.environ["DIRCWD"];
# os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')
# f= open(DIRCWD+'/__config/config.py'); CFG= dict2(dict(CFG,  **eval(f.read()))); f.close()


DIRCWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(DIRCWD)
sys.path.append(DIRCWD + "/aapackage")


# __path__= DIRCWD +'/aapackage/'
# __version__= "1.0.0"
# __file__= "util.py"

####################################################################################################
# print(os.environ)


# Headless PhantomJS ##############################################################################
def web_get_url_loginpassword(
    url_list=None,
    browser="phantomjs",
    login="",
    password="",
    phantomjs_path="D:/_devs/webserver/phantomjs-1.9.8/phantomjs.exe",
    pars=None,
):
    """
   from selenium import webdriver
   import time
   # Issue with recent selenium on firefox...
   # conda install -c conda-forge selenium ==2.53.6 
 """
    if pars is None:
        pars = {
            "url_login": "https://github.com/login",
            "login_field": "username",
            "password_field": "password",
            "submit_field": "commit",
        }
    if url_list is None:
        url_list = ["url1", "url2"]
    pa = dict2(pars)

    from selenium import webdriver
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

    DesiredCapabilities.PHANTOMJS[
        "phantomjs.page.settings.userAgent"
    ] = "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:16.0) Gecko/20121026 Firefox/16.0"

    if browser == "firefox":
        driver = webdriver.Firefox()
    else:
        driver = webdriver.PhantomJS(
            phantomjs_path
        )  # r"D:/_devs/webserver/phantomjs-1.9.8/phantomjs.exe"

    ######## Login      ######
    if login != "" and password != "":
        driver.get(pa.url_login)
        username = driver.find_element_by_id(pa.login_field)
        password = driver.find_element_by_id(pa.password_field)
        username.clear()
        username.send_keys(login)
        password.clear()
        password.send_keys(password)
        driver.find_element_by_name(pa.submit_field).click()

    # Search Query
    for url in url_list:
        driver.get(url)
        html = driver.page_source
        print(html)

    '''
  os_folder_create(outputfolder)

 # INSERT KEYWORDS
 kw_query = ''
 for kw in keywords:  kw_query = kw_query + '%22' + kw + '%22+'

 print("Search results:", flush=True)
 box_id = 0
 list_of_dicts = []
 try :
  for page in range(page_start, page_end+1):
        print("\nPage "+str(page)+": ", end=' ',  flush=True)

        base_url = ('https://github.com/search?l=Python&p='  + str(page) + '&q=' + kw_query +
            '&type=Code&utf8=%E2%9C%93')
        driver.get(base_url)
        html1 = driver.page_source
        soup = BeautifulSoup(html1, 'lxml')

        #Scraping
        for desc, blob in zip(soup.findAll('div', class_='d-inline-block col-10'),
                              soup.findAll('div', class_='file-box blob-wrapper')):

            box_id = box_id + 1
            print(box_id, end=' ',  flush=True)

            dict1 = {"url_scrape": '', "keywords": keywords, "language": 'Python', "page": '',
                'box_id': '', 'box_date': '', 'box_text': '', 'box_reponame': '', 'box_repourl': '',
                'box_filename': '', 'box_fileurl': '', 'url_scrape': base_url, 'page': str(page)}

            urls = desc.findAll('a')
            dict1['box_repourl'] = 'https://github.com' + urls[0]['href']
            dict1['box_fileurl'] = 'https://github.com' + urls[1]['href']
            driver.get(dict1['box_fileurl'])

            ######### DOWNLOADING    #############################################
            if isdownload_file:
               driver.find_element_by_xpath('//*[@id="raw-url"]').click()
               if isdebug : print(driver.current_url,  flush=True)
               wget.download(driver.current_url, outputfolder)

            dict1['box_id'] =       box_id
            dict1['box_reponame'] = desc.text.strip().split(' ')[0].split('/')[-1].strip('\n')
            dict1['box_filename'] = desc.text.strip().split('\n      –\n      ')[1].split('\n')[0]
            dict1['box_date'] = desc.text.strip().split(
                '\n      –\n      ')[1].split('\n')[3].strip('Last indexed on ')
            blob_code = """ """
            for k in blob.findAll('td', class_='blob-code blob-code-inner'):
                aux= k.text.rstrip()
                if len(aux) > 1 :   blob_code= blob_code +  "\n" + aux
            dict1['box_text'] = blob_code

            list_of_dicts.append(dict1)
 except  Exception as e : print(e)
 driver.quit()

 df = pd.DataFrame(list_of_dicts)
 print("Nb elements:"+str(len(df)))

 df.to_csv(outputfolder +"/"+outputfile, index=False, mode='w')
 if isreturn_df :    return df
 
 '''


# Internet data connect- #################################################################
"""
https://moz.com/
    devblog/benchmarking-python-content-extraction-algorithms-dragnet-readability-goose-and-eatiht/
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

"""


def web_restapi_toresp(apiurl1):

    resp = requests.get(apiurl1)
    if resp.status_code != 200:  # This means something went wrong.
        raise requests.ConnectionError("GET /tasks_folder/ {}".format(resp.status_code))
    return resp


def web_getrawhtml(url1):

    resp = requests.get(url1)
    if resp.status_code != 200:  # This means something went wrong.
        raise requests.ConnectionError("GET /tasks_folder/ {}".format(resp.status_code))
    else:
        return resp.text


def web_importio_todataframe(apiurl1, isurl=1):

    resp = None
    if isurl:
        resp = requests.get(apiurl1)
        if resp.status_code != 200:  # This means something went wrong.
            raise requests.ConnectionError("GET /tasks_folder/ {}".format(resp.status_code))
    au = resp.json()
    txt = au["extractorData"]["data"][0]["group"]
    colname = []
    i = -1
    for row in txt:
        i += 1
        if i == 1:
            break
        for key, value in list(row.items()):
            if i == 0:
                colname.append(str(key))
    colname = np.array(colname)
    colmax = len(colname)

    dictlist = np.empty((5000, colmax), dtype=np.object)
    i = -1
    for row in txt:
        j = 0
        i += 1
        for key, value in list(row.items()):
            dictlist[i, j] = str(value[0]["text"])
            j += 1

    dictlist = dictlist[0 : i + 1, :]
    df = util.pd_createdf(dictlist, col1=colname, idx1=np.arange(0, len(dictlist)))
    return df


def web_getjson_fromurl(url):
    import json

    http = urllib3.connection_from_url(url)
    jsonurl = http.urlopen("GET", url)

    # soup = BeautifulSoup(page)
    print(jsonurl)
    data = json.loads(jsonurl.read())

    return data

    # return the title and the text of the article at the specified url


def web_gettext_fromurl(url, htmltag="p"):
    http = urllib3.connection_from_url(url)
    page = http.urlopen("GET", url).data.decode("utf8")

    soup = BeautifulSoup(page, "lxml")
    text = " \n\n".join([p.text for p in soup.find_all("p")])
    return soup.title.text + "\n\n" + text


def web_gettext_fromhtml(file1, htmltag="p"):
    with open(file1, "r", encoding="UTF-8") as f:
        page = f.read()

    soup = BeautifulSoup(page, "lxml")
    text = " \n\n".join([p.text for p in soup.find_all(htmltag)])
    return soup.title.text + "\n\n" + text


"""
I know its been said already, 
but I'd highly recommend the Requests python package
: http://docs.python-requests.org/en/latest/index.html

If you've used languages other than python, 
you're probably thinking urllib and urllib2 are easy to use, 
not much code, and highly capable, that's how I used to think. 
But the Requests package is so unbelievably useful and 
short that everyone should be using it.

First, it supports a fully restful API, and is as easy as:

...

resp = requests.get('http://www.mywebsite.com/user')
resp = requests.post('http://www.mywebsite.com/user')
resp = requests.put('http://www.mywebsite.com/user/put')
resp = requests.delete('http://www.mywebsite.com/user/delete')
Regardless of whether GET/POST you never have to encode parameters again, it simply takes
a dictionary as an argument and is good to go.

userdata = {"firstname": "John", "lastname": "Doe", "password": "jdoe123"}
resp = requests.post('http://www.mywebsite.com/user', params=userdata)
Plus it even has a built in json decoder (again, i know json.loads() isn't a lot more to write,
but this sure is convenient):

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
"""


def web_getlink_fromurl(url):
    http = urllib3.connection_from_url(url)
    page = http.urlopen("GET", url).data.decode("utf8")
    soup = BeautifulSoup(page, "lxml")
    soup.prettify()
    links = []
    for anchor in soup.findAll("a", href=True):
        lnk = anchor["href"]
        links.append(anchor["href"])

    return set(links)


def web_send_email(
    FROM,
    recipient,
    subject,
    body,
    login1="mizenjapan@gmail.com",
    pss1="sophieelise237",
    server1="smtp.gmail.com",
    port1=465,
):
    """  # send_email("Kevin", "brookm291@gmail.com", "JapaneseText:" , "txt") """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    #    TO = recipient if type(recipient) is list else [recipient]
    TO = recipient
    msg = MIMEMultipart("alternative")
    msg.set_charset("utf-8")
    msg["Subject"] = subject
    msg["From"] = FROM
    msg["To"] = TO
    part2 = MIMEText(body, "plain", "utf-8")
    msg.attach(part2)

    try:  # SMTP_SSL Example
        server_ssl = smtplib.SMTP_SSL(server1, port1)
        server_ssl.ehlo()  # optional, called by login()
        server_ssl.login(login1, pss1)
        server_ssl.sendmail(FROM, [TO], msg.as_string())
        server_ssl.close()
        print("successfully sent the mail")
        return 1
    except:
        print("failed to send mail")
        return -1


def web_send_email_tls(
    FROM,
    recipient,
    subject,
    body,
    login1="mizenjapan@gmail.com",
    pss1="sophieelise237",
    server1="smtp.gmail.com",
    port1=465,
):
    # send_email("Kevin", "brookm291@gmail.com", "JapaneseText:" , "txt")
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    #    TO = recipient if type(recipient) is list else [recipient]
    TO = recipient
    msg = MIMEMultipart("alternative")
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

        print("successfully sent the mail")
        return 1
    except:
        print("failed to send mail")
        return -1


def web_sendurl(url1):
    # Send Text by email
    mm = web_gettext_fromurl(url1)
    web_send_email("Python", "brookm291@gmail.com", mm[0:30], url1 + "\n\n" + mm)
