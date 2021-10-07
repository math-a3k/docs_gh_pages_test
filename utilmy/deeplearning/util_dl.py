# -*- coding: utf-8 -*-
HELP="""

utils in DL

"""
import os,io, numpy as np, sys, glob, time, copy, json, functools, pandas as pd
from typing import Union
from box import Box
import diskcache as dc



################################################################################################
verbose = 0

def log(*s):
    print(*s, flush=True)


def log2(*s):
    if verbose >1 : print(*s, flush=True)


def help():
    from utilmy import help_create
    ss  = HELP
    ss += help_create("utilmy.deeplearning.util_dl")
    print(ss)



################################################################################################
def test():
    pass




################################################################################################
################################################################################################
def tensorboard_log(pars_dict:dict=None,  writer=None,  verbose=True):
    """
    #### Usage 1
    logdir = 'logs/params'

    cc = {'arbitray dict' : 1 }

    from tensorboardX import SummaryWriter
    # from tensorboard import SummaryWriter
    tb_writer = SummaryWriter(logdir)
    tensorboard_log(cc, writer= tb_writer)

    %reload_ext tensorboard
    %tensorboard --logdir logs/params/
    """
    import collections
    def dict_flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(dict_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


    flatten_box = dict_flatten(pars_dict)
    if verbose:
        print(flatten_box)


    for k, v in flatten_box.items():
        if isinstance(v, (int, float)):
            writer.add_scalar(str(k), v, 0)
        else :
            writer.add_text(str(k), str(v), 0)

    writer.close()
    return writer



        
def tf_check():
    #### python prepro.py check_tf 
    import tensorflow as tf
    print( tf.config.list_physical_devices())


    
def gpu_usage(): 
    
   cmd = "nvidia-smi --query-gpu=pci.bus_id,utilization.gpu --format=csv"

   from utilmy import os_system    
   res = os_system(cmd)
   print(res)
        
   ## cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv"
   ## cmd2= " nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv  "
    
    
def gpu_free():  
    cmd = "nvidia-smi --query-gpu=pci.bus_id,utilization.gpu --format=csv  "
    from utilmy import os_system    
    ss = os_system(cmd)

    # ss   = ('pci.bus_id, utilization.gpu [%]\n00000000:01:00.0, 37 %\n00000000:02:00.0, 0 %\n00000000:03:00.0, 0 %\n00000000:04:00.0, 89 %\n', '')
    ss   = ss[0]
    ss   = ss.split("\n")
    ss   = [ x.split(",") for x in ss[1:] if len(x) > 4 ]
    print(ss)
    deviceid_free = []
    for ii, t in enumerate(ss) :
        if  " 0 %"  in t[1]:
            deviceid_free.append( ii )
    print( deviceid_free )        
    return deviceid_free
            
            
   

def down_page(query, out_dir="query1", genre_en='', id0="", cat="", npage=1) :
    """
        python prepro.py down_page  'メンス+ポロシャツ'    --out_dir men_fs_blue  


    """
    import time, os, json, csv, requests, sys, urllib
    from bs4 import BeautifulSoup as bs
    from urllib.request import Request, urlopen
    import urllib.parse


    path = "/datrakuten/" + out_dir + "/"
    os.makedirs(path, exist_ok=True)
    # os.chdir(path)

    query2     = urllib.parse.quote(query, encoding='utf-8')
    url_prefix = 'httpl/' + query2
    ### https://search.rakuten.co.jp/search/mall/%E3%83%A1%E3%8384+blue+/?p=2
    print(url_prefix)
    print(path)

    csv_file   = open( path + 'ameta.csv','w',encoding="utf-8")
    csv_writer = csv.writer(csv_file, delimiter='\t')
    csv_writer.writerow(['path', 'id0', 'cat', 'genre_en', 'image_name', 'price','shop','item_url','page_url',  ])

    page  = 1
    count = 0
    while page < npage+1 :
        try:
            rakuten_url = url_prefix  + f"/?p=+{page}"
            req    = Request(url=rakuten_url)
            source = urlopen(req).read()
            soup   = bs(source,'lxml')

            print('page', page, str(soup)[:5], str(rakuten_url)[-20:],  )

            for individual_item in soup.find_all('div',class_='searchresultitem'):
                count += 1
                save = 0
                shopname     = 'nan'
                count_review = 'nan'

                for names in individual_item.find_all('div',class_='title'):
                    product_name = names.h2.a.text
                    break

                for price in individual_item.find_all('div',class_='price'):
                    product_price = price.span.text
                    product_price = product_price .replace("円", "").replace(",", "") 
                    break
                
                for url in individual_item.find_all('div',class_='image'):
                    product_url = url.a.get('href')
                    break

                for images in individual_item.find_all('div',class_='image'):
                    try:
                        product_image = images.a.img.get('src')
                        urllib.request.urlretrieve(product_image, path + str(count)+".jpg")
                        # upload_to_drive(str(count)+'.jpg')
                        count += 1
                        break
                    except:
                        save = 1
                        print(product_url + " Error Detected")
                    
                for simpleshop in individual_item.find_all('div',class_='merchant'):
                    shopname = simpleshop.a.text
                    break

                for review in individual_item.find_all('a',class_='dui-rating-filter'):
                    count_review = review.text

                if save == 0:
                    csv_writer.writerow([str(count)+'.jpg', id0, cat, genre_en,  product_name, product_price, shopname, product_url, rakuten_url, ])

        except Exception as e :
            print(e)
            time.sleep(2)
            continue

        page += 1

    print("Success", page-1, count)






