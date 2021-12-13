from bs4 import BeautifulSoup as bs
from urllib.request import Request, urlopen
from Credentials import myToken, parent
import urllib
import time
import os
import json
import csv
import requests
import sys

# def upload_to_drive(filename):
#     headers = {"Authorization": "Bearer {authorizationToken}".format(authorizationToken=myToken)}
#     para = {
#         "name": filename,
#         "parents":[parent]
#     }
#     files = {
#         'data': ('metadata', json.dumps(para), 'application/json; charset=UTF-8'),
#         'file': open(filename, "rb")
#     }
#     r = requests.post(
#         "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
#         headers=headers,
#         files=files
#     )
#     print(r.text)

current_dir = os.getcwd()
path = os.path.join(current_dir,'Product_Images_and_CSV')
os.mkdir(path)
os.chdir(path)

csv_file = open('Rakuten.csv','w',encoding="utf-8")
csv_writer = csv.writer(csv_file, delimiter='\t')
csv_writer.writerow(['Image Name','Product Title', 'Price','Shop Name','Review Count','product_url','scrape_page_url'])

page = 1
count = 0
while page!=151:
    try:
        rakuten_url = 'https://search.rakuten.co.jp/search/mall/-/555087/?p='+str(page)
        req = Request(url=rakuten_url)
        source = urlopen(req).read()
        soup = bs(source,'lxml')

        for individual_item in soup.find_all('div',class_='searchresultitem'):
            count += 1
            save = 0
            shopname = 'Shop Not Available'
            count_review = 'Number of Reviews Not Available'

            for names in individual_item.find_all('div',class_='title'):
                product_name = names.h2.a.text
                break

            for price in individual_item.find_all('div',class_='price'):
                product_price = price.span.text
                break
            
            for url in individual_item.find_all('div',class_='image'):
                product_url = url.a.get('href')
                break

            for images in individual_item.find_all('div',class_='image'):
                try:
                    product_image = images.a.img.get('src')
                    urllib.request.urlretrieve(product_image,str(count)+".jpg")
                    # upload_to_drive(str(count)+'.jpg')
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
                csv_writer.writerow([str(count)+'.jpg',product_name,product_price,shopname,count_review,product_url,rakuten_url])
    except:
        time.sleep(5)
        continue

    page += 1

# upload_to_drive("Rakuten.csv")
print("Site Scraped Successfully")
