import requests
from bs4 import BeautifulSoup
import pandas as pd



def test_extract_to_pandas():
    url = 'http://my-trade.in/'
    page = requests.get(url)
    df = extract_to_pandas(page.content, 'MainContent_dataGridView1')
    print(df.head())
	

def extract_to_pandas(html, table_id=None):
	
    soup = BeautifulSoup(html, 'html.parser')
    if table_id:
        tbl = soup.find("table", {'id': table_id})
    else:
        tbl = soup.find("table")
    df = pd.read_html(str(tbl))[0]
    return df	
	

	"""
      HTML --> return pandas dataframe

      columns = 
                <th>logic ID</th>
                <th>access</th>
                <th>impression</th>
                <th>click</th>
                <th>purchase</th>
                <th>in-shop</th>
                <th>gms</th>
                <th>gms/purchase</th>
                <th>in-shop</th>
                <th>cost</th>
                <th>exposure</th>
                <th>CTR</th>
                <th>CVR</th>
                <th>in-shop</th>
                datetime   (YYYY MM DD HH:SS:MM)



	"""


def download_page(url):
    """
	URL --> return HTML page in text format
    """





html1 ="""

<!DOCTYPE html>
<html lang="en">

<head>
    <title> Dashboard </title>
    <link rel="shortcut icon" href="/static/favicon.ico">
    
        <script src="/static/js/jquery.min.js" ></script>
        <script src="/static/js/jquery.jcarousel.min.js"></script>
        <script src="/static/js/bootstrap.min.js"></script>
        <script src="/static/js/d3.v3.min.js"></script>
        <script src="/static/google-code-prettify/prettify.js"></script>
        <script src="/static/js/d3_scoupon.js"></script>

    
    
        <link rel="stylesheet" href="/static/css/bootstrap.min.css">
        <link rel="stylesheet" href="/static/css/bootstrap-theme.min.css">
        <link rel="stylesheet" href="/static/css/jcarousel.basic.css">
        <link rel="stylesheet" href="/static/google-code-prettify/prettify.css">

        <style>
            a[name]:before {
              display: block;
              content: " ";
              margin-top: -40px;
              height: 40px;
              visibility: hidden;
            }
            
            
        </style>
    
    
    
</head>
<body role="document" onload="prettyPrint()">
    <nav class="navbar navbar-inverse navbar-static-top">
        <div class="container-fluid">
            <div class="navbar-header">
                <a class="navbar-brand" href="/">SmartCoupon</a>
            </div>
            <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                <ul class="nav navbar-nav">
                    
                        <li><a href="/scheler">Campaign Sler</a></li>
                    
                    
                        <li><a href="/conthboard">Control Dad</a></li>
                    
                    
                        <li><a href="/monng">Monitoring Dashboards</a></li>
                    
                    
                    
                        <li><a href="/iory">Inventory</a></li>
                    

                </ul>
                <ul class="nav navbar-nav navbar-right">
                    <li><a href="/logout">Logout scoupon</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <div class="container-fluid">
        
            
        
        
    
       <h1>Active campaign: 20211213</h1>
    


    <ul class="nav nav-tabs">
        
            <li class="active"><a>Active</a></li>
        
        <li><a href="/monitng/list">List</a></li>
    </ul>

    <h2>
        Sales:                  169 /
        GMS:              942,485 &yen;
    </h2>

    <ul class="nav nav-tabs">
        
            <li class="active"><a>Top</a></li>
            <li><a href="/monich">Search</a></li>
            <li><a href="/movvre">Genre</a></li>
        
    </ul>

    <table class="table">
        <thead>
            <tr>
                <th>logic ID</th>
                <th>access</th>
                <th>impression</th>
                <th>click</th>
                <th>purchase</th>
                <th>in-shop</th>
                <th>gms</th>
                <th>gms/purchase</th>
                <th>in-shop</th>
                <th>cost</th>
                <th>exposure</th>
                <th>CTR</th>
                <th>CVR</th>
                <th>in-shop</th>
                
                    <th>action</th>
                
            </tr>
        </thead>
        <tbody>
        
            <tr class="info">
                <td>k83</td>
                <td>             262,141</td>
                <td>             260,403</td>
                <td>               2,618</td>
                <td>                  15</td>
                <td>                  23</td>
                <td>              92,165 &yen;</td>
                <td>               6,144 &yen;</td>
                <td>             156,286 &yen;</td>
                <td>              11,143 &yen;</td>
                <td>99.34 %</td>
                <td>1.01 %</td>
                <td>0.57 %</td>
                <td>0.88 %</td>
                
                <td><a href="/mongister" class="btn btn-danger">X</a></td>
                
            </tr>
        
            <tr class="info">
                <td>si1213</td>
                <td>             260,755</td>
                <td>             258,945</td>
                <td>               2,375</td>
                <td>                  21</td>
                <td>                  24</td>
                <td>              82,659 &yen;</td>
                <td>               3,936 &yen;</td>
                <td>             140,069 &yen;</td>
                <td>               4,999 &yen;</td>
                <td>99.31 %</td>
                <td>0.92 %</td>
                <td>0.88 %</td>
                <td>1.01 %</td>
                
                <td><a href="/moniister" class="btn btn-danger">X</a></td>
                
            </tr>
        
            <tr class="info">
                <td>kvn-20211213</td>
                <td>              29,061</td>
                <td>              28,504</td>
                <td>                 305</td>
                <td>                   0</td>
                <td>                   0</td>
                <td>                   0 &yen;</td>
                <td>                   0 &yen;</td>
                <td>                   0 &yen;</td>
                <td>                   0 &yen;</td>
                <td>98.08 %</td>
                <td>1.07 %</td>
                <td>0.0 %</td>
                <td>0.0 %</td>
                
                <td><a href="/moster" class="btn btn-danger">X</a></td>
                
            </tr>
        
            <tr>
                <th>total</th>
                <th>             551,957</th>
                <th>             547,852</th>
                <th>               5,298</th>
                <th>                  36</th>
                <th>                  47</th>
                <th>             174,824 &yen;</th>
                <th>              10,080 &yen;</th>
                <th>             296,355 &yen;</th>
                <th>              16,142 &yen;</th>
            </tr>
        </tbody>
    </table>

    
        <p>
            <a href="/monigister" class="btn btn-primary">REGISTER LOGIC</a>
        </p>
    


    </div>
</body>
</html>

"""
