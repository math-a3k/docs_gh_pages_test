from utilmy.viz import vizhtml as vi
import os, sys, random, numpy as np, pandas as pd, time
from datetime import datetime
from typing import List
from tqdm import tqdm
from box import Box
 
      
def test_getdata(verbose=True):
    """data = test_get_data()
    df   = data['housing.csv']
    df.head(3)
    https://github.com/szrlee/Stock-Time-Series-Analysis/tree/master/data
    """
    import pandas as pd
    flist = [
        'https://raw.githubusercontent.com/samigamer1999/datasets/main/titanic.csv',
        'https://github.com/subhadipml/California-Housing-Price-Prediction/raw/master/housing.csv',
        'https://raw.githubusercontent.com/AlexAdvent/high_charts/main/data/stock_data.csv',
        'https://raw.githubusercontent.com/samigamer1999/datasets/main/cars.csv',
        'https://raw.githubusercontent.com/samigamer1999/datasets/main/sales.csv',
        'https://raw.githubusercontent.com/AlexAdvent/high_charts/main/data/weatherdata.csv'
    ]
    data = {}
    for url in flist :
       fname =  url.split("/")[-1]
       print( "\n", "\n", url, )
       df = pd.read_csv(url)
       data[fname] = df
       if verbose: print(df)
       # df.to_csv(fname , index=False)
    print(data.keys() )
    return data

def test1(verbose=False):
    ####  Test Datatable
    doc = vi.htmlDoc(dir_out="", title="hello", format='myxxxx', cfg={})
    # check add css
    css = """.intro { background-color: yellow;} """
    doc.add_css(css)
    # test create table
    df = test_getdata()['titanic.csv']
    doc.h1(" Table test ")
    doc.table(df, use_datatable=True, table_id="test", custom_css_class='intro')
    if verbose: doc.print()
    doc.save(dir_out="testdata/test_viz_table.html")
    doc.open_browser()  # Open myfile.html

def test2(verbose=False):
    """
      # pip install --upgrade utilmy
      from util.viz import vizhtml as vi
      vi.test2()
    """
    data = test_getdata(verbose=verbose)
    doc = vi.htmlDoc(title='Weather report', dir_out="", cfg={} )
    doc.h1(' Weather report')
    doc.hr()

    # create time series chart. mode highcharts
    doc.h2('Plot of weather data') 
    doc.plot_tseries(data['weatherdata.csv'].iloc[:1000, :],coldate=  'Date',date_format =  '%m/%d/%Y',
                      coly1   =  ['Temperature'],coly2   =  ["Rainfall"],
                      # xlabel='Date',  y1label=  "Temperature", y2label=  "Rainfall", 
                     title = "Weather",cfg={}, mode='highcharts')
    doc.hr() 
    doc.h3('Weather data') 
    doc.table(data['weatherdata.csv'].iloc[:10 : ], use_datatable=True )

    # create histogram chart. mode highcharts
    doc.plot_histogram(data['housing.csv'].iloc[:1000, :], col="median_income",
                       xaxis_label= "x-axis",yaxis_label="y-axis",cfg={}, mode='highcharts', save_img=False)

     # Testing with example data sets (Titanic)
    cfg = {"title" : "Titanic", 'figsize' : (20, 7)}

    # create scatter chart. mode highcharts
    doc.plot_scatter(data['titanic.csv'].iloc[:50, :], colx='Age', coly='Fare',
                         collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                         figsize=(20,7),cfg=cfg, mode='highcharts')

    doc.save('viz_test3_all_graphs.html')
    doc.open_browser()
    html1 = doc.get_html()
    # html_show(html1)

def test3(verbose=False):
    # pip install box-python    can use .key or ["mykey"]  for dict
    data = test_getdata(verbose=verbose)
    df2  = data['sales.csv']
    from box import Box
    cfg = Box({})
    cfg.tseries = {"title": 'ok'}
    cfg.scatter = {"title" : "Titanic", 'figsize' : (12, 7)}
    cfg.histo   = {"title": 'ok'}
    cfg.use_datatable = True

    df = pd.DataFrame([[1, 2]])
    df2_list = [df, df, df]
    print(df2_list)
    doc = vi.htmlDoc(dir_out="", title="hello", format='myxxxx', cfg=cfg)

    doc.h1('My title')  # h1
    doc.sep()
    doc.br()  # <br>
    doc.tag('<h2> My graph title </h2>')
    doc.plot_scatter(data['titanic.csv'], colx='Age', coly='Fare',
                     collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                     cfg=cfg.scatter, mode='matplot', save_img='')
    doc.hr()  # doc.sep() line separator
    # for df2_i in df2_list:
    #      print(df2_i)
    #      col2 =df2_i.columns
    #      # doc.h3(f" plot title: {df2_i['category'].values[0]}")
    #      doc.plot_tseries(df2_i, coldate= col2[0], coly1= col2[1],   cfg = cfg.tseries, mode='highcharts')

    doc.tag('<h2> My histo title </h2>')
    doc.plot_histogram(df2,col='Unit Cost',mode='matplot', save_img="")
    doc.plot_histogram(df2,col='Unit Price',cfg =  cfg.histo,title="Price", mode='matplot', save_img="")

    doc.save(dir_out="myfile.html")
    doc.open_browser()  # Open myfile.html

def test4(verbose=False):
    data = test_getdata(verbose=verbose)
    from box import Box
    cfg = Box({})
    cfg.tseries = {"title": 'ok'}
    cfg.scatter = {"title" : "Titanic", 'figsize' : (12, 7)}
    cfg.histo   = {"title": 'ok'}
    cfg.use_datatable = True
    doc = vi.htmlDoc(dir_out="", title="hello", format='myxxxx', cfg=cfg)
    # table
    doc.h1(" Table test ")
    doc.table(data['titanic.csv'], use_datatable=True, table_id="test", custom_css_class='intro')
    doc.hr()
    # histogram
    doc.h1(" histo test ")
    doc.plot_histogram(data['sales.csv'],col='Unit Price',colormap='RdYlBu',cfg =  cfg.histo,title="Price",ylabel="Unit price", mode='matplot', save_img="")
    doc.plot_histogram(data['housing.csv'].iloc[:1000, :], col="median_income",xaxis_label= "x-axis",yaxis_label="y-axis",cfg={}, mode='highcharts', save_img=False)
    doc.hr()
    #  scatter plot
    doc.tag('<h2> Scater Plot </h2>')
    doc.plot_scatter( data['titanic.csv'], colx='Age', coly='Fare',
                     collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                     cfg=cfg.scatter, mode='matplot', save_img='')
    doc.plot_scatter(data['titanic.csv'].iloc[:50, :], colx='Age', coly='Fare',
                         collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                         figsize=(20,7),cfg=cfg, mode='highcharts',)
    
    # create time series chart. mode highcharts
    doc.h2('Plot of weather data') 
    doc.plot_tseries(data['weatherdata.csv'].iloc[:1000, :],coldate='Date',date_format =  '%m/%d/%Y',
                      coly1   =  ['Temperature'],coly2   =  ["Rainfall"],
                      # xlabel=     'Date', y1label=  "Temperature", y2label=  "Rainfall", 
                     title ="Weather",cfg={},mode='highcharts')
    doc.hr()
    # plot network
    doc.h1(" plot network test ")
    df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'], 'weight':[1, 2, 1,5]})
    doc.pd_plot_network(df, cola='from', colb='to', coledge='col_edge',colweight="weight")

    doc.save('test4.html')
    doc.open_browser()
    html1 = doc.get_html()
    
def test_scatter_and_histogram_matplot(verbose=False):
  data = test_getdata(verbose=verbose)
  cfg = Box({})
  cfg.tseries = {"title": 'ok'}
  cfg.scatter = {"title" : "Titanic", 'figsize' : (12, 7)}
  cfg.histo   = {"title": 'ok'}
  cfg.use_datatable = True

  doc = vi.htmlDoc(dir_out="", title="hello", format='myxxxx', cfg=cfg)
  doc.h1('My title')  # h1
  doc.sep()
  doc.br()  # <br>

  doc.tag('<h2> Test Histogram and Scatter </h2>')
  doc.plot_scatter(data['titanic.csv'], colx='Age', coly='Fare',
                    collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                    cfg=cfg.scatter, mode='matplot', save_img='')
  doc.hr()  # doc.sep() line separator
  doc.plot_histogram(data['sales.csv'],col='Unit Cost',mode='matplot', save_img="")
  doc.save(dir_out="myfile.html")
  doc.open_browser()  # Open myfile.html-

def test_pd_plot_network(verbose=False):
  df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'], 'weight':[1, 2, 1,5]})
  html_code = vi.pd_plot_network(df, cola='from', colb='to', coledge='col_edge',colweight="weight")
  if verbose: print(html_code)

def test_cssname(verbose=False,css_name="a4"):
    # pip install box-python    can use .key or ["mykey"]  for dict
    data = test_getdata(verbose=verbose)
    doc = vi.htmlDoc(title="hello",css_name=css_name, format='myxxxx')

    doc.h1('My title')  # h1
    doc.sep()
    doc.br()  # <br>

    doc.tag('<h2> Test Cssname </h2>')
    doc.plot_scatter(data['titanic.csv'], colx='Age', coly='Fare',
                     collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                     mode='matplot', save_img='')
    doc.save(dir_out="myfile.html")
    doc.open_browser()  # Open myfile.html

def test_external_css():
  from box import Box
  cfg = Box({})
  cfg.tseries = {"title": 'ok'}
  cfg.scatter = {"title" : "Titanic", 'figsize' : (12, 7)}
  cfg.histo   = {"title": 'ok'}
  cfg.use_datatable = True
  # loading border style from external css
  doc = vi.htmlDoc(title="hello", format='myxxxx',css_name='None',css_file='https://alexadvent.github.io/style.css')
  data = test_getdata()
  # table
  doc.h1(" Table test ")
  doc.table(data['titanic.csv'], use_datatable=True, table_id="test", custom_css_class='intro')
  doc.hr()
  # histogram
  doc.h1(" histo test ")
  doc.plot_histogram(data['housing.csv'].iloc[:1000, :], col="median_income", mode='highcharts')
  doc.hr()
  doc.save('test4.html')
  doc.open_browser()
  html1 = doc.get_html()

def test_table():
   url = 'https://raw.githubusercontent.com/AlexAdvent/high_charts/main/table_data.csv'
   df = pd.read_csv(url)
   vi.log( df.head() )
   vi.show_table_image(df, colimage='image_url', colgroup='name', title='test_table')
   
def test_page():
    # get data
    data = test_getdata(verbose=False)
    
    from box import Box
    cfg = Box({})
    cfg.tseries = {"title": 'tseries_title'}
    cfg.scatter = {"title" : "scatter_title", 'figsize' : (12, 7)}
    cfg.histo   = {"title": 'histo_title'}

    # initialize htmldoc
    doc = vi.htmlDoc(title='Weather report', dir_out="", cfg={}, css_name= "a4")

    # test_table
    doc.h1("Test Table")
    doc.table(data['titanic.csv'][0:30], use_datatable=True, table_id="test", custom_css_class='intro', format='grey_dark')  
    doc.table(data['stock_data.csv'][0:10], use_datatable=False, table_id="test_false_datatable", custom_css_class='intro', format='orange_dark')
    doc.sep()

    # plot histogram
    doc.h1("Test Histogram")
    doc.h3("histogram highchart")
    doc.plot_histogram(data['housing.csv'].iloc[:1000, :], col="median_income",
                       xaxis_label= "x-axis",yaxis_label="y-axis",cfg={}, mode='highcharts',title="test_histo", save_img=False)
    doc.h4("nbin")
    doc.plot_histogram(data['housing.csv'].iloc[:1000, :], col="median_income",
                       xaxis_label= "x-axis",yaxis_label="y-axis",cfg={},nbin=30, mode='highcharts',title="test_histo",color="lightblue", save_img=False)
    doc.h4("bin width")
    doc.plot_histogram(data['housing.csv'].iloc[:1000, :], col="median_income",
                       xaxis_label= "x-axis",yaxis_label="y-axis",cfg={},binWidth=3, mode='highcharts',title="test_histo",color="orangered", save_img=False)
    doc.h3("histogram matplot")
    doc.plot_histogram(data['sales.csv'],col='Unit Price',colormap='RdYlBu',cfg =  cfg.histo,title="test_histo",ylabel="Unit price", mode='matplot', save_img="")
    doc.h4("nbin")
    doc.plot_histogram(data['sales.csv'],col='Unit Price',colormap='RdYlBu',nbin=20,cfg = cfg.histo,title="test_histo",ylabel="Unit price", mode='matplot', save_img="")
    doc.br()

    # plot scatter
    doc.h1("Test Scatter")
    doc.h3("scatter highchart")
    doc.plot_scatter( data['titanic.csv'].iloc[:50, :], colx='Age', coly='Fare',
                     collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                     cfg=cfg.scatter, mode='matplot', save_img='')
    doc.h3("scatter matplot")
    doc.plot_scatter(data['titanic.csv'].iloc[:50, :], colx='Age', coly='Fare',
                         collabel='Name', colclass1='Fare', colclass2='Age', colclass3='Sex',
                         figsize=(20,7),cfg=cfg, mode='highcharts',)
    
    
    # plot tseries
    doc.h1("Test TSeries")
    doc.h3("matplot tseries")
    doc.plot_tseries(data['stock_data.csv'],coldate = 'Date', date_format = '%m/%d/%Y', coly1 = ['Open', 'High', 'Low', 'Close'], coly2  = ['Turnover (Lacs)'],title = "Stock",mode='highcharts')
    doc.h4("only one axis")
    doc.plot_tseries(data['weatherdata.csv'],coldate = 'Date', date_format = '%m/%d/%Y', xlabel='date', y1label="quantity", coly1 = ['Temperature', 'Temperature'], title = "weather",mode='highcharts')

    # network
    df = pd.DataFrame({ 'from':['A', 'B', 'C','A'], 'to':['D', 'A', 'E','C'], 'weight':[1, 2, 1,5]})
    doc.pd_plot_network(df, cola='from', colb='to', coledge='col_edge',colweight="weight")
    
    vi.html_show(doc.get_html())
    doc.save('test_page.html')
 
 
def test_colimage_table():
  doc = vi.htmlDoc(dir_out="", title="hello", format='myxxxx', cfg={})
  url = 'https://raw.githubusercontent.com/AlexAdvent/high_charts/main/table_data.csv'
  df = pd.read_csv(url)
  doc.h1(" Table without colimage")
  doc.table(df, use_datatable=False, table_id="test", custom_css_class='intro')
  doc.h1(" Table with colimage")
  doc.table(df, use_datatable=False, table_id="testwithcolimage", custom_css_class='intro', colimage='image_url')
  doc.save(dir_out="testdata/test_viz_table.html")
  doc.open_browser()  # Open myfile.html   
  
  
def test_tseries_dateformat():
  data = test_getdata(verbose=False)
  from box import Box
  cfg = Box({})
  cfg.tseries = {"title": 'tseries_title'}
  cfg.scatter = {"title" : "scatter_title", 'figsize' : (12, 7)}
  cfg.histo   = {"title": 'histo_title'}

  # initialize htmldoc
  doc = vi.htmlDoc(title='Stock report', dir_out="", cfg={}, css_name= "a4")
  doc.h1("Test TSeries")
  doc.h3("matplot tseries")   
  doc.plot_tseries(data['stock_data.csv'],coldate = 'date_space', date_format = '%Y %m %d', coly1 = ['Open', 'High', 'Low', 'Close'], coly2  = ['Turnover (Lacs)'],title = "Stock",mode='highcharts')
  doc.plot_tseries(data['stock_data.csv'],coldate = 'date_string', date_format = '%Y%m%d', coly1 = ['Open', 'High', 'Low', 'Close'], coly2  = ['Turnover (Lacs)'],title = "Stock",mode='highcharts')
  doc.plot_tseries(data['stock_data.csv'],coldate = 'date_timestamp', date_format = None, coly1 = ['Open', 'High', 'Low', 'Close'], coly2  = ['Turnover (Lacs)'],title = "Stock",mode='highcharts')
  doc.plot_tseries(data['stock_data.csv'],coldate = 'date_year', date_format = '%Y', coly1 = ['Open'], coly2  = ['Turnover (Lacs)'],title = "Stock",mode='highcharts')
  doc.plot_tseries(data['stock_data.csv'],coldate = 'date_month_year', date_format = None, coly1 = ['Open'], coly2  = ['Turnover (Lacs)'],title = "Stock",mode='highcharts')
  doc.plot_tseries(data['stock_data.csv'],coldate = 'Date', date_format = None, coly1 = ['Open', 'High', 'Low', 'Close'], coly2  = ['Turnover (Lacs)'],title = "Stock",mode='highcharts')

  vi.html_show(doc.get_html())
  doc.save('test_date_format.html')
