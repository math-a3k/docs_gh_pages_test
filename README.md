
[![Build and Test , Package PyPI](https://github.com/arita37/myutil/actions/workflows/build%20and%20release.yml/badge.svg)](https://github.com/arita37/myutil/actions/workflows/build%20and%20release.yml)

[     https://pypi.org/project/utilmy/#history ](https://pypi.org/project/utilmy/#history)




# myutil
One liner utilities


# Looking for contributors

   This package is looking for contributors.
   Documentation is here:  https://github.com/arita37/myutil/issues/9
   
 
 # Usage
 ```
 #### Generate automatic docs/stats
 cd yourrepo
 doc-gen   --prefix   https://github.com/arita37/yourepo/tree/main/
 
 
 
 ```
 


# myutil
    https://pypi.org/project/utilmy/#history


## Install

    pip install utilmy 

    git clone  https://github.com/arita37/myutil.git
    cd myutil
    pip install -e .
    
    
    

 # Code
 ```

def test2():
    """
      # pip install --upgrade utilmy
      from util.viz import vizhtml as vi
      vi.test2()

    """
    data = test_getdata()

    doc = htmlDoc(title='Weather report', dir_out="", cfg={} )
    doc.h1(' Weather report')
    doc.hr() ; doc.br()

    doc.h2('Plot of weather data') 
    doc.plot_tseries(data['weatherdata.csv'].iloc[:1000, :],
                      coldate     =  'Date',
                      date_format =  '%m/%d/%Y',
                      cols_axe1   =  ['Temperature'],
                      cols_axe2   =  ["Rainfall"],
                      # x_label=     'Date', 
                      # axe1_label=  "Temperature",
                      # axe2_label=  "Rainfall", 
                     title =      "Weather",
                     cfg={},             
                     mode='highcharts'
                     )

    doc.hr() ; doc.br()
    doc.h3('Weather data') 
    doc.table(data['weatherdata.csv'].iloc[:10 : ], use_datatable=True )


    doc.plot_histogram(data['housing.csv'].iloc[:1000, :], col="median_income",
                       xaxis_label= "x-axis",yaxis_label="y-axis",cfg={}, mode='highcharts', save_img=False)


     # Testing with example data sets (Titanic)
    cfg = {"title" : "Titanic", 'figsize' : (20, 7)}

    doc.plot_scatter(data['titanic.csv'].iloc[:50, :], colx='Age', coly='Fare',
                         collabel='Name', colclass1='Sex', colclass2='Age', colclass3='Sex',
                         cfg=figsize, mode='highcharts',                         
                         )

    doc.save('viz_test3_all_graphs.html')
    doc.open_browser()
    html1 = doc.get_html()
    # print(html1)
    # html_show(html1)



 
 
 
 ```
 



