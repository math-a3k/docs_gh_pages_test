import xlwings as xw

@xw.func
def load_csv(csvfile):
    """Reads input csv file and populates the caller"""
    import pandas as pd
    df = pd.read_csv(csvfile)
    
    app = xw.apps.active
    wb = app.books.active 
    
    cellrange = wb.app.selection
    rownum=cellrange.row
    colnum=cellrange.column
    
    xw.Range((rownum,colnum)).options(pd.DataFrame, header=1, index=True, expand='table').value = df

@xw.func
def invokenumpy():
    """Prints 9 equally spaced numbers b/w 0 and 2 using numpy"""
    import numpy as np
    app = xw.apps.active
    wb = app.books.active 
    
    cellrange = wb.app.selection
    rownum=cellrange.row
    colnum=cellrange.column
    
    xw.Range((rownum,colnum)).value = np.linspace(0,2,9)
    
@xw.func
def invokesklearn():
    """Loads 10 rows from iris dataset to excel sheet"""
    from sklearn import datasets
    import pandas as pd
    import numpy as np
    iris = datasets.load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    
    app = xw.apps.active
    wb = app.books.active 
    
    cellrange = wb.app.selection
    rownum=cellrange.row
    colnum=cellrange.column
    
    xw.Range((rownum,colnum)).options(pd.DataFrame, header=1, index=True, expand='table').value=df.head(10)
  
 
@xw.func
def loaddf():
    """Load data from excel to pandas dataframe"""
    import pandas as pd
    wb = xw.Book.caller()
    
    ws = xw.sheets.active
    cellrange = wb.app.selection
    df = ws.range(cellrange).options(pd.DataFrame, header=1, index=True).value
    
    #Just *** FOR TESTING *** writeback to excel
    xw.Range('N1').options(pd.DataFrame, header=1, index=True, expand='table').value=df
    
    