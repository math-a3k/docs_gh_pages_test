# -*- coding: utf-8 -*-
MNAME = "utilmy.tabular.util_uncertainty"
HELP = """ utils for uncertainty estimation
#### Uncertainy interval.
https://mapie.readthedocs.io/en/latest/tutorial_classification.html


clf = GaussianNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)

mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
mapie_score.fit(X_cal, y_cal)
alpha = [0.2, 0.1, 0.05]
y_pred_score, y_ps_score = mapie_score.predict(X_test_mesh, alpha=alpha)




"""
import os, numpy as np, glob, pandas as pd, matplotlib.pyplot as plt
from box import Box

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn import metrics

try :  ### for pytest
   from mapie.classification import MapieClassifier
   from mapie.metrics import classification_coverage_score
except : pass

#### Types


#############################################################################################
from utilmy import log, log2
from mapie.regression import MapieRegressor
from numpy import ndarray
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.linear_model._base import LinearRegression
from sklearn.tree._classes import DecisionTreeClassifier
from typing import List, Optional, Tuple, Type, Union

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    ss = HELP + help_create(MNAME)
    print(ss)



#############################################################################################
def test_all() -> None:
    """function test_all
    Args:
    Returns:
        
    """
    log(MNAME)
    test1()
    test2()


def test1() -> None:
  """function test1
  Args:
  Returns:
      
  """
  from sklearn.linear_model import LinearRegression
  d = Box({})
  d.X_train, d.X_test, d.y_train, d.y_test, d.feat_names = test_data_regression_boston()
  
  mlist = [
        ('mapie.regression.MapieRegressor', LinearRegression(), 
           {'method':'naive', 'cv': 'prefit', 'n_jobs':1, 'verbose': 50 },  #### mapie pars
           {'alpha':[0.05, 0.32]}),  ### pred_pars
        ('mapie.regression.MapieRegressor', LinearRegression(), {'method':'base'}  ,{'alpha':[0.05, 0.32]}),
        ('mapie.regression.MapieRegressor', LinearRegression(), {'method':'plus'}  ,{'alpha':[0.05, 0.32]}),
        ('mapie.regression.MapieRegressor', LinearRegression(), {'method':'minmax'}  ,{'alpha':[0.05, 0.32]}),
  ]

  for ii,m in enumerate(mlist) :  
      print(str(m[1]))
      d.task_type = 'regressor' if 'Regressor' in m[0] else 'classifier'
      model1= model_fit(name = m[0], model = m[1], mapie_pars = m[2], predict_pars = m[3], data_pars=d, do_prefit=True, do_eval=True)

      model_save(model1, f'./mymodel_{ii}/')
      model2 = model_load( f'./mymodel_{ii}/')


def test2() -> None:
  """function test2
  Args:
  Returns:
      
  """
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.tree import DecisionTreeClassifier
  d = Box({})
  d.X_train, d.X_test, d.y_train, d.y_test, d.feat_names = test_data_classifier_digits()
  
  mlist = [
        ('mapie.classification.MapieClassifier', DecisionTreeClassifier(), 
            {'method':'score', 'cv': 'prefit', 'n_jobs': 1, 'verbose':50 }  ,  ### mapie model
            {'alpha':[0.05, 0.32]}),  ## Preds
        ('mapie.classification.MapieClassifier', RandomForestClassifier(), {'method':'cumulated_score'}  ,  {'alpha':[0.05, 0.32]}),
        # ('mapie.classification.MapieClassifier', RandomForestClassifier(), {'method':'top_k'}  ,  {'alpha':[0.05, 0.32]}),  ## NG
  ]

  for ii,m in enumerate(mlist) :  
      print(str(m[1]))
      d.task_type = 'regressor' if 'Regressor' in m[0] else 'classifier'
      model1= model_fit(name = m[0], model = m[1], mapie_pars = m[2], predict_pars = m[3], data_pars=d, do_prefit=True, do_eval=True)
      model_save(model1, f'./mymodel_{ii}/')
      model2 = model_load( f'./mymodel_{ii}/')


def test5():
  """function test5
  Args:
  Returns:
      
  """
  from sklearn.naive_bayes import GaussianNB
  from mapie.classification import MapieClassifier
  from mapie.metrics import classification_coverage_score
  clf = GaussianNB().fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  y_pred_proba = clf.predict_proba(X_test)
  y_pred_proba_max = np.max(y_pred_proba, axis=1)
  mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
  mapie_score.fit(X_cal, y_cal)
  alpha = [0.2, 0.1, 0.05]
  y_pred_score, y_ps_score = mapie_score.predict(X_test_mesh, alpha=alpha)



  
#############################################################################################  
def model_fit(name: str = 'mapie.regression.MapieRegressor', model: Optional[Union[RandomForestClassifier, DecisionTreeClassifier, LinearRegression]]=None, mapie_pars:dict=None, predict_pars:dict=None, data_pars:dict=None, 
              do_prefit: bool=False, do_eval: bool=True, test_size: float=0.3) -> Union[MapieClassifier, MapieRegressor]:
    """  Fit Mapie
        model :          DecisionTreeClassifier()
        mapie_pars : {'method':'naive', 'cv': 'prefit', 'n_jobs':1, 'verbose': 50 },  #### mapie pars
        predict_pars : 'alpha':[0.05, 0.32]

    """
    d = Box(data_pars) if data_pars is not None else Box({})

    #### Normalize name
    name = name.split(".")
    name = ".".join(name[:-1]) + ":" + name[-1]
    mapie_model = load_function_uri(name)


    log(model)
    task = d.get('task_type', 'classifier')

    if do_prefit :
        X_train, X_cal, y_train, y_cal = train_test_split(d.X_train, d.y_train, test_size= test_size)
        model = model.fit(X_train, y_train)
    else :
        X_cal, y_cal = d.X_train, d.y_train

    if task == 'classifier' :
       final_model = mapie_model(model,**mapie_pars).fit(X_cal, y_cal)   ### .astype(int)

    elif task == 'regressor' :
       final_model = mapie_model(model,**mapie_pars).fit(X_cal, y_cal)   ### .astype(int)

    else:
       final_model = mapie_model(model,**mapie_pars).fit(X_cal, y_cal) 

    if do_eval :
      model_evaluate(final_model, data_pars, predict_pars)
      
    return final_model


def model_save(model: Union[MapieClassifier, MapieRegressor], path: Optional[str]=None, info: None=None) -> None:
    """function model_save
    Args:
        model (  Union[MapieClassifier ) :   
        MapieRegressor]:   
        path (  Optional[str] ) :   
        info (  None ) :   
    Returns:
        
    """
    import pickle
    os.makedirs(path, exist_ok=True)
    filename = 'model.pkl'
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    info = {} if info is None else info
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))   # ,protocol=pickle.HIGHEST_PROTOCOL )


def model_load(path: str="") -> Union[MapieClassifier, MapieRegressor]:
    """function model_load
    Args:
        path (  str ) :   
    Returns:
        
    """
    import pickle
    model0 = pickle.load(open(f'{path}/model.pkl', mode='rb'))
    return model0


def model_predict(model, X_test, predict_pars:dict=None, interval=True):
    """ Convenien wrapper

    """
    if interval :
      y_pred, y_pis = model.predict(X_test, **predict_pars)

    else :
      y_pred, _ = model.predict(X_test, **predict_pars)


def model_evaluate(model: Union[MapieClassifier, MapieRegressor], data_pars:dict, predict_pars:dict) -> None:
    """ Evaluate model
    """
    d = Box(data_pars) if data_pars is not None else Box({})
    y_pred, y_pis = model.predict(d.X_test, **predict_pars)
    
    task_type = d.get('task_type', 'classifier')
    if task_type == 'classifier' :
      model_viz_classification_preds(y_pred, d.y_test)
    else :
      # get test performance
      print(f'test r2: {metrics.r2_score(d.y_test, y_pred):0.2f}')


def model_viz_classification_preds(preds: ndarray, y_test: ndarray) -> None:
    '''look at prediction breakdown
    '''
    from sklearn import metrics
    disp1 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.show()


def model_eval2(clf, Xval, yval, dirout=""):
  """function model_eval2
  Args:
      clf:   
      Xval:   
      yval:   
      dirout:   
  Returns:
      
  """
  from mapie.classification import MapieClassifier
  from mapie.metrics import classification_coverage_score
  
  mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
  mapie_score.fit(Xval, yval)
  alpha = [0.2, 0.1, 0.05]
  y_pred_score, y_ps_score = mapie_score.predict(Xval, alpha=alpha)

  from utilmy import save
  save(mapie_score, dirout)


#############################################################################################
def test_data_regression_boston() -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    '''load (regression) data on boston housing prices
    '''
    from sklearn.datasets import load_boston
    X_reg, y_reg = load_boston(return_X_y=True)
    feature_names = load_boston()['feature_names']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.25) # split
    return X_train_reg, X_test_reg, y_train_reg, y_test_reg, feature_names


def test_data_classifier_digits() -> Tuple[ndarray, ndarray, ndarray, ndarray, List[str]]:
    '''load (classification) data on diabetes
    '''
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    X, y     = digits.data, digits.target
    feature_names = load_digits()['feature_names']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y) # split
    return X_train, X_test, y_train, y_test, feature_names


def load_function_uri(uri_name: str="path_norm") -> Union[Type[MapieRegressor], Type[MapieClassifier]]:
    """ Load dynamically function from URI
    ###### Pandas CSV case : Custom MLMODELS One
    #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"

    ###### External File processor :
    #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
    """
    import importlib, sys
    from pathlib import Path
    if ":" in uri_name :
       pkg = uri_name.split(":")
       assert len(pkg) > 1, "  Missing :   in  uri_name module_name:function_or_class "
       package, name = pkg[0], pkg[1]

    else :
       pkg = uri_name.split(".")
       package = ".".join(pkg[:-1])      
       name    = pkg[-1]   

    
    try:
        #### Import from package mlmodels sub-folder
        return  getattr(importlib.import_module(package), name)

    except Exception as e1:
        try:
            ### Add Folder to Path and Load absoluate path module
            path_parent = str(Path(package).parent.parent.absolute())
            sys.path.append(path_parent)
            #log(path_parent)

            #### import Absolute Path model_tf.1_lstm
            model_name   = Path(package).stem  # remove .py
            package_name = str(Path(package).parts[-2]) + "." + str(model_name)
            #log(package_name, model_name)
            return  getattr(importlib.import_module(package_name), name)

        except Exception as e2:
            raise NameError(f"Module {pkg} notfound, {e1}, {e2}")





###################################################################################################
if __name__ == "__main__":
    import fire

    fire.Fire()


