# -*- coding: utf-8 -*-
"""  All module here for include  """

def aa_isanaconda():
 import sys; 
 txt= sys.version
 if txt.find('Continuum') > 0 : return True
 else: return False
 
import numpy as np,  pandas as pd, copy, scipy as sci, matplotlib.pyplot as plt, math as m


import requests,  re; from bs4 import BeautifulSoup

from tabulate import tabulate;
from datetime import datetime; from datetime import timedelta; from calendar import isleap


import portfolio as pf, util


import sklearn as sk; from sklearn import linear_model; from sklearn import covariance
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier
# from sklearn import cross_validation    # DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
from sklearn.metrics import confusion_matrix

from collections import OrderedDict



if aa_isanaconda() :  #Import the Packages for Ananconda
  import PyGMO as py
  
  
  


############################################################################
#---------------------             --------------------







############################################################################











############################################################################
#---------------------             --------------------




############################################################################





















############################################################################
#---------------------             --------------------






























