# -*- coding: utf-8 -*-
"""
Methods for ML models, model ensembels, metrics etc.
util_model : input/output is numpy

"""
import copy
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy as sci
from dateutil.parser import parse

import sklearn as sk
from sklearn import covariance, linear_model, model_selection, preprocessing
from sklearn.cluster import dbscan, k_means
from sklearn.decomposition import PCA, pca
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, make_scorer,
                             mean_absolute_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

####################################################################################################
DIRCWD = os.getcwd()
print("os.getcwd", os.getcwd())




####################################################################################################
class dict2(object):
    ## Dict with attributes
    def __init__(self, d):
        """ dict2:__init__
        Args:
            d:     
        Returns:
           
        """
        self.__dict__ = d




####################################################################################################
def np_conditional_entropy(x, y):
    """
    Calculates the conditional entropy of x given y: S(x|y)
    Wikipedia: https://en.wikipedia.org/wiki/Conditional_entropy
    **Returns:** float
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of measurements
    """
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x, y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy)
    return entropy


def np_correl_cat_cat_cramers_v(x, y):
    """
    Calculates Cramer's V statistic for categorical-categorical association.
    Uses correction from Bergsma and Wicher, Journal of the Korean Statistical Society 42 (2013):
    This is a symmetric coefficient: V(x,y) = V(y,x)
    Original function taken from: https://stackoverflow.com/a/46498792/5863503
    Wikipedia: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = sci.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def np_correl_cat_cat_theils_u(x, y):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-categorical association.
    This is the uncertainty of x given y: value is on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    """
    s_xy = np_conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = sci.stats.entropy(p_x)
    if s_x == 0:
        return 1
    return (s_x - s_xy) / s_x


def np_correl_cat_num_ratio(cat_array, num_array):
    """
    Calculates the Correlation Ratio (sometimes marked by the greek letter Eta) for categorical-continuous association.
    Answers the question - given a continuous value of a measurement, is it possible to know which category is it
    associated with?
    Value is in the range [0,1], where 0 means a category cannot be determined by a continuous measurement, and 1 means
    a category can be determined with absolute certainty.
    Wikipedia: https://en.wikipedia.org/wiki/Correlation_ratio
    **Returns:** float in the range of [0,1]
    Parameters
    ----------
    cat_array : list / NumPy ndarray / Pandas Series   A sequence of categorical measurements
    num_array : list / NumPy ndarray / Pandas Series  A sequence of continuous measurements
    """
    cat_array = convert(cat_array, "array")
    num_array = convert(num_array, "array")
    fcat, _ = pd.factorize(cat_array)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = num_array[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(num_array, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def pd_num_correl_associations(
    df, colcat=None, mark_columns=False, theil_u=False, plot=True, return_results=False, **kwargs
):
    """
    Calculate the correlation/strength-of-association of features in data-set with both categorical (eda_tools) and
    continuous features using:
     * Pearson's R for continuous-continuous cases
     * Correlation Ratio for categorical-continuous cases
     * Cramer's V or Theil's U for categorical-categorical cases
    **Returns:** a DataFrame of the correlation/strength-of-association between all features
    **Example:** see `associations_example` under `dython.examples`
    Parameters
    ----------
    df : NumPy ndarray / Pandas DataFrame
        The data-set for which the features' correlation is computed
    colcat : string / list / NumPy ndarray
        Names of columns of the data-set which hold categorical values. Can also be the string 'all' to state that all
        columns are categorical, or None (default) to state none are categorical
    mark_columns : Boolean, default = False
        if True, output's columns' names will have a suffix of '(nom)' or '(con)' based on there type (eda_tools or
        continuous), as provided by colcat
    theil_u : Boolean, default = False
        In the case of categorical-categorical feaures, use Theil's U instead of Cramer's V
    plot : Boolean, default = True
        If True, plot a heat-map of the correlation matrix
    return_results : Boolean, default = False
        If True, the function will return a Pandas DataFrame of the computed associations
    kwargs : any key-value pairs
        Arguments to be passed to used function and methods
    """
    # df = convert(df, "dataframe")
    col = df.columns
    if colcat is None:
        colcat = list()
    elif colcat == "all":
        colcat = col
    corr = pd.DataFrame(index=col, columns=col)
    for i in range(0, len(col)):
        for j in range(i, len(col)):
            if i == j:
                corr[col[i]][col[j]] = 1.0
            else:
                if col[i] in colcat:
                    if col[j] in colcat:
                        if theil_u:
                            corr[col[j]][col[i]] = np_correl_cat_cat_theils_u(
                                df[col[i]], df[col[j]]
                            )
                            corr[col[i]][col[j]] = np_correl_cat_cat_theils_u(
                                df[col[j]], df[col[i]]
                            )
                        else:
                            cell = np_correl_cat_cat_cramers_v(df[col[i]], df[col[j]])
                            corr[col[i]][col[j]] = cell
                            corr[col[j]][col[i]] = cell
                    else:
                        cell = np_correl_cat_num_ratio(df[col[i]], df[col[j]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
                else:
                    if col[j] in colcat:
                        cell = np_correl_cat_num_ratio(df[col[j]], df[col[i]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
                    else:
                        cell, _ = sci.stats.pearsonr(df[col[i]], df[col[j]])
                        corr[col[i]][col[j]] = cell
                        corr[col[j]][col[i]] = cell
    corr.fillna(value=np.nan, inplace=True)
    if mark_columns:
        marked_columns = [
            "{} (nom)".format(col) if col in colcat else "{} (con)".format(col) for col in col
        ]
        corr.columns = marked_columns
        corr.index = marked_columns
    if plot:
        pass
        """
        plt.figure(figsize=kwargs.get('figsize',None))
        sns.heatmap(corr, annot=kwargs.get('annot',True), fmt=kwargs.get('fmt','.2f'))
        plt.show()
        """
    if return_results:
        return corr





def stat_hypothesis_test_permutation(df, variable, classes, repetitions):

    """Test whether two numerical samples
    come from the same underlying distribution,
    using the absolute difference between the means.
    table: name of table containing the sample
    variable: label of column containing the numerical variable
    classes: label of column containing names of the two samples
    repetitions: number of random permutations"""

    t = df[[ variable, classes]]

    # Find the observed test statistic
    means_table = t.groupby(classes).agg(np.mean)
    obs_stat = abs(means_table.column(1).item(0) - means_table.column(1).item(1))

    # Assuming the null is true, randomly permute the variable
    # and collect all the generated test statistics
    stats = make_array()
    for i in np.arange(repetitions):
        shuffled_var = t.select(variable).sample(with_replacement=False).column(0)
        shuffled = t.select(classes).with_column('Shuffled Variable', shuffled_var)
        m_tbl = shuffled.group(classes, np.mean)
        new_stat = abs(m_tbl.column(1).item(0) - m_tbl.column(1).item(1))
        stats = np.append(stats, new_stat)

    # Find the empirical P-value:
    emp_p = np.count_nonzero(stats >= obs_stat)/repetitions

    # Draw the empirical histogram of the tvd's generated under the null,
    # and compare with the value observed in the original sample
    Table().with_column('Test Statistic', stats).hist(bins=20)
    plots.title('Empirical Distribution Under the Null')
    print('Observed statistic:', obs_stat)
    print('Empirical P-value:', emp_p)
    
    
    








def np_transform_pca(X, dimpca=2, whiten=True):
    """Project ndim data into dimpca sub-space  """
    pca = PCA(n_components=dimpca, whiten=whiten).fit(X)
    return pca.transform(X)


def sk_distribution_kernel_bestbandwidth(X, kde):
    """Find best Bandwidht for a  given kernel
  :param kde:
  :return:
 """
    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(
        kde, {"bandwidth": np.linspace(0.1, 1.0, 30)}, cv=20
    )  # 20-fold cross-validation
    grid.fit(X[:, None])
    return grid.best_params_


def sk_distribution_kernel_sample(kde=None, n=1):
    """
  kde = sm.nonparametric.KDEUnivariate(np.array(Y[Y_cluster==0],dtype=np.float64))
  kde = sm.nonparametric.KDEMultivariate()  # ... you already did this
 """

    from scipy.optimize import brentq

    samples = np.zeros(n)

    # 1-d root-finding  F-1(U) --> Sample
    def func(x):
        return kde.cdf([x]) - u

    for i in range(0, n):
        u = np.random.random()  # sample
        samples[i] = brentq(func, -999, 999)  # read brentq-docs about these constants
    return samples
