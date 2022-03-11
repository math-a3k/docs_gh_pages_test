### drift
# coding=utf-8
HELP="""


https://thirdeyedata.io/unsupervised-concept-drift-detection-techniques-for-machine-learning-models-with-examples-in-python/

https://github.com/pranab/beymani


https://pkghosh.wordpress.com/2020/12/24/concept-drift-detection-techniques-with-python-implementation-for-supervised-machine-learning-models/



https://github.com/topics/hypothesis-testing?l=python&o=desc&s=stars

https://pypi.org/project/pysie/#description



"""
import os, sys, pandas as pd, numpy as np
from utilmy.utilmy import pd_generate_data
from utilmy.prepro.util_feature import  pd_colnum_tocat, pd_colnum_tocat_stat


#################################################################################################
from utilmy.utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    print( HELP + help_create("utilmy.tabular") )

    
#################################################################################################
def test_all():
    """
    #### python test.py   test_tabular
    """
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    model = DecisionTreeRegressor(random_state=1)

    df = pd.read_csv("./testdata/tmp/test/crop.data.csv")
    y = df.fertilizer
    X = df[["yield","density","block"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    def test():
        log("Testing normality...")
        # import utilmy.tabular as m
        test_normality(df["yield"])
        
        
        df1 = pd_generate_data(7, 100)
        test_anova(df1,'cat1','cat2')
        test_normality2(df1, '0', "Shapiro")
        test_plot_qqplot(df1, '1')

        
        log("Testing heteroscedacity...")
        # from utilmy.tabular import test_heteroscedacity
        log(test_heteroscedacity(y_test,y_pred))
    
        log("Testing test_mutualinfo()...")
        # from utilmy.tabular import test_mutualinfo
        df1 = pd_generate_data(7, 100)

        test_mutualinfo(df1["0"],df1[["1","2","3"]],colname="test")

        log("Testing hypothesis_test()...")
        #from utilmy.tabular import test_hypothesis
        log(test_hypothesis(X_train, X_test,"chisquare"))

    def custom_stat(values, axis=1):
        #stat_val = np.mean(np.asmatrix(values),axis=axis)
        # # stat_val = np.std(np.asmatrix(values),axis=axis)p.mean
        stat_val = np.sqrt(np.mean(np.asmatrix(values*values),axis=axis))
        return stat_val

    def test_estimator():
        log("Testing estimators()...")
        # from utilmy.tabular import estimator_std_normal,estimator_boostrap_bayes,estimator_bootstrap
        log(estimator_std_normal(y_pred))
        log(estimator_boostrap_bayes(y_pred))
        estimator_bootstrap(y_pred, custom_stat=custom_stat)

       
    
    def test_pd_utils():
        log("Testing pd_utils ...")
        #from utilmy.tabular import pd_train_test_split_time,pd_to_scipy_sparse_matrix,pd_stat_correl_pair,\
        #    pd_stat_pandas_profile,pd_stat_distribution_colnum,pd_stat_histogram,pd_stat_shift_trend_changes,\
        #    pd_stat_shift_trend_correlation,pd_stat_shift_changes
        from utilmy.prepro.util_feature import pd_colnum_tocat_stat

        pd_train_test_split_time(df, coltime="block")
        pd_to_scipy_sparse_matrix(df)
        '''TODO: git test failling here
        this bug is caused due to typecasting mismatch in the function.
        However, even typecasting the arrays manually in the function is not solving
        the problem.
        '''
        # log(pd_stat_correl_pair(df,coltarget=["fertilizer"],colname=["yield"]))
        
        pd_stat_pandas_profile(df,savefile="./testdata/tmp/test/report.html", title="Pandas profile")
        pd_stat_distribution_colnum(df, nrows=len(df))
        pd_stat_histogram(df, bins=50, coltarget="yield")
        _,df_grouped = pd_colnum_tocat_stat(df,"density","block",10)
        pd_stat_shift_trend_changes(df_grouped,"density","block")

        _, X_train_grouped =  pd_colnum_tocat_stat(X_train,"yield","block",10)
        _, X_test_grouped =  pd_colnum_tocat_stat(X_test,"yield","block",10)
        pd_stat_shift_trend_correlation(X_train_grouped, X_test_grouped,"yield","block")

        '''TODO: TypeError: pd_colnum_tocat_stat() got an unexpected keyword argument 'colname',
        This function needs complete rewrite there are many bugs and logical errors.
        pd_stat_shift_changes(df,"yield", features_list=["density","block"])
        '''

    def test_drift_detect():
        import tensorflow as tf
        from tensorflow.keras.layers import Dense,InputLayer,Dropout
        # from utilmy.tabular import pd_data_drift_detect_alibi

        input_size = X_train.shape[1]
        output_size = y_train.nunique()
        model = tf.keras.Sequential(
            [
                InputLayer(input_shape=(input_size)),
                Dense(16,activation=tf.nn.relu),
                Dropout(0.3),
                Dense(1)
            ]
        )
        model.compile(optimizer='adam',loss='mse')
        model.fit(X_train,y_train,epochs=1)

        pd_data_drift_detect_alibi(X_train, X_test,'regressoruncertaintydrift','tensorflow',model=model)
        pd_data_drift_detect_alibi(X_train, X_test,'learnedkerneldrift','tensorflow',model=model)
        pd_data_drift_detect_alibi(X_train, X_test,'spotthediffdrift','tensorflow',model=model)
        pd_data_drift_detect_alibi(X_train, X_test,'spotthediffdrift','tensorflow')
        pd_data_drift_detect_alibi(X_train, X_test,'ksdrift','tensorflow')
        pd_data_drift_detect_alibi(X_train, X_test,'mmddrift','tensorflow')
        pd_data_drift_detect_alibi(X_train, X_test,'chisquaredrift','tensorflow')
        pd_data_drift_detect_alibi(X_train, X_test,'tabulardrift','tensorflow')

        input_size = X_train.shape[1]
        output_size = y_train.nunique()
        model = tf.keras.Sequential(
            [
                InputLayer(input_shape=(input_size)),
                Dense(16,activation=tf.nn.relu),
                Dropout(0.3),
                Dense(output_size,activation=tf.nn.softmax)
            ]
        )
        model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy())
        model.fit(X_train,tf.one_hot(y_train,output_size),epochs=1)
        pd_data_drift_detect_alibi(X_train,X_test,'classifieruncertaintydrift','tensorflow',model=model)


    def test_np_utils():
        log("Testing np_utils ...")
        # from utilmy.tabular import np_col_extractname, np_conv_to_one_col, np_list_remove
        import numpy as np
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        np_col_extractname(["aa_","bb-","cc"])
        np_list_remove(arr,[1,2,3], mode="exact")
        np_conv_to_one_col(arr)

  
    test()
    test_estimator()
    # test_pd_utils()
    # test_drift_detect()
    test_np_utils()



def test0():
    """function test0
    Args:
    Returns:
        
    """
    df = pd_generate_data(7, 100)
    test_anova(df, 'cat1', 'cat2')
    test_normality2(df, '0', "Shapiro")
    test_plot_qqplot(df, '1')
    '''TODO: import needed
    NameError: name 'pd_colnum_tocat' is not defined
    test_mutualinfo(df["0"],df[["1","2","3"]],colname="test")
    '''

def test1():
    """function test1
    Args:
    Returns:
        
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("../testdata/tmp/test/crop.data.csv")
    model = DecisionTreeRegressor(random_state=1)
    y = df.fertilizer
    X = df[["yield","density","block"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_normality(df["yield"])
    log(test_heteroscedacity(y_test,y_pred))
    log(test_hypothesis(X_train, X_test,"chisquare"))
    log(estimator_std_normal(y_pred))
    log(estimator_boostrap_bayes(y_pred))
    '''TODO: need to check this one
    estimator_bootstrap(y_pred, custom_stat=custom_stat(y_pred))
    '''


def test3():
    """function test3
    Args:
    Returns:
        
    """
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    np_col_extractname(["aa_","bb-","cc"])
    np_list_remove(arr,[1,2,3], mode="exact")
    np_conv_to_one_col(arr)


def log(*s) -> None:
    """function log
    Args:
        *s:   
    Returns:
        
    """
    print(s)




#########################################################################
def pd_data_drift_detect_alibi(
    df:pd.DataFrame,      ### Reference dataset
    df_new:pd.DataFrame,  ### Test dataset to be checked
    method:str="'regressoruncertaintydrift','classifieruncertaintydrift','ksdrift','mmddrift','learnedkerneldrift','chisquaredrift','tabulardrift', 'classifierdrift','spotthediffdrift'",
    backend:str='tensorflow,pytorch',
    model=None,  ### Pre-trained model
    p_val=0.05,  **kwargs):
    
    """ Detecting drift in the dataset using alibi
    https://docs.seldon.io/projects/alibi-detect/en/latest/api/modules.html
    
    :param df:    dfframe test dataset to check for drift
    :param dfnew: dfframe test dataset to check for drift    
    :param backend: str "tensorflow" or "pytorch"
    :param model:  trained pytorch or tensorflow model.
    :param p_val: p value float 

    example:
    model = tf.keras.Sequential([InputLayer(input_shape=(input_size)),Dropout(0.3),Dense(1)])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train,y_train,epochs=1)

    cd, is_drift_preds = pd_data_drift_detect(X_train, X_test,'regressoruncertaintydrift','tensorflow',model=model)

    from utilmy import import_function
    myclass = import_function(fun_name='KSDrift', module_name='alibi_detect.cd')  
    mdrift = myclass(df.values,p_val=p_val,**kwargs)
    
    
    """
    methods = ['regressoruncertaintydrift','classifieruncertaintydrift','ksdrift',
                'mmddrift','learnedkerneldrift','chisquaredrift','tabulardrift',
                'classifierdrift','spotthediffdrift']
    
    if len(method) > 25 :
        log('Using KSDrift as default')
        method = 'ksdrift'
        backend = 'tensorflow'
        
    assert method in methods, f"method is invalid, methods available {methods}"

    from utilmy import import_function
    mc = import_function(fun_name= 'KSDrift', module_name='alibi_detect.cd')  
    mdrift = mc(df.values,p_val=p_val,**kwargs)
    
    
    if method == "regressoruncertaintydrift":
        from alibi_detect.cd import RegressorUncertaintyDrift as mc
        mdrift = mc(df.values,model=model,p_val=p_val, backend=backend,**kwargs)
    
    if method == 'classifieruncertaintydrift':
        from alibi_detect.cd import ClassifierUncertaintyDrift as mc
        mdrift = mc(df.values,model=model,p_val=p_val, backend=backend,preds_type='probs',**kwargs)
    
    if method == 'ksdrift':
        from alibi_detect.cd import KSDrift as mc
        mdrift = mc(df.values,p_val=p_val,**kwargs)
    
    if method == 'mmddrift':
        from alibi_detect.cd import MMDDrift as mc
        mdrift = mc(df.values,backend=backend,p_val=0.05,**kwargs)

    if method == 'learnedkerneldrift':
        from alibi_detect.cd import LearnedKernelDrift
        if backend == "tensforflow":
            from alibi_detect.utils.tensorflow.kernels import DeepKernel
            kernel = DeepKernel(model)
            mdrift = LearnedKernelDrift(df.values, kernel, backend=backend, p_val=p_val, **kwargs)
            
        if backend == "pytorch":
            from alibi_detect.utils.pytorch.kernels import DeepKernel
            kernel = DeepKernel(model)
            mdrift = LearnedKernelDrift(df.values, kernel, backend=backend, p_val=p_val, **kwargs)
    
    if method == 'chisquaredrift':
        from alibi_detect.cd import ChiSquareDrift as mc
        mdrift = mc(df.values, p_val=p_val,**kwargs)
    
    if method == 'tabulardrift':
        from alibi_detect.cd import TabularDrift as mc
        mdrift = mc(df.values, p_val=p_val,**kwargs)
    
    if method == 'classifierdrift':
        from alibi_detect.cd import ClassifierDrift as mc
        mdrift = mc(df.values, model, p_val=p_val,backend=backend,**kwargs)
    
    if method == 'spotthediffdrift':
        from alibi_detect.cd import SpotTheDiffDrift 

        if backend == 'tensorflow' and model is not None:
            from alibi_detect.utils.tensorflow.kernels import DeepKernel as mc
            kernel = mc(model)
            mdrift = SpotTheDiffDrift(df.values.astype('float32'),backend=backend,p_val=p_val,kernel=kernel)

        if backend == 'pytorch' and model is not None:
            from alibi_detect.utils.pytorch.kernels import DeepKernel as mc
            kernel = mc(model)
            mdrift = SpotTheDiffDrift(df.values.astype('float32'),backend=backend,p_val=p_val,kernel=kernel)
        
        mdrift = SpotTheDiffDrift(df.values.astype('float32'),backend=backend,p_val=p_val)

        
    is_drift_pvalue_scores = mdrift.predict(df_new.values)
    return mdrift, is_drift_pvalue_scores



#############################################################################
#############################################################################
def y_adjuster_log(y_true, y_pred_log, error_func, **kwargs):
    """
       Adjustment of log, exp transfrmation for yt= y + error
       https://www.inovex.de/de/blog/honey-i-shrunk-the-target-variable/
       
       log(y) = u =sigma**2
    
    """
    import scipy as sp

    def cost_func(delta):
        return error_func(np.exp(delta + y_pred_log), y_true)

    res = sp.optimize.minimize(cost_func, 0., **kwargs)
    if res.success:
        return res.x
    else:
        raise RuntimeError(f"Finding correction term failed!\n{res}")


# conduct multiple comparisons
from tqdm import tqdm
from typing import Dict, Optional, Union, List
from scipy import stats
from numpy import float64, ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series

def test_multiple_comparisons(data: pd.DataFrame, label='y', adjuster=True) -> List[float]:
    """Run multiple t tests.
       p_values = multiple_comparisons(data)
       
       # bonferroni correction
        print('Total number of discoveries is: {:,}'
              .format(sum([x[1] < threshold / n_trials for x in p_values])))
        print('Percentage of significant results: {:5.2%}'
              .format(sum([x[1] < threshold / n_trials for x in p_values]) / n_trials))

        # Benjamini–Hochberg procedure
        p_values.sort(key=lambda x: x[1])

        for i, x in enumerate(p_values):
            if x[1] >= (i + 1) / len(p_values) * threshold:
                break
        significant = p_values[:i]

        print('Total number of discoveries is: {:,}'
              .format(len(significant)))
        print('Percentage of significant results: {:5.2%}'
              .format(len(significant) / n_trials))

    """
    p_values = []
    for c in tqdm(data.columns):
        if c.startswith(label): 
            continue
        group_a = data[data[c] == 0][label]
        group_b = data[data[c] == 1][label]

        _, p = stats.ttest_ind(group_a, group_b, equal_var=True)
        p_values.append((c, p))
    
    if adjuster:
        p_values.sort(key=lambda x: x[1])
        for i, x in enumerate(p_values):
            if x[1] >= (i + 1) / len(p_values) * threshold:
                break
        significant = p_values[:i]    
        return significant
    
    return p_values



    
def test_anova(df: DataFrame, col1: str, col2: str) -> None:
    """
    ANOVA test two categorical features
    Input dfframe, 1st feature and 2nd feature
    """
    import scipy.stats as stats

    ov=pd.crosstab(df[col1],df[col2])
    edu_frame=df[[col1, col2]]
    groups = edu_frame.groupby(col1).groups
    edu_class=edu_frame[col2]
    lis_group = groups.keys()
    lg=[]
    for i in groups.keys():
        globals()[i]  = edu_class[groups[i]].values
        lg.append(globals()[i])
    dfd = 0
    for m in lis_group:
        dfd=len(m)-1+dfd
    print(stats.f_oneway(*lg))
    stat_val = stats.f_oneway(*lg)[0]
    crit_val = stats.f.ppf(q=1-0.05, dfn=len(lis_group)-1, dfd=dfd)
    if stat_val >= crit_val :
         print('Reject null hypothesies and conclude that atleast one group is different and the feature is releavant to the class.')
    else:
         print('Accept null hypothesies and conclude that atleast one group is same and the feature is not releavant to the class.')



def test_normality2(df: DataFrame, column: str, test_type: str) -> None:
    """
    Function to check Normal Distribution of a Feature by 3 methods
    Input dfframe, feature name, and a test type
    Three types of test
    1)'Shapiro'
    2)'Normal'
    3)'Anderson'

    output the statistical test score and result whether accept or reject
    Accept mean the feature is Gaussain
    Reject mean the feature is not Gaussain
    """
    from scipy.stats import shapiro
    from scipy.stats import normaltest
    from scipy.stats import anderson
    if  test_type == 'Shapiro':
        stat, p = shapiro(df[column])
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print(column,' looks Gaussian (fail to reject H0)')
        else:
            print(column,' does not look Gaussian (reject H0)')
    if  test_type == 'Normal':
        stat, p = normaltest(df[column])
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print(column,' looks Gaussian (fail to reject H0)')
        else:
            print(column,' does not look Gaussian (reject H0)')
        # normality test
    if  test_type == 'Anderson':
        result = anderson(df[column])
        print('Statistic: %.3f' % result.statistic)
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < result.critical_values[i]:
                print(sl,' : ',cv,' ',column,' looks normal (fail to reject H0)')
            else:
                print(sl,' : ',cv,' ',column,' does not looks normal (fail to reject H0)')


def test_plot_qqplot(df: DataFrame, col_name: str) -> None:
    """
    Function to plot boxplot, histplot and qqplot for numerical feature analyze
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    fig.suptitle('Numerical Analysis'+" "+col_name)
    sns.boxplot(ax=axes[0], data=df,x=col_name)
    sns.histplot(ax=axes[1],data=df, x=col_name, kde=True)
    sm.qqplot(ax=axes[2],data=df[col_name], line ='45')
    print(df[col_name].describe())



####################################################################################################
def test_heteroscedacity(y: Series, y_pred: ndarray, pred_value_only: int=1) -> Dict[str, Dict[str, float64]]:
    """function test_heteroscedacity
    Args:
        y (  Series ) :   
        y_pred (  ndarray ) :   
        pred_value_only (  int ) :   
    Returns:
        
    """
    ss = """
       Test  Heteroscedacity :  Residual**2  = Linear(X, Pred, Pred**2)
       F pvalues < 0.01 : Null is Rejected  ---> Not Homoscedastic
       het_breuschpagan

    """
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white
    error    = y_pred - y

    ypred_df = pd.DataFrame({"pcst": [1.0] * len(y), "pred": y_pred, "pred2": y_pred * y_pred})
    labels   = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    test1    = het_breuschpagan(error * error, ypred_df.values)
    test2    = het_white(error * error, ypred_df.values)
    ddict    = {"het-breuschpagan": dict(zip(labels, test1)),
             "het-white": dict(zip(labels, test2)),
             }

    return ddict


def test_normality(error: Series, distribution: str="norm", test_size_limit: int=5000) -> Dict[str, Dict[str, Union[float, float64, ndarray]]]:
    """
       Test  Is Normal distribution
       F pvalues < 0.01 : Rejected

    """
    from scipy.stats import shapiro, anderson, kstest

    error2 = error

    error2 = error2[np.random.choice(len(error2), 5000)]  # limit test
    test1  = shapiro(error2)
    ddict1 = dict(zip(["shapiro", "W-p-value"], test1))

    test2  = anderson(error2, dist=distribution)
    ddict2 = dict(zip(["anderson", "p-value", "P critical"], test2))

    test3  = kstest(error2, distribution)
    ddict3 = dict(zip(["kstest", "p-value"], test3))

    ddict  = dict(zip(["shapiro", "anderson", "kstest"], [ddict1, ddict2, ddict3]))

    return ddict


def test_mutualinfo(error: Series, Xtest: DataFrame, colname: Optional[str]=None, bins: int=5) -> Dict[str, float64]:
    """
       Test  Error vs Input Variable Independance byt Mutual ifno
       sklearn.feature_selection.mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)

    """
    from sklearn.feature_selection import mutual_info_classif
    error = pd.DataFrame({"error": error})
    error_dis, _ = pd_colnum_tocat(error, bins=bins, method="quantile")
    # print(error_dis)

    res = mutual_info_classif(Xtest.values, error_dis.values.ravel())

    return dict(zip(colname, res))


def test_hypothesis(df_obs: DataFrame, df_ref: DataFrame, method: str='', **kw):
    """
    https://github.com/aschleg/hypothetical/blob/master/tests/test_contingency.py

    """
    try:
       from hypothetical.contingency import (ChiSquareContingency, CochranQ, McNemarTest,
            table_margins, expected_frequencies )
    except :
       print(' pip install hypothetical ')

    if method == 'chisquare' :
        c = ChiSquareContingency(df_obs, df_ref)
        return c



####################################################################################################
####################################################################################################
def estimator_std_normal(err, alpha=0.05, ):
    """function estimator_std_normal
    Args:
        err:   
        alpha:   
        :   
    Returns:
        
    """
    # estimate_std( err, alpha=0.05, )
    from scipy import stats
    n = len(err)  # sample sizes
    s2 = np.var(err, ddof=1)  # sample variance
    df = n - 1  # degrees of freedom
    upper = np.sqrt((n - 1) * s2 / stats.chi2.ppf(alpha / 2, df))
    lower = np.sqrt((n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, df))

    return np.sqrt(s2), (lower, upper)


def estimator_boostrap_bayes(err, alpha=0.05, ):
    """function estimator_boostrap_bayes
    Args:
        err:   
        alpha:   
        :   
    Returns:
        
    """
    from scipy.stats import bayes_mvs
    mean, var, std = bayes_mvs(err, alpha=alpha)
    return mean, var, std


def estimator_bootstrap(err, custom_stat=None, alpha=0.05, n_iter=10000):
    """
      def custom_stat(values, axis=1):
      # stat_val = np.mean(np.asmatrix(values),axis=axis)
      # stat_val = np.std(np.asmatrix(values),axis=axis)p.mean
      stat_val = np.sqrt(np.mean(np.asmatrix(values*values),axis=axis))
      return stat_val
    """
    import bootstrapped.bootstrap as bs
    res = bs.bootstrap(err, stat_func=custom_stat, alpha=alpha, num_iterations=n_iter)
    return res



def np_col_extractname(col_onehot):
    """
    Column extraction from onehot name
    :param col_onehotp
    :return:
    """
    colnew = []
    for x in col_onehot:
        if len(x) > 2:
            if x[-2] == "_":
                if x[:-2] not in colnew:
                    colnew.append(x[:-2])

            elif x[-2] == "-":
                if x[:-3] not in colnew:
                    colnew.append(x[:-3])

            else:
                if x not in colnew:
                    colnew.append(x)
    return colnew


def np_list_remove(cols, colsremove, mode="exact"):
    """
    """
    if mode == "exact":
        for x in colsremove:
            try:
                cols.remove(x)
            except BaseException:
                pass
        return cols

    if mode == "fuzzy":
        cols3 = []
        for t in cols:
            flag = 0
            for x in colsremove:
                if x in t:
                    flag = 1
                    break
            if flag == 0:
                cols3.append(t)
        return cols3


def np_conv_to_one_col(np_array, sep_char="_"):
    """
    converts string/numeric columns to one string column
    :param np_array: the numpy array with more than one column
    :param sep_char: the separator character
    """
    def row2string(row_):
        return sep_char.join([str(i) for i in row_])

    np_array_=np.apply_along_axis(row2string,1,np_array)
    return np_array_[:,None]





################################################################################
################################################################################
if __name__ == '__main__':
    #import fire
    #fire.Fire()
    test_all()


    
    
    
