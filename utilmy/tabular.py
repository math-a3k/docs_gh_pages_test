HELP="""

https://github.com/topics/hypothesis-testing?l=python&o=desc&s=stars

https://pypi.org/project/pysie/#description



"""
import os, sys, pandas as pd, numpy as np

def log(*s):
    print(s)


#############################################################################
#############################################################################
def y_adjustment():
    """
       Adjustment of log, exp transfrmation for yt= y + error
       https://www.inovex.de/de/blog/honey-i-shrunk-the-target-variable/
       
       log(y) = u =sigma**2
    
    """
    def adjuter_log(y_true, y_pred_log, error_func, **kwargs):
        """Determine correction delta for exp transformation"""
        def cost_func(delta):
            return error_func(np.exp(delta + y_pred_log), y_true)
        res = sp.optimize.minimize(cost_func, 0., **kwargs)
        if res.success:
            return res.x
        else:
            raise RuntimeError(f"Finding correction term failed!\n{res}")


            
            
    
    
def test_anova(df, col1, col2):
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



def test_normality2(df, column, test_type):
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


def test_plot_qqplot(df, col_name):
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
def test_heteroscedacity(y, y_pred, pred_value_only=1):
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


def test_normality(error, distribution="norm", test_size_limit=5000):
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


def test_mutualinfo(error, Xtest, colname=None, bins=5):
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


def test_hypothesis(df_obs, df_ref, method='', **kw):
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
    # estimate_std( err, alpha=0.05, )
    from scipy import stats
    n = len(err)  # sample sizes
    s2 = np.var(err, ddof=1)  # sample variance
    df = n - 1  # degrees of freedom
    upper = np.sqrt((n - 1) * s2 / stats.chi2.ppf(alpha / 2, df))
    lower = np.sqrt((n - 1) * s2 / stats.chi2.ppf(1 - alpha / 2, df))

    return np.sqrt(s2), (lower, upper)


def estimator_boostrap_bayes(err, alpha=0.05, ):
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



#############################################################################################################
def pd_train_test_split_time(df, test_period = 40, cols=None , coltime ="time_key", sort=True, minsize=5,
                     n_sample=5,  verbose=False) :
   cols = list(df.columns) if cols is None else cols
   if sort :
       df   = df.sort_values( coltime, ascending=1 )
   #imax = len(df) - test_period
   colkey = [ t for t in cols if t not in [coltime] ]  #### All time reference be removed
   if verbose : log(colkey)
   imax = test_period * n_sample ## Over sampling
   df1  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[:max(minsize, len(dfi) -imax), :] ).reset_index(colkey, drop=True).reset_index(drop=True)
   df2  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[max(minsize,  len(dfi) -imax):, :] ).reset_index(colkey, drop=True).reset_index(drop=True)
   return df1, df2


def pd_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    import numpy as np
    from scipy.sparse import lil_matrix
    arr = lil_matrix(df.shape, dtype=np.float32)
    for i, col in enumerate(df.columns):
        ix = df[col] != 0
        arr[np.where(ix), i] = 1

    return arr.tocsr()


def pd_stat_correl_pair(df, coltarget=None, colname=None):
    """
      Genearte correletion between the column and target column
      df represents the dataframe comprising the column and colname comprising the target column
    :param df:
    :param colname: list of columns
    :param coltarget : target column
    :return:
    """
    from scipy.stats import pearsonr

    colname = colname if colname is not None else list(df.columns)
    target_corr = []
    for col in colname:
        target_corr.append(pearsonr(df[col].values, df[coltarget].values)[0])

    df_correl = pd.DataFrame({"colx": [""] * len(colname), "coly": colname, "correl": target_corr})
    df_correl[coltarget] = colname
    return df_correl


def pd_stat_pandas_profile(df, savefile="report.html", title="Pandas Profile"):
    """ Describe the tables
        #Pandas-Profiling 2.0.0
        df.profile_report()
    """
    print("start profiling")
    profile = df.profile_report(title=title)
    profile.to_file(output_file=savefile)
    colexclude = profile.get_rejected_variables(threshold=0.98)
    return colexclude


def pd_stat_distribution_colnum(df, nrows=2000, verbose=False):
    """ Stats the tables
    """
    df = df.sample(n=nrows)
    coldes = ["col", "coltype",  "count", "min", "max",  "median", "mean",
              "std", "25%", "75%", "nb_na", "pct_na" ]

    def getstat(col):
        """max, min, nb, nb_na, pct_na, median, qt_25, qt_75,
           nb, nb_unique, nb_na, freq_1st, freq_2th, freq_3th
           s.describe()
        """
        ss    = [col, str(df[col].dtype)]
        ss    = ss +list(df[col].describe().values)

        nb_na = df[col].isnull().sum()
        ntot  = len(df)
        ss    = ss + [nb_na, nb_na / (ntot + 0.0)]

        return pd.DataFrame( [ss],  columns=coldes, )

    dfdes = pd.DataFrame([], columns=coldes)
    cols  = df.columns
    for col in cols:
        dtype1 = str(df[col].dtype)
        if dtype1[0:3] in ["int", "flo"]:
            try :
              row1  = getstat(col)
              dfdes = pd.concat((dfdes, row1), axis=0)
            except Exception as e:
              print('error', col, e)

        if dtype1 == "object":
            pass

    dfdes.index = np.arange(0, len(dfdes))
    if verbose : print('Stats\n', dfdes)
    return dfdes


def pd_stat_histogram(df, bins=50, coltarget="diff"):
    """
    :param df:
    :param bins:
    :param coltarget:
    :return:
    """
    hh = np.histogram(
        df[coltarget].values, bins=bins, range=None, normed=None, weights=None, density=None
    )
    hh2 = pd.DataFrame({"bins": hh[1][:-1], "freq": hh[0]})
    hh2["density"] = hh2["freqall"] / hh2["freqall"].sum()
    return hh2


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


####################################################################################################
def pd_stat_shift_trend_changes(df, feature, target_col, threshold=0.03):
    """
    Calculates number of times the trend of feature wrt target changed direction.
    :param df: df_grouped dataset
    :param feature: feature column name
    :param target_col: target column
    :param threshold: minimum % difference required to count as trend change
    :return: number of trend chagnes for the feature
    """
    df                            = df.loc[df[feature] != 'Nulls', :].reset_index(drop=True)
    target_diffs                  = df[target_col + '_mean'].diff()
    target_diffs                  = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff                      = df[target_col + '_mean'].max() - df[target_col + '_mean'].min()
    target_diffs_mod              = target_diffs.fillna(0).abs()
    low_change                    = target_diffs_mod < threshold * max_diff
    target_diffs_norm             = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm             = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2             = target_diffs_norm.diff()
    changes                       = target_diffs_lvl2.fillna(0).abs() / 2
    tot_trend_changes             = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return (tot_trend_changes)


def pd_stat_shift_trend_correlation(df, df_test, colname, target_col):
    """
    Calculates correlation between train and test trend of colname wrt target.
    :param df: train df data
    :param df_test: test df data
    :param colname: colname column name
    :param target_col: target column name
    :return: trend correlation between train and test
    """
    df      = df[df[colname] != 'Nulls'].reset_index(drop=True)
    df_test = df_test[df_test[colname] != 'Nulls'].reset_index(drop=True)

    if df_test.loc[0, colname] != df.loc[0, colname]:
        df_test[colname]        = df_test[colname].cat.add_categories(df.loc[0, colname])
        df_test.loc[0, colname] = df.loc[0, colname]
    df_test_train = df.merge(df_test[[colname, target_col + '_mean']], on=colname,
                             how='left',
                             suffixes=('', '_test'))
    nan_rows = pd.isnull(df_test_train[target_col + '_mean']) | pd.isnull(
        df_test_train[target_col + '_mean_test'])
    df_test_train = df_test_train.loc[~nan_rows, :]
    if len(df_test_train) > 1:
        trend_correlation = np.corrcoef(df_test_train[target_col + '_mean'],
                                        df_test_train[target_col + '_mean_test'])[0, 1]
    else:
        trend_correlation = 0
        print("Only one bin created for " + colname + ". Correlation can't be calculated")

    return (trend_correlation)


def pd_stat_shift_changes(df, target_col, features_list=0, bins=10, df_test=0):
    """
    Calculates trend changes and correlation between train/test for list of features
    :param df: dfframe containing features and target columns
    :param target_col: target column name
    :param features_list: by default creates plots for all features. If list passed, creates plots of only those features.
    :param bins: number of bins to be created from continuous colname
    :param df_test: test df which has to be compared with input df for correlation
    :return: dfframe with trend changes and trend correlation (if test df passed)
    """

    if type(features_list) == int:
        features_list = list(df.columns)
        features_list.remove(target_col)

    stats_all = []
    has_test = type(df_test) == pd.core.frame.DataFrame
    ignored = []
    for colname in features_list:
        if df[colname].dtype == 'O' or colname == target_col:
            ignored.append(colname)
        else:
            cuts, df_grouped = pd_colnum_tocat_stat(df=df, colname=colname, target_col=target_col, bins=bins)
            trend_changes    = pd_stat_shift_trend_correlation(df=df_grouped, colname=colname, target_col=target_col)
            if has_test:
                df_test            = pd_colnum_tocat_stat(df=df_test.reset_index(drop=True), colname=colname,
                                                          target_col  = target_col, bins=bins, cuts=cuts)
                trend_corr         = pd_stat_shift_trend_correlation(df_grouped, df_test, colname, target_col)
                trend_changes_test = pd_stat_shift_changes(df=df_test, colname=colname,
                                                           target_col=target_col)
                stats = [colname, trend_changes, trend_changes_test, trend_corr]
            else:
                stats = [colname, trend_changes]
            stats_all.append(stats)
    stats_all_df = pd.DataFrame(stats_all)
    stats_all_df.columns = ['colname', 'Trend_changes'] if has_test == False else ['colname', 'Trend_changes',
                                                                                   'Trend_changes_test',
                                                                                   'Trend_correlation']
    if len(ignored) > 0:
        print('Categorical features ' + str(ignored) + ' ignored. Categorical features not supported yet.')

    print('Returning stats for all numeric features')
    return (stats_all_df)


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
