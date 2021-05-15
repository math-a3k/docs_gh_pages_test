"""

https://github.com/topics/hypothesis-testing?l=python&o=desc&s=stars

https://pypi.org/project/pysie/#description



"""
import os, sys, pandas as pd, numpy as np

def log(*s):
    print(s)


#############################################################################
#############################################################################
def pd_text_hash_create_lsh(df, col, sep=" ", threshold=0.7, num_perm=10):
    '''
    For each of the entry create a hash function
    '''
    from datasketch import MinHash, MinHashLSH
    #Create LSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    #Intialize list
    hash_lines = []
    
    ll = df[col].values
    for index, sentence in enumerate(ll):

        #Get tokens of individual elements
        tokens = sentence.split(sep)

        #Create local hash funtion
        v = MinHash(num_perm=num_perm)

        for j in set(tokens):
            v.update(j.encode('utf8'))

        #Append
        hash_lines.append(v)
        lsh.insert(str(index), v)
    return hash_lines, lsh



def pd_text_getcluster(df, col, threshold, num_perm):
    '''
    For each of the hash function find a cluster and assign unique id to the dataframe cluster_id
    '''
    #MAster cluster ids
    all_cluster_ids = []

    #REturn from hash list
    hash_lines, lsh = pd_text_hash_create_lsh(df, col=col, threshold = threshold, num_perm = num_perm)

    #For each local hash find the cluster ids and assign to the dataframe and return dataframe
    cluster_count = 1
    for ind, i in enumerate(hash_lines):

        if ind in all_cluster_ids:
            continue

        x_duplicate     = lsh.query(i)
        x_duplicate_int = list(map(int, x_duplicate))
        #print(x_duplicate_int)
        df.at[x_duplicate_int, 'cluster_id'] = cluster_count
        cluster_count   += 1
        all_cluster_ids += x_duplicate_int

    return df


def test_lsh():

    ll = [ 'aa bb cc', 'a b c', 'cc bb cc']
    column_name = "sentence"
    threshold   = 0.7
    num_perm    = 10
    num_items   = 100000

    df   = pd.DataFrame(ll, columns = [column_name])
    df1  = pd_text_getcluster(df.head(num_items), column_name, threshold, num_perm)
    print(df1)




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
