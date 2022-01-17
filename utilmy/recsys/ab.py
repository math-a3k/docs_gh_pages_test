""""
All about abtest



https://github.com/arita37/myutil/blob/main/docs/doc_index.py#L1582-L1590


estimator_boostrap_bayes(err, alpha = 0.05, )
estimator_bootstrap(err, custom_stat = None, alpha = 0.05, n_iter = 10000)
estimator_std_normal(err, alpha = 0.05, )
help()
log(*s)
np_col_extractname(col_onehot)
pd_stat_correl_pair(df, coltarget = None, colname = None)
pd_stat_distribution_colnum(df, nrows = 2000, verbose = False)
pd_stat_histogram(df, bins = 50, coltarget = "diff")
pd_stat_pandas_profile(df, savefile = "report.html", title = "Pandas Profile")
pd_stat_shift_changes(df, target_col, features_list = 0, bins = 10, df_test = 0)
pd_stat_shift_trend_changes(df, feature, target_col, threshold = 0.03)
pd_stat_shift_trend_correlation(df, df_test, colname, target_col)
pd_to_scipy_sparse_matrix(df)
pd_train_test_split_time(df, test_period  =  40, cols = None, coltime  = "time_key", sort = True, minsize = 5, n_sample = 5, verbose = False)


test_anova(df, col1, col2)
test_heteroscedacity(y, y_pred, pred_value_only = 1)
test_hypothesis(df_obs, df_ref, method = '', **kw)
test_multiple_comparisons(data: pd.DataFrame, label = 'y', adjuster = True)
test_mutualinfo(error, Xtest, colname = None, bins = 5)
test_normality(df, column, test_type)
test_normality2(df, column, test_type)
test_plot_qqplot(df, col_name)
y_adjuster_log(y_true, y_pred_log, error_func, **kwargs)


https://pypi.org/project/abracadabra/

https://pypi.org/project//

https://github.com/aschleg/hypothetical

Binomial
https://github.com/aschleg/hypothetical/blob/master/hypothetical/hypothesis.py#L48

"""
import os, sys, random, numpy as np, pandas as pd, fire, time
from typing import List
from tqdm import tqdm
from box import Box

try :
  import abra, hypothetical 
except :
   from utilmy.utilmy import sys_install
   pkg = "  abracadabra   hypothetical  "  
   sys_install(cmd= f"pip install {pkg}  --upgrade-strategy only-if-needed")      
   1/0  ### exit Gracefully !





################################################################################################################
def test():
  from abra.utils import generate_fake_observations

  # generate demo data
  experiment_observations = generate_fake_observations(
      distribution='bernoulli',
      n_treatments=3,
      n_attributes=4,
      n_observations=120
  )

  experiment_observations.head()

  # Running an AB Test is as easy as 1, 2, 3
  from abra import Experiment, HypothesisTest

  # 1. Initialize the `Experiment`
  # We (optionally) name the experiment "Demo"
  exp = Experiment(data=experiment_observations, name='Demo')

  # 2. Define the `HypothesisTest`
  # Here, we test that the variation "C" is "larger" than the control "A",
  # based on the values of the "metric" column, using a Frequentist z-test,
  # as parameterized by `inference_method="proportions_delta"`
  ab_test = HypothesisTest(
      metric='metric',
      treatment='treatment',
      control='A', variation='C',
      inference_method='proportions_delta',
      hypothesis='larger'
  )



