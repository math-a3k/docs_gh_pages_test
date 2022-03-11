# -*- coding: utf-8 -*-
MNAME = "utilmy.recsys.ab"
HELP = """"
All about abtest

cd utilmy/recsys/ab
python ab.py ab_getstat --df  /mypath/data.parquet   



_pooled_prob(N_A, N_B, X_A, X_B)
_pooled_SE(N_A, N_B, X_A, X_B)
_p_val(N_A, N_B, p_A, p_B)
np_calculate_z_val(sig_level=0.05, two_tailed=True)
np_calculate_confidence_interval(sample_mean=0, sample_std=1, sample_size=1, sig_level=0.05)
np_calculate_ab_dist(stderr, d_hat=0, group_type='control')
pd_generate_ctr_data(N_A, N_B, p_A, p_B, days=None, control_label='A'
np_calculate_min_sample_size(bcr, mde, power=0.8, sig_level=0.05)
plot_confidence_interval(ax, mu, s, sig_level=0.05, color='grey')
plot_norm_dist(ax, mu, std, with_CI=False, sig_level=0.05, label=None)
plot_binom_dist(ax, A_converted, A_cr, A_total, B_converted, B_cr, B_total)
plot_null_hypothesis_dist(ax, stderr)
plot_alternate_hypothesis_dist(ax, stderr, d_hat)
show_area(ax, d_hat, stderr, sig_level, area_type='power')
plot_ab(ax, N_A, N_B, bcr, d_hat, sig_level=0.05, show_power=False
zplot(ax, area=0.95, two_tailed=True, align_right=False)
abplot_CI_bars(N, X, sig_level=0.05, dmin=None)
funnel_CI_plot(A, B, sig_level=0.05)
ab_getstat(df,treatment_col,measure_col,attribute_cols,control_label,variation_label,inference_method,hypothesis,alpha,experiment_name)


https://pypi.org/project/abracadabra/


"""
import os, sys, random, numpy as np, pandas as pd, fire, time
from typing import List
from tqdm import tqdm
from box import Box
import scipy.stats as scs
import matplotlib.pyplot as plt


### from utilmy.stats.hypothetical import 

try :
  import abra
except :
   from utilmy.utilmy import sys_install
   pkg = "  abracadabra   hypothetical  "
   print('Installing pip install ' + pkg) ; time.time(7)
   # sys_install(cmd= f"pip install {pkg}  --upgrade-strategy only-if-needed")
   1/0  ### exit Gracefully !


##################################################################################################
from utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    ss = HELP + help_create(MNAME)
    print(ss)

   

    
#################################################################################################
def test_ab_getstat():
    """function test_ab_getstat
    Args:
    Returns:
        
    """
    from abra.utils import generate_fake_observations
    
    # generate demo data
    df_log = generate_fake_observations(
        distribution='bernoulli',
        n_treatments=2,
        n_attributes=3,
        n_observations=120
    )
    log(df_log)

    result = ab_getstat(df=df_log,
                        treatment_col='treatment',
                        measure_col='metric',
                        attribute_cols=['attr_0','attr_1'],
                        control_label='A',
                        variation_label='B',
                        inference_method=['proportions_delta'],
                        hypothesis=['larger', 'smaller'],
                        dirout=None,
                        )
    result = str(result.to_dict(orient='records'))
    expected_df = "[{'metric': 'metric', 'hypothesis': 'B is larger', 'model_name': 'proportions_delta', " \
                  "'accept_hypothesis': True, 'control_name': 'A', 'control_nobs': 61.0, 'control_mean': 0.5409836065573771, 'control_ci': (0.4148941954376071, 0.6670730176771471), 'control_var': 0.24832034399355019, 'variation_name': 'B', 'variation_nobs': 59.0, 'variation_mean': 0.6779661016949152, 'variation_ci': (0.5577150512709282, 0.7982171521189022), 'variation_var': 0.218328066647515, 'delta': 0.13698249513753813, 'delta_relative': 25.32100667693886, 'effect_size': 0.28343191705927406, 'alpha': 0.05, 'segmentation': None, 'warnings': None, 'test_type': 'frequentist', 'p': 0.03811217367380087, 'p_interpretation': 'p-value', 'delta_ci': (-0.009399936102360451, inf), 'ntiles_ci': (0.05, inf), 'delta_relative_ci': (-1.7375639461938985, inf), 'ci_interpretation': 'Confidence Interval', 'p_value': 0.03811217367380087, 'power': 0.4630913221833776, 'statistic_name': 'z', 'statistic_value': 1.7730263098116528, 'df': None, 'mc_correction': None}, {'metric': 'metric', 'hypothesis': 'B is smaller', 'model_name': 'proportions_delta', 'accept_hypothesis': False, 'control_name': 'A', 'control_nobs': 61.0, 'control_mean': 0.5409836065573771, 'control_ci': (0.4148941954376071, 0.6670730176771471), 'control_var': 0.24832034399355019, 'variation_name': 'B', 'variation_nobs': 59.0, 'variation_mean': 0.6779661016949152, 'variation_ci': (0.5577150512709282, 0.7982171521189022), 'variation_var': 0.218328066647515, 'delta': 0.13698249513753813, 'delta_relative': 25.32100667693886, 'effect_size': 0.28343191705927406, 'alpha': 0.05, 'segmentation': None, 'warnings': None, 'test_type': 'frequentist', 'p': 0.9618878263261992, 'p_interpretation': 'p-value', 'delta_ci': (-inf, 0.28336492637743665), 'ntiles_ci': (-inf, 0.95), 'delta_relative_ci': (-inf, 52.37957730007161), 'ci_interpretation': 'Confidence Interval', 'p_value': 0.9618878263261992, 'power': 0.0006941837287176945, 'statistic_name': 'z', 'statistic_value': 1.7730263098116528, 'df': None, 'mc_correction': None}]"
    assert result == expected_df

    ab_getstat(df=df_log,
                        treatment_col='treatment',
                        measure_col='metric',
                        attribute_cols=['attr_0','attr_1'],
                        control_label='A',
                        variation_label='B',
                        inference_method=['proportions_delta'],
                        hypothesis=['larger', 'smaller'],
                        experiment_name='Experiment',
                        dirout='./results',
                        tag='test'
                        )
    assert os.path.exists('./results_test/abplot_inference=proportions_delta_hypothesis=larger.png')
    assert os.path.exists('./results_test/abplot_inference=proportions_delta_hypothesis=smaller.png')
    assert os.path.exists('./results_test/abstats.parquet')

    
def test_np_calculate_z_val():
    """function test_np_calculate_z_val
    Args:
    Returns:
        
    """
    from numpy.testing import assert_almost_equal
    assert_almost_equal(np_calculate_z_val(sig_level=0.05), 1.9599, decimal=4)
    assert_almost_equal(np_calculate_z_val(sig_level=0.1), 1.6448, decimal=4)


def test_np_calculate_confidence_interval():
    """function test_np_calculate_confidence_interval
    Args:
    Returns:
        
    """
    from numpy.testing import assert_almost_equal
    ci_1 = np_calculate_confidence_interval(sample_mean=5,
                                            sample_std=1,
                                            sample_size=50,
                                            sig_level=0.05)
    assert_almost_equal(ci_1[0], 4.7228, decimal=4)
    assert_almost_equal(ci_1[1], 5.2771, decimal=4)
    assert len(ci_1) == 2

    ci_2 = np_calculate_confidence_interval(sample_mean=10,
                                            sample_std=3,
                                            sample_size=10,
                                            sig_level=0.1)
    assert_almost_equal(ci_2[0], 8.4395, decimal=4)
    assert_almost_equal(ci_2[1], 11.5604, decimal=4)
    assert len(ci_2) == 2


def test_np_calculate_ab_dist():
    """function test_np_calculate_ab_dist
    Args:
    Returns:
        
    """
    dist_1 = np_calculate_ab_dist(stderr=5, d_hat=1, group_type='control')
    assert dist_1.mean() == 0.0
    assert dist_1.std() == 5.0

    dist_2 = np_calculate_ab_dist(stderr=5, d_hat=1, group_type='test')
    assert dist_2.mean() == 1.0
    assert dist_2.std() == 5.0


def test_pd_generate_ctr_data():
    """function test_pd_generate_ctr_data
    Args:
    Returns:
        
    """
    ab_data, ab_summary = pd_generate_ctr_data(1000,1000,0.4,0.7, seed=35)
    ab_summary = ab_summary.reset_index().to_dict(orient='records')

    expected_df = [{'converted': 426, 'group': 'A', 'rate': 0.4108003857280617, 'total': 1037},
                {'converted': 671, 'group': 'B', 'rate': 0.6967808930425753, 'total': 963}]

    assert ab_summary == expected_df


def test_np_calculate_min_sample_size():
    """function test_np_calculate_min_sample_size
    Args:
    Returns:
        
    """
    assert np_calculate_min_sample_size(bcr=0.7, mde=0.2) == 62
    assert np_calculate_min_sample_size(bcr=0.7, mde=0.3) == 22


def get_ab_test_data(vars_also=False):
    """function get_ab_test_data
    Args:
        vars_also:   
    Returns:
        
    """
    # A is control; B is test
    N_A = 1000
    N_B = 1000
    bcr = 0.10  # baseline conversion rate
    d_hat = 0.02  # difference between the groups
    seed = 35

    vars = {'N_A': 1000, 'N_B': 1000, 'bcr': 0.10 , 'd_hat': 0.02, 'seed': 35}
    data, summary = pd_generate_ctr_data(N_A, N_B, bcr, bcr+d_hat, seed=seed)

    vars['A_converted'] = summary.loc['A','converted']
    vars['A_total'] = summary.loc['A','total']
    vars['A_cr'] = summary.loc['A','rate']

    vars['B_converted'] = summary.loc['B','converted']
    vars['B_total'] = summary.loc['B','total']
    vars['B_cr'] = summary.loc['B','rate']

    if vars_also:
        return (data, vars)
    return data


def test_plot_binom_dist():
    """function test_plot_binom_dist
    Args:
    Returns:
        
    """
    from numpy.testing import assert_almost_equal
    _, vars = get_ab_test_data(vars_also=True)

    fig, ax = plt.subplots(figsize=(12,6))
    plot_binom_dist(ax,
                    vars['A_converted'],
                    vars['A_cr'],
                    vars['A_total'],
                    vars['B_converted'],
                    vars['B_cr'],
                    vars['B_total'])
    
    fig.canvas.draw()
    plt.close()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_mean = data.mean()
    assert_almost_equal(fig_mean, 242.1494, decimal=4)


def test_plot_ab():
    """function test_plot_ab
    Args:
    Returns:
        
    """
    from numpy.testing import assert_almost_equal
    _, vars = get_ab_test_data(vars_also=True)
    bcr = vars['A_cr']
    d_hat = vars['B_cr'] - vars['A_cr']

    fig, ax = plt.subplots(figsize=(12,6))
    plot_ab(ax, vars['N_A'], vars['N_B'], bcr, d_hat)
                    
    fig.canvas.draw()
    plt.close()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_mean = data.mean()
    assert_almost_equal(fig_mean, 251.3217, decimal=4)

    fig, ax = plt.subplots(figsize=(12,6))
    plot_ab(ax, vars['N_A'], vars['N_B'], bcr, d_hat, show_power=True)
                    
    fig.canvas.draw()
    plt.close()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_mean = data.mean()
    assert_almost_equal(fig_mean, 249.4322, decimal=4)

    fig, ax = plt.subplots(figsize=(12,6))
    plot_ab(ax, vars['N_A'], vars['N_B'], bcr, d_hat, show_alpha=True)
                    
    fig.canvas.draw()
    plt.close()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_mean = data.mean()
    assert_almost_equal(fig_mean, 251.0622, decimal=4)

    fig, ax = plt.subplots(figsize=(12,6))
    plot_ab(ax, vars['N_A'], vars['N_B'], bcr, d_hat, show_beta=True)
                    
    fig.canvas.draw()
    plt.close()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_mean = data.mean()
    assert_almost_equal(fig_mean, 248.5483, decimal=4)


def test_zplot():
    """function test_zplot
    Args:
    Returns:
        
    """
    from numpy.testing import assert_almost_equal

    fig, ax = plt.subplots(figsize=(12,6))
    zplot(ax, area=0.95, two_tailed=True, align_right=False)
                    
    fig.canvas.draw()
    plt.close()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_mean = data.mean()
    assert_almost_equal(fig_mean, 248.3205, decimal=4)

    fig, ax = plt.subplots(figsize=(12,6))
    zplot(ax, area=0.80, two_tailed=False, align_right=True)
                    
    fig.canvas.draw()
    plt.close()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_mean = data.mean()
    assert_almost_equal(fig_mean, 248.9381, decimal=4)



def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    test_np_calculate_z_val()
    test_np_calculate_confidence_interval()
    test_np_calculate_ab_dist()
    test_pd_generate_ctr_data()
    test_np_calculate_min_sample_size()
    test_plot_binom_dist()
    test_plot_ab()
    test_zplot()
    test_ab_getstat


################################################################################################################
def ab_getstat(df,
               treatment_col='treatment',
               measure_col='metric',
               attribute_cols='attrib',
               control_label='A',
               variation_label='B',
               inference_method='means_delta',
               hypothesis=None,
               alpha=.05,
               experiment_name='exp',
               dirout=None,
               tag=None,
               **kwargs
               ):
    """ Wrapper function for running AB Tests.
    Args:
        df (DataFrame): the tabular data to analyze, must have columns that 
            correspond with `treatment`, `measures`, `attributes`, and `enrollment` 
            if any of those are defined.
        treatment_col (str): the column in `data` that identifies the association of each
            enrollment in the experiment with one of the experiment conditions.
        measure_col (str): the column in the dataset that is associated with indicator 
            measurements.
        attribute_cols (list[str]): the columns in `data` that define segmenting attributes
            associated with each enrollment in the experiment.
        control_label (str): the name of the control treatment.
        variation_label (str): the name of the experimental treatment.

        inference_method (str, list[str]): a single or a list of inference method, one for each test.
            Each item in the list is the name of the inference method used to perform the hypothesis test.
            Can be one of the following:
            Frequentist Inference:
                - 'means_delta'         Continuous
                - 'proprortions_delta'  Proportions
                - 'rates_ratio'         Counts / rates

            Bayesian Inference:
                - 'gaussian'            Continuous
                - 'exp_student_t'       Continuous
                - 'bernoulli'           Proportions / binary
                - 'beta_binomial'       Proportions
                - 'binomial'            Proportions
                - 'gamma_poisson'       Counts / rates
        hypothesis (str, list[str]): a single or a list of hypothesis names to be tested, one for each test
        alpha (float): the Type I error rate
        exp_name (str): the name of the experiment.
        dirout (str): disk path where results and plots will be saved.
        tag (str): suffix that gets appended to the dirout
            if none is provided, timestamp will be used as suffix
    Returns:
        DataFrame: result of the AB test experiment.
    """
    from abra import Experiment, HypothesisTest

    inference_method = [inference_method] if isinstance(inference_method, str) else inference_method
    hypothesis       = [hypothesis] if isinstance(hypothesis, str) else hypothesis

    if len(inference_method)>1:
        if len(hypothesis)>1:
            assert len(inference_method) == len(hypothesis), 'Length of inference_method and hypothesis should be same'
        else:
            hypothesis = hypothesis*len(inference_method)
    else:
        if len(hypothesis)>1:
            inference_method = inference_method*len(hypothesis)

    if isinstance(df, str):
      from utilmy import pd_read_file
      df = pd_read_file(df)   #### 
      log(df.shape, df.columns)

    if dirout is not None:
        if tag is None:
            tag = str(int(time.time()))
        dirout = dirout + "_" + tag
        os.makedirs(dirout, exist_ok=True)
            
    exp = Experiment(data=df,  treatment=treatment_col,
                     measures=measure_col,
                     attributes=attribute_cols,
                     name=experiment_name)
    
    abtest_res = pd.DataFrame()
    for i, h in zip(inference_method, hypothesis):
        ab_test = HypothesisTest(metric=measure_col,
                                treatment=treatment_col,
                                control=control_label,
                                variation=variation_label,
                                inference_method=i,
                                hypothesis=h)
        
        ab_test_result = exp.run_test(ab_test, alpha=alpha)

        if dirout is not None:
            outfile = os.path.join(dirout, f'abplot_inference={i}_hypothesis={h}')
            ab_test_result.visualize(outfile=outfile)

        ab_test_result_df = ab_test_result.to_dataframe()
        abtest_res = pd.concat([abtest_res, ab_test_result_df], ignore_index=True)

    if dirout is not None:
        from utilmy import pd_to_file
        save_path = os.path.join(dirout, 'abstats.parquet')
        pd_to_file(abtest_res.astype('str'), save_path, show=1)
    else :  
      return abtest_res
  
  
  
  

def np_calculate_z_val(sig_level=0.05, two_tailed=True):
    """Returns the z value for a given significance level"""
    z_dist = scs.norm()
    if two_tailed:
        sig_level = sig_level/2
        area = 1 - sig_level
    else:
        area = 1 - sig_level

    z = z_dist.ppf(area)
    return z


def np_calculate_confidence_interval(sample_mean=0, sample_std=1, sample_size=1, sig_level=0.05):
    """Returns the confidence interval as a tuple"""
    z = np_calculate_z_val(sig_level)

    left = sample_mean - z * sample_std / np.sqrt(sample_size)
    right = sample_mean + z * sample_std / np.sqrt(sample_size)

    return (left, right)


def np_calculate_ab_dist(stderr, d_hat=0, group_type='control'):
    """Returns a distribution object depending on group type
    Examples:
    Parameters:
        stderr (float): pooled standard error of two independent samples
        d_hat (float): the mean difference between two independent samples
        group_type (string): 'control' and 'test' are supported
    Returns:
        dist (scipy.stats distribution object)
    """
    if group_type == 'control':
        sample_mean = 0

    elif group_type == 'test':
        sample_mean = d_hat

    # create a normal distribution which is dependent on mean and std dev
    dist = scs.norm(sample_mean, stderr)
    return dist


def pd_generate_ctr_data(N_A, N_B, p_A, p_B, days=None, control_label='A',
                  test_label='B', seed=None):
    """Returns a pandas dataframe with fake CTR data
    Example:
    Parameters:
        N_A (int): sample size for control group
        N_B (int): sample size for test group
            Note: final sample size may not match N_A provided because the
            group at each row is chosen at random (50/50).
        p_A (float): conversion rate; conversion rate of control group
        p_B (float): conversion rate; conversion rate of test group
        days (int): optional; if provided, a column for 'ts' will be included
            to divide the data in chunks of time
            Note: overflow data will be included in an extra day
        control_label (str)
        test_label (str)
        seed (int)
    Returns:
        pd.DataFrame: the generated ctr dataframe
        pd.DataFrame: summary dataframe
    """
    if seed:
        np.random.seed(seed)

    # initiate empty container
    data = []

    # total amount of rows in the data
    N = N_A + N_B

    # distribute events based on proportion of group size
    group_bern = scs.bernoulli(N_A / (N_A + N_B))

    # initiate bernoulli distributions from which to randomly sample
    A_bern = scs.bernoulli(p_A)
    B_bern = scs.bernoulli(p_B)

    for idx in range(N):
        # initite empty row
        row = {}
        # for 'ts' column
        if days is not None:
            if type(days) == int:
                row['ts'] = idx // (N // days)
            else:
                raise ValueError("Provide an integer for the days parameter.")
        # assign group based on 50/50 probability
        row['group'] = group_bern.rvs()

        if row['group'] == 0:
            # assign conversion based on provided parameters
            row['converted'] = A_bern.rvs()
        else:
            row['converted'] = B_bern.rvs()
        # collect row into data container
        data.append(row)

    # convert data into pandas dataframe
    df = pd.DataFrame(data)

    # transform group labels of 0s and 1s to user-defined group labels
    df['group'] = df['group'].apply(
        lambda x: control_label if x == 0 else test_label)
    
    # summary dataframe
    ab_summary = df.pivot_table(values='converted', index='group', aggfunc=np.sum)
    # add additional columns to the pivot table
    ab_summary['total'] = df.pivot_table(values='converted', index='group', aggfunc=lambda x: len(x))
    ab_summary['rate'] = df.pivot_table(values='converted', index='group')

    return df, ab_summary


def np_calculate_min_sample_size(bcr, mde, power=0.8, sig_level=0.05):
    """Returns the minimum sample size to set up a split test
    Arguments:
        bcr (float): probability of success for control, sometimes
        referred to as baseline conversion rate
        mde (float): minimum change in measurement between control
        group and test group if alternative hypothesis is true, sometimes
        referred to as minimum detectable effect
        power (float): probability of rejecting the null hypothesis when the
        null hypothesis is false, typically 0.8
        sig_level (float): significance level often denoted as alpha,
        typically 0.05
    Returns:
        min_N: minimum sample size (float)
    References:
        Stanford lecture on sample sizes
        http://statweb.stanford.edu/~susan/courses/s141/hopower.pdf
    """
    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)

    # find Z_beta from desired power
    Z_beta = standard_norm.ppf(power)

    # find Z_alpha
    Z_alpha = standard_norm.ppf(1-sig_level/2)

    # average of probabilities from both groups
    pooled_prob = (bcr + bcr+mde) / 2

    min_N = int((2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2
             / mde**2))

    return min_N


def plot_confidence_interval(ax, mu, s, sig_level=0.05, color='grey'):
    """Calculates the two-tailed confidence interval and adds the plot to
    an axes object.
    Example:
        plot_confidence_interval(ax, mu=0, s=stderr, sig_level=0.05)
    Parameters:
        ax (matplotlib axes)
        mu (float): mean
        s (float): standard deviation
    Returns:
        None: the function adds a plot to the axes object provided
    """
    left, right = np_calculate_confidence_interval(sample_mean=mu, sample_std=s,
                                      sig_level=sig_level)
    ax.axvline(left, c=color, linestyle='--', alpha=0.5)
    ax.axvline(right, c=color, linestyle='--', alpha=0.5)


def plot_norm_dist(ax, mu, std, with_CI=False, sig_level=0.05, label=None):
    """Adds a normal distribution to the axes provided
    Example:
        plot_norm_dist(ax, 0, 1)  # plots a standard normal distribution
    Parameters:
        ax (matplotlib axes)
        mu (float): mean of the normal distribution
        std (float): standard deviation of the normal distribution
    Returns:
        None: the function adds a plot to the axes object provided
    """
    x = np.linspace(mu - 12 * std, mu + 12 * std, 1000)
    y = scs.norm(mu, std).pdf(x)
    ax.plot(x, y, label=label)

    if with_CI:
        plot_confidence_interval(ax, mu, std, sig_level=sig_level)


def plot_binom_dist(ax, A_converted, A_cr, A_total, B_converted, B_cr, B_total):
    """Adds a binomial distribution to the axes provided
    Parameters:
        ax (matplotlib axes)
        A_converted
        A_cr
        A_total
        B_converted
        B_cr
        B_total
    Returns:
        None: the function adds a plot to the axes object provided
    """
    xA = np.linspace(A_converted-49, A_converted+50, 100)
    yA = scs.binom(A_total, A_cr).pmf(xA)
    ax.bar(xA, yA, alpha=0.5)
    xB = np.linspace(B_converted-49, B_converted+50, 100)
    yB = scs.binom(B_total, B_cr).pmf(xB)
    ax.bar(xB, yB, alpha=0.5)
    plt.xlabel('converted')
    plt.ylabel('probability')


def plot_null_hypothesis_dist(ax, stderr):
    """Plots the null hypothesis distribution where if there is no real change,
    the distribution of the differences between the test and the control groups
    will be normally distributed.
    The confidence band is also plotted.
    Example:
        plot_null_hypothesis_dist(ax, stderr)
    Parameters:
        ax (matplotlib axes)
        stderr (float): the pooled standard error of the control and test group
    Returns:
        None: the function adds a plot to the axes object provided
    """
    plot_norm_dist(ax, 0, stderr, label="Null")
    plot_confidence_interval(ax, mu=0, s=stderr, sig_level=0.05)


def plot_alternate_hypothesis_dist(ax, stderr, d_hat):
    """Plots the alternative hypothesis distribution where if there is a real
    change, the distribution of the differences between the test and the
    control groups will be normally distributed and centered around d_hat
    The confidence band is also plotted.
    Example:
        plot_alternate_hypothesis_dist(ax, stderr, d_hat=0.025)
    Parameters:
        ax (matplotlib axes)
        stderr (float): the pooled standard error of the control and test group
    Returns:
        None: the function adds a plot to the axes object provided
    """
    plot_norm_dist(ax, d_hat, stderr, label="Alternative")


def show_area(ax, d_hat, stderr, sig_level, area_type='power'):
    """Fill between upper significance boundary and distribution for
    alternative hypothesis
    """
    left, right = np_calculate_confidence_interval(sample_mean=0, sample_std=stderr,
                                      sig_level=sig_level)
    x = np.linspace(-12 * stderr, 12 * stderr, 1000)
    null = np_calculate_ab_dist(stderr, 'control')
    alternative = np_calculate_ab_dist(stderr, d_hat, 'test')

    # if area_type is power
    # Fill between upper significance boundary and distribution for alternative
    # hypothesis
    if area_type == 'power':
        ax.fill_between(x, 0, alternative.pdf(x), color='green', alpha=0.25,
                        where=(x > right))
        ax.text(-3 * stderr, null.pdf(0),
                'power = {0:.3f}'.format(1 - alternative.cdf(right)),
                fontsize=12, ha='right', color='k')

    # if area_type is alpha
    # Fill between upper significance boundary and distribution for null
    # hypothesis
    if area_type == 'alpha':
        ax.fill_between(x, 0, null.pdf(x), color='green', alpha=0.25,
                        where=(x > right))
        ax.text(-3 * stderr, null.pdf(0),
                'alpha = {0:.3f}'.format(1 - null.cdf(right)),
                fontsize=12, ha='right', color='k')

    # if area_type is beta
    # Fill between distribution for alternative hypothesis and upper
    # significance boundary
    if area_type == 'beta':
        ax.fill_between(x, 0, alternative.pdf(x), color='green', alpha=0.25,
                        where=(x < right))
        ax.text(-3 * stderr, null.pdf(0),
                'beta = {0:.3f}'.format(alternative.cdf(right)),
                fontsize=12, ha='right', color='k')


def plot_ab(ax, N_A, N_B, bcr, d_hat, sig_level=0.05, show_power=False,
           show_alpha=False, show_beta=False, show_p_value=False,
           show_legend=True):
    """Example plot of AB test
    Example:
        abplot(n=4000, bcr=0.11, d_hat=0.03)
    Parameters:
        n (int): total sample size for both control and test groups (N_A + N_B)
        bcr (float): base conversion rate; conversion rate of control
        d_hat: difference in conversion rate between the control and test
            groups, sometimes referred to as **minimal detectable effect** when
            calculating minimum sample size or **lift** when discussing
            positive improvement desired from launching a change.
    Returns:
        None: the function plots an AB test as two distributions for
        visualization purposes
    """

    # define parameters to find pooled standard error
    X_A = bcr * N_A
    X_B = (bcr + d_hat) * N_B
    stderr = _pooled_SE(N_A, N_B, X_A, X_B)

    # plot the distribution of the null and alternative hypothesis
    plot_null_hypothesis_dist(ax, stderr)
    plot_alternate_hypothesis_dist(ax, stderr, d_hat)

    # set extent of plot area
    ax.set_xlim(-8 * stderr, 8 * stderr)

    # shade areas according to user input
    if show_power:
        show_area(ax, d_hat, stderr, sig_level, area_type='power')
    if show_alpha:
        show_area(ax, d_hat, stderr, sig_level, area_type='alpha')
    if show_beta:
        show_area(ax, d_hat, stderr, sig_level, area_type='beta')

    # show p_value based on the binomial distributions for the two groups
    if show_p_value:
        null = np_calculate_ab_dist(stderr, 'control')
        p_value = _p_val(N_A, N_B, bcr, bcr+d_hat)
        ax.text(3 * stderr, null.pdf(0),
                'p-value = {0:.3f}'.format(p_value),
                fontsize=12, ha='left')

    # option to show legend
    if show_legend:
        plt.legend()

    plt.xlabel('d')
    plt.ylabel('PDF')


def zplot(ax, area=0.95, two_tailed=True, align_right=False):
    """Plots a z distribution with common annotations
    Example:
        zplot(area=0.95)
        zplot(area=0.80, two_tailed=False, align_right=True)
    Parameters:
        area (float): The area under the standard normal distribution curve.
        align (str): The area under the curve can be aligned to the center
            (default) or to the left.
    Returns:
        None: A plot of the normal distribution with annotations showing the
        area under the curve and the boundaries of the area.
    """
    # create normal distribution
    norm = scs.norm()
    # create data points to plot
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)

    ax.plot(x, y)

    # code to fill areas
    # for two-tailed tests
    if two_tailed:
        left = norm.ppf(0.5 - area / 2)
        right = norm.ppf(0.5 + area / 2)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')

        ax.fill_between(x, 0, y, color='grey', alpha=0.25,
                        where=(x > left) & (x < right))
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12,
                 rotation=90, va="bottom", ha="right")
        plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                 fontsize=12, rotation=90, va="bottom", ha="left")
    # for one-tailed tests
    else:
        # align the area to the right
        if align_right:
            left = norm.ppf(1-area)
            ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha=0.25,
                            where=x > left)
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left),
                     fontsize=12, rotation=90, va="bottom", ha="right")
        # align the area to the left
        else:
            right = norm.ppf(area)
            ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha=0.25,
                            where=x < right)
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                     fontsize=12, rotation=90, va="bottom", ha="left")

    # annotate the shaded area
    plt.text(0, 0.1, "shaded area = {0:.3f}".format(area), fontsize=12,
             ha='center')
    # axis labels
    plt.xlabel('z')
    plt.ylabel('PDF')


def abplot_CI_bars(N, X, sig_level=0.05, dmin=None):
    """Returns a confidence interval bar plot for multivariate tests
    Parameters:
        N (list or tuple): sample size for all groups
        X (list or tuple): number of conversions for each variant
        sig_level (float): significance level
        dmin (float): minimum desired lift; a red and green dashed lines are
            shown on the plot if dmin is provided.
    Returns:
        None: A plot of the confidence interval bars is returned inline.
    """

    # initiate plot object
    fig, ax = plt.subplots(figsize=(12, 3))

    # get control group values
    N_A = N[0]
    X_A = X[0]

    # initiate containers for standard error and differences
    SE = []
    d = []
    # iterate through X and N and calculate d and SE
    for idx in range(1, len(N)):
        X_B = X[idx]
        N_B = N[idx]
        d.append(X_B / N_B - X_A / N_A)
        SE.append(_pooled_SE(N_A, N_B, X_A, X_B))

    # convert to numpy arrays
    SE = np.array(SE)
    d = np.array(d)

    y = np.arange(len(N)-1)

    # get z value
    z = np_calculate_z_val(sig_level)
    # confidence interval values
    ci = SE * z

    # bar to represent the confidence interval
    ax.hlines(y, d-ci, d+ci, color='blue', alpha=0.35, lw=10, zorder=1)
    # marker for the mean
    ax.scatter(d, y, s=300, marker='|', lw=10, color='magenta', zorder=2)

    # vertical line to represent 0
    ax.axvline(0, c='grey', linestyle='-')

    # plot veritcal dashed lines if dmin is provided
    if dmin is not None:
        ax.axvline(-dmin, c='red', linestyle='--', alpha=0.75)
        ax.axvline(dmin, c='green', linestyle='--', alpha=0.75)

    # invert y axis to show variant 1 at the top
    ax.invert_yaxis()
    # label variants on y axis
    labels = ['variant{}'.format(idx+1) for idx in range(len(N)-1)]
    plt.yticks(np.arange(len(N)-1), labels)


def funnel_CI_plot(A, B, sig_level=0.05):
    """Returns a confidence interval bar plot for multivariate tests
    Parameters:
        A (list of tuples): (sample size, conversions) for control group funnel
        B (list of tuples): (sample size, conversions) for test group funnel
        sig_level (float): significance level
    Returns:
        None: A plot of the confidence interval bars is returned inline.
    """

    # initiate plot object
    fig, ax = plt.subplots(figsize=(12, 3))

    # initiate containers for standard error and differences
    SE = []
    d = []
    # iterate through X and N and calculate d and SE
    for idx in range(len(A)):
        X_A = A[idx][1]
        N_A = A[idx][0]
        X_B = B[idx][1]
        N_B = B[idx][0]
        d.append(X_B / N_B - X_A / N_A)
        SE.append(_pooled_SE(N_A, N_B, X_A, X_B))

    # convert to numpy arrays
    SE = np.array(SE)
    d = np.array(d)
    print(d)

    y = np.arange(len(A))

    # get z value
    z = np_calculate_z_val(sig_level)
    # confidence interval values
    ci = SE * z

    # bar to represent the confidence interval
    ax.hlines(y, d-ci, d+ci, color='blue', alpha=0.35, lw=10, zorder=1)
    # marker for the mean
    ax.scatter(d, y, s=300, marker='|', lw=10, color='magenta', zorder=2)

    # vertical line to represent 0
    ax.axvline(0, c='grey', linestyle='-')

    # invert y axis to show variant 1 at the top
    ax.invert_yaxis()
    # label variants on y axis
    labels = ['metric{}'.format(idx+1) for idx in range(len(A))]
    plt.yticks(np.arange(len(A)), labels)

    
  


def _pooled_prob(N_A, N_B, X_A, X_B):
    """Returns pooled probability for two samples"""
    return (X_A + X_B) / (N_A + N_B)


def _pooled_SE(N_A, N_B, X_A, X_B):
    """Returns the pooled standard error for two samples"""
    p_hat = _pooled_prob(N_A, N_B, X_A, X_B)
    SE = np.sqrt(p_hat * (1 - p_hat) * (1 / N_A + 1 / N_B))
    return SE


def _p_val(N_A, N_B, p_A, p_B):
    """Returns the p-value for an A/B test"""
    return scs.binom(N_A, p_A).pmf(p_B * N_B)


###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()  
  
