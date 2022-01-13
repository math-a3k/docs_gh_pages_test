# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-

"""
Generic template for new model.


(n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2,
 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
 max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, 
 warm_start=False, ccp_alpha=0.0, max_samples=None)[source]


"""
import os
import pandas as pd, numpy as np, scipy as sci

import sklearn
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.tree import *
from lightgbm import LGBMModel, LGBMRegressor,LGBMClassifier


####################################################################################################
VERBOSE = False
# MODEL_URI = get_model_uri(__file__)


# from mlmodels.util import log, path_norm, get_model_uri
def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session

def init(*kw, **kwargs) :
    global model, session
    model   = Model(*kw, **kwargs)
    session = None

    
class Model(object):
    def __init__(self, model_pars=None,  data_pars=None, compute_pars=None ):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None :
            self.model = None
        else :
            model_class = globals()[model_pars["model_name"]]
            self.model  = model_class(**model_pars['model_pars'])
            if VERBOSE : log(model_class, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest,  ytest = get_dataset(data_pars, task_type="train")
    if VERBOSE : log( Xtrain.shape, model.model)

    if  "LGBM" in model.model_pars['model_name'] :
       model.model.fit(Xtrain, ytrain, eval_set= [ (Xtest, ytest)], **compute_pars.get("compute_pars", {}) )
    
    else :
       model.model.fit(Xtrain, ytrain, **compute_pars.get("compute_pars", {}) )

        

def fit_metrics(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics of the model when fitted.
    """
    global model, session
    data_pars['train'] = True
    Xval, yval = get_dataset(data_pars, task_type="eval")
    ypred      = model.model.predict(Xval)
    
    # log(data_pars)
    mpars = compute_pars.get("metrics_pars", {})
    
    scorer = {
              "rmse" : sklearn.metrics.mean_squared_error,
              "mae"  : sklearn.metrics.mean_absolute_error      
            }[ mpars['metric_name']]

    mpars2    = mpars.get("metrics_pars", {})  ##Specific to score
    score_val = scorer(yval, ypred,  **mpars2 )

    ddict = {"score_val":score_val}

    return ddict


def predict(data_pars=None, compute_pars=None, out_pars=None, **kw):
    global model, session
    data_pars['train'] = False
    Xpred = get_dataset(data_pars, task_type="predict")
    ypred = model.model.predict(Xpred)
    return ypred


def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)
    
    filename = "model.pkl"
    pickle.dump(model, open( f"{path}/{filename}" , mode='wb')) #, protocol=pickle.HIGHEST_PROTOCOL )
    
    filename = "info.pkl"
    pickle.dump(info, open( f"{path}/{filename}" , mode='wb')) #,protocol=pickle.HIGHEST_PROTOCOL )   
    
    

def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open( f"{path}/model.pkl", mode='rb') )
 
    model = Model() # Empty model    
    model.model        = model0.model
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars    
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl") :   
      if not "model.pkl" in fp :  
        obj = pickle.load(open( fp, mode='rb') )
        key = fp.split("/")[-1]
        dd[key] = obj
    return dd


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import  make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain,  ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'] )
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest =  train_test_split(dfX.values, dfy.values)
        return Xtrain,  ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'] )
        dfX = df[prepro_pars['colX']]
        
        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest



####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  : 
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram') 
    if data_type == "ram"  :
        if task_type == "predict"  :
            d = data_pars[task_type]
            return d["X"]
        
        if task_type == "eval"  :
            d = data_pars[task_type]
            return d["X"], d["y"]
        
        if task_type == "train"  :
            d = data_pars[task_type]
            return d["Xtrain"], d["ytrain"],  d["Xtest"], d["ytest"]

    elif data_type == "file"  :   
        raise Exception(f' {data_type} data_type Not implemented ')

        
        
    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



def get_params(param_pars={}, **kw):
    import json
    #from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")





################################################################################################
########## Tests are normalized Do not Change ##################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test
    global model, session

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)

    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    init(model_pars, data_pars, compute_pars)
    fit(data_pars, compute_pars, out_pars)

    log("#### save the trained model  #######################################")

    log("#### Predict   #####################################################")
    ypred = predict(data_pars, compute_pars, out_pars)

    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(data_pars, compute_pars, out_pars)
    log(metrics_val)

    log("#### Plot   ########################################################")

    log("#### Save#####   ###################################################")
    save_pars = {"path" : out_pars['model_path'] }
    save(save_pars)

    log("#### Load   ########################################################")    
    model, session = load_model( save_pars)
    log(model.model)
    # ypred = predict(model2, data_pars, compute_pars, out_pars)
    metrics_val = fit_metrics(data_pars, compute_pars, out_pars)
    log(metrics_val)


if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"

    ### Local fixed params
    # test(pars_choice="test01")




"""
This is the class and function reference of scikit-learn. Please refer to the full user guide for further details, as the class and function raw specifications may not be enough to give full guidelines on their uses. For reference on concepts repeated across the API, see Glossary of Common Terms and API Elements.
sklearn.base: Base classes and utility functions
Base classes for all estimators.
Used for VotingClassifier
Base classes
base.BaseEstimator
base.BiclusterMixin
base.ClassifierMixin
base.ClusterMixin
base.DensityMixin
base.RegressorMixin
base.TransformerMixin
Functions
base.clone(estimator[, safe])
base.is_classifier(estimator)
base.is_regressor(estimator)
config_context(\*\*new_config)
get_config()
set_config([assume_finite, working_memory, …])
show_versions()
sklearn.calibration: Probability Calibration
Calibration of predicted probabilities.
User guide: See the Probability calibration section for further details.
calibration.CalibratedClassifierCV([…])
calibration.calibration_curve(y_true, y_prob)
sklearn.cluster: Clustering
The sklearn.cluster module gathers popular unsupervised clustering algorithms.
User guide: See the Clustering and Biclustering sections for further details.
Classes
cluster.AffinityPropagation([damping, …])
cluster.AgglomerativeClustering([…])
cluster.Birch([threshold, branching_factor, …])
cluster.DBSCAN([eps, min_samples, metric, …])
cluster.FeatureAgglomeration([n_clusters, …])
cluster.KMeans([n_clusters, init, n_init, …])
cluster.MiniBatchKMeans([n_clusters, init, …])
cluster.MeanShift([bandwidth, seeds, …])
cluster.OPTICS([min_samples, max_eps, …])
cluster.SpectralClustering([n_clusters, …])
cluster.SpectralBiclustering([n_clusters, …])
cluster.SpectralCoclustering([n_clusters, …])
Functions
cluster.affinity_propagation(S[, …])
cluster.cluster_optics_dbscan(reachability, …)
cluster.cluster_optics_xi(reachability, …)
cluster.compute_optics_graph(X, min_samples, …)
cluster.dbscan(X[, eps, min_samples, …])
cluster.estimate_bandwidth(X[, quantile, …])
cluster.k_means(X, n_clusters[, …])
cluster.mean_shift(X[, bandwidth, seeds, …])
cluster.spectral_clustering(affinity[, …])
cluster.ward_tree(X[, connectivity, …])
sklearn.compose: Composite Estimators
Meta-estimators for building composite models with transformers
In addition to its current contents, this module will eventually be home to refurbished versions of Pipeline and FeatureUnion.
User guide: See the Pipelines and composite estimators section for further details.
compose.ColumnTransformer(transformers[, …])
compose.TransformedTargetRegressor([…])
compose.make_column_transformer(…)
compose.make_column_selector([pattern, …])
sklearn.covariance: Covariance Estimators
The sklearn.covariance module includes methods and algorithms to robustly estimate the covariance of features given a set of points. The precision matrix defined as the inverse of the covariance is also estimated. Covariance estimation is closely related to the theory of Gaussian Graphical Models.
User guide: See the Covariance estimation section for further details.
covariance.EmpiricalCovariance([…])
covariance.EllipticEnvelope([…])
covariance.GraphicalLasso([alpha, mode, …])
covariance.GraphicalLassoCV([alphas, …])
covariance.LedoitWolf([store_precision, …])
covariance.MinCovDet([store_precision, …])
covariance.OAS([store_precision, …])
covariance.ShrunkCovariance([…])
covariance.empirical_covariance(X[, …])
covariance.graphical_lasso(emp_cov, alpha[, …])
covariance.ledoit_wolf(X[, assume_centered, …])
covariance.oas(X[, assume_centered])
covariance.shrunk_covariance(emp_cov[, …])
sklearn.cross_decomposition: Cross decomposition
User guide: See the Cross decomposition section for further details.
cross_decomposition.CCA([n_components, …])
cross_decomposition.PLSCanonical([…])
cross_decomposition.PLSRegression([…])
cross_decomposition.PLSSVD([n_components, …])
sklearn.datasets: Datasets
The sklearn.datasets module includes utilities to load datasets, including methods to load and fetch popular reference datasets. It also features some artificial data generators.
User guide: See the Dataset loading utilities section for further details.
Loaders
datasets.clear_data_home([data_home])
datasets.dump_svmlight_file(X, y, f[, …])
datasets.fetch_20newsgroups([data_home, …])
datasets.fetch_20newsgroups_vectorized([…])
datasets.fetch_california_housing([…])
datasets.fetch_covtype([data_home, …])
datasets.fetch_kddcup99([subset, data_home, …])
datasets.fetch_lfw_pairs([subset, …])
datasets.fetch_lfw_people([data_home, …])
datasets.fetch_olivetti_faces([data_home, …])
datasets.fetch_openml([name, version, …])
datasets.fetch_rcv1([data_home, subset, …])
datasets.fetch_species_distributions([…])
datasets.get_data_home([data_home])
datasets.load_boston([return_X_y])
datasets.load_breast_cancer([return_X_y])
datasets.load_diabetes([return_X_y])
datasets.load_digits([n_class, return_X_y])
datasets.load_files(container_path[, …])
datasets.load_iris([return_X_y])
datasets.load_linnerud([return_X_y])
datasets.load_sample_image(image_name)
datasets.load_sample_images()
datasets.load_svmlight_file(f[, n_features, …])
datasets.load_svmlight_files(files[, …])
datasets.load_wine([return_X_y])
Samples generator
datasets.make_biclusters(shape, n_clusters)
datasets.make_blobs([n_samples, n_features, …])
datasets.make_checkerboard(shape, n_clusters)
datasets.make_circles([n_samples, shuffle, …])
datasets.make_classification([n_samples, …])
datasets.make_friedman1([n_samples, …])
datasets.make_friedman2([n_samples, noise, …])
datasets.make_friedman3([n_samples, noise, …])
datasets.make_gaussian_quantiles([mean, …])
datasets.make_hastie_10_2([n_samples, …])
datasets.make_low_rank_matrix([n_samples, …])
datasets.make_moons([n_samples, shuffle, …])
datasets.make_multilabel_classification([…])
datasets.make_regression([n_samples, …])
datasets.make_s_curve([n_samples, noise, …])
datasets.make_sparse_coded_signal(n_samples, …)
datasets.make_sparse_spd_matrix([dim, …])
datasets.make_sparse_uncorrelated([…])
datasets.make_spd_matrix(n_dim[, random_state])
datasets.make_swiss_roll([n_samples, noise, …])
sklearn.decomposition: Matrix Decomposition
The sklearn.decomposition module includes matrix decomposition algorithms, including among others PCA, NMF or ICA. Most of the algorithms of this module can be regarded as dimensionality reduction techniques.
User guide: See the Decomposing signals in components (matrix factorization problems) section for further details.
decomposition.DictionaryLearning([…])
decomposition.FactorAnalysis([n_components, …])
decomposition.FastICA([n_components, …])
decomposition.IncrementalPCA([n_components, …])
decomposition.KernelPCA([n_components, …])
decomposition.LatentDirichletAllocation([…])
decomposition.MiniBatchDictionaryLearning([…])
decomposition.MiniBatchSparsePCA([…])
decomposition.NMF([n_components, init, …])
decomposition.PCA([n_components, copy, …])
decomposition.SparsePCA([n_components, …])
decomposition.SparseCoder(dictionary[, …])
decomposition.TruncatedSVD([n_components, …])
decomposition.dict_learning(X, n_components, …)
decomposition.dict_learning_online(X[, …])
decomposition.fastica(X[, n_components, …])
decomposition.non_negative_factorization(X)
decomposition.sparse_encode(X, dictionary[, …])
sklearn.discriminant_analysis: Discriminant Analysis
Linear Discriminant Analysis and Quadratic Discriminant Analysis
User guide: See the Linear and Quadratic Discriminant Analysis section for further details.
discriminant_analysis.LinearDiscriminantAnalysis([…])
discriminant_analysis.QuadraticDiscriminantAnalysis([…])
sklearn.dummy: Dummy estimators
User guide: See the Metrics and scoring: quantifying the quality of predictions section for further details.
dummy.DummyClassifier([strategy, …])
dummy.DummyRegressor([strategy, constant, …])
sklearn.ensemble: Ensemble Methods
The sklearn.ensemble module includes ensemble-based methods for classification, regression and anomaly detection.
User guide: See the Ensemble methods section for further details.
ensemble.AdaBoostClassifier([…])
ensemble.AdaBoostRegressor([base_estimator, …])
ensemble.BaggingClassifier([base_estimator, …])
ensemble.BaggingRegressor([base_estimator, …])
ensemble.ExtraTreesClassifier([…])
ensemble.ExtraTreesRegressor([n_estimators, …])
ensemble.GradientBoostingClassifier([loss, …])
ensemble.GradientBoostingRegressor([loss, …])
ensemble.IsolationForest([n_estimators, …])
ensemble.RandomForestClassifier([…])
ensemble.RandomForestRegressor([…])
ensemble.RandomTreesEmbedding([…])
ensemble.StackingClassifier(estimators[, …])
ensemble.StackingRegressor(estimators[, …])
ensemble.VotingClassifier(estimators[, …])
ensemble.VotingRegressor(estimators[, …])
ensemble.HistGradientBoostingRegressor([…])
ensemble.HistGradientBoostingClassifier([…])
sklearn.exceptions: Exceptions and warnings
The sklearn.exceptions module includes all custom warnings and error classes used across scikit-learn.
exceptions.ChangedBehaviorWarning
exceptions.ConvergenceWarning
exceptions.DataConversionWarning
exceptions.DataDimensionalityWarning
exceptions.EfficiencyWarning
exceptions.FitFailedWarning
exceptions.NotFittedError
exceptions.NonBLASDotWarning
exceptions.UndefinedMetricWarning
sklearn.experimental: Experimental
The sklearn.experimental module provides importable modules that enable the use of experimental features or estimators.
The features and estimators that are experimental aren’t subject to deprecation cycles. Use them at your own risks!
experimental.enable_hist_gradient_boosting
experimental.enable_iterative_imputer
sklearn.feature_extraction: Feature Extraction
The sklearn.feature_extraction module deals with feature extraction from raw data. It currently includes methods to extract features from text and images.
User guide: See the Feature extraction section for further details.
feature_extraction.DictVectorizer([dtype, …])
feature_extraction.FeatureHasher([…])
From images
The sklearn.feature_extraction.image submodule gathers utilities to extract features from images.
feature_extraction.image.extract_patches_2d(…)
feature_extraction.image.grid_to_graph(n_x, n_y)
feature_extraction.image.img_to_graph(img[, …])
feature_extraction.image.reconstruct_from_patches_2d(…)
feature_extraction.image.PatchExtractor([…])
From text
The sklearn.feature_extraction.text submodule gathers utilities to build feature vectors from text documents.
feature_extraction.text.CountVectorizer([…])
feature_extraction.text.HashingVectorizer([…])
feature_extraction.text.TfidfTransformer([…])
feature_extraction.text.TfidfVectorizer([…])
sklearn.feature_selection: Feature Selection
The sklearn.feature_selection module implements feature selection algorithms. It currently includes univariate filter selection methods and the recursive feature elimination algorithm.
User guide: See the Feature selection section for further details.
feature_selection.GenericUnivariateSelect([…])
feature_selection.SelectPercentile([…])
feature_selection.SelectKBest([score_func, k])
feature_selection.SelectFpr([score_func, alpha])
feature_selection.SelectFdr([score_func, alpha])
feature_selection.SelectFromModel(estimator)
feature_selection.SelectFwe([score_func, alpha])
feature_selection.RFE(estimator[, …])
feature_selection.RFECV(estimator[, step, …])
feature_selection.VarianceThreshold([threshold])
feature_selection.chi2(X, y)
feature_selection.f_classif(X, y)
feature_selection.f_regression(X, y[, center])
feature_selection.mutual_info_classif(X, y)
feature_selection.mutual_info_regression(X, y)
sklearn.gaussian_process: Gaussian Processes
The sklearn.gaussian_process module implements Gaussian Process based regression and classification.
User guide: See the Gaussian Processes section for further details.
gaussian_process.GaussianProcessClassifier([…])
gaussian_process.GaussianProcessRegressor([…])
Kernels:
gaussian_process.kernels.CompoundKernel(kernels)
gaussian_process.kernels.ConstantKernel([…])
gaussian_process.kernels.DotProduct([…])
gaussian_process.kernels.ExpSineSquared([…])
gaussian_process.kernels.Exponentiation(…)
gaussian_process.kernels.Hyperparameter
gaussian_process.kernels.Kernel
gaussian_process.kernels.Matern([…])
gaussian_process.kernels.PairwiseKernel([…])
gaussian_process.kernels.Product(k1, k2)
gaussian_process.kernels.RBF([length_scale, …])
gaussian_process.kernels.RationalQuadratic([…])
gaussian_process.kernels.Sum(k1, k2)
gaussian_process.kernels.WhiteKernel([…])
sklearn.impute: Impute
Transformers for missing value imputation
User guide: See the Imputation of missing values section for further details.
impute.SimpleImputer([missing_values, …])
impute.IterativeImputer([estimator, …])
impute.MissingIndicator([missing_values, …])
impute.KNNImputer([missing_values, …])
sklearn.inspection: inspection
The sklearn.inspection module includes tools for model inspection.
inspection.partial_dependence(estimator, X, …)
inspection.permutation_importance(estimator, …)
Plotting
inspection.PartialDependenceDisplay(…)
inspection.plot_partial_dependence(…[, …])
sklearn.isotonic: Isotonic regression
User guide: See the Isotonic regression section for further details.
isotonic.IsotonicRegression([y_min, y_max, …])
isotonic.check_increasing(x, y)
isotonic.isotonic_regression(y[, …])
sklearn.kernel_approximation Kernel Approximation
The sklearn.kernel_approximation module implements several approximate kernel feature maps base on Fourier transforms.
User guide: See the Kernel Approximation section for further details.
kernel_approximation.AdditiveChi2Sampler([…])
kernel_approximation.Nystroem([kernel, …])
kernel_approximation.RBFSampler([gamma, …])
kernel_approximation.SkewedChi2Sampler([…])
sklearn.kernel_ridge Kernel Ridge Regression
Module sklearn.kernel_ridge implements kernel ridge regression.
User guide: See the Kernel ridge regression section for further details.
kernel_ridge.KernelRidge([alpha, kernel, …])
sklearn.linear_model: Linear Models
The sklearn.linear_model module implements a variety of linear models.
User guide: See the Linear Models section for further details.
The following subsections are only rough guidelines: the same estimator can fall into multiple categories, depending on its parameters.
Linear classifiers
linear_model.LogisticRegression([penalty, …])
linear_model.LogisticRegressionCV([Cs, …])
linear_model.PassiveAggressiveClassifier([…])
linear_model.Perceptron([penalty, alpha, …])
linear_model.RidgeClassifier([alpha, …])
linear_model.RidgeClassifierCV([alphas, …])
linear_model.SGDClassifier([loss, penalty, …])
Classical linear regressors
linear_model.LinearRegression([…])
linear_model.Ridge([alpha, fit_intercept, …])
linear_model.RidgeCV([alphas, …])
linear_model.SGDRegressor([loss, penalty, …])
Regressors with variable selection
The following estimators have built-in variable selection fitting procedures, but any estimator using a L1 or elastic-net penalty also performs variable selection: typically SGDRegressor or SGDClassifier with an appropriate penalty.
linear_model.ElasticNet([alpha, l1_ratio, …])
linear_model.ElasticNetCV([l1_ratio, eps, …])
linear_model.Lars([fit_intercept, verbose, …])
linear_model.LarsCV([fit_intercept, …])
linear_model.Lasso([alpha, fit_intercept, …])
linear_model.LassoCV([eps, n_alphas, …])
linear_model.LassoLars([alpha, …])
linear_model.LassoLarsCV([fit_intercept, …])
linear_model.LassoLarsIC([criterion, …])
linear_model.OrthogonalMatchingPursuit([…])
linear_model.OrthogonalMatchingPursuitCV([…])
Bayesian regressors
linear_model.ARDRegression([n_iter, tol, …])
linear_model.BayesianRidge([n_iter, tol, …])
Multi-task linear regressors with variable selection
These estimators fit multiple regression problems (or tasks) jointly, while inducing sparse coefficients. While the inferred coefficients may differ between the tasks, they are constrained to agree on the features that are selected (non-zero coefficients).
linear_model.MultiTaskElasticNet([alpha, …])
linear_model.MultiTaskElasticNetCV([…])
linear_model.MultiTaskLasso([alpha, …])
linear_model.MultiTaskLassoCV([eps, …])
Outlier-robust regressors
Any estimator using the Huber loss would also be robust to outliers, e.g. SGDRegressor with loss='huber'.
linear_model.HuberRegressor([epsilon, …])
linear_model.RANSACRegressor([…])
linear_model.TheilSenRegressor([…])
Miscellaneous
linear_model.PassiveAggressiveRegressor([C, …])
linear_model.enet_path(X, y[, l1_ratio, …])
linear_model.lars_path(X, y[, Xy, Gram, …])
linear_model.lars_path_gram(Xy, Gram, n_samples)
linear_model.lasso_path(X, y[, eps, …])
linear_model.orthogonal_mp(X, y[, …])
linear_model.orthogonal_mp_gram(Gram, Xy[, …])
linear_model.ridge_regression(X, y, alpha[, …])
sklearn.manifold: Manifold Learning
The sklearn.manifold module implements data embedding techniques.
User guide: See the Manifold learning section for further details.
manifold.Isomap([n_neighbors, n_components, …])
manifold.LocallyLinearEmbedding([…])
manifold.MDS([n_components, metric, n_init, …])
manifold.SpectralEmbedding([n_components, …])
manifold.TSNE([n_components, perplexity, …])
manifold.locally_linear_embedding(X, …[, …])
manifold.smacof(dissimilarities[, metric, …])
manifold.spectral_embedding(adjacency[, …])
manifold.trustworthiness(X, X_embedded[, …])
sklearn.metrics: Metrics
See the Metrics and scoring: quantifying the quality of predictions section and the Pairwise metrics, Affinities and Kernels section of the user guide for further details.
The sklearn.metrics module includes score functions, performance metrics and pairwise metrics and distance computations.
Model Selection Interface
See the The scoring parameter: defining model evaluation rules section of the user guide for further details.
metrics.check_scoring(estimator[, scoring, …])
metrics.get_scorer(scoring)
metrics.make_scorer(score_func[, …])
Classification metrics
See the Classification metrics section of the user guide for further details.
metrics.accuracy_score(y_true, y_pred[, …])
metrics.auc(x, y)
metrics.average_precision_score(y_true, y_score)
metrics.balanced_accuracy_score(y_true, y_pred)
metrics.brier_score_loss(y_true, y_prob[, …])
metrics.classification_report(y_true, y_pred)
metrics.cohen_kappa_score(y1, y2[, labels, …])
metrics.confusion_matrix(y_true, y_pred[, …])
metrics.dcg_score(y_true, y_score[, k, …])
metrics.f1_score(y_true, y_pred[, labels, …])
metrics.fbeta_score(y_true, y_pred, beta[, …])
metrics.hamming_loss(y_true, y_pred[, …])
metrics.hinge_loss(y_true, pred_decision[, …])
metrics.jaccard_score(y_true, y_pred[, …])
metrics.log_loss(y_true, y_pred[, eps, …])
metrics.matthews_corrcoef(y_true, y_pred[, …])
metrics.multilabel_confusion_matrix(y_true, …)
metrics.ndcg_score(y_true, y_score[, k, …])
metrics.precision_recall_curve(y_true, …)
metrics.precision_recall_fscore_support(…)
metrics.precision_score(y_true, y_pred[, …])
metrics.recall_score(y_true, y_pred[, …])
metrics.roc_auc_score(y_true, y_score[, …])
metrics.roc_curve(y_true, y_score[, …])
metrics.zero_one_loss(y_true, y_pred[, …])
Regression metrics
See the Regression metrics section of the user guide for further details.
metrics.explained_variance_score(y_true, y_pred)
metrics.max_error(y_true, y_pred)
metrics.mean_absolute_error(y_true, y_pred)
metrics.mean_squared_error(y_true, y_pred[, …])
metrics.mean_squared_log_error(y_true, y_pred)
metrics.median_absolute_error(y_true, y_pred)
metrics.r2_score(y_true, y_pred[, …])
metrics.mean_poisson_deviance(y_true, y_pred)
metrics.mean_gamma_deviance(y_true, y_pred)
metrics.mean_tweedie_deviance(y_true, y_pred)
Multilabel ranking metrics
See the Multilabel ranking metrics section of the user guide for further details.
metrics.coverage_error(y_true, y_score[, …])
metrics.label_ranking_average_precision_score(…)
metrics.label_ranking_loss(y_true, y_score)
Clustering metrics
See the Clustering performance evaluation section of the user guide for further details.
The sklearn.metrics.cluster submodule contains evaluation metrics for cluster analysis results. There are two forms of evaluation:
supervised, which uses a ground truth class values for each sample.
unsupervised, which does not and measures the ‘quality’ of the model itself.
metrics.adjusted_mutual_info_score(…[, …])
metrics.adjusted_rand_score(labels_true, …)
metrics.calinski_harabasz_score(X, labels)
metrics.davies_bouldin_score(X, labels)
metrics.completeness_score(labels_true, …)
metrics.cluster.contingency_matrix(…[, …])
metrics.fowlkes_mallows_score(labels_true, …)
metrics.homogeneity_completeness_v_measure(…)
metrics.homogeneity_score(labels_true, …)
metrics.mutual_info_score(labels_true, …)
metrics.normalized_mutual_info_score(…[, …])
metrics.silhouette_score(X, labels[, …])
metrics.silhouette_samples(X, labels[, metric])
metrics.v_measure_score(labels_true, labels_pred)
Biclustering metrics
See the Biclustering evaluation section of the user guide for further details.
metrics.consensus_score(a, b[, similarity])
Pairwise metrics
See the Pairwise metrics, Affinities and Kernels section of the user guide for further details.
metrics.pairwise.additive_chi2_kernel(X[, Y])
metrics.pairwise.chi2_kernel(X[, Y, gamma])
metrics.pairwise.cosine_similarity(X[, Y, …])
metrics.pairwise.cosine_distances(X[, Y])
metrics.pairwise.distance_metrics()
metrics.pairwise.euclidean_distances(X[, Y, …])
metrics.pairwise.haversine_distances(X[, Y])
metrics.pairwise.kernel_metrics()
metrics.pairwise.laplacian_kernel(X[, Y, gamma])
metrics.pairwise.linear_kernel(X[, Y, …])
metrics.pairwise.manhattan_distances(X[, Y, …])
metrics.pairwise.nan_euclidean_distances(X)
metrics.pairwise.pairwise_kernels(X[, Y, …])
metrics.pairwise.polynomial_kernel(X[, Y, …])
metrics.pairwise.rbf_kernel(X[, Y, gamma])
metrics.pairwise.sigmoid_kernel(X[, Y, …])
metrics.pairwise.paired_euclidean_distances(X, Y)
metrics.pairwise.paired_manhattan_distances(X, Y)
metrics.pairwise.paired_cosine_distances(X, Y)
metrics.pairwise.paired_distances(X, Y[, metric])
metrics.pairwise_distances(X[, Y, metric, …])
metrics.pairwise_distances_argmin(X, Y[, …])
metrics.pairwise_distances_argmin_min(X, Y)
metrics.pairwise_distances_chunked(X[, Y, …])
Plotting
See the Visualizations section of the user guide for further details.
metrics.plot_confusion_matrix(estimator, X, …)
metrics.plot_precision_recall_curve(…[, …])
metrics.plot_roc_curve(estimator, X, y[, …])
metrics.ConfusionMatrixDisplay(…)
metrics.PrecisionRecallDisplay(precision, …)
metrics.RocCurveDisplay(fpr, tpr, roc_auc, …)
sklearn.mixture: Gaussian Mixture Models
The sklearn.mixture module implements mixture modeling algorithms.
User guide: See the Gaussian mixture models section for further details.
mixture.BayesianGaussianMixture([…])
mixture.GaussianMixture([n_components, …])
sklearn.model_selection: Model Selection
User guide: See the Cross-validation: evaluating estimator performance, Tuning the hyper-parameters of an estimator and Learning curve sections for further details.
Splitter Classes
model_selection.GroupKFold([n_splits])
model_selection.GroupShuffleSplit([…])
model_selection.KFold([n_splits, shuffle, …])
model_selection.LeaveOneGroupOut
model_selection.LeavePGroupsOut(n_groups)
model_selection.LeaveOneOut
model_selection.LeavePOut(p)
model_selection.PredefinedSplit(test_fold)
model_selection.RepeatedKFold([n_splits, …])
model_selection.RepeatedStratifiedKFold([…])
model_selection.ShuffleSplit([n_splits, …])
model_selection.StratifiedKFold([n_splits, …])
model_selection.StratifiedShuffleSplit([…])
model_selection.TimeSeriesSplit([n_splits, …])
Splitter Functions
model_selection.check_cv([cv, y, classifier])
model_selection.train_test_split(\*arrays, …)
Hyper-parameter optimizers
model_selection.GridSearchCV(estimator, …)
model_selection.ParameterGrid(param_grid)
model_selection.ParameterSampler(…[, …])
model_selection.RandomizedSearchCV(…[, …])
model_selection.fit_grid_point(X, y, …[, …])
Model validation
model_selection.cross_validate(estimator, X)
model_selection.cross_val_predict(estimator, X)
model_selection.cross_val_score(estimator, X)
model_selection.learning_curve(estimator, X, y)
model_selection.permutation_test_score(…)
model_selection.validation_curve(estimator, …)
sklearn.multiclass: Multiclass and multilabel classification
Multiclass and multilabel classification strategies
This module implements multiclass learning algorithms:
one-vs-the-rest / one-vs-all
one-vs-one
error correcting output codes
The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. For example, it is possible to use these estimators to turn a binary classifier or a regressor into a multiclass classifier. It is also possible to use these estimators with multiclass estimators in the hope that their accuracy or runtime performance improves.
All classifiers in scikit-learn implement multiclass classification; you only need to use this module if you want to experiment with custom multiclass strategies.
The one-vs-the-rest meta-classifier also implements a predict_proba method, so long as such a method is implemented by the base classifier. This method returns probabilities of class membership in both the single label and multilabel case. Note that in the multilabel case, probabilities are the marginal probability that a given sample falls in the given class. As such, in the multilabel case the sum of these probabilities over all possible labels for a given sample will not sum to unity, as they do in the single label case.
User guide: See the Multiclass and multilabel algorithms section for further details.
multiclass.OneVsRestClassifier(estimator[, …])
multiclass.OneVsOneClassifier(estimator[, …])
multiclass.OutputCodeClassifier(estimator[, …])
sklearn.multioutput: Multioutput regression and classification
This module implements multioutput regression and classification.
The estimators provided in this module are meta-estimators: they require a base estimator to be provided in their constructor. The meta-estimator extends single output estimators to multioutput estimators.
User guide: See the Multiclass and multilabel algorithms section for further details.
multioutput.ClassifierChain(base_estimator)
multioutput.MultiOutputRegressor(estimator)
multioutput.MultiOutputClassifier(estimator)
multioutput.RegressorChain(base_estimator[, …])
sklearn.naive_bayes: Naive Bayes
The sklearn.naive_bayes module implements Naive Bayes algorithms. These are supervised learning methods based on applying Bayes’ theorem with strong (naive) feature independence assumptions.
User guide: See the Naive Bayes section for further details.
naive_bayes.BernoulliNB([alpha, binarize, …])
naive_bayes.CategoricalNB([alpha, …])
naive_bayes.ComplementNB([alpha, fit_prior, …])
naive_bayes.GaussianNB([priors, var_smoothing])
naive_bayes.MultinomialNB([alpha, …])
sklearn.neighbors: Nearest Neighbors
The sklearn.neighbors module implements the k-nearest neighbors algorithm.
User guide: See the Nearest Neighbors section for further details.
neighbors.BallTree
neighbors.DistanceMetric
neighbors.KDTree
neighbors.KernelDensity([bandwidth, …])
neighbors.KNeighborsClassifier([…])
neighbors.KNeighborsRegressor([n_neighbors, …])
neighbors.KNeighborsTransformer([mode, …])
neighbors.LocalOutlierFactor([n_neighbors, …])
neighbors.RadiusNeighborsClassifier([…])
neighbors.RadiusNeighborsRegressor([radius, …])
neighbors.RadiusNeighborsTransformer([mode, …])
neighbors.NearestCentroid([metric, …])
neighbors.NearestNeighbors([n_neighbors, …])
neighbors.NeighborhoodComponentsAnalysis([…])
neighbors.kneighbors_graph(X, n_neighbors[, …])
neighbors.radius_neighbors_graph(X, radius)
sklearn.neural_network: Neural network models
The sklearn.neural_network module includes models based on neural networks.
User guide: See the Neural network models (supervised) and Neural network models (unsupervised) sections for further details.
neural_network.BernoulliRBM([n_components, …])
neural_network.MLPClassifier([…])
neural_network.MLPRegressor([…])
sklearn.pipeline: Pipeline
The sklearn.pipeline module implements utilities to build a composite estimator, as a chain of transforms and estimators.
pipeline.FeatureUnion(transformer_list[, …])
pipeline.Pipeline(steps[, memory, verbose])
pipeline.make_pipeline(\*steps, \*\*kwargs)
pipeline.make_union(\*transformers, \*\*kwargs)
sklearn.preprocessing: Preprocessing and Normalization
The sklearn.preprocessing module includes scaling, centering, normalization, binarization methods.
User guide: See the Preprocessing data section for further details.
preprocessing.Binarizer([threshold, copy])
preprocessing.FunctionTransformer([func, …])
preprocessing.KBinsDiscretizer([n_bins, …])
preprocessing.KernelCenterer()
preprocessing.LabelBinarizer([neg_label, …])
preprocessing.LabelEncoder
preprocessing.MultiLabelBinarizer([classes, …])
preprocessing.MaxAbsScaler([copy])
preprocessing.MinMaxScaler([feature_range, copy])
preprocessing.Normalizer([norm, copy])
preprocessing.OneHotEncoder([categories, …])
preprocessing.OrdinalEncoder([categories, dtype])
preprocessing.PolynomialFeatures([degree, …])
preprocessing.PowerTransformer([method, …])
preprocessing.QuantileTransformer([…])
preprocessing.RobustScaler([with_centering, …])
preprocessing.StandardScaler([copy, …])
preprocessing.add_dummy_feature(X[, value])
preprocessing.binarize(X[, threshold, copy])
preprocessing.label_binarize(y, classes[, …])
preprocessing.maxabs_scale(X[, axis, copy])
preprocessing.minmax_scale(X[, …])
preprocessing.normalize(X[, norm, axis, …])
preprocessing.quantile_transform(X[, axis, …])
preprocessing.robust_scale(X[, axis, …])
preprocessing.scale(X[, axis, with_mean, …])
preprocessing.power_transform(X[, method, …])
sklearn.random_projection: Random projection
Random Projection transformers
Random Projections are a simple and computationally efficient way to reduce the dimensionality of the data by trading a controlled amount of accuracy (as additional variance) for faster processing times and smaller model sizes.
The dimensions and distribution of Random Projections matrices are controlled so as to preserve the pairwise distances between any two samples of the dataset.
The main theoretical result behind the efficiency of random projection is the Johnson-Lindenstrauss lemma (quoting Wikipedia):
In mathematics, the Johnson-Lindenstrauss lemma is a result concerning low-distortion embeddings of points from high-dimensional into low-dimensional Euclidean space. The lemma states that a small set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved. The map used for the embedding is at least Lipschitz, and can even be taken to be an orthogonal projection.
User guide: See the Random Projection section for further details.
random_projection.GaussianRandomProjection([…])
random_projection.SparseRandomProjection([…])
random_projection.johnson_lindenstrauss_min_dim(…)
sklearn.semi_supervised Semi-Supervised Learning
The sklearn.semi_supervised module implements semi-supervised learning algorithms. These algorithms utilized small amounts of labeled data and large amounts of unlabeled data for classification tasks. This module includes Label Propagation.
User guide: See the Semi-Supervised section for further details.
semi_supervised.LabelPropagation([kernel, …])
semi_supervised.LabelSpreading([kernel, …])
sklearn.svm: Support Vector Machines
The sklearn.svm module includes Support Vector Machine algorithms.
User guide: See the Support Vector Machines section for further details.
Estimators
svm.LinearSVC([penalty, loss, dual, tol, C, …])
svm.LinearSVR([epsilon, tol, C, loss, …])
svm.NuSVC([nu, kernel, degree, gamma, …])
svm.NuSVR([nu, C, kernel, degree, gamma, …])
svm.OneClassSVM([kernel, degree, gamma, …])
svm.SVC([C, kernel, degree, gamma, coef0, …])
svm.SVR([kernel, degree, gamma, coef0, tol, …])
svm.l1_min_c(X, y[, loss, fit_intercept, …])
sklearn.tree: Decision Trees
The sklearn.tree module includes decision tree-based models for classification and regression.
User guide: See the Decision Trees section for further details.
tree.DecisionTreeClassifier([criterion, …])
tree.DecisionTreeRegressor([criterion, …])
tree.ExtraTreeClassifier([criterion, …])
tree.ExtraTreeRegressor([criterion, …])
tree.export_graphviz(decision_tree[, …])
tree.export_text(decision_tree[, …])
Plotting
tree.plot_tree(decision_tree[, max_depth, …])
sklearn.utils: Utilities
The sklearn.utils module includes various utilities.
Developer guide: See the Utilities for Developers page for further details.
utils.arrayfuncs.min_pos()
utils.as_float_array(X[, copy, force_all_finite])
utils.assert_all_finite(X[, allow_nan])
utils.check_X_y(X, y[, accept_sparse, …])
utils.check_array(array[, accept_sparse, …])
utils.check_scalar(x, name, target_type[, …])
utils.check_consistent_length(\*arrays)
utils.check_random_state(seed)
utils.class_weight.compute_class_weight(…)
utils.class_weight.compute_sample_weight(…)
utils.deprecated([extra])
utils.estimator_checks.check_estimator(Estimator)
utils.estimator_checks.parametrize_with_checks(…)
utils.extmath.safe_sparse_dot(a, b[, …])
utils.extmath.randomized_range_finder(A, …)
utils.extmath.randomized_svd(M, n_components)
utils.extmath.fast_logdet(A)
utils.extmath.density(w, \*\*kwargs)
utils.extmath.weighted_mode(a, w[, axis])
utils.gen_even_slices(n, n_packs[, n_samples])
utils.graph.single_source_shortest_path_length(…)
utils.graph_shortest_path.graph_shortest_path()
utils.indexable(\*iterables)
utils.metaestimators.if_delegate_has_method(…)
utils.multiclass.type_of_target(y)
utils.multiclass.is_multilabel(y)
utils.multiclass.unique_labels(\*ys)
utils.murmurhash3_32()
utils.resample(\*arrays, \*\*options)
utils._safe_indexing(X, indices[, axis])
utils.safe_mask(X, mask)
utils.safe_sqr(X[, copy])
utils.shuffle(\*arrays, \*\*options)
utils.sparsefuncs.incr_mean_variance_axis(X, …)
utils.sparsefuncs.inplace_column_scale(X, scale)
utils.sparsefuncs.inplace_row_scale(X, scale)
utils.sparsefuncs.inplace_swap_row(X, m, n)
utils.sparsefuncs.inplace_swap_column(X, m, n)
utils.sparsefuncs.mean_variance_axis(X, axis)
utils.sparsefuncs.inplace_csr_column_scale(X, …)
utils.sparsefuncs_fast.inplace_csr_row_normalize_l1()
utils.sparsefuncs_fast.inplace_csr_row_normalize_l2()
utils.random.sample_without_replacement()
utils.validation.check_is_fitted(estimator)
utils.validation.check_memory(memory)
utils.validation.check_symmetric(array[, …])
utils.validation.column_or_1d(y[, warn])
utils.validation.has_fit_parameter(…)
utils.all_estimators([…])
Utilities from joblib:
utils.parallel_backend(backend[, n_jobs, …])
utils.register_parallel_backend(name, factory)
Recently deprecated
To be removed in 0.23
utils.Memory(**kwargs)
utils.Parallel(**kwargs)
utils.cpu_count()
utils.delayed(function[, check_pickle])
metrics.calinski_harabaz_score(X, labels)
metrics.jaccard_similarity_score(y_true, y_pred)
linear_model.logistic_regression_path(X, y)
utils.safe_indexing(X, indices[, axis])
ensemble.partial_dependence.partial_dependence(…)
ensemble.partial_dependence.plot_partial_dependence(…)
"""
