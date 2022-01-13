# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from jsoncomment import JsonComment ; json = JsonComment()


import sklearn



def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)





####################################################################################################
def metrics_eval(metric_list=["mean_squared_error"], ytrue=None, ypred=None, ypred_proba=None, return_dict=0):
    """
        ytrue = np.random.randint(0,2, 10)
        ypred = np.random.randint(0,5, 10)
        ypred_proba = np.random.uniform(0, 1, 10)
        metrics_eval(["roc_auc_score", "accuracy_score"], ytrue, ypred, ypred_proba)


        ##### Classification metrics
        accuracy_score(y_true, y_pred,...)  Accuracy classification score.
        auc(x, y)   Compute Area Under the Curve (AUC) using the trapezoidal rule
        average_precision_score(y_true, y_score)    Compute average precision (AP) from prediction scores
        balanced_accuracy_score(y_true, y_pred) Compute the balanced accuracy
        brier_score_loss(y_true, y_prob,...)    Compute the Brier score.
        classification_report(y_true, y_pred)   Build a text report showing the main classification metrics
        cohen_kappa_score(y1, y2,...)   Cohenâ€™s kappa: a statistic that measures inter-annotator agreement.
        confusion_matrix(y_true, y_pred,...)    Compute confusion matrix to evaluate the accuracy of a classification.
        dcg_score(y_true, y_score , k, ...  )   Compute Discounted Cumulative Gain.
        f1_score(y_true, y_pred,...)    Compute the F1 score, also known as balanced F-score or F-measure
        fbeta_score(y_true, y_pred, beta,...)   Compute the F-beta score
        hamming_loss(y_true, y_pred,...)    Compute the average Hamming loss.
        hinge_loss(y_true, pred_decision,...)   Average hinge loss (non-regularized)
        jaccard_score(y_true, y_pred,...)   Jaccard similarity coefficient score
        log_loss(y_true, y_pred , eps, ...  )   Log loss, aka logistic loss or cross-entropy loss.
        matthews_corrcoef(y_true, y_pred,...)   Compute the Matthews correlation coefficient (MCC)
        multilabel_confusion_matrix(y_true, ...)    Compute a confusion matrix for each class or sample
        ndcg_score(y_true, y_score , k, ...  )  Compute Normalized Discounted Cumulative Gain.
        precision_recall_curve(y_true, ...) Compute precision-recall pairs for different probability thresholds
        precision_recall_fscore_support(...)    Compute precision, recall, F-measure and support for each class
        precision_score(y_true, y_pred,...) Compute the precision
        recall_score(y_true, y_pred,...)    Compute the recall
        roc_auc_score(y_true, y_score,...)  Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        roc_curve(y_true, y_score,...)  Compute Receiver operating characteristic (ROC)
        zero_one_loss(y_true, y_pred,...)   Zero-one classification loss.



        ##### Regression metrics
        explained_variance_score(y_true, y_pred)    Explained variance regression score function
        max_error(y_true, y_pred)   max_error metric calculates the maximum residual error.
        mean_absolute_error(y_true, y_pred) Mean absolute error regression loss
        mean_squared_error(y_true, y_pred,...)  Mean squared error regression loss
        mean_squared_log_error(y_true, y_pred)  Mean squared logarithmic error regression loss
        median_absolute_error(y_true, y_pred)   Median absolute error regression loss
        r2_score(y_true, y_pred,...)    R^2 (coefficient of determination) regression score function.
        mean_poisson_deviance(y_true, y_pred)   Mean Poisson deviance regression loss.
        mean_gamma_deviance(y_true, y_pred) Mean Gamma deviance regression loss.
        mean_tweedie_deviance(y_true, y_pred)   Mean Tweedie deviance regression loss.



        ##### Multilabel ranking metrics
        coverage_error(y_true, y_score,...) Coverage error measure
        label_ranking_average_precision_score(...)  Compute ranking-based average precision
        label_ranking_loss(y_true, y_score) Compute Ranking loss measure



        ##### Clustering metrics
        supervised, which uses a ground truth class values for each sample.
        unsupervised, which does not and measures the â€˜qualityâ€™ of the model itself.

        adjusted_mutual_info_score(...,...) Adjusted Mutual Information between two clusterings.
        adjusted_rand_score(labels_true, ...)   Rand index adjusted for chance.
        calinski_harabasz_score(X, labels)  Compute the Calinski and Harabasz score.
        davies_bouldin_score(X, labels) Computes the Davies-Bouldin score.
        completeness_score(labels_true, ...)    Completeness metric of a cluster labeling given a ground truth.
        cluster.contingency_matrix(...,...) Build a contingency matrix describing the relationship between labels.
        fowlkes_mallows_score(labels_true, ...) Measure the similarity of two clusterings of a set of points.
        homogeneity_completeness_v_measure(...) Compute the homogeneity and completeness and V-Measure scores at once.
        homogeneity_score(labels_true, ...) Homogeneity metric of a cluster labeling given a ground truth.
        mutual_info_score(labels_true, ...) Mutual Information between two clusterings.
        normalized_mutual_info_score(...,...)   Normalized Mutual Information between two clusterings.
        silhouette_score(X, labels,...) Compute the mean Silhouette Coefficient of all samples.
        silhouette_samples(X, labels , metric  )    Compute the Silhouette Coefficient for each sample.
        v_measure_score(labels_true, labels_pred)   V-measure cluster labeling given a ground truth.


        ##### Biclustering metrics
        consensus_score(a, b , similarity  )    The similarity of two sets of biclusters.

        ##### Pairwise metrics
        pairwise.additive_chi2_kernel(X , Y  )  Computes the additive chi-squared kernel between observations in X and Y
        pairwise.chi2_kernel(X , Y, gamma  )    Computes the exponential chi-squared kernel X and Y.
        pairwise.cosine_similarity(X , Y, ...  )    Compute cosine similarity between samples in X and Y.
        pairwise.cosine_distances(X , Y  )  Compute cosine distance between samples in X and Y.
        pairwise.distance_metrics() Valid metrics for pairwise_distances.
        pairwise.euclidean_distances(X , Y, ...  )  Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors.
        pairwise.haversine_distances(X , Y  )   Compute the Haversine distance between samples in X and Y
        pairwise.kernel_metrics()   Valid metrics for pairwise_kernels
        pairwise.laplacian_kernel(X , Y, gamma  )   Compute the laplacian kernel between X and Y.
        pairwise.linear_kernel(X , Y, ...  )    Compute the linear kernel between X and Y.
        pairwise.manhattan_distances(X , Y, ...  )  Compute the L1 distances between the vectors in X and Y.
        pairwise.nan_euclidean_distances(X) Calculate the euclidean distances in the presence of missing values.
        pairwise.pairwise_kernels(X , Y, ...  ) Compute the kernel between arrays X and optional array Y.
        pairwise.polynomial_kernel(X , Y, ...  )    Compute the polynomial kernel between X and Y.
        pairwise.rbf_kernel(X , Y, gamma  ) Compute the rbf (gaussian) kernel between X and Y.
        pairwise.sigmoid_kernel(X , Y, ...  )   Compute the sigmoid kernel between X and Y.
        pairwise.paired_euclidean_distances(X, Y)   Computes the paired euclidean distances between X and Y
        pairwise.paired_manhattan_distances(X, Y)   Compute the L1 distances between the vectors in X and Y.
        pairwise.paired_cosine_distances(X, Y)  Computes the paired cosine distances between X and Y
        pairwise.paired_distances(X, Y , metric  )  Computes the paired distances between X and Y.
        pairwise_distances(X , Y, metric, ...  )    Compute the distance matrix from a vector array X and optional Y.
        pairwise_distances_argmin(X, Y,...) Compute minimum distances between one point and a set of points.
        pairwise_distances_argmin_min(X, Y) Compute minimum distances between one point and a set of points.
        pairwise_distances_chunked(X , Y, ...  )    Generate a distance matrix chunk by chunk with optional reduction

    """
    import pandas as pd, importlib
    mdict = {  "metric_name": [],
               "metric_val":  [],
               "n_sample" :   [len(ytrue)] * len(metric_list)}

    if isinstance(metric_list, str) :
        metric_list = [metric_list]

    for metric_name in metric_list :
      mod = "sklearn.metrics"  
      metric_scorer = getattr(importlib.import_module(mod), metric_name)
    
      if metric_name in [ "roc_auc_score" ]  :
         mval = metric_scorer( ytrue, ypred_proba)
      else :
         mval = metric_scorer( ytrue, ypred)

      mdict["metric_name"].append(metric_name)
      mdict["metric_val"].append(mval)

    if return_dict : return mdict

    mdict = pd.DataFrame(mdict)
    return mdict




def test():
	import numpy as np
	ytrue = np.random.randint(0,5, 10)
	ypred = np.random.randint(0,5, 10)
	ypred_proba = None
	log( metrics_eval(["mean_squared_error", "mean_absolute_error"], ytrue, ypred, ypred_proba) )


	ytrue = np.random.randint(0,2, 10)
	ypred = np.random.randint(0,5, 10)
	ypred_proba = np.random.uniform(0, 1, 10)
	log( metrics_eval(["roc_auc_score", "accuracy_score"], ytrue, ypred, ypred_proba) )




if __name__ == "__main__":
    test()



