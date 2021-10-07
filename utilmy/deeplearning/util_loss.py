# -*- coding: utf-8 -*-
import io, os, numpy as np
from typing import Union



###################################################################################################
###################################################################################################
from importall import *
from sklearn.metrics import accuracy_score


def log(*s):
    print(*s, flush=True)

    
    
def metric_accuracy(y_test, y_pred, dd):
   test_accuracy = {} 
   for k,(ytruei, ypredi) in enumerate(zip(y_test, y_pred)) : 
       ytruei = np.argmax(ytruei,         axis=-1)
       ypredi = np.argmax(ypredi.numpy(), axis=-1)
       # log(ytruei, ypredi ) 
       test_accuracy[ dd.labels_col[k] ] = accuracy_score(ytruei, ypredi )
        
   log('accuracy', test_accuracy)     
   return test_accuracy 
    


def clf_loss_macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y     = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp    = tf.reduce_sum(y_hat * y, axis=0)
    fp    = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn    = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost    = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost
    




