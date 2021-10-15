# -*- coding: utf-8 -*-
import os,io, numpy as np, sys, glob, time, copy, json, pandas as pd, functools, sys
from typing import Union
from sklearn.metrics import accuracy_score
from box import Box
import tensorflow as tf
import tensorflow_addons as tfa

###################################################################################
def log(*s):
    print(*s, flush=True)



cc = Box({})
dd = Box({})


###################################################################################
def metric_accuracy(y_test, y_pred, dd):
   test_accuracy = {} 
   for k,(ytruei, ypredi) in enumerate(zip(y_test, y_pred)) : 
       ytruei = np.argmax(ytruei,         axis=-1)
       ypredi = np.argmax(ypredi.numpy(), axis=-1)
       # log(ytruei, ypredi ) 
       test_accuracy[ dd.labels_col[k] ] = accuracy_score(ytruei, ypredi )
        
   log('accuracy', test_accuracy)     
   return test_accuracy 
    


def clf_loss_macro_soft_f1(y, y_hat):   #name
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
    


class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule", path=None):
        # compute the set of learning rates for each corresponding
        # epoch
        pass


#####  Learning Rate Schedule   ##################################################
def learning_rate_schedule(mode="step", epoch=1, cc=None):
    if mode == "step" :
        # compute the learning rate for the current epoch
        if   epoch % 30  < 7 :  return 1e-3 * np.exp(-epoch * 0.005)  ### 0.74 every 10 epoch
        elif epoch % 30  < 17:  return 7e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 25:  return 3e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 30:  return 5e-5 * np.exp(-epoch * 0.005)
        else :  return 1e-4
    
    if mode == "random" :
        # randomize to prevent overfit... and reset learning
        if   epoch < 10 :  return 1e-3 * np.exp(-epoch * 0.004)  ### 0.74 every 10 epoch
        else :
            if epoch % 10 == 0 :
               ll = np.array([ 1e-3, 7e-4, 6e-4, 3e-4, 2e-4, 1e-4, 5e-5 ]) * np.exp(-epoch * 0.004)            
               ix = np.random.randint(len(ll))        
               cc.lr_actual =  ll[ix]            
            return cc.lr_actual
        

        
def loss_schedule(mode="step", epoch=1):
    if mode == "step" :
        ####  Classifier Loss :   2.8920667 2.6134858 
        ####  {'gender': 0.8566666, 'masterCategory': 0.99, 'subCategory': 0.9166, 'articleType': 0.7, 'baseColour': 0.5633333 }
        cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 90.0  ]
        cc.loss.ww_triplet = 1.0

        if epoch % 10 == 0 : dd.best_loss = 10000.0

        if epoch % 30 < 10 :
            cc.loss.ww_triplet  = 0.5   * 1.0
            cc.loss.ww_clf      = 2.0   * 5.0      ### 2.67
            cc.loss.ww_vae      = 100.0 * 10.0      ### 5.4
            cc.loss.ww_percep   = 10.0  * 1.0      ### 5.8   0.015    #### original: 0.015
            cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 100.0  ]

        elif epoch % 30 < 20 :
            cc.loss.ww_triplet  = 0.5   * 5.0
            cc.loss.ww_clf      = 2.0   * 5.0      ### 2.67
            cc.loss.ww_vae      = 100.0 * 5.0      ### 5.4
            cc.loss.ww_percep   = 10.0  * 1.0      ### 5.8   0.015    #### original: 0.015
            cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 100.0  ]

        elif epoch % 30 < 30 :
            cc.loss.ww_triplet  = 0.5   * 20.0
            cc.loss.ww_clf      = 2.0   * 5.0      ### 2.67
            cc.loss.ww_vae      = 100.0 * 1.0      ### 5.4
            cc.loss.ww_percep   = 10.0  * 5.0      ### 5.8   0.015    #### original: 0.015
            cc.loss.ww_clf_head = [ 1.0 , 1.0, 1.0, 200.0, 100.0  ]



cc.loss= {}

###### Loss definition ##########################################################
####  reduction=tf.keras.losses.Reduction.NONE  for distributed GPU
clf_loss_global    =  tf.keras.losses.BinaryCrossentropy()
### Classification distance
triplet_loss_global = tfa.losses.TripletSemiHardLoss( margin=  1.0,    distance_metric='L2',    name= 'triplet',)
recons_loss_global  = tf.keras.losses.MeanAbsoluteError()  # reduction="sum"
percep_loss_global  = tf.keras.losses.MeanSquaredError()


def perceptual_loss_function(x, x_recon, z_mean, z_logsigma, kl_weight=0.00005,
                             y_label_heads=None, y_pred_heads=None, clf_loss_fn=None):
    ### log( 'x_recon.shae',  x_recon.shape )
    ### VAE Loss  :  Mean Square : 0.054996297 0.046276666   , Huber: 0.0566 
    ### m = 0.00392156862  # 1/255
    ###   recons_loss = tf.reduce_mean( tf.reduce_mean(tf.abs(x-x_recon), axis=(1,2,3)) )
    ###   recons_loss = tf.reduce_mean( tf.reduce_mean(tf.square(x-x_recon), axis=(1,2,3) ) )    ## MSE error
    #     recons_loss = tf.reduce_mean( tf.square(x-x_recon), axis=(1,2,3) )     ## MSE error    
    recons_loss = recons_loss_global(x, x_recon)
    latent_loss = tf.reduce_mean( 0.5 * tf.reduce_sum(tf.exp(z_logsigma) + tf.square(z_mean) - 1.0 - z_logsigma, axis=1) )
    loss_vae    = kl_weight*latent_loss + recons_loss

    
    ### Efficient Head Loss : Input Need to Scale into 0-255, output is [0,1], 1280 vector :  0.5819792 0.5353247 
    ### https://stackoverflow.com/questions/65452099/keras-efficientnet-b0-use-input-values-between-0-and-255
    ### loss_percep = tf.reduce_mean(tf.square(  tf.subtract(tf.stop_gradient(percep_model(x * 255.0 )), percep_model(x_recon * 255.0  )  )))
    loss_percep = percep_loss_global( tf.stop_gradient(percep_model(x * 255.0 )),   percep_model(x_recon * 255.0  )  )
    
    
    #### Update  cc.loss  weights
    loss_schedule(mode="step", epoch=epoch)

        
    ### Triplet Loss: ####################################################################################################
    loss_triplet = 0.0 
    if cc.loss.ww_triplet > 0.0 :   ### 1.9  (*4)     5.6 (*1)
        ### `y_true` to be provided as 1-D integer `Tensor` with shape `[batch_size]  
        ### `y_pred` must be 2-D float `Tensor` of l2 normalized embedding vectors.
        z1     = tf.math.l2_normalize(z_mean, axis=1)  # L2 normalize embeddings
        loss_triplet = 6*triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[0], axis = -1) ,   y_pred=z1)  + 4*triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[2], axis = -1) ,   y_pred=z1)  +  2 * triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[3], axis = -1) ,   y_pred=z1)   +  triplet_loss_global(y_true= tf.keras.backend.argmax(y_label_heads[4], axis = -1) ,   y_pred=z1)  

    
    ####  Classifier Loss :   2.8920667 2.6134858 
    ####  {'gender': 0.8566666, 'masterCategory': 0.99, 'subCategory': 0.9166, 'articleType': 0.7, 'baseColour': 0.5633333 }
    if y_label_heads is not None:
        loss_clf = []
        for i in range(len(y_pred_heads)):
            head_loss = clf_loss_fn(y_label_heads[i], y_pred_heads[i])
            loss_clf.append(head_loss * cc.loss.ww_clf_head[i] )
        loss_clf = tf.reduce_mean(loss_clf)
        
        ####     0.05    ,  0.5                              2.67
        loss_all = loss_vae * cc.loss.ww_vae  +  loss_percep * cc.loss.ww_percep +  loss_clf * cc.loss.ww_clf  + loss_triplet * cc.loss.ww_triplet
        #  loss_all = loss_triplet 
        #### 0.20  
    else :
        loss_all = loss_vae * cc.loss.ww_vae  +  loss_percep * cc.loss.ww_percep
    return loss_all



class StepDecay(LearningRateDecay):
    def __init__(self, init_lr=0.01, factor=0.25, drop_every=5):
        # store the base initial learning rate, drop factor, and epochs to drop every
        self.init_lr    = init_lr
        self.factor     = factor
        self.drop_every = drop_every

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        if   epoch % 30  < 7 :  return 1e-3 * np.exp(-epoch * 0.005)  ### 0.74 every 10 epoch
        elif epoch % 30  < 17:  return 7e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 25:  return 3e-4 * np.exp(-epoch * 0.005)
        elif epoch % 30  < 30:  return 5e-5 * np.exp(-epoch * 0.005)
        else :  return 1e-5



################TEST#####################
if cc.schedule_type == 'step':
    print("Using 'step-based' learning rate decay")
    schedule = StepDecay(init_lr=cc.learning_rate, factor=0.70, drop_every= 3 )
    
    
    
    
