# -*- coding: utf-8 -*-
"""sentence_tansformer.ipynb

cd deeplearning/torch/
python sentrans.py  test

Original file is located at
    https://colab.research.google.com/drive/13jklIi81IT8B3TrIOhWSLwk48Qf2Htmc
**This Notebook has been created by Ali Hamza (9th January, 2022) to train Sentence Transformer with different Losses such as:**
> Softmax Loss
> Cusine Loss
> TripletHard Loss
> MultpleNegativesRanking Loss
"""

#!pip3 install tensorflow
from google.colab import drive
drive.mount('/content/drive')
import sys
# sys.path.append('drive/sent_tans')

#!pip3 install python-box
from box import Box

# !pip install sentence-transformers

from sentence_transformers import SentenceTransformer, SentencesDataset, losses, util
from sentence_transformers import models, losses, datasets
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import math 
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random
import torch
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']='2,3'


#####################################################################################
def log(*s):
    print(*s, flush=True)


#####################################################################################
def test():
  #  Run Various test suing strans_former,
  # Mostly Single sentence   ---> Classification
    cc = Box({})
    cc.epoch = 3
    cc.lr = 1E-5
    cc.warmup = 100

    cc.n_sample  = 1000
    cc.batch_size=16

    cc.mode = 'cpu/gpu'
    cc.ncpu =5
    cc.ngpu= 2  
    
  ### Classifier with Cosinus Loss
    log("Classifier with Cosinus Loss ")
    sentrans_train(modelname_or_path ="distilbert-base-nli-mean-tokens",
                taskname="classifier", 
                lossname="cosinus",
                train_path="/content/sample_data/fake_train_data_v2.csv",
                val_path="/content/sample_data/fake_train_data_v2.csv",
                eval_path = "/content/sample_data/stsbenchmark.csv",
                metricname='cosinus',
                dirout= "/content/sample_data/results/cosinus",cc=cc) 
    

  ### Classifier with Triplet Hard  Loss
    log("Classifier with Triplet Hard  Loss")
    sentrans_train(modelname_or_path ="distilbert-base-nli-mean-tokens",
                taskname="classifier", 
                lossname="triplethard",
                train_path="/content/sample_data/fake_train_data_v2.csv",
                eval_path = "/content/sample_data/stsbenchmark.csv",
                metricname='tripletloss',
                dirout= "/content/sample_data/results/triplethard",cc=cc) 


   ### Classifier with Softmax Loss
    # log("Classifier with Softmax Loss")
    sentrans_train(modelname_or_path ="distilbert-base-nli-mean-tokens",
                taskname="classifier", 
                lossname="softmax",
                train_path="/content/sample_data/fake_train_data_v2.csv",
                val_path="/content/sample_data/fake_train_data_v2.csv",
                eval_path = "/content/sample_data/stsbenchmark.csv",
                metricname='softmax',
                dirout= "/content/sample_data/results/softmax",cc=cc) 

   ### Classifier with MultpleNegativesRankingLoss Loss
    log("Classifier with MultpleNegativesRanking Loss")
    sentrans_train(modelname_or_path ="distilbert-base-nli-mean-tokens",
                taskname="classifier", 
                lossname="MultpleNegativesRankingLoss",
                train_path="/content/sample_data/fake_train_data_v2.csv",
                val_path="content/sample_data/fake_train_data_v2.csv",
                eval_path = "/content/sample_data/stsbenchmark.csv",
                metricname='MultpleNegativesRankingLoss',
                dirout= "/content/sample_data/results/MultpleNegativesRankingLoss",cc=cc)    
   ### Ranking with Cosinus Loss
   # I have written the script for evaluation in the bottom of sentrans_train() function



def model_evaluate(model ="modelname OR path OR model object", fIn='', cc:dict= None):

    df = pd.read_csv(fIn, error_bad_lines=False)
    test_samples = []

    for i, row in df.iterrows():
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    
    if isinstance(model, str): 
       if "/" in model : model= model_load(model)
       else :    model = SentenceTransformer(model)
        
        
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=16, name='sts-test')
    test_evaluator(model, output_path=model)


def model_load(path):
    #### reload model
    model = SentenceTransformer(path)
    model.eval()
    return model

def model_save(model,path, reload=True):
    torch.save(model, path)
    log(path)
    
    if reload:
        #### reload model  + model something   
        model1 = model_load(path)
        model1


def create_evaluator(dname='sts', dirin='/content/sample_data/', cc:dict=None):
    if dname == 'sts':
        ###Read STSbenchmark dataset and use it as development set
        nli_dataset_path = dirin + 'fake_train_data.csv'
        sts_dataset_path = dirin + 'stsbenchmark.csv'
        
        log("Read STSbenchmark dev dataset")

        dev_samples = []
        df = pd.read_csv(sts_dataset_path, error_bad_lines=False)
        
        for i,row in df.iterrows():
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=cc.batch_size, name='sts-dev')
        return dev_evaluator



def load_dataloader(path_or_df = "", cc:dict= None):
    
    if isinstance(path_or_df, str):
        dftrain = pd.read_csv(path_or_df)
        
    elif isinstance(path_or_df, pd.DataFrame):
        dftrain = path_or_df
    else : 
        raise Exception('need')
    
    train_samples = []
    for i,row in dftrain.iterrows():
      train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=row['label']))
      train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=cc.batch_size)
                
    return train_dataloader



def load_loss(model ='', lossname ='cusinus', dftrain ="", coly='label', cc:dict= None):
        
    if lossname == 'MultpleNegativesRankingLoss':
      train_loss = losses.MultipleNegativesRankingLoss(model)
      
    elif lossname == 'softmax':
      train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
                                      num_labels=len(pd.unique(dftrain[coly])))
    elif lossname =='cusinus':
      train_loss = losses.CosineSimilarityLoss(model)
      
    elif lossname =='triplethard':
      train_loss = losses.BatchHardTripletLoss(model=model)
        
    return train_loss



def sentrans_train(modelname_or_path="",
                 taskname="classifier", 
                 lossname="cosinus",
                 train_path="train/*.csv",
                 val_path="val/*.csv",
                 eval_path ="eval/*.csv",
                 metricname='cosinus',
                 dirout ="mymodel_save/",
                 cc:dict= None):
  #  """"
  #     load a model,
  #     load a loss/task
  #     load dataset
  #     fine tuning train the model
  #     evaluate
  #     save on disk
  #     reload the model for check.
  # """

  
    cc = Box(cc)   #### can use cc.epoch   cc.lr
  # """  
  # cc = Box({})
  # cc.epoch = 3
  # cc.lr = 1E-5
  # cc.warmup = 100
  # cc.n_sample  = 1000
  # cc.batch_size=16
  # cc.mode = 'cpu/gpu'
  # cc.ncpu =5
  # cc.ngpu= 2
  # """
  

  ### load model form disk or from internet

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    ### datalodaer
    df      = pd.read_csv(train_path, error_bad_lines=False)
    dftrain = df[[ 'sentence1', 'sentence2', 'label'  ]].values
    train_dataloader,train_loss = load_dataloader( df,cc)

    
    dfval = pd.read_csv(train_path, error_bad_lines=False)
    dfval = dfval[[ 'sentence1', 'sentence2', 'label'  ]].values

    
    ### create loss    
    train_loss = load_losses(model,lossname, df,cc)  
    log(len(train_dataloader))

    if taskname == 'classifier':
        # Configure the training
        warmup_steps = math.ceil(len(train_dataloader) * cc.epoch * 0.1) #10% of train data for warm-up.  
        log("Warmup-steps: {}".format(warmup_steps))
    
        dev_evaluator = create_evaluator('sts', '/content/sample_data/', cc)
       
 
         # Tell pytorch to run this model on the multiple GPUs if available otherwise use all CPUs.
        if cc.get('use_gpu', 0) > 0 :        ### default is CPU
            device = torch.device("cuda:0")
        else :
            device = 'cpu'
        log('device', device)
        
        if device == 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
        elif torch.cuda.device_count() > 1:
          log("Let's use", torch.cuda.device_count(), "GPU")
          model = DDP(model)

        model.to(device)
        
        
        log('########## train')
        model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=cc.epoch,
          evaluation_steps= cc.n_sample,
          warmup_steps=cc.warmup,
          output_path=dirout,
          use_amp=True          #Set to True, if your GPU supports FP16 operations
          )

        log("\n******************< finish training > ********************")
        
        log("## save the model  ")
        model_save(model, dirout, reload=True)


        log('# show metrics')
        model_evaluate(model, eval_path)
        
        

        log("\n******************< finish  > ********************")


##########################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()


