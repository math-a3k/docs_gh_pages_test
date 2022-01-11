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
#!pip3 install python-box
# !pip install sentence-transformers
#!pip3 install tensorflow
"""
# from google.colab import drive
# drive.mount('/content/drive')

import sys, os, gzip, csv, random, math, logging, pandas as pd
from datetime import datetime
from box import Box

# sys.path.append('drive/sent_tans')

from sentence_transformers import SentenceTransformer, SentencesDataset, losses, util
from sentence_transformers import models, losses, datasets
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from tensorflow.keras.metrics import SparseCategoricalAccuracy

from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

os.environ['CUDA_VISIBLE_DEVICES']='2,3'

#### read data on disk
from utilmy import pd_read_file


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
    cc.warmup = 10

    cc.n_sample  = 50
    cc.batch_size=8

    cc.mode = 'cpu/gpu'
    cc.use_gpu = 0
    cc.ncpu =5
    cc.ngpu= 2

    #### Data
    cc.data_nclass = 5



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
    log("Classifier with cosinus Loss")
    sentrans_train(modelname_or_path ="distilbert-base-nli-mean-tokens",
                taskname="classifier", 
                lossname="cosinus",
                train_path="/content/sample_data/fake_train_data_v2.csv",
                val_path="content/sample_data/fake_train_data_v2.csv",
                eval_path = "/content/sample_data/stsbenchmark.csv",
                metricname='MultpleNegativesRankingLoss',
                dirout= "/content/sample_data/results/MultpleNegativesRankingLoss",cc=cc)

###################################################################################################################
def model_evaluate(model ="modelname OR path OR model object", dirdata='./*.csv', dirout='./', cc:dict= None, batch_size=16, name='sts-test'):
    ### Evaluate Model
    df = pd.read_csv(dirdata, error_bad_lines=False)
    test_samples = []
    for i, row in df.iterrows():
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    model= model_load(model)

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=batch_size, name=name)
    test_evaluator(model, output_path=dirout)


def model_load(path_or_name_or_object):
    #### Reload model or return the model itself
    if isintance(path_or_name_or_object, str) :
       # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
       model = SentenceTransformer(path_or_name_or_object)
       model.eval()
    
    return model


def model_save(model,path, reload=True):
    model.save( path)
    log(path)
    
    if reload:
        #### reload model  + model something   
        model1 = model_load(path)
        log(model1)


def model_setup_compute(model, use_gpu=0, ngpu=1, ncpu=1):
     # Tell pytorch to run this model on the multiple GPUs if available otherwise use all CPUs.
    if cc.get('use_gpu', 0) > 0 :        ### default is CPU
        if torch.cuda.device_count() < 0 :
            log('no gpu')
            device = 'cpu'
            torch.set_num_threads(ncpu)
            log('cpu used:', ncpu, " / " ,torch.get_num_threads())
            model = nn.DataParallel(model)            
        else :    
            log("Let's use", torch.cuda.device_count(), "GPU")
            device = torch.device("cuda:0")
            model = DDP(model)        
    else :
            device = 'cpu'
            torch.set_num_threads(ncpu)
            log('cpu used:', ncpu, " / " ,torch.get_num_threads())
            model = nn.DataParallel(model)  
        
    log('device', device)
    model.to(device)
    return model




###################################################################################################################
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

    log('Nelements', len(train_dataloader))
    return train_dataloader



def load_loss(model ='', lossname ='cosinus',  cc:dict= None):

    if lossname == 'MultpleNegativesRankingLoss':
      train_loss = losses.MultipleNegativesRankingLoss(model)

    elif lossname == 'softmax':
      nclass     =  cc.get('data_nclass', -1)
      train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                      num_labels=nclass )
    elif lossname =='cosinus':
      train_loss = losses.CosineSimilarityLoss(model)

    elif lossname =='triplethard':
      train_loss = losses.BatchHardTripletLoss(model=model)


    return train_loss

### function to compute cosinue similarity
def calculate_cosine_similarity(sentence1 = "sentence 1" , sentence2 = "sentence 2", model_id = "model name or path or object"):
    
  model = model_load(model_id)

  #Compute embedding for both lists
  embeddings1 = model.encode(sentence1, convert_to_tensor=True)
  embeddings2 = model.encode(sentence2, convert_to_tensor=True)

  #Compute cosine-similarity
  cosine_scores = util.cos_sim(embeddings1, embeddings2)
  print("{} \t {} \n cosine-similarity Score: {:.4f}".format(sentence1, sentence2, cosine_scores[0][0]))



def sentrans_train(modelname_or_path='distilbert-base-nli-mean-tokens',
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
    cc = Box(cc)   #### can use cc.epoch   cc.lr


    ##### load model form disk or from internet
    model = model_load(modelname_or_path)

    
    ##### datalodaer
    df = pd.read_csv(train_path, error_bad_lines=False)
    # df = pd_read_file(train_path,  error_bad_lines=False)
    train_dataloader = load_dataloader( df, cc)

    
    ##### Use in the code ?????????
    dfval = pd.read_csv(train_path, error_bad_lines=False)
    # dfval = dfval[[ 'sentence1', 'sentence2', 'label'  ]].values

    
    ##### create loss
    if 'data_nclass' not in cc :
        cc.data_nclass = df['label'].nunique()

    train_loss = load_loss(model,lossname,  cc= cc)


    if taskname == 'classifier':
        # print calculate_cosine_similarity before training
        log(" calculate_cosine_similarity before training")  
        calculate_cosine_similarity(df['sentence1'][0], df['sentence2'][0])
        
        # Configure the training
        cc.warmup_steps = math.ceil(len(train_dataloader) * cc.epoch * 0.1) #10% of train data for warm-up.
        log("Warmup-steps: {}".format(cc.warmup_steps))
    
        #### 
        dev_evaluator = create_evaluator('sts', '/content/sample_data/', cc)
       

        # Tell pytorch to run this model on the multiple GPUs if available otherwise use all CPUs.
        model = model_setup_compute(model, use_gpu=cc.get('use_gpu', 0)  , ngpu= cc.get('ngpu', 0) , ncpu= cc.get('ncpu', 1) )


        log('########## train')
        model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=cc.epoch,
          evaluation_steps= cc.n_sample,
          warmup_steps=cc.warmup_steps,
          output_path=dirout,
          use_amp=True          #Set to True, if your GPU supports FP16 operations
          )

        log("\n******************< Eval similarity > ********************")
         # print calculate_cosine_similarity after training
        log(" calculate_cosine_similarity after training")    
        calculate_cosine_similarity(df['sentence1'][0], df['sentence2'][0])
        
        log("### Save the model  ")
        model_save(model, dirout, reload=True)
        model = model_load(dirout)

        log('### Show eval metrics')
        model_evaluate(model, eval_path)
        
       
        log("\n******************< finish  > ********************")


##########################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()


