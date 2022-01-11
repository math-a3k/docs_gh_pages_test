# -*- coding: utf-8 -*-
"""sentence_tansformer.ipynb
cd deeplearning/torch/
python sentences.py  test

Original file is located at
    https://colab.research.google.com/drive/13jklIi81IT8B3TrIOhWSLwk48Qf2Htmc

train Sentence Transformer with different Losses such as:**
> Softmax Loss
> Cusine Loss
> TripletHard Loss
> MultpleNegativesRanking Loss
#!pip3 install python-box
# !pip install sentence-transformers
#!pip3 install tensorflow
"""
import sys, os, gzip, csv, random, math, logging, pandas as pd
from datetime import datetime
from box import Box

import sentence_transformers as st

from sentence_transformers import SentenceTransformer, SentencesDataset, losses, util
from sentence_transformers import models, losses, datasets
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

#vfrom tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances


#### read data on disk
from utilmy import pd_read_file


#####################################################################################
def log(*s):
    print(*s, flush=True)

    

#####################################################################################
def test():
    #  Run Various test suing strans_former,
    # Mostly Single sentence   ---> Classification
    os.environ['CUDA_VISIBLE_DEVICES']='2,3'
  
    cc = Box({})
    cc.epoch = 3
    cc.lr = 1E-5
    cc.warmup = 10

    cc.eval_steps  = 50
    cc.batch_size=8

    cc.mode = 'cpu/gpu'
    cc.use_gpu = 0
    cc.ncpu =5
    cc.ngpu= 2

    #### Data
    cc.data_nclass = 5


    dirdata ='ztest/'
    modelid = "distilbert-base-nli-mean-tokens"
    
    dataset_download(dirout= dirdata)
    dataset_fake(dirdata)
    
    lloss = [ 'cosine', 'triplethard',"softmax", 'MultpleNegativesRankingLoss' ]
    
    for lname in lloss :
        log("Classifier with Loss ", lname)
        sentrans_train(modelname_or_path = modelid,
                    taskname  = "classifier", 
                    lossname  = lname,
                    train_path= dirdata + f"/data_fake.parquet",
                    val_path=   dirdata + f"/data_fake.parquet",
                    eval_path = dirdata + f"/data_fake.parquet",
                    metricname='cosinus',
                    dirout= dirdata + f"/results/" + lloss, cc=cc) 
    


###################################################################################################################        
def dataset_fake(dirdata):        
    nli_dataset_path = dirdata + '/AllNLI.tsv.gz'
    sts_dataset_path = dirdata + '/stsbenchmark.tsv.gz'

    # Read the AllNLI.tsv.gz file and create the training dataset
    df = pd_read(nli_dataset_path, npool=1) 
    df.iloc[:50, :].to_parquet(dirdata +"/fake_data.parquet")



def dataset_fake2():
    # This function load the fake dataset if it's already existed otherwise downloads it first.
    # then Preprocess the data for MultpleNegativesRanking loss function and return it as dataloader

    def add_to_samples(sent1, sent2, label):
        if sent1 not in train_data:
            train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
            train_data[sent1][label].add(sent2)

    train_data = {}
    with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'train':
                sent1 = row['sentence1'].strip()
                sent2 = row['sentence2'].strip()

                df.append(sent1, sent2, row['label'])
                df.append(sent2, sent1, row['label'])  #Also add the opposite


    train_samples = []
    for sent1, others in train_data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            train_samples.append(InputExample(texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            train_samples.append(InputExample(texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

    
def dataset_download(dirout='/content/sample_data/sent_tans/'):
    #### Check if dataset exsist. If not, download and extract  it    
    nli_dataset_path = dirout + '/AllNLI.tsv.gz'
    sts_dataset_path = dirout + '/stsbenchmark.tsv.gz'
    os.makedirs(dirout, exist_ok=False)    
    if not os.path.exists(nli_dataset_path):
        util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
        


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
def pd_read(path_or_df='./myfile.csv', npool=1,  **kw):
    if isinstance(path_or_df, str):
        if '.tsv' in path_or_df or '.csv' in  path_or_df  :
            dftrain = pd_read_file(path_or_df, npool=npool)
        else :    
            dftrain = pd.read_csv(path_or_df, error_bad_lines=False)
        
    elif isinstance(path_or_df, pd.DataFrame):
        dftrain = path_or_df
    else : 
        raise Exception('need' path_or_df)
    return dftrain    
        
        
def load_evaluator(name='sts', path_or_df="", dname='sts', cc:dict=None):
    if dname == 'sts':        
        log("Read STSbenchmark dev dataset")
        df = pd_read(path_or_df) 
        if 'nsample' in cc : df = df.iloc[:cc.nsample,:]
     
        dev_samples = []        
        for i,row in df.iterrows():
            if row['split'] == 'dev':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size= cc.batch_size, name=name)
        return dev_evaluator


def load_dataloader(name='sts', path_or_df = "", cc:dict= None, npool=4):    
    df = pd_read(path_or_df, npool=npool) 
    
    if 'nsample' in cc : df = df.iloc[:cc.nsample,:]
    
    train_samples = []
    for i,row in df.iterrows():
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


def metrics_cosine_sim(sentence1 = "sentence 1" , sentence2 = "sentence 2", model_id = "model name or path or object"):
  ### function to compute cosinue similarity      
  model = model_load(model_id)

  #Compute embedding for both lists
  embeddings1 = model.encode(sentence1, convert_to_tensor=True)
  embeddings2 = model.encode(sentence2, convert_to_tensor=True)

  #Compute cosine-similarity
  cosine_scores = util.cos_sim(embeddings1, embeddings2)
  log( f"{sentence1} \t {sentence2} \n cosine-similarity Score: {cosine_scores[0][0]}" )



def sentrans_train(modelname_or_path='distilbert-base-nli-mean-tokens',
                 taskname="classifier",   lossname="cosinus",
                 datasetname = 'sts',  
                   
                 train_path="train/*.csv", val_path  ="val/*.csv",  eval_path ="eval/*.csv",
                   
                 metricname='cosinus',
                 dirout ="mymodel_save/",
                 cc:dict= None):
  #  """"
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
    
    if taskname == 'classifier':
        log(" metrics_cosine_similarity before training")  
        metrics_cosine_sim(df['sentence1'][0], df['sentence2'][0], model)
        
        
        ##### dataloader train, evaluator
        if 'data_nclass' not in cc :
            cc.data_nclass = df['label'].nunique()
        
        train_dataloader = load_dataloader(datasetname, train_path, cc)        
        val_evaluator    = load_evaluator(datasetname,  eval_path,  cc)
    
        ##### Task Loss
        train_loss       = load_loss(model, lossname,  cc= cc)        
        
        ##### Configure the training
        cc.warmup_steps = math.ceil(len(train_dataloader) * cc.epoch * 0.1) #10% of train data for warm-up.
        log("Warmup-steps: {}".format(cc.warmup_steps))
          
        model = model_setup_compute(model, use_gpu=cc.get('use_gpu', 0)  , ngpu= cc.get('ngpu', 0) , ncpu= cc.get('ncpu', 1) )
        
        
        log('########## train')
        model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator =val_evaluator,
          epochs    =cc.epoch,
          evaluation_steps = cc.eval_steps,
          warmup_steps     = cc.warmup_steps,
          output_path=dirout,
          use_amp=True          #Set to True, if your GPU supports FP16 operations
          )

        log("\n******************< Eval similarity > ********************")
         # print calculate_cosine_similarity after training
        log(" cosine_similarity after training")    
        metrics_cosine_sim(df['sentence1'][0], df['sentence2'][0])
        
        log("### Save the model  ")
        model_save(model, dirout, reload=True)
        model = model_load(dirout)

        log('### Show eval metrics')
        model_evaluate(model, dirout)
        
        log("\n******************< finish  > ********************")


##########################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()


