# -*- coding: utf-8 -*-
"""sentence transformer main.ipynb
Original file is located at
https://colab.research.google.com/drive/1oc1fzgNEDLDEDn4Lfa-72Bzx_SIA20A-?usp=sharing



"""

#!pip3 install tensorflow
from google.colab import drive
drive.mount('/content/drive')
import sys



### pip install python-box
from box import Box


##### Train params  #################
def test():
  """
    Run Various test suing strans_former,

    Mostly Single sentence   ---> Classification

  """
  ### Classifier with Cosinus Loss
  cc = Box({}) 
  cc = Box({})
  cc.epoch = 3
  cc.lr = 1E-5
  cc.warmup = 100

  cc.n_sample  = 1000
  cc.batch_size=16

  cc.mode = 'cpu/gpu'
  cc.ncpu =5
  cc.ngpu= 2  

  sentrans_train(    , cc=cc)



  ### Classifier with Triplet Hard loss Loss








  ### Classifier with Softmax Loss





  ### Ranking with Cosinus Loss




  ###




"""  
cc = Box({})
cc.epoch = 3
cc.lr = 1E-5
cc.warmup = 100

cc.n_sample  = 1000
cc.batch_size=16

cc.mode = 'cpu/gpu'
cc.ncpu =5
cc.ngpu= 2
"""







##### Please use those tempalte code
def log(*s): print(*s, flush=True)



def metric_evaluate(model, )
    df = pd.read_csv(fIn, delimiter='\t',)
    test_samples = []

    for i, row in df.iterrows():
        if row['split'] == 'test':
          score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
          test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    """
    ### show metrics
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
      reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
      for row in reader:
        if row['split'] == 'test':
          score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
          test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    """

    model = SentenceTransformer(modelname_or_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=16, name='sts-test')
    test_evaluator(model, output_path=modelname_or_path)




def model_load(path):
    #### reload model
    model = SentenceTransformer(path)
    model.eval()
    return model


def model_save(path):
    #### reload model
    model = laodd ...
    model.eval()
        


def create_evaluator(dname='sts', dirin='/content/sample_data/sent_tans/', cc:dict=None):
    if dname ='sts':
        ###Read STSbenchmark dataset and use it as development set
        download_dataset()
        nli_dataset_path = dirin + 'AllNLI.tsv.gz'
        sts_dataset_path = dirin + '/stsbenchmark.tsv.gz'
        
        log("Read STSbenchmark dev dataset")

        dev_samples = []
        df = pd_read_csv( sts_dataset_path, delimiter='\t', quoting=csv.QUOTE_NONE )
        for i,row in df.iterrows():
          if row['split'] == 'dev':
              score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
              dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

        """
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'dev':
                    score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                    dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        """
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=cc.batch_size, name='sts-dev')
        return dev_evaluator

















def sentrans_train(modelname_or_path="",
                 taskname="classifier", 
                 lossname="",
                 train_path="train/*.csv",
                 val_path="val/*.csv",
                 metricname='cosinus',
                 dirout ="mymodel_save/",
                 cc:dict= None
                
  )
  """"
      load a model,
      load a loss/task
      load dataset
      fine tuning train the model
      evaluate
      save on disk
      reload the model for check.


  """
  cc = Box(cc)   #### can use cc.epoch   cc.lr

  """  
  cc = Box({})
  cc.epoch = 3
  cc.lr = 1E-5
  cc.warmup = 100

  cc.n_sample  = 1000
  cc.batch_size=16

  cc.mode = 'cpu/gpu'
  cc.ncpu =5
  cc.ngpu= 2
  """



  ### load model form disk or from internet



  ## dataloader
  dftrain = pd.read_csv(  dir_train )
  dftrain = dftrain[[ 'text1', 'text2', 'label'  ]].values

  dfval = pd.read_csv(  dir_train )
  dfval = dfval[[ 'text1', 'text2', 'label'  ]].values





  ### create task and Loss function
  if lossname == 'cosinus':  loss = 


  if taskname == 'classifier ':
      task =  



  ## train model with multi-cpus  or multi-gpus 




  ### show metrics


  ### save model


  #### reload model






























############################################################################
"""sentence-transformers/distiluse-base-multilingual-cased-v
!pip install sentence-transformers

"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

train_examples = [InputExample(texts=['野菜は健康的です', '運動はあなたにとって良いことです'], label=0.8),
    InputExample(texts=['空は曇りです', '食べ物はテーブルの上にあります'], label=0.3)]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)




##### """Before Training:
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

## uncomment to create a new model or reuse from previous block (trained) ###
#model = CrossEncoder('cl-tohoku/bert-base-japanese-v2', num_labels=4)
    # Smartphones
a1  = "ゾニ畏ゲ"
a2 =   "レゑけい何"
      # Weather
b1 =   "流コァ依"
b2 =   "し維せ科ま逸"
   
    # Food and health
c1 =  "臆デ夜ッヶバ鬱れ"
c2 = "うユ威ぱ"

encode_a1 = model.tokenizer.encode([a1], return_tensors='pt')
encode_a2 = model.tokenizer.encode([c1], return_tensors='pt')
a1_v = model.model.bert(encode_a1).pooler_output[0]
a2_v = model.model.bert(encode_a2).pooler_output[0]
cosine_similarity(a1_v.detach().numpy().reshape(1,-1),a2_v.detach().numpy().reshape(1,-1))



#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)



#####After Training:
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

## uncomment to create a new model or reuse from previous block (trained) ###
#model = CrossEncoder('cl-tohoku/bert-base-japanese-v2', num_labels=4)
    # Smartphones
a1  = "ゾニ畏ゲ"
a2 =   "レゑけい何"
      # Weather
b1 =   "流コァ依"
b2 =   "し維せ科ま逸"
   
    # Food and health
c1 =  "臆デ夜ッヶバ鬱れ"
c2 = "うユ威ぱ"

encode_a1 = model.tokenizer.encode([a1], return_tensors='pt')
encode_a2 = model.tokenizer.encode([c1], return_tensors='pt')
a1_v = model.model.bert(encode_a1).pooler_output[0]
a2_v = model.model.bert(encode_a2).pooler_output[0]
cosine_similarity(a1_v.detach().numpy().reshape(1,-1),a2_v.detach().numpy().reshape(1,-1))



#########################################
"""Before training:
a1 - a2 : 0.999
a1 - c1: 0.999


after training: 
a1 - a2 : 0.82244647

a1 - c1: 0.67907935

"""


#### Check
import numpy as np
result = model.tokenizer.encode(['hello'], return_tensors='pt')
print(result)
s1 = model.model.bert(result).pooler_output

print(model.model.bert(result).pooler_output.shape)

# ! wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -q

# ! wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz -q















#################################################################################
#################################################################################
"""Cross Encoder example"""

"""
This examples trains a CrossEncoder for the NLI task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it learns to predict the labels: "contradiction": 0, "entailment": 1, "neutral": 2.
It does NOT produce a sentence embedding and does NOT work for individual sentences.
Usage:
python training_nli.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv

queries = ['whats the color of the sky?','what is the capitol of France?','How many bytes in kb?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?']
contexts = ['the color of the sky is blue','transformers are used for embeddings','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte']
labels = [ 1,0,0,1,1,1,1,1,1,1]


queries_val = ['whats the color of the sky?','what is the capitol of France?','How many bytes in kb?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?']
contexts_val = ['the color of the sky is blue','transformers are used for embeddings','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte']
labels_val = [ 1,0,0,1,1,1,1,1,1,1]

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

train_samples = []
dev_samples = []
for query, label_id in zip(X, labels):
    train_samples.append(InputExample(texts=[query, ""], label=label_id))

#for query, context, label_id in zip(queries_val, contexts_val, labels_val):
    #dev_samples.append(InputExample(texts=[query, context], label=label_id))


train_batch_size = 1
num_epochs = 1


#Define our CrossEncoder model. We use distilroberta-base as basis and setup it up to predict 3 labels
model = CrossEncoder('cl-tohoku/bert-base-japanese-v2', num_labels=4)


#We wrap train_samples, which is a list ot InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

#During training, we use CESoftmaxAccuracyEvaluator to measure the accuracy on the dev set.
evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev_samples)


warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))
model_save_path= 'outputs/saved_models/'












































#################################################################################
#################################################################################
#################################################################################
######## Fine tuning in Keras  ##################################################
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense,Softmax, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from absl import logging
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense,Softmax, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

import tensorflow as tf

import tensorflow_hub as hub
## create a model on top Universal Sentence Encoder
class SentenceEncoder(tf.keras.Model):
  def __init__(self, num_labels=None):
      super().__init__(name="sentence_encoder")
      module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
      se = hub.load(module_url)
      hub_layer = hub.KerasLayer(se, input_shape=[], dtype=tf.string, trainable=True)
      self.model = tf.keras.Sequential()
      self.model.add(hub_layer)
      self.model.add(tf.keras.layers.Dense(256, activation='relu'))
      self.model.add(tf.keras.layers.Dense(num_labels))
  def call(self, inputs, **kwargs):
        # two outputs from BERT
        return self.model(inputs)

class ReRanker(tf.keras.Model):
  def __init__(self):
      super().__init__(name="sentence_encoder")
      module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
      se = hub.load(module_url)
      hub_layer = hub.KerasLayer(se, input_shape=[], dtype=tf.string, trainable=True)
      self.model = tf.keras.Sequential()
      self.model.add(hub_layer)
      self.model.add(tf.keras.layers.Dense(512, activation='relu'))
      self.model.add(tf.keras.layers.Dense(2, activation='softmax'))
  def call(self, inputs, **kwargs):
        # two outputs from BERT
        return self.model(inputs)
               
model = SentenceEncoder(num_labels=4)





module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
query   = tf.keras.layers.Input(shape=(128,))
context = tf.keras.layers.Input(shape=(512,))
classifier = tf.keras.Sequential([
    tf.keras.layers.concatenate(),                              
    hub.KerasLayer("model", input_shape=[]),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')
])



se = hub.load(module_url)
hub_layer = hub.KerasLayer(se, input_shape=[], dtype=tf.string, trainable=True)(query)
x = Dense(512, activation='relu')(hub_layer)
outputs  = Dense(2, activation='softmax')(x)
model = Model(inputs=[query, context], outputs=[var_test])
opt = Adam(learning_rate=3e-5, epsilon=1e-08)
losses = [SparseCategoricalCrossentropy(from_logits=False)]
losses = [tfr.keras.losses.PairwiseHingeLoss()]
metrics = [SparseCategoricalAccuracy("accuracy")]
# compile model
model.compile(optimizer=opt, loss=losses, metrics=metrics)

def build_model():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    query = tf.keras.Input(shape=(), name="query", dtype=tf.string)
    context = tf.keras.Input(shape=(),name="context", dtype=tf.string)
    embed = hub.load(module_url)
    hub_emb = hub.KerasLayer(embed, input_shape=(), output_shape = (512), dtype=tf.string, trainable=True)
    query_emb = hub_emb(query)
    context_emb = hub_emb(context)
    emb = tf.keras.layers.Concatenate()([query_emb, context_emb])
    dense = tf.keras.layers.Dense(256, activation="relu")(emb)
    classifier = tf.keras.layers.Dense(2)(dense)
    model = tf.keras.Model(inputs=[query, context], outputs=classifier, name="ranker_model")
    losses = [tf.losses.SparseCategoricalCrossentropy(from_logits=True)]
    model.compile(loss=losses, optimizer="adam", metrics=['accuracy'])
    return model
model = build_model()

model = SentenceEncoder(num_labels=4)
q = ['whats the color of the sky?','what is the capitol of France?','How many bytes in kb?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?']
q = np.array(q)
context = ['the color of the sky is blue','transformers are used for embeddings','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte']
#context = tf.convert_to_tensor(context)
context = np.array(context)
y = [ 1,0,0,1,1,1,1,1,1,1]
print(len(q) == len(context))
history = model.fit([q,context], np.array(y), epochs=2, batch_size=1, verbose=1)

q = ['whats the color of the sky?','what is the capitol of France?','How many bytes in kb?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?','what is the capitol of France?']
context = ['the color of the sky is blue','transformers are used for embeddings','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte','there are 1000 bytes in one kilo byte']
y = [ 1,0,0,1,1,1,1,1,1,1]

for a, b, label in zip(q,context,y):
   print(a)
   print(b)

!pip install fugashi
!pip install unidic_lite

### create sample data with four labels in total ###
X = [
    # Smartphones
    "ゾニ畏ゲ",
    "レゑけい何",
    "り雲アテ",

    # Weather
    "流コァ依",
    "し維せ科ま逸",
    "咽こ遺ク",

    # Food and health
    "臆デ夜ッヶバ鬱れ",
    "うユ威ぱ",
    "ブ屋ぐ鋭",
     

    # Asking about age
    "ニ逸欧河",
    "ゆヘ畝",
]

y = [0,0,0,1,1,1,2,2,2,3,3]




### create sample data with four labels in total ###
X = [
    # Smartphones
    "ゾニ畏ゲ",
    "レゑけい何",
    "り雲アテ",

    # Weather
    "流コァ依",
    "し維せ科ま逸",
    "咽こ遺ク",

    # Food and health
    "臆デ夜ッヶバ鬱れ",
    "うユ威ぱ",
    "ブ屋ぐ鋭",
     

    # Asking about age
    "ニ逸欧河",
    "ゆヘ畝",
]

y = [0,0,0,1,1,1,2,2,2,3,3]

## build, compile and fit the model on the training data
model = SentenceEncoder(num_labels=4)

opt = Adam(learning_rate=3e-5, epsilon=1e-08)


losses = [SparseCategoricalCrossentropy(from_logits=True)]

metrics = [SparseCategoricalAccuracy("accuracy")]
# compile model
model.compile(optimizer=opt, loss=losses, metrics=metrics)
history = model.fit(X,
                    y,
                    epochs=2,
                    batch_size=32,
                    verbose=1)

## path to save/load the model
model_path = '/content/drive/MyDrive/MultilingualModel-Yaki/sentence_encoder'

## extract the SentenceEncoderHubLayer
embedding = model.layers[0].layers[0]
Weather_A = "ァだ依尉ド乙"
Weather_B = "ぅ　て暗虞煙つッペ液"
Health_C = "億ンじ違"
Health_D = "はま衣"

### get the embeddings ####
embedding_a = embedding([Weather_A])[0].numpy()      
embedding_b = embedding([Weather_B])[0].numpy()                             


from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

## check cosine similarity ##
result=cosine_similarity(embedding_a.reshape(1,-1),embedding_b.reshape(1,-1))
print("before saving: ")
print(result)

## save the model ###
model.save(model_path)

### load the model ####
model = tf.keras.models.load_model(model_path)
embedding = model.layers[0].layers[0]

## recompute the embeddings using the loaded model ##
embedding_a = embedding([Weather_A])[0].numpy()      
embedding_b = embedding([Weather_B])[0].numpy()                             

result=cosine_similarity(embedding_a.reshape(1,-1),embedding_b.reshape(1,-1))

print("after saving: ")

print(result)

"""Compare embeddings before/after fine-tuning
> 
Load dataset train_file_sim.csv 

"""

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/MultilingualModel-Yaki/train_file_sim.csv')
print(df.head())



"""Extract the embeddings from sample features"""

model = SentenceEncoder(num_labels=2)

embedding_layer = model.layers[0].layers[0]
## extract 100 examples with class 1
samples_a = df.loc[df['class'] == 0][:100]
## extract 100 examples with class 2
samples_b = df.loc[df['class'] == 1][:100]

## create batches for comparison (embed_a & embed b belong to class 0, embed_c & embed_d belong to class 1)
embed_a = embedding_layer(samples_a[:50]['features'].values).numpy()      
embed_b = embedding_layer(samples_a[50:100]['features'].values).numpy()    
embed_c = embedding_layer(samples_b[0:50]['features'].values).numpy()      
embed_d = embedding_layer(samples_b[50:100]['features'].values).numpy()

"""Compare the embeddings of A & B , A & D """

from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np
embedding_layer = model.layers[0].layers[0]

res_same_class = []
for a,b,  in zip(embed_a, embed_b):
    result=cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))
    res_same_class.append(result)
print("avg similarity score: same class")

print(np.mean(res_same_class))
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np
diff_same_class = []
for a,b,  in zip(embed_a, embed_d):
    result=cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))
    diff_same_class.append(result)
print("avg similarity score: different classes")
print(np.mean(diff_same_class))

"""Fine-tune the model on the training dataframe"""

model = SentenceEncoder(num_labels=2)

opt = Adam(learning_rate=1e-5, epsilon=1e-08)

# two outputs, one for slots, another for intents
# we have to fine tune for both
losses = [SparseCategoricalCrossentropy(from_logits=True)]

metrics = [SparseCategoricalAccuracy("accuracy")]
# compile model
model.compile(optimizer=opt, loss=losses, metrics=metrics)

history = model.fit(df['features'].values,
                    df['class'].values,
                    epochs=10,
                    batch_size=32,
                    verbose=1)

np.unique(df['class'].values)

"""Recompute embeddings after fine-tuining"""

embedding_layer = model.layers[0].layers[0]

## create batches for comparison (embed_a & embed b belong to class 0, embed_c & embed_d belong to class 1)
embed_a = embedding_layer(samples_a[:50]['features'].values).numpy()      
embed_b = embedding_layer(samples_a[50:100]['features'].values).numpy()    
embed_c = embedding_layer(samples_b[0:50]['features'].values).numpy()      
embed_d = embedding_layer(samples_b[50:100]['features'].values).numpy()

res_same_class = []
for a,b,  in zip(embed_a, embed_b):
    result=cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))
    res_same_class.append(result)
print("after fine-tuning: avg similarity score: same class")

print(np.mean(res_same_class))
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np
diff_same_class = []
for a,b,  in zip(embed_a, embed_d):
    result=cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))
    diff_same_class.append(result)
print("after fine-tuning: avg similarity score: different classes")
print(np.mean(diff_same_class))









