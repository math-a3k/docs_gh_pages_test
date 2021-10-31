# -*- coding: utf-8 -*-
HELP = """  sentence --> generate vectors


"""
import os, sys, time, datetime,inspect, json, yaml, gc, glob, pandas as pd, numpy as np
from absl import logging

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense,Softmax, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

import tensorflow_hub as hub

###################################################################################
from utilmy.parallel import pd_read_file, os_makedirs
from utilmy.utilmy import log, log2


def help():
    from utilmy import help_create
    ss = help_create("utilmy.nlp.util_sentence") + HELP
    print(ss)


###################################################################################
def test_all():
    pass
    




def test2():
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

  model = SentenceEncoder(num_labels=4)
  model_finetune_classifier(model_path, df, n_labels=4, lrate=1e-5)
  model = model_load(model)
  embed_compare_class_sim(model, embed_a, embed_b, embed_c, embed_d)




def test1():
  import pandas as pd
  df = pd.read_csv('/train_file_sim.csv')
  print(df.head())


  model = SentenceEncoder(num_labels=2)
  model_finetune_classifier(model_path, df, n_labels=4, lrate=1e-5)


  """Extract the embeddings from sample features"""
  embedding_layer = model.layers[0].layers[0]

  samples_a = df.loc[df['class'] == 0][:100]    ## extract 100 examples with class 1
  samples_b = df.loc[df['class'] == 1][:100]      ## extract 100 examples with class 2


  ## create batches for comparison (embed_a & embed b belong to class 0, embed_c & embed_d belong to class 1)
  embed_a = embedding_layer(samples_a[:50]['features'].values).numpy()      
  embed_b = embedding_layer(samples_a[50:100]['features'].values).numpy()    
  embed_c = embedding_layer(samples_b[0:50]['features'].values).numpy()      
  embed_d = embedding_layer(samples_b[50:100]['features'].values).numpy()






###################################################################################
class SentenceEncoder(tf.keras.Model):
  ## create a model on top Universal Sentence Encoder
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


def model_load(model_path):
  ### model load
  model     = tf.keras.models.load_model(model_path)
  model_embedding = model.layers[0].layers[0]
  return model


def model_get_embed(model):
    ### model_embedding('my sentence')    
    model_embed = model.layers[0].layers[0]
    return model_embed


def get_embed(model_emb, word) :
  return model_emb(word).to_numpy()


def model_finetune_classifier(model_path, df, n_labels=3, lrate=1e-5):
  """ Fine-tune the model on the training dataframe


  """  
  model = SentenceEncoder(num_labels=n_labels)
  opt   = Adam(learning_rate=lrate, epsilon=1e-08)

  # two outputs, one for slots, another for intents,  we have to fine tune for both
  losses  = [SparseCategoricalCrossentropy(from_logits=True)]
  metrics = [SparseCategoricalAccuracy("accuracy")]

  model.compile(optimizer=opt, loss=losses, metrics=metrics)
  history = model.fit(df['features'].values,
                      df['class'].values,
                      epochs=10,
                      batch_size=32,
                      verbose=1)

  np.unique(df['class'].values)

  model.save(model_path)
  return model


def embed_compare_class_sim(model, embed_a, embed_b, embed_c, embed_d):
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



  
  
  
  
  
  
  
  


###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()



