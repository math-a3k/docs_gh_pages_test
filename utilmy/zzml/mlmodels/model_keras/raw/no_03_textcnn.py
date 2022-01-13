#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 23:34:15 2020

@author: user
"""

from sklearn.model_selection import train_test_split
from collections import Counter
import keras.layers as Layer
import pandas as pd
import numpy as np
import os, inspect
import gensim
import keras
from jsoncomment import JsonComment ; json = JsonComment()
import re

"""
## Models List
|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  TextCNN Model  | [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)             |
"""
"""
Requirements:   
    gensim
    numpy
    keras
    sklearn
Download the pre-train word2vec (https://code.google.com/archive/p/word2vec/), and decompression the file to the "./dataset/" path.
Download  MR data  from this link(http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz) and decompression data to the "./dataset/" path
 
"""


VERBOSE = False
def log(*s, n=0, m=1):
  sspace = "#" * n
  sjump = "\n" * m
  print(sjump, sspace, s, sspace, flush=True)

####################################################################################################
def os_module_path():
  current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
  return current_dir

def path_setup(out_folder="", sublevel=0, data_path="dataset/"):
    current_dir= os_module_path()
    data_path = os.path.join(current_dir, data_path)
    out_path = os.path.join(current_dir, out_folder)
    os.makedirs(out_path, exist_ok=True)
    model_path = os.path.join(out_path,"model/")
    os.makedirs(model_path, exist_ok=True)
    log(data_path, out_path, model_path)
    return data_path, out_path, model_path

def get_params(choice="", data_path="./dataset/", config_mode="test", **kw):
    if choice == "json":
        data_path = "./params.json"
        with open(data_path) as config_f:
            config = json.load(config_f)
            print(config)
            c      = config[config_mode]

        model_pars, data_pars  = c[ "model_pars" ], c[ "data_pars" ]
        compute_pars, out_pars = c[ "compute_pars" ], c[ "out_pars" ]
        return model_pars, data_pars, compute_pars, out_pars


    if choice == "test01":
        log("#### Path params   #################")
        data_path, out_path, model_path = path_setup(out_folder="output/", sublevel=0,
                                                     data_path=data_path)
       
        positive_data_file_path = os.path.join(data_path,"rt-polaritydata/rt-polarity.pos")
        negative_data_file_path = os.path.join(data_path,"rt-polaritydata/rt-polarity.neg")
        word2vec_model_path = os.path.join(data_path,"GoogleNews-vectors-negative300.bin")
        out_path = os.path.join(out_path,"output.csv")
        modelpath = os.path.join(model_path,"model.h5")
        model_pars   = {"learning_rate": 0.001,"sequence_length": 56,"num_classes": 2,"drop_out" : 0.5,"l2_reg_lambda" : 0.0,"optimization" : "adam","embedding_size" : 300,"filter_sizes": [3, 4, 5],"num_filters" : 128}
        data_pars = {"positive_data_file":positive_data_file_path ,"negative_data_file": negative_data_file_path,"DEV_SAMPLE_PERCENTAGE": 0.1,"data_type": "pandas","size": [0, 0, 6],"output_size": [0, 6],"train": "True","word2vec_model_path":word2vec_model_path}
        compute_pars = {"epochs": 1,"batch_size" : 128,"return_pred" : "True"}
        out_pars     = {"out_path": out_path,"data_type": "pandas","size": [0, 0, 6],"output_size": [0, 6],"modelpath": modelpath}

    return model_pars, data_pars, compute_pars, out_pars

class data_loader:

  def __init__(self, data_pars=None
               ):
      self.data_pars = data_pars
      
  def clean_str(self,string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

  def load_data_and_labels(self):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(self.data_pars["positive_data_file"], "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(self.data_pars["negative_data_file"], "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [self.clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

  def as_matrix(self, sequences, max_len, index2word):
    matrix = np.full((len(sequences), max_len), 0)
    for i, seq in enumerate(sequences):
        row_ix = [index2word.index(w) for w in seq.split(' ')]
        matrix[i, :len(row_ix)] = row_ix
    return matrix
  def Generate_data(self, data_pars=None):

    x_text, y = self.load_data_and_labels()
    print('Total records of the MR data set: ', len(x_text))
    max_doc_length = max([len(x.split(' ')) for x in x_text])
    print("Max document length: ", max_doc_length)

    tokens = [t for doc in x_text for t in doc.split(' ')]
    print("Total tokens in the MR data set: ", len(tokens))
    counter = Counter(tokens)
    index2word = list(counter.keys())
    index2word.insert(0, 'PAD')
    print("Vocabulary size in MR data set(contains 'PAD' as first): ", len(index2word))
    x_matrix = self.as_matrix(x_text, max_doc_length, index2word)
    x_train, x_test, y_train, y_test = train_test_split(x_matrix, y, test_size=self.data_pars["DEV_SAMPLE_PERCENTAGE"])
    print('Train records: ', len(x_train))
    print('Test records:', len(x_test))

    return x_train, x_test, y_train, y_test ,index2word 


class data_provider:
  def __init__(self,  data_loader, data_pars=None
                ):
        self.data_pars = data_pars
        self.data_loader = data_loader
        self.x_train, self.x_test, self.y_train, self.y_test, self.index2word = self.data_loader.Generate_data()
  def get_dataset(self, **kw):
        """
          JSON data_pars to get dataset
          "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
          "size": [0, 0, 6], "output_size": [0, 6] },
        """

        if self.data_pars['train'] :
          Xtrain, ytrain = self.x_train, self.y_train  # data for training.
          return Xtrain, ytrain, self.index2word

        else :
          Xtest, ytest = self.x_test, self.y_test  # data for test.
          return Xtest, ytest

def get_pre_train_word2vec(model, index2word, vocab_size):
    embedding_size = model.vector_size
    pre_train_word2vec = dict(zip(model.vocab.keys(), model.vectors))
    word_embedding_2dlist = [[]] * vocab_size    # [vocab_size, embedding_size]
    word_embedding_2dlist[0] = np.zeros(embedding_size)    # assign empty for first word:'PAD'
    pre_count = 0    # vocabulary in pre-train word2vec
    # loop for all vocabulary, note that the first is 'PDA'
    for i in range(1, vocab_size):
        if index2word[i] in pre_train_word2vec:
            word_embedding_2dlist[i] = pre_train_word2vec[index2word[i]]
            pre_count += 1
        else:
            # initilaize randomly if vocabulary not exits in pre-train word2vec
            word_embedding_2dlist[i] = np.random.uniform(-0.1, 0.1, embedding_size)
    return np.array(word_embedding_2dlist), pre_count


class Model:

  def __init__(self, embedding_matrix=None, 
               vocab_size=None, 
               model_pars=None
 
               ):
    ### Model Structure        ################################
    self.embedding_matrix = embedding_matrix
    self.vocab_size = vocab_size
    if model_pars is None:
            self.model = None

    else:
            self.model_pars = model_pars
            
  def model(self):
      input_x = Layer.Input(shape=(self.model_pars["sequence_length"],), name='input_x')

        # embedding layer
      if self.embedding_matrix is None:
          embedding = Layer.Embedding(self.vocab_size, self.model_pars["embedding_size"], name='embedding')(input_x)
      else:
          embedding = Layer.Embedding(self.vocab_size, self.model_pars["embedding_size"], weights=[self.embedding_matrix], name='embedding')(input_x)
      expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
      embedding_chars = Layer.Reshape(expend_shape)(embedding)

      # conv->max pool
      pooled_outputs = []
      for i, filter_size in enumerate(self.model_pars["filter_sizes"]):
          conv = Layer.Conv2D(filters=self.model_pars["num_filters"], 
                          kernel_size=[filter_size, self.model_pars["embedding_size"]],
                          strides=1,
                          padding='valid',
                          activation='relu',
                          kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                          bias_initializer=keras.initializers.constant(value=0.1),
                          name=('conv_%d' % filter_size))(embedding_chars)
  
          max_pool = Layer.MaxPool2D(pool_size=[self.model_pars["sequence_length"] - filter_size + 1, 1],
                                  strides=(1, 1),
                                  padding='valid',
                                  name=('max_pool_%d' % filter_size))(conv)
          pooled_outputs.append(max_pool)


      # combine all the pooled features
      num_filters_total = self.model_pars["num_filters"] * len(self.model_pars["filter_sizes"])
      h_pool = Layer.Concatenate(axis=3)(pooled_outputs)
      h_pool_flat = Layer.Reshape([num_filters_total])(h_pool)
      # add dropout
      dropout = Layer.Dropout(self.model_pars["drop_out"])(h_pool_flat)

      # output layer
      output = Layer.Dense(self.model_pars["num_classes"],
                        kernel_initializer='glorot_normal',
                        bias_initializer=keras.initializers.constant(0.1),
                        activation='softmax',
                        name='output')(dropout)

      model = keras.models.Model(inputs=input_x, outputs=output)
      model.compile(self.model_pars["optimization"], 'categorical_crossentropy', metrics=['accuracy'])
      return model

def fit(model, Xtrain, ytrain, compute_pars=None, **kw):
  """
  :param model:    Class model
  :param data_pars:  dict of
  :param out_pars:
  :param compute_pars:
  :param kwargs:
  :return:
  """
  model = model.fit(Xtrain, ytrain,
		                  batch_size=compute_pars["batch_size"],epochs=compute_pars["epochs"])
  

  return model

def metrics(ytrue, ypred, data_pars=None, out_pars=None, **kw):
    """
       Return metrics 
    """
    ytrue = np.argmax(ytrue, axis=1)
    ypred = np.argmax(ypred, axis=1)
    true_count = sum(ytrue == ypred)
    ddict= true_count / len(ytrue)
     
    return ddict

def predict(model, Xtest, ytest, data_pars=None, out_pars=None, compute_pars=None, **kw):
     
  ##### Get Data ###############################################
  

  #### Do prediction
  ypred = model.predict(Xtest)

  ### Save Results
  df = pd.DataFrame(list(zip(ytest, ypred)), 
               columns =['y_real', 'y_pred']) 
  print(df)
  df.to_csv(out_pars["out_path"], index = False, header=True)
  ### Return val
  if compute_pars["return_pred"]:  
    return ypred


def reset_model():
  pass

def save(model, path) :
    model.save(path)  # creates a HDF5 file 
    del model  # deletes the existing model

def load(path) :
  model = Model()
  model.model = None
  return model   

def test(data_path="dataset/" ,pars_choice="json", reset=True):
    ###loading the command line arguments
    ###CNN-non-static

    log("#### Loading params   #################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice, data_path= data_path)
                                                                       
    log("#### Loading train data   #############")
    loader = data_loader(data_pars)
    data_prov = data_provider(loader, data_pars)
    Xtrain, ytrain, index2word = data_prov.get_dataset()
    
    log("#### Loading test data   ###############")
    data_pars["train"]=False 
    Xtest, ytest =  data_prov.get_dataset()

    log("#### Load_wors2vec Module init   ########")
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(data_pars["word2vec_model_path"], binary=True)
    
    log("#### create word embedding  ##############")
    word_embedding, _ = get_pre_train_word2vec(word2vec_model, index2word, len(index2word))
    

    log("#### CNN-non-static Model init   #########")
    model_ = Model(word_embedding, len(index2word), model_pars)
    log(model_)
    model= model_.model()

    log("#### Fit   ###############################")
    history = fit(model, Xtrain, ytrain, compute_pars)
    print("precision", history)

    log("#### Predict   ############################")
    ypred = predict(model, Xtest, ytest, data_pars, out_pars, compute_pars)
    print("ypred", ypred)
    


    log("#### Get  metrics   #######################")
    metrics_val = metrics(ytest, ypred, compute_pars, out_pars)
    print("precision", metrics_val)



    log("#### Save/Load   ###########################")
    save(model, out_pars['modelpath'])
    model2 = load(out_pars['modelpath'])
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)

def test2(data_path="dataset/" ,pars_choice="json", reset=True):
    ###loading the command line arguments
   

    log("#### Loading params   #######################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice, data_path=data_path)
#                                                                      
    
    log("#### Loading train data   ####################")
    loader = data_loader(data_pars)
    data_prov = data_provider(loader, data_pars)
    Xtrain, ytrain, index2word = data_prov.get_dataset()
    
    log("#### Loading test data   #####################")
    data_pars["train"]=False 
    Xtest, ytest =  data_prov.get_dataset()

    log("#### CNN-rand Model init   ###################")
    model_ = Model(None, len(index2word), model_pars)
    log(model_)
    model= model_.model()
    
    log("#### Fit   ###################################")
    history = fit(model, Xtrain, ytrain, compute_pars)
    print(history)

    log("#### Predict   ###############################")
    ypred = predict(model, Xtest, ytest, data_pars, out_pars, compute_pars)
    print("ypred", ypred)
    


    log("#### Get  metrics   ##########################")
    metrics_val = metrics(ytest, ypred, compute_pars, out_pars)
    print("precision", metrics_val)

    log("#### Save/Load   ##############################")
    save(model, out_pars['modelpath'])
    model2 = load(out_pars['modelpath'])

    print(model2)


if __name__ == '__main__':
    VERBOSE = True   
    ### Local
    test(pars_choice= "test01")
    test(pars_choice= "json")
    test2(pars_choice= "test01")
    test2(pars_choice= "json")

