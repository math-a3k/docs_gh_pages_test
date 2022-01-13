""""
Related to data procesisng

TO DO :
   Normalize datasetloader and embedding loading


1) embeddings can be trainable or fixed  : True
2) embedding are model data, not not split train/test 





"""
import os, gc, copy, sys
from pathlib import Path
import pandas as pd, numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from mlmodels.util import path_norm, log
from mlmodels.util import load_function_uri as load_function

###############################################################################################################
def log2(*v, **kw) :
  if VERBOSE : log(*v, **kw)

VERBOSE = True



###############################################################################################################
###############################################################################################################
def torch_datasets_wrapper(sets, args_list = None, **args):
    """

    """
    if not isinstance(sets,list) and not isinstance(sets,tuple):
        sets = [sets]
    import torch
    if args_list is None:
        return [torch.utils.data.DataLoader(x,**args) for x in sets]
    return [torch.utils.data.DataLoader(x,**a,**args) for a,x in zip(args_list,sets)]



def pandas_reader(task, path, colX, coly, path_eval=None, train_size=0.8):
   """ Simple loader for tabular dataset
                    "uri"  : "mlmodels.preprocess.generic.pandas_reader",
                "args"  : {
                           "task" : "train", 
                           "path" : "dataset/tabular/titanic_train_preprocessed.csv",
                           "colX" : ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S", "Title"],
                           "coly" : "Survived",
                           "path_eval" : "",
                           "train_size" : 0.8,
                          },
                "out"   :  [  "Xtrain", "ytrain", "Xtest", "ytest" ]  
   
   """
   df = pd.read_csv(path)
   
   if task == "train":
      from sklearn.model_selection import train_test_split
      if path_eval is None :
         dftrain, dftest = train_test_split(df, train_size = train_size)
      else :
         dftrain = df
         dftest  = pd.read_csv(path_eval)
         del df; gc.collect()

      Xtrain, ytrain = dftrain[colX], dftrain[coly]         
      Xtest, ytest   = dftest[colX], dftest[coly]
      return Xtrain, ytrain,Xtest, ytest,
      
   if task == "eval":
      Xtest, ytest = df[colX], df[coly]
      return Xtest, ytest
   
   if task == "eval":
      X= df[colX]
      return X
   else :
      raise Exception(" {task} is unknown task")
   
   
   
def tf_dataset_download(data_info, **args):
    """
       Save in numpy compressez format TF Datasets
       data_info ={ "dataset" : "mnist", "batch_size" : 5000,"data_path" : "dataset/vision/mnist2/"}
        args{"n_train": 500, "n_test": 500 }
       tf_dataset_download(dataset_pars)            
   
   """
    import tensorflow_datasets as tfds
    import numpy as np
    

    dataset    = data_info.get("dataset", "")
    out_path   = data_info.get("data_path", "")
    n_train    = args.get("train_samples", 500)
    n_test     = args.get("test_samples", 50)
    batch_size = args.get("batch_size", 10)

   
    if len(dataset) < 1 or len(out_path) < 1 :
        raise Exception("['data_path','dataset'] is required field, please add these ['data_path','dataset'] in data_info") 
    log(dataset)
  
  
    dataset  = dataset.lower()
    out_path = path_norm(out_path)
    if dataset.find('.') > -1:
        dataset = dataset.split('.')[0]
        
    name     = dataset  #.replace(".","-")
    os.makedirs(out_path, exist_ok=True)
    log("Dataset Name is : ", name)
 
 
    #### Split is more Complex in TF Dataset
    train_ds = tfds.as_numpy( tfds.load(dataset, split= f"train[0:{n_train}]") )
    test_ds  = tfds.as_numpy( tfds.load(dataset, split= f"test[0:{n_test}]") )
    # val_ds  = tfds.as_numpy( tfds.load(dataset_id, split= f"test[0:{n_test}]", batch_size=batch_size) )
    # log("train", train_ds.shape )
    # log("test",  test_ds.shape )
 

    def get_keys(x):  ### TF Dataset
        if "image" in x.keys() : xkey = "image"
        if "text" in x.keys() : xkey = "text"    
        return xkey
 

    log("##############", "Saving train dataset", "###############################")
    Xtemp, ytemp = [], []
    for x in train_ds:
        #log(x)
        xkey =  get_keys(x)
        Xtemp.append(x[xkey])
        ytemp.append(x.get('label')) 
    Xtemp = np.array(Xtemp)
    ytemp = np.array(ytemp)  ### Beware of None
    
    trainPath = os.path.join(out_path,'train')
    os.makedirs(trainPath, exist_ok=True)
    np.savez_compressed(os.path.join(trainPath, f"{name}") , X = Xtemp, y = ytemp )    
 

    log("##############", "Saving test dataset", "###############################") 
    Xtemp, ytemp = [], []
    for x in test_ds:
        #log(x)
        Xtemp.append(x[xkey])
        ytemp.append(x.get('label'))
    Xtemp = np.array(Xtemp)
    ytemp = np.array(ytemp)

    testPath = os.path.join(out_path,'test')
    os.makedirs(testPath, exist_ok=True)
    ### Multiple files
    np.savez_compressed(os.path.join(testPath, f"{name}"), X = Xtemp, y = ytemp)
    
    log("Saved", out_path, os.listdir( out_path ))



 
def get_dataset_torch(data_info, **args):
    """
      From URI path, get dataloader for Pytorch Models.
      Use Transform if necessary
      Return train_loader, valid_loader


      torchvison.datasets
         MNIST Fashion-MNIST KMNIST EMNIST QMNIST  FakeData COCO Captions Detection LSUN ImageFolder DatasetFolder 
         ImageNet CIFAR STL10 SVHN PhotoTour SBU Flickr VOC Cityscapes SBD USPS Kinetics-400 HMDB51 UCF101 CelebA

      torchtext.datasets
         Sentiment Analysis:    SST IMDb Question Classification TREC Entailment SNLI MultiNLI 
         Language Modeling:     WikiText-2 WikiText103  PennTreebank 
         Machine Translation :  Multi30k IWSLT WMT14 
         Sequence Tagging    :  UDPOS CoNLL2000Chunking 
         Question Answering  :  BABI20

    """
    
    target_transform_info = args.get('target_transform', None)
    transform_info        = args.get('transform', None)
    to_image              = args.get('to_image', True)
    shuffle               = args.get('shuffle', True)
    dataloader            = args.get("dataloader", "torchvision.datasets:MNIST")
 
    dataset    = data_info.get("dataset", None)
    data_path  = data_info.get("data_path", None)
    train      = data_info.get("train", True)
    batch_size = data_info.get("batch_size", 1)
    data_type  = data_info.get('data_type', "tch_dataset")
 
    if not dataset or not data_path:
        raise Exception("please add these 'data_path','dataset' in data_info")
 
    log("#### If transformer URI is Provided", transform_info)
    transform = None
    if transform_info :
        transform_uri = transform_info.get("uri", "mlmodels.preprocess.image:torch_transform_mnist" )
        try:
            transform_args = transform_info.get("args", None)
            trans_pass     = transform_info.get("pass_data_pars", False)   

            if trans_pass:   ### Maybe no need,  transform_args is not None ???
               transform = load_function(transform_uri)(**transform_args)
            else:
               transform = load_function(transform_uri)()
               
        except Exception as e :
            transform = None
            print("transform", e)
 
    log("#### Loading dataloader URI")           
    dset = load_function(dataloader)
    log("dataset : ",dset)           

    # dset = load_function(d.get("dataset", "torchvision.datasets:MNIST") ) 



    if data_type != "tch_dataset":
        ###### Custom Build Dataset   ####################################################
        
        ### Romove conflict(Duplicate) Arguments  
        entriesToRemove = ('download', 'transform','target_transform')
        for k in entriesToRemove:
            args.pop(k, None)

        dset_inst    = dset(os.path.join(data_path,'train'), train=True, download=True, transform=transform, data_info=data_info, **args)
        # train_loader = DataLoader( dset_inst, batch_size=batch_size, shuffle= shuffle)
        train_loader = Custom_DataLoader( dset_inst, batch_size=batch_size, shuffle= shuffle)
        
        dset_inst    = dset(os.path.join(data_path,'test'), train=False, download=False, transform=transform, data_info=data_info, **args)
        # valid_loader = DataLoader( dset_inst, batch_size=batch_size, shuffle=shuffle)
        valid_loader = Custom_DataLoader( dset_inst, batch_size=batch_size, shuffle=shuffle)

    else :
        ###### Pre Built Dataset available  #############################################
        dset_inst    = dset(data_path, train=True, download=True, transform=transform)
        # train_loader = DataLoader( dset_inst, batch_size=batch_size, shuffle=shuffle)
        train_loader = Custom_DataLoader( dset_inst, batch_size=batch_size, shuffle=shuffle)
        
        dset_inst    = dset(data_path, train=False, download=False, transform=transform)
        # valid_loader = DataLoader( dset_inst, batch_size=batch_size, shuffle=shuffle)
        valid_loader = Custom_DataLoader( dset_inst, batch_size=batch_size, shuffle=shuffle)


    return train_loader, valid_loader  



####Not Yet tested
def get_dataset_keras(data_info, **args):

    """"
   #### Write someple
   from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataloader
   from mlmodels.preprocess.keras_dataloader.dataset import Dataset as kerasDataset
 
   class TensorDataset(kerasDataset):
 
       def __getitem__(self, index):
           # time.sleep(np.random.randint(1, 3))
           return np.random.rand(3), np.array([index])
 
       def __len__(self):
           return 100
           
   model.compile('adam', loss='mse')
   data_loader = kerasDataloader(TensorDataset(), batch_size=20, num_workers=0)
   model.fit_generator(generator=data_loader, epochs=1, verbose=1)
 
 
   ##### MNIST case : TorchVison TorchText Pre-Built
   "dataset"       : "torchvision.datasets:MNIST"
   "transform_uri" : "mlmodels.preprocess.image:torch_transform_mnist"
 
 
   ##### Pandas CSV case : Custom MLMODELS One
   "dataset"        : "mlmodels.preprocess.generic:pandasDataset"
   "transform_uri"  : "mlmodels.preprocess.text:torch_fillna"
 
 
   ##### External File processor :
   "dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
   "transform_uri"  : "MyFolder/preprocess/myfile.py:torch_fillna"
 
 
   """
    from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataLoader
    

    target_transform_info = args.get('target_transform', None)
    transform_info        = args.get('transform', None)
    shuffle               = args.get('shuffle', True)
    dataloader            = args.get("dataloader", "mlmodels.preprocess.datasets:MNIST")
 
    dataset               = data_info.get("dataset", None)
    data_path             = data_info.get("data_path", None)
    train                 = data_info.get("train", True)
    batch_size            = data_info.get("batch_size", 1)
   
    transform = None
    if transform_info :
        transform_uri = transform_info.get("uri", "mlmodels.preprocess.image:keras_transform_mnist")
        try:
            transform_args = transform_info.get("args", None)
            trans_pass     = transform_info.get("pass_data_pars", False)
            if trans_pass:
               transform = load_function(transform_uri)(**transform_args)
            else:
               transform = load_function(transform_uri)()
        except Exception as e :
            transform = None
            print(e)
 

    #### from mlmodels.preprocess.image import pandasDataset
    ### Duplicate from preivous one
    dset = load_function(data_info.get("dataset", "mlmodels.preprocess.datasets:MNIST") )
 
 
    ######  Dataset Downloader  #############################################
    dset_inst    = dset(data_path, train=True, download=True, transform= transform, data_info = data_info, **args)
    train_loader = kerasDataLoader( dset_inst, batch_size=batch_size, shuffle= shuffle)
 
    dset_inst    = dset(data_path, train=False, download=False, transform= transform,  data_info = data_info, **args)
    valid_loader = kerasDataLoader( dset_inst, batch_size=batch_size, shuffle= shuffle)
 
 
    return train_loader, valid_loader




def get_model_embedding(data_info, **args):
    """"
     Mostly Embedding data, it can be external data used in the model.
 
     INDEPENDANT OF Framework BUT Follows PyTorch Logic
 
   ##### MNIST case : TorchVison TorchText Pre-Built
   "dataset"       : "torchvision.datasets:MNIST"
   "transform_uri" : "mlmodels.preprocess.image:torch_transform_mnist"
 
 
   ##### Pandas CSV case : Custom MLMODELS One
   "dataset"        : "mlmodels.preprocess.generic:pandasDataset"
   "transform_uri"  : "mlmodels.preprocess.text:torch_fillna"
 
 
   ##### External File processor :
   "dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
   "transform_uri"  : "MyFolder/preprocess/myfile.py:torch_fillna"
 
 
   """
   
    model_pars = args.get("model_pars",{})
    d          = model_pars

    # args       = args("args", {})
    train      = args.get('train', True)
    download   = args.get("download", True)


    log("############## Get mbedding Loader URI  ")
    transform = None
    if  len(args.get("embedding_transform_uri", ""))  > 1 :
        transform = load_function( d.get("embedding_transform_uri", "mlmodels.preprocess.text:torch_transform_glove" ))()
 
 
    log("#### Get Embedding t embeddingLoader ")
    dset = load_function(d.get("embedding_dataset", "torchtext.embedding:glove") )
 
    dloader = None
    if len(d.get('embedding_path', "")) > 1 :
        ###### Custom Build Dataset   ####################################################
        dloader    = dset(d['embedding_path'], train=train, download=download, transform= transform, model_pars=model_pars, 
                          args = args,  data_info = data_info)
        
    else :
        ###### Pre Built Dataset available  #############################################
        dloader    = dset(d['embedding_path'], train=train, download=download, transform= transform)
 
 
    return dloader
 


class Custom_DataLoader:
    """
      Mini Batch Dataloader to overload PyTorch one

    """
    def __init__(self, dataset=None, batch_size=-1, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.drop_last = drop_last
        if batch_size == -1:
            self.batch_size = len(self.dataset)
        else:
            self.batch_size = batch_size
        assert self.batch_size > 0, "batch_size must not be negative, except -1 for full memory load."
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            index_generator = iter(np.random.permutation(len(self.dataset)))
        else:
            index_generator = iter(range(len(self.dataset)))
            
        batch = []
        while(True):
            try:
                batch.append(self.dataset[index_generator.__next__()])
                if len(batch) == self.batch_size:
                    output = list(zip(*batch))  # Outputs resolved this way for (image, label) pairs
                    output[1] = torch.tensor(output[1])
                    yield output 
                    batch=[]
                    
            except StopIteration:
                if not self.drop_last:
                    output = list(zip(*batch)) # Outputs resolved this way for (image, label) pairs
                    output[1] = torch.tensor(output[1])
                    yield output
                break
            except:
                break
    
    def __len__(self):
            length = round(len(self.dataset) / self.batch_size)
            if self.drop_last:
                length = length - 1
            return length




class pandasDataset(Dataset):
    """
   Defines a dataset composed of sentiment text and labels
   Attributes:
       df (Dataframe): Dataframe of the CSV from teh path
       sample_weights(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
       data (list[int, [int]]): The data in the set

   """
   
    def __init__(self, root="", train=True, transform=None, target_transform=None, 
                 download=False, data_info={}, **args):
        # import torch

        self.train            = train
        self.transform        = transform
        self.target_transform = target_transform
        self.download         = download
        self.data_info        = data_info
        

        if len(data_info) < 1:
            raise Exception("'data_info' is required fields in pandasDataset")
        
        dataset = data_info.get('dataset', "")
        if len(dataset) < 1:
           raise Exception("'dataset' is required field, please add it in data_info key", dataset)
        
        if dataset.find('.') > -1:
            dataset_name = dataset.split('.')[0]
            filename = dataset
        else:
            dataset_name = dataset
            filename = dataset if dataset.find('.') > -1 else dataset + '.csv'  ## CSV file

        if len(root) > 0: # root has a value
            path =  root # os.path.join(root,'train' if train else 'test') #TODO: need re-organize dataset later
        else:
            path = data_info.get("data_path","")



        #### DataFrame Load  ##############################################
        # df = torch.load(os.path.join(path, filename))
        file_path = path_norm(os.path.join(path, filename))

        if not os.path.exists(file_path):
            # file_path = path_norm(os.path.join(path, dataset_name, 'train.csv' if train else 'test.csv'))
            file_path = path_norm(os.path.join(path, dataset_name, filename if train else filename))
            if not os.path.exists(file_path):
                # file_path = path_norm(os.path.join(path, dataset_name, 'train/train.csv' if train else 'test/test.csv'))
                file_path = path_norm(os.path.join(path, dataset_name, 'train' if train else 'test', filename))
        df = pd.read_csv(file_path, **args.get("read_csv_parm",{}))
        self.df = df

 
        #### Split  #######################################################
        colX   = args.get('colX', list( df.columns) )  ### All columns
        coly   = args.get('coly', [])
        X      = df[ colX ]
        labels = df[ coly ]

 
        #### Compute sample weights from inverse class frequencies, pytorch not needed.
        import torch
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight) 
        # BUG weight[labels] >> IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

 
        #### Data Joining  ################################################
        self.data = list(zip(X, labels))
 
 
    def __len__(self):
        return len(self.data)
 
 
    def __getitem__(self, index):
        """
       Args:
           index (int): Index
       Returns:
           tuple: (image, target) where target is index of the target class.
       """
        X, target = self.data[index], int(self.targets[index])
                        
        if self.transform is not None:
            X = self.transform(X)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
 
        return X, target
 
    def shuffle(self, frac=1.0, random_state=123):
        self.df = self.df.sample(frac=frac, random_state=random_state)
        

    def get_data(self):
        #### TODO : Mini Batch Load 
        return self.df



from PIL import Image
class NumpyDataset(Dataset):
    """
  Defines a dataset composed of Features and labels
 
  Attributes:
      data_pars{
          data_path: the folder path that it cotains numpy files
          transforms: operation you wanna apply on image
 
          example:
              data_info= {'data_path': 'mlmodels/dataset/vision/cifar10/', 'dataset':'clfar10'}
               args = { 'to_image':True}
                                   
      }        
  """
    
 
    def __init__(self, root="", train=True, transform=None, target_transform=None,
                 download=False, data_info={}, **args):
        
        if len(data_info) < 1:
            raise Exception("'data_info' is required fields in NumpyDataset")
            
            
        dataset = data_info.get('dataset', None)
        if not dataset:
            raise Exception("'dataset' is required field, please add it in data_info key")
        
        try:
            dataset = dataset.lower()
        except:
            raise Exception(f"Datatype error 'dataset': {dataset}")
        
        if len(root) == 0:
            root      = data_info.get('data_path', "")
            root      = os.path.join(root,'train' if train else 'test')
           
        self.target_transform = target_transform
        self.transform  = transform
        self.to_image   = args.get('to_image', True)
 
       
        # file_path      =   os.path.join(root,'train' if train else 'test', f"{dataset}.npz")
        if not f"{dataset}".endswith(".npz"):  # TODO: re-organize train test dataset folder later
            file_path   = os.path.join(root, f"{dataset}.npz")
        else:
            file_path   = os.path.join(root, dataset)
        
        file_path = path_norm( file_path)
        print("Dataset File path : ", file_path)
        data            = np.load( file_path,**args.get("numpy_loader_args", {}))
        
        
        if data_info.get("data_type",None) == 'tf_dataset':
            self.features   = data['X']
            self.classes    = data['y']
        
        self.data = tuple(data[x] for x in sorted(data.files))
        data.close()
 
 
    def __getitem__(self, index):
 
        X, y = self.features[index], self.classes[index]
        # X =  np.stack((X, X, X)) # gray to rgb 64x64 to 3x64x64
 
        if self.to_image :
            X = Image.fromarray(np.uint8(X))
 
        if self.transform is not None:
            X = self.transform(X)
 
        if self.target_transform is not None:
            y = self.target_transform(y)
 
        return X, y
 
    def __len__(self):
        return len(self.features)


    def get_data(self):
        return self.data






def get_model_embedding(data_info, **args):
    """"
     Get Embedding data to be used in the model, it can be external data used in the model.
     INDEPENDANT OF Framework BUT Follows PyTorch Logic

     embedding_transform_uri
     embedding_path
 
 
   ##### MNIST case : TorchVison TorchText Pre-Built
   "dataset"       : "torchvision.datasets:MNIST"
   "transform_uri" : "mlmodels.preprocess.image:torch_transform_mnist"
 
 
   ##### Pandas CSV case : Custom MLMODELS One
   "dataset"        : "mlmodels.preprocess.generic:pandasDataset"
   "transform_uri"  : "mlmodels.preprocess.text:torch_fillna"
 
 
   ##### External File processor :
   "dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
   "transform_uri"  : "MyFolder/preprocess/myfile.py:torch_fillna"
 
 
   """
   
    model_pars = args.get("model_pars",{})
    d          = model_pars

    # args       = args("args", {})
    train      = args.get('train', True)
    download   = args.get("download", True)


    log("############## Get Embedding Loader URI  #############################")
    transform = None
    if  len(args.get("embedding_transform_uri", ""))  > 1 :
        transform = load_function( d.get("embedding_transform_uri", "mlmodels.preprocess.text:torch_transform_glove" ))()
 
 
    log("#### Get Embedding t embeddingLoader #################################")
    dset = load_function(d.get("embedding_dataset", "torchtext.embedding:glove") )
 
    dloader = None
    if len(d.get('embedding_path', "")) > 1 :
        ###### Custom Build Dataset   ####################################################
        dloader    = dset(d['embedding_path'], train=train, download=download, transform= transform, model_pars=model_pars, 
                          args = args,  data_info = data_info)
        
    else :
        ###### Pre Built Dataset available  #############################################
        dloader    = dset(d['embedding_path'], train=train, download=download, transform= transform)
 
 
    return dloader
 
 










def text_create_tabular_dataset(path_train, path_valid,   lang='en', pretrained_emb='glove.6B.300d'):
    import spacy
    import torchtext
    from torchtext.data import Field
    from torchtext.data import TabularDataset
    from torchtext.vocab import GloVe
    from torchtext.data import Iterator, BucketIterator
    import torchtext.datasets
    from time import sleep
    import re


    def clean_str(string):
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
        return string.strip()


    #### Tokenizer  ################################################
    disable = [ 'tagger', 'parser', 'ner', 'textcat'
        'entity_ruler', 'sentencizer', 
        'merge_noun_chunks', 'merge_entities',
        'merge_subtokens']
    try :
      spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)

    except :
       #### Very hacky to get Glove Data 
       log( f"Download {lang}")
       os.system( f"python -m spacy download {lang}")
       sleep(60)
       spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  

    def tokenizer(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]


    # Creating field for text and label  ###########################
    TEXT  = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False)


    log('Preprocessing the text...')
    TEXT.preprocessing = torchtext.data.Pipeline(clean_str)



    log('Creating tabular datasets...It might take a while to finish!')
    train_datafield = [('text', TEXT), ('label', LABEL)]
    tabular_train   = TabularDataset(path=path_train, format='csv', skip_header=True, fields=train_datafield)

    valid_datafield = [('text', TEXT), ('label', LABEL)]
    tabular_valid   = TabularDataset(path=path_valid, format='csv', skip_header=True, fields=valid_datafield)


    log('Building vocaulary...')
    TEXT.build_vocab(tabular_train, vectors=pretrained_emb)
    LABEL.build_vocab(tabular_train)


    return tabular_train, tabular_valid, TEXT.vocab






def create_kerasDataloader():
    """
    keras dataloader
    DataLoader for keras

    Usage example
    from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataloader
    from mlmodels.preprocess.keras_dataloader.dataset import Dataset as kerasDataset


    class TensorDataset(kerasDataset):

        def __getitem__(self, index):
            # time.sleep(np.random.randint(1, 3))
            return np.random.rand(3), np.array([index])

        def __len__(self):
            return 100
            
    model = Sequential()
    model.add(Dense(units=4, input_dim=3))
    model.add(Dense(units=1))
    model.compile('adam', loss='mse')

    data_loader = kerasDataGenerator(TensorDataset(), batch_size=20, num_workers=0)

    model.fit_generator(generator=data_loader, epochs=1, verbose=1)
    """
    #### Write someple
    from mlmodels.preprocess.keras_dataloader.dataloader import DataGenerator as kerasDataloader
    from mlmodels.preprocess.keras_dataloader.dataset import Dataset as kerasDataset



    class TensorDataset(kerasDataset):

        def __getitem__(self, index):
            # time.sleep(np.random.randint(1, 3))
            return np.random.rand(3), np.array([index])

        def __len__(self):
            return 100
            
    #model = Sequential()
    #model.add(Dense(units=4, input_dim=3))
    #model.add(Dense(units=1))
    #model.compile('adam', loss='mse')

    data_loader = kerasDataloader(TensorDataset(), batch_size=20, num_workers=0)

    return data_loader
    # model.fit_generator(generator=data_loader, epochs=1, verbose=1)





########################################################################################
########################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test
    from jsoncomment import JsonComment ; json = JsonComment()
    log("#### Test unit Dataloader/Dataset   ####################################")
    
    js = json.load(open(path_norm(data_path), mode='r'))

    for key,ddict in js.items() :
      log("\n\n\n", "#"*20, key)
      try :
        data_info = ddict['data_pars']['data_info']

        ###Only ONE TASK  !!!!
        prepro    = ddict['data_pars']['preprocessors'][0]   
        uri       = prepro['uri']
        args      = prepro['args']

        log(key, uri, args)
        ffun = load_function(uri)
        ffun = ffun(**args, data_info = data_info)
      except Exception as e:
        log("[Error] ", key, e)
      



if __name__ == "__main__":
    test(data_path="dataset/test_json/test_preprocess-generic.json", pars_choice="", config_mode="")











'''
class numpyDataset(Dataset):
    """
    Defines a dataset composed of sentiment text and labels
    Attributes:
        X: numpy tensor of the path
        y: numpy for labels
        sample_weights(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
        data (list[int, [int]]): The data in the set


    """
   
    def __init__(self,root="", train=True, transform=None, target_transform=None,
                 download=False, data_pars=None, ):
        import torch
        import numpy as np
        self.data_pars        = data_pars
        self.transform        = transform
        self.target_transform = target_transform
        self.download         = download
        d = data_pars


        path = d['train_path'] if train else d['test_path']
        #filename = d['X_filename'], d['y_filename']
        #colX =d['colX']


        # df = torch.load(os.path.join(path, filename))
        X      = np.load(os.path.join(path, d['X_filename']))
        labels = np.load(os.path.join(path, d['y_filename'] )) 
        # self.X = X
        # self.labels = labels


        #### Split  ####################
        #X = df[ colX ]
        #labels = df[ d["coly"] ]


        #### Compute sample weights from inverse class frequencies
        class_sample_count = np.unique(labels, return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight[labels])


        #### Data Joining  ############
        self.data = list(zip(X, labels))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        X, target = self.data[index], int(self.targets[index])


        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target

'''
