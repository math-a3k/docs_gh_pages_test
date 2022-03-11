# coding: utf-8
"""
https://github.com/leohsuofnthu/Pytorch-TextCNN/blob/master/textCNN_IMDB.ipynb



"""

import os
from jsoncomment import JsonComment ; json = JsonComment()
import shutil
from pathlib import Path
from time import sleep
import re

import pandas as pd
from numpy.random import RandomState

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator
import torchtext.datasets

import spacy


####################################################################################################
import mlmodels.models as M

from mlmodels.util import  os_package_root_path, log, path_norm, get_model_uri

VERBOSE = False
MODEL_URI = get_model_uri(__file__)




###########################################################################################################
###########################################################################################################
def _train(m, device, train_itr, optimizer, epoch, max_epoch):
    """function _train
    Args:
        m:   
        device:   
        train_itr:   
        optimizer:   
        epoch:   
        max_epoch:   
    Returns:
        
    """
    m.train()
    corrects, train_loss = 0.0,0
    for batch in train_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text,0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        optimizer.zero_grad()
        logit = m(text)
        
        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(train_itr.dataset)
    train_loss /= size 
    accuracy = 100.0 * corrects/size
  
    return train_loss, accuracy
    
def _valid(m, device, test_itr):
    """function _valid
    Args:
        m:   
        device:   
        test_itr:   
    Returns:
        
    """
    m.eval()
    corrects, test_loss = 0.0,0
    for batch in test_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text,0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        
        logit = m(text)
        loss = F.cross_entropy(logit, target)

        
        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(test_itr.dataset)
    test_loss /= size 
    accuracy = 100.0 * corrects/size
    
    return test_loss, accuracy

def _get_device():
    """function _get_device
    Args:
    Returns:
        
    """
    # use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # return device
    return "cpu"


def get_config_file():
    """function get_config_file
    Args:
    Returns:
        
    """
    return os.path.join(
        os_package_root_path(__file__, 1),
        'config', 'model_tch', 'textcnn.json')


def get_data_file():
    """function get_data_file
    Args:
    Returns:
        
    """
    return os.path.join(
        os_package_root_path(__file__), 'dataset', 'IMDB_Dataset.txt')


def analyze_datainfo_paths(data_info):
    """function analyze_datainfo_paths
    Args:
        data_info:   
    Returns:
        
    """
    data_path  = path_norm(data_info.get("data_path", None))
    dataset    = data_info.get("dataset", None)

    if not dataset or not data_path:
        raise Exception("please add these 'data_path','dataset' in data_info")
    
    if dataset.find('.') > -1:
        dataset_name = dataset.split('.')[0]
    else:
        raise Exception("please add dataset Extension like that : dataset.txt")
    
    path_train = os.path.join(data_path, 'train')
    os.makedirs(path_train, exist_ok=True)

    path_train_dataset = os.path.join(path_train, dataset_name + '.csv')
    

    path_valid = os.path.join(data_path, 'test')
    os.makedirs(path_valid, exist_ok=True)

    path_valid_dataset = os.path.join(path_valid, dataset_name + '.csv')

    dataset_path = os.path.join(data_path, dataset)

    return dataset_path, path_train_dataset, path_valid_dataset

# def split_train_valid(path_data, path_train, path_valid, frac=0.7):
#     df = pd.read_csv(path_data)
#     rng = RandomState()
#     tr = df.sample(frac=0.7, random_state=rng)
#     tst = df.loc[~df.index.isin(tr.index)]
#     print("Spliting original file to train/valid set...")
#     tr.to_csv(path_train, index=False)
#     tst.to_csv(path_valid, index=False)


def split_train_valid(data_info, **args):
    """function split_train_valid
    Args:
        data_info:   
        **args:   
    Returns:
        
    """
    frac= args.get("frac", 0.7)

    dataset_path, path_train_dataset, path_valid_dataset = analyze_datainfo_paths(data_info)

    df = pd.read_csv(dataset_path)

    rng = RandomState()
    train_data = df.sample(frac=frac, random_state=rng)
    test_data = df.loc[~df.index.isin(train_data.index)]

    print("Spliting original file to train/valid set...")
    train_data.to_csv(path_train_dataset, index=False)
    test_data.to_csv(path_valid_dataset, index=False)

    return train_data, test_data



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


def create_tabular_dataset(data_info, **args):
    """function create_tabular_dataset
    Args:
        data_info:   
        **args:   
    Returns:
        
    """
    disable = [
        'tagger', 'parser', 'ner', 'textcat'
        'entity_ruler', 'sentencizer', 
        'merge_noun_chunks', 'merge_entities',
        'merge_subtokens']
    
    lang= args.get('lang','en')
    pretrained_emb= args.get('pretrained_emb', 'glove.6B.300d')

    _, path_train_dataset, path_valid_dataset = analyze_datainfo_paths(data_info)

    try :
      spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)

    except :
       log( f"Download {lang}")
       import importlib

       os.system( f"python -m spacy download {lang}")
       spacy_en = importlib.import_module(f'{lang}_core_web_sm').load( disable= disable) 
       
    #    sleep(60)
    #    spacy_en = spacy.load( f'{lang}_core_web_sm', disable= disable)  


    def tokenizer(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # Creating field for text and label
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False)

    print('Preprocessing the text...')
    # clean the text
    TEXT.preprocessing = torchtext.data.Pipeline(clean_str)

    print('Creating tabular datasets...It might take a while to finish!')
    train_datafield = [('text', TEXT), ('label', LABEL)]
    tabular_train = TabularDataset(
        path=path_train_dataset, format='csv',
        skip_header=True, fields=train_datafield)

    valid_datafield = [('text', TEXT), ('label', LABEL)]

    tabular_valid = TabularDataset(path=path_valid_dataset, 
                                   format='csv',
                                   skip_header=True,
                                   fields=valid_datafield)

    print('Building vocaulary...')
    TEXT.build_vocab(tabular_train, vectors=pretrained_emb)
    LABEL.build_vocab(tabular_train)

    return tabular_train, tabular_valid, TEXT.vocab


# def create_data_iterator(tr_batch_size, val_batch_size, tabular_train, tabular_valid, d):
#     # Create the Iterator for datasets (Iterator works like dataloader)

#     train_iter = Iterator(tabular_train, batch_size=tr_batch_size, device=d, sort_within_batch=False, repeat=False)
#     valid_iter = Iterator(tabular_valid, batch_size=val_batch_size, device=d, sort_within_batch=False, repeat=False) 

#     return train_iter, valid_iter


def create_data_iterator(batch_size, tabular_train, tabular_valid, d):
    """function create_data_iterator
    Args:
        batch_size:   
        tabular_train:   
        tabular_valid:   
        d:   
    Returns:
        
    """
    # Create the Iterator for datasets (Iterator works like dataloader)

    train_iter = Iterator(tabular_train, batch_size=batch_size, device=d, sort_within_batch=False, repeat=False)
    valid_iter = Iterator(tabular_valid, batch_size=batch_size, device=d, sort_within_batch=False, repeat=False) 

    return train_iter, valid_iter



###########################################################################################################
###########################################################################################################
class TextCNN(nn.Module):

    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        """ TextCNN:__init__
        Args:
            model_pars:     
            data_pars:     
            compute_pars:     
            **kwargs:     
        Returns:
           
        """
        print(model_pars)
        kernel_wins = [int(x) for x in model_pars["kernel_height"]]
        super(TextCNN, self).__init__()
        
        # load pretrained embedding in embedding layer.
        emb_dim  = model_pars.get("emb_dim", 300) 
        self.emb_size = model_pars.get("emb_size", 20000)

        # emb_dim = 300
        #self.embed = nn.Embedding(20000, emb_dim)
        # self.embed.weight.data.copy_(vocab_built.vectors)

        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, model_pars["dim_channel"], (w, emb_dim))
             for w in kernel_wins])
        # Dropout layer
        self.dropout = nn.Dropout(model_pars["dropout_rate"])

        # FC layer
        self.fc = nn.Linear(
            len(kernel_wins) * model_pars["dim_channel"], model_pars["num_class"])


    def rebuild_embed(self, vocab_built):
        """ TextCNN:rebuild_embed
        Args:
            vocab_built:     
        Returns:
           
        """
        self.embed = nn.Embedding(*vocab_built.vectors.shape)
        self.embed.weight.data.copy_(vocab_built.vectors)

        # print("vocab_built.vectors shape : ", *vocab_built.vectors.shape)


    def forward(self, x):
        """ TextCNN:forward
        Args:
            x:     
        Returns:
           
        """
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]

        fc_x = torch.cat(pool_x, dim=1)

        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit



#Model = TextCNN
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None ):
        """ Model:__init__
        Args:
            model_pars:     
            data_pars:     
            compute_pars:     
        Returns:
           
        """

        self.model_pars   = model_pars
        self.data_pars    = data_pars
        self.compute_pars = compute_pars

        ### Model Structure        ################################
        if model_pars is None :
            self.model = None
            return None

        self.model = TextCNN(model_pars, data_pars, compute_pars, )



def evaluate(model, session=None, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    """function evaluate
    Args:
        model:   
        session:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    # return metrics on full dataset
    device = _get_device()
    data_pars = data_pars.copy()
    data_pars.update(frac=1)
    test_iter, _, vocab = get_dataset(data_pars, out_pars)
    model.model.rebuild_embed(vocab)

    return _valid(model.model, device, test_iter)



def fit(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    """function fit
    Args:
        model:   
        sess:   
        data_pars:   
        compute_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """

    model0        = model.model
    lr            = compute_pars['learning_rate']
    epochs        = compute_pars["epochs"]
    device        = _get_device()
    train_loss    = []
    train_acc     = []
    test_loss     = []
    test_acc      = []
    best_test_acc = -1
    optimizer     = optim.Adam(model0.parameters(), lr=lr)
    train_iter, valid_iter, vocab = get_dataset(data_pars, out_pars)
    
    # load word embeddings to model
    model0.rebuild_embed(vocab)

    for epoch in range(1, epochs + 1):

        tr_loss, tr_acc = _train(model0, device, train_iter, optimizer, epoch, epochs)
        print( f'Train Epoch: {epoch} \t Loss: {tr_loss} \t Accuracy: {tr_acc}')

        ts_loss, ts_acc = _valid(model0, device, valid_iter)
        print( f'Train Epoch: {epoch} \t Loss: {ts_loss} \t Accuracy: {ts_acc}')


        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            log( f"model saves at {best_test_acc}% accuracy")
            os.makedirs(out_pars["checkpointdir"], exist_ok=True)
            torch.save(model0.state_dict(),  os.path.join(out_pars["checkpointdir"], "best_accuracy"))


        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)

    model.model = model0    
    return model, None



# def get_dataset(data_pars=None, out_pars=None, **kwargs):
#     device         = _get_device()
#     path           = path_norm( data_pars['data_path'])
#     frac           = data_pars['frac']
#     lang           = data_pars['lang']
#     pretrained_emb = data_pars['pretrained_emb']
#     train_exists   = os.path.isfile(data_pars['train_path'])
#     valid_exists   = os.path.isfile(data_pars['valid_path'])

#     if not (train_exists and valid_exists) or data_pars['split_if_exists']:
#         split_train_valid( path, data_pars['train_path'], data_pars['valid_path'], frac )

#     trainset, validset, vocab = create_tabular_dataset( data_pars['train_path'], data_pars['valid_path'], lang, pretrained_emb)
#     train_iter, valid_iter = create_data_iterator( data_pars['batch_size'], data_pars['val_batch_size'],
#         trainset, validset, device
#     )
#     return train_iter, valid_iter, vocab


def get_dataset(data_pars=None, out_pars=None, **kwargs):
    """function get_dataset
    Args:
        data_pars:   
        out_pars:   
        **kwargs:   
    Returns:
        
    """
    device         = _get_device()
    dataset        = data_pars['data_info'].get('dataset', None)
    batch_size     = data_pars['data_info'].get('batch_size', 64)

    from mlmodels.dataloader import DataLoader

    loader = DataLoader(data_pars)

    if dataset:
        loader.compute()
        try:
            dataset, internal_states  = loader.get_data()
            trainset, validset, vocab = dataset
            train_iter, valid_iter    = create_data_iterator(batch_size, trainset, validset, device)

        except:
            raise Exception("the last Preprocessor have to return (trainset, validset, vocab), internal_states.")
            
        return train_iter, valid_iter, vocab

    else:
        raise Exception("dataset not provided ")
        return 0



def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None, return_ytrue=1):
    """function predict
    Args:
        model:   
        session:   
        data_pars:   
        compute_pars:   
        out_pars:   
        return_ytrue:   
    Returns:
        
    """
    data_pars = data_pars.copy()
    data_pars.update(frac=1)
    print(data_pars)
    test_iter, _, vocab = get_dataset(data_pars, out_pars)
    # get a batch of data
    it = next(iter(test_iter))
    # print("Test data:", it)

    x_test = it.text
    # print("x_test shape: ", x_test.shape)
    
    x_test = torch.transpose(x_test, 0, 1)
    # print("transpose x_test shape: ", x_test.shape)

    model0 = model.model
    model0.rebuild_embed(vocab)
    ypred = model0(x_test)
    ypred = torch.max(ypred, 1)[1]
    if return_ytrue:
        label = it.label
        return ypred, label
    else:
        return ypred, None


def save(model, session=None, save_pars=None):
    """function save
    Args:
        model:   
        session:   
        save_pars:   
    Returns:
        
    """
    from mlmodels.util import save_tch
    save_tch(model, save_pars= save_pars )
    # return torch.save(model, path)


def load(load_pars= None ):
    """function load
    Args:
        load_pars:   
    Returns:
        
    """
    from mlmodels.util import load_tch
    return load_tch(load_pars), None
    # return torch.load(path)



def get_params(param_pars=None, **kw):
    """function get_params
    Args:
        param_pars:   
        **kw:   
    Returns:
        
    """
    from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']


    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, 'r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path  = path_norm( "dataset/text/imdb.csv"  )   
        out_path   = path_norm( "ztest/model_tch/textcnn/" )
        model_path = os.path.join(out_path , "model")

        data_pars= {
            "data_path":   path_norm("dataset/recommender/IMDB_sample.txt"),
            "train_path":  path_norm("dataset/recommender/IMDB_train.csv"),
            "valid_path":  path_norm("dataset/recommender/IMDB_valid.csv"),

 
            "split_if_exists": True,
            "frac": 0.99,
            "lang": "en",
            "pretrained_emb": "glove.6B.300d",
            "batch_size": 64,
            "val_batch_size": 64,
        }


        model_pars= {"dim_channel": 100, "kernel_height": [3,4,5], "dropout_rate": 0.5, "num_class": 2 }

        compute_pars= {"learning_rate": 0.001, "epochs": 1, "checkpointdir": out_path + "/checkpoint/"}

        out_pars= {
            "path" : model_path,
            "checkpointdir": out_path + "/checkpoint/"
        }

        return model_pars, data_pars, compute_pars, out_pars







###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    """function test
    Args:
        data_path:   
        pars_choice:   
        config_mode:   
    Returns:
        
    """
    ### Local test
    from mlmodels.util import path_norm
    data_path = path_norm(data_path)
    log("Json file path: ", data_path)

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(model_pars, data_pars, compute_pars, out_pars )


    log("#### Loading dataset   #############################################")
    Xtuple = get_dataset(data_pars)
    print(len(Xtuple))


    log("#### Model init       #############################################")
    session = None
    model   = Model(model_pars, data_pars, compute_pars)


    log("#### Model fit        #############################################")
    data_pars["train"] = 1
    model, session = fit(model, session, data_pars, compute_pars, out_pars)


    log("#### Save   ########################################################")
    save_pars = {"path": out_pars['path'] + "/model.pkl"}
    save(model, session, save_pars=save_pars)


    log("#### Load   ########################################################")
    model2, session2 = load(save_pars)


    log("#### Predict from Load   ###########################################")
    data_pars["train"] = 0
    ypred, _ = predict(model2, session2, data_pars, compute_pars, out_pars)



    log("#### Predict   #####################################################")
    data_pars["train"] = 0
    ypred, _ = predict(model, session, data_pars, compute_pars, out_pars)
    # print("ypred : ", ypred)
    # print("ypred shape: ", ypred.shape)


    log("#### metrics   #####################################################")
    metrics_val = evaluate(model, session, data_pars, compute_pars, out_pars)
    log(metrics_val)


    log("#### Plot   ########################################################")




if __name__ == '__main__':
    # test( data_path="model_tch/textcnn.json", pars_choice = "test01" )
    test( data_path="dataset/json/refactor/textcnn.json", pars_choice="json", config_mode="test" )





