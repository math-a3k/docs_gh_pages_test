# -*- coding: utf-8 -*-
"""

https://github.com/graph4ai/graph4nlp_demo/blob/main/SIGIR2021_demo/text_classification.ipynb




cd deeplearning/torch/
python graphnlp.py  test


"""
import sys, os, gzip, csv, random, math, logging, pandas as pd, argparse
from datetime import datetime
from box import Box
from tqdm.notebook import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim


from graph4nlp.pytorch.modules.config import get_basic_args
from graph4nlp.pytorch.modules.utils.config_utils import update_values, get_yaml_config
from torch.utils.data import DataLoader
from graph4nlp.pytorch.data.data import to_batch
from graph4nlp.pytorch.datasets.mawps import MawpsDatasetForTree
from graph4nlp.pytorch.modules.graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_embedding import *
from graph4nlp.pytorch.models.graph2tree import Graph2Tree
from graph4nlp.pytorch.modules.utils.tree_utils import Tree, prepare_oov


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
        graphnlp_train(modelname_or_path = modelid,
                    taskname  = "classifier", 
                    lossname  = lname,
                    train_path= dirdata + f"/data_fake.parquet",
                    val_path=   dirdata + f"/data_fake.parquet",
                    eval_path = dirdata + f"/data_fake.parquet",
                    metricname='cosinus',
                    dirout= dirdata + f"/results/" + lloss, cc=cc) 
    


###################################################################################################################        
def dataset_fake(dirdata):        
    nli_dataset_path = dirdata + '/A.tsv.gz'
    sts_dataset_path = dirdata + '/sk.tsv.gz'

    # Read the AllNLI.tsv.gz file and create the training dataset
    df = pd_read(nli_dataset_path, npool=1) 
    df.iloc[:50, :].to_parquet(dirdata +"/fake_data.parquet")



    
def dataset_download(dirout='/content/sample_data/sent_tans/'):
    #### Check if dataset exsist. If not, download and extract  it    
    nli_dataset_path = dirout + '/AllNLI.tsv.gz'
    sts_dataset_path = dirout + '/stsbenchmark.tsv.gz'
    os.makedirs(dirout, exist_ok=False)    
    if not os.path.exists(nli_dataset_path):
        wget('h', nli_dataset_path)

    if not os.path.exists(sts_dataset_path):
        wget('htt', sts_dataset_path)
        


###################################################################################################################        
def model_evaluate(model ="modelname OR path OR model object", dirdata='./*.csv', dirout='./', cc:dict= None, batch_size=16, name='sts-test'):
    ### Evaluate Model
    df = pd.read_csv(dirdata, error_bad_lines=False)

    model= model_load(model)
    
    #### Evaluate


def model_load(path_or_name_or_object):
    #### Reload model or return the model itself
    if isinstance(path_or_name_or_object, str) :
       # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
       model = graphnlp(path_or_name_or_object)
       model.eval()
    
    return model


def model_save(model,path, reload=True):
    model.save( path)
    log(path)
    
    if reload:
        #### reload model  + model something   
        model1 = model_load(path)
        log(model1)


def model_setup_compute(model, use_gpu=0, ngpu=1, ncpu=1, cc:dict=None):
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
        raise Exception('need path_or_df')
    return dftrain    
        

def load_evaluator(name='sts', path_or_df="", dname='sts', cc:dict=None):
    if dname == 'sts':        
        log("Read  dataset")
        df = pd_read(path_or_df) 
        if 'nsample' in cc : df = df.iloc[:cc.nsample,:]
          
        #### Evaluator  
    
        return dev_evaluator


def load_dataloader(name='sts', path_or_df = "", cc:dict= None, npool=4):    
    df = pd_read(path_or_df, npool=npool) 
    if 'nsample' in cc : df = df.iloc[:cc.nsample,:]
        
    #### Setup dataloader

    log('Nelements', len(train_dataloader))
    return train_dataloader



def load_loss(model ='', lossname ='cosinus',  cc:dict= None):

    if lossname == 'MultpleNegativesRankingLoss':
      train_loss = 

    elif lossname == 'softmax':
      nclass     =  cc.get('data_nclass', -1)
      train_loss = 
    elif lossname =='cosinus':
      train_loss = 

    elif lossname =='triplethard':
      train_loss = 

    return train_loss




def graphnlp_train(modelname_or_path='distilbert-base-nli-mean-tokens',
                 taskname="classifier",   lossname="cosinus",
                 datasetname = 'sts',  
                   
                 train_path="train/*.csv", val_path  ="val/*.csv",  eval_path ="eval/*.csv",
                   
                 metricname='cosinus',
                 dirout ="mymodel_save/",
                 cc:dict= None):
  #  """"
  # https://github.com/graph4ai/graph4nlp_demo/blob/main/SIGIR2021_demo/text_classification.ipynb
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


    # config setup
    config_file =  cc.config_file  # '../config/trec/graphsage_bi_fuse_static_dependency.yaml'
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    print_config(config)
    
    if taskname == 'classifier':
        ##### load model form disk or from internet        
        modelrun = ModelHandler(config)
    
    
        ##### dataloader train, evaluator

    
        ##### Task Loss
        loss = 

        ##### optimizer Loss
        optimizer = 
        scheduler =
        stopper = 
        
        
        log('########## train')
        val_acc = modelrun.train()
       
        

        ### return self.stopper.best_score

        log("\n******************< Eval > ********************")
         test_acc = modelrun.test() 
        
        log("### Save+ Reload the model  ")



        
        log("\n******************< finish  > ********************")

        
        
##########################################################################################
def metrics_cosine_sim(sentence1 = "sentence 1" , sentence2 = "sentence 2", model_id = "model name or path or object"):
  ### function to compute cosinue similarity      
  model = model_load(model_id)

  #Compute embedding for both lists
  embeddings1 = model.encode(sentence1, convert_to_tensor=True)
  embeddings2 = model.encode(sentence2, convert_to_tensor=True)

  #Compute cosine-similarity
  cosine_scores = util.cos_sim(embeddings1, embeddings2)
  log( f"{sentence1} \t {sentence2} \n cosine-similarity Score: {cosine_scores[0][0]}" )

        
        
##########################################################################################
if __name__ == '__main__':
    import fire
    fire.Fire()



    
    
    
"""    
Text Classification Tutorial
===================================


Introduction
------------


In this tutorial demo, we will use the Graph4NLP library to build a GNN-based text classification model. The model consists of

- graph construction module (e.g., dependency based static graph)
- graph embedding module (e.g., Bi-Fuse GraphSAGE)
- predictoin module (e.g., graph pooling + MLP classifier)

We will use the built-in module APIs to build the model, and evaluate it on the TREC dataset. The full example can be downloaded from `text classification notebook <https://github.com/graph4ai/graph4nlp_demo/blob/main/SIGIR2021_demo/text_classification.ipynb>`__.



Environment setup
------

Please follow the instructions `here <https://github.com/graph4ai/graph4nlp_demo#environment-setup>`__ to set up the environment.




Build the text classifier
------

Let's first build the GNN-based text classifier which contains three major components including graph construction module, graph embedding module and graph prediction module.

For graph construction module, the Graph4NLP library provides built-in APIs to support both static graph construction methods (e.g., `dependency graph`, `constituency graph`, `IE graph`) and dynamic graph construction methods (e.g., `node embedding based graph`, `node embedding based refined graph`). When calling the graph construction API, users should also specify the `embedding style` (e.g., word2vec, BiLSTM, BERT) to initalize the node/edge embeddings. Both single-token and multi-token node/edge graphs are supported.

For graph embedding module, the Graph4NLP library provides builti-in APIs to support both `undirectional` and `bidirectinal` versions for common GNNs such as `GCN`, `GraphSAGE`, `GAT` and `GGNN`.

For graph prediction module, the Graph4NLP library provides a high-level graph classification prediction module which consists of a graph pooling component (e.g., average pooling, max pooling) and a multilayer perceptron (MLP).


.. code-block:: python
"""
    class TextClassifier(nn.Module):
        def __init__(self, vocab, label_model, config):
            super(TextClassifier, self).__init__()
            self.config = config
            self.vocab = vocab
            self.label_model = label_model

            # Specify embedding style to initialize node/edge embeddings
            embedding_style = {'single_token_item': True if config['graph_type'] != 'ie' else False,
                                'emb_strategy': config.get('emb_strategy', 'w2v_bilstm'),
                                'num_rnn_layers': 1,
                                'bert_model_name': config.get('bert_model_name', 'bert-base-uncased'),
                                'bert_lower_case': True
                               }

            assert not (config['graph_type'] in ('node_emb', 'node_emb_refined') and config['gnn'] == 'gat'), \
                                    'dynamic graph construction does not support GAT'

            use_edge_weight = False


            # Set up graph construction module
            if config['graph_type'] == 'dependency':
                self.graph_topology = DependencyBasedGraphConstruction(embedding_style=embedding_style,
                                       vocab=vocab.in_word_vocab, hidden_size=config['num_hidden'],
                                       word_dropout=config['word_dropout'], rnn_dropout=config['rnn_dropout'],
                                       fix_word_emb=not config['no_fix_word_emb'], fix_bert_emb=not config.get('no_fix_bert_emb', False))
            elif config['graph_type'] == 'constituency':
                self.graph_topology = ConstituencyBasedGraphConstruction(embedding_style=embedding_style,
                                       vocab=vocab.in_word_vocab, hidden_size=config['num_hidden'],
                                       word_dropout=config['word_dropout'], rnn_dropout=config['rnn_dropout'],
                                       fix_word_emb=not config['no_fix_word_emb'], fix_bert_emb=not config.get('no_fix_bert_emb', False))
            elif config['graph_type'] == 'ie':
                self.graph_topology = IEBasedGraphConstruction(embedding_style=embedding_style,
                                       vocab=vocab.in_word_vocab, hidden_size=config['num_hidden'],
                                       word_dropout=config['word_dropout'], rnn_dropout=config['rnn_dropout'],
                                       fix_word_emb=not config['no_fix_word_emb'], fix_bert_emb=not config.get('no_fix_bert_emb', False))
            elif config['graph_type'] == 'node_emb':
                self.graph_topology = NodeEmbeddingBasedGraphConstruction(vocab.in_word_vocab,
                                       embedding_style, sim_metric_type=config['gl_metric_type'],
                                       num_heads=config['gl_num_heads'], top_k_neigh=config['gl_top_k'],
                                       epsilon_neigh=config['gl_epsilon'], smoothness_ratio=config['gl_smoothness_ratio'],
                                       connectivity_ratio=config['gl_connectivity_ratio'], sparsity_ratio=config['gl_sparsity_ratio'],
                                       input_size=config['num_hidden'], hidden_size=config['gl_num_hidden'],
                                       fix_word_emb=not config['no_fix_word_emb'], fix_bert_emb=not config.get('no_fix_bert_emb', False),
                                       word_dropout=config['word_dropout'], rnn_dropout=config['rnn_dropout'])
                use_edge_weight = True
            elif config['graph_type'] == 'node_emb_refined':
                self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(vocab.in_word_vocab,
                                        embedding_style, config['init_adj_alpha'],
                                        sim_metric_type=config['gl_metric_type'], num_heads=config['gl_num_heads'],
                                        top_k_neigh=config['gl_top_k'], epsilon_neigh=config['gl_epsilon'],
                                        smoothness_ratio=config['gl_smoothness_ratio'], connectivity_ratio=config['gl_connectivity_ratio'],
                                        sparsity_ratio=config['gl_sparsity_ratio'], input_size=config['num_hidden'],
                                        hidden_size=config['gl_num_hidden'], fix_word_emb=not config['no_fix_word_emb'],
                                        fix_bert_emb=not config.get('no_fix_bert_emb', False),
                                        word_dropout=config['word_dropout'], rnn_dropout=config['rnn_dropout'])
                use_edge_weight = True
            else:
                raise RuntimeError('Unknown graph_type: {}'.format(config['graph_type']))

            if 'w2v' in self.graph_topology.embedding_layer.word_emb_layers:
                self.word_emb = self.graph_topology.embedding_layer.word_emb_layers['w2v'].word_emb_layer
            else:
                self.word_emb = WordEmbedding(self.vocab.in_word_vocab.embeddings.shape[0],
                                self.vocab.in_word_vocab.embeddings.shape[1], pretrained_word_emb=self.vocab.in_word_vocab.embeddings,
                                fix_emb=not config['no_fix_word_emb'], device=config['device']).word_emb_layer


            # Set up graph embedding module
            if config['gnn'] == 'gat':
                heads = [config['gat_num_heads']] * (config['gnn_num_layers'] - 1) + [config['gat_num_out_heads']]
                self.gnn = GAT(config['gnn_num_layers'], config['num_hidden'], config['num_hidden'], config['num_hidden'],
                            heads, direction_option=config['gnn_direction_option'], feat_drop=config['gnn_dropout'],
                            attn_drop=config['gat_attn_dropout'], negative_slope=config['gat_negative_slope'],
                            residual=config['gat_residual'], activation=F.elu)
            elif config['gnn'] == 'graphsage':
                self.gnn = GraphSAGE(config['gnn_num_layers'], config['num_hidden'], config['num_hidden'], config['num_hidden'],
                            config['graphsage_aggreagte_type'], direction_option=config['gnn_direction_option'], feat_drop=config['gnn_dropout'],
                            bias=True, norm=None, activation=F.relu, use_edge_weight=use_edge_weight)
            elif config['gnn'] == 'ggnn':
                self.gnn = GGNN(config['gnn_num_layers'], config['num_hidden'], config['num_hidden'], config['num_hidden'],
                            feat_drop=config['gnn_dropout'], direction_option=config['gnn_direction_option'], bias=True, use_edge_weight=use_edge_weight)
            else:
                raise RuntimeError('Unknown gnn type: {}'.format(config['gnn']))


            # Set up graph prediction module
            self.clf = FeedForwardNN(2 * config['num_hidden'] if config['gnn_direction_option'] == 'bi_sep' else config['num_hidden'],
                        config['num_classes'], [config['num_hidden']], graph_pool_type=config['graph_pooling'],
                        dim=config['num_hidden'], use_linear_proj=config['max_pool_linear_proj'])

            self.loss = GeneralLoss('CrossEntropy')


        def forward(self, graph_list, tgt=None, require_loss=True):
            # build graph topology
            batch_gd = self.graph_topology(graph_list)

            # run GNN encoder
            self.gnn(batch_gd)

            # run graph classifier
            self.clf(batch_gd)
            logits = batch_gd.graph_attributes['logits']

            if require_loss:
                loss = self.loss(logits, tgt)
                return logits, loss
            else:
                return logits

        @classmethod
        def load_checkpoint(cls, model_path):
            return torch.load(model_path)


"""
Build the model handler
----


Next, let's build a model handler which will do a bunch of things including setting up dataloader, model, optimizer, evaluation metrics, train/val/test loops, and so on.

When setting up the dataloader, users will need to call the dataset API which will preprocess the data, e.g., calling the graph construction module, building the vocabulary, tensorizing the data. Users will need to specify the graph construction type when calling the dataset API.

Users can build their customized dataset APIs by inheriting our low-level dataset APIs. We provide low-level dataset APIs to support various scenarios (e.g., `Text2Label`, `Sequence2Labeling`, `Text2Text`, `Text2Tree`, `DoubleText2Text`).


.. code-block:: python
"""
    class ModelHandler:
        def __init__(self, config):
            super(ModelHandler, self).__init__()
            self.config = config
            self.logger = Logger(self.config['out_dir'], config={k:v for k, v in self.config.items() if k != 'device'}, overwrite=True)
            self.logger.write(self.config['out_dir'])
            self._build_device()
            self._build_dataloader()
            self._build_model()
            self._build_optimizer()
            self._build_evaluation()

        def _build_device(self):
            if not self.config['no_cuda'] and torch.cuda.is_available():
                print('[ Using CUDA ]')
                self.config['device'] = torch.device('cuda' if self.config['gpu'] < 0 else 'cuda:%d' % self.config['gpu'])
                torch.cuda.manual_seed(self.config['seed'])
                torch.cuda.manual_seed_all(self.config['seed'])
                torch.backends.cudnn.deterministic = True
                cudnn.benchmark = False
            else:
                self.config['device'] = torch.device('cpu')

        def _build_dataloader(self):
            dynamic_init_topology_builder = None
            if self.config['graph_type'] == 'dependency':
                topology_builder = DependencyBasedGraphConstruction
                graph_type = 'static'
                merge_strategy = 'tailhead'
            elif self.config['graph_type'] == 'constituency':
                topology_builder = ConstituencyBasedGraphConstruction
                graph_type = 'static'
                merge_strategy = 'tailhead'
            elif self.config['graph_type'] == 'ie':
                topology_builder = IEBasedGraphConstruction
                graph_type = 'static'
                merge_strategy = 'global'
            elif self.config['graph_type'] == 'node_emb':
                topology_builder = NodeEmbeddingBasedGraphConstruction
                graph_type = 'dynamic'
                merge_strategy = None
            elif self.config['graph_type'] == 'node_emb_refined':
                topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
                graph_type = 'dynamic'
                merge_strategy = 'tailhead'

                if self.config['init_graph_type'] == 'line':
                    dynamic_init_topology_builder = None
                elif self.config['init_graph_type'] == 'dependency':
                    dynamic_init_topology_builder = DependencyBasedGraphConstruction
                elif self.config['init_graph_type'] == 'constituency':
                    dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
                elif self.config['init_graph_type'] == 'ie':
                    merge_strategy = 'global'
                    dynamic_init_topology_builder = IEBasedGraphConstruction
                else:
                    raise RuntimeError('Define your own dynamic_init_topology_builder')
            else:
                raise RuntimeError('Unknown graph_type: {}'.format(self.config['graph_type']))

            topology_subdir = '{}_graph'.format(self.config['graph_type'])
            if self.config['graph_type'] == 'node_emb_refined':
                topology_subdir += '_{}'.format(self.config['init_graph_type'])


            # Call the TREC dataset API
            dataset = TrecDataset(root_dir=self.config.get('root_dir', self.config['root_data_dir']),
                                  pretrained_word_emb_name=self.config.get('pretrained_word_emb_name', "840B"),
                                  merge_strategy=merge_strategy, seed=self.config['seed'], thread_number=4,
                                  port=9000, timeout=15000, word_emb_size=300, graph_type=graph_type,
                                  topology_builder=topology_builder, topology_subdir=topology_subdir,
                                  dynamic_graph_type=self.config['graph_type'] if \
                                      self.config['graph_type'] in ('node_emb', 'node_emb_refined') else None,
                                  dynamic_init_topology_builder=dynamic_init_topology_builder,
                                  dynamic_init_topology_aux_args={'dummy_param': 0})

            self.train_dataloader = DataLoader(dataset.train, batch_size=self.config['batch_size'], shuffle=True,
                                               num_workers=self.config['num_workers'], collate_fn=dataset.collate_fn)
            if hasattr(dataset, 'val')==False:
                dataset.val = dataset.test
            self.val_dataloader = DataLoader(dataset.val, batch_size=self.config['batch_size'], shuffle=False,
                                              num_workers=self.config['num_workers'], collate_fn=dataset.collate_fn)
            self.test_dataloader = DataLoader(dataset.test, batch_size=self.config['batch_size'], shuffle=False,
                                              num_workers=self.config['num_workers'], collate_fn=dataset.collate_fn)
            self.vocab = dataset.vocab_model
            self.label_model = dataset.label_model
            self.config['num_classes'] = self.label_model.num_classes
            self.num_train = len(dataset.train)
            self.num_val = len(dataset.val)
            self.num_test = len(dataset.test)
            print('Train size: {}, Val size: {}, Test size: {}'
                .format(self.num_train, self.num_val, self.num_test))
            self.logger.write('Train size: {}, Val size: {}, Test size: {}'
                .format(self.num_train, self.num_val, self.num_test))

        def _build_model(self):
            self.model = TextClassifier(self.vocab, self.label_model, self.config).to(self.config['device'])

        def _build_optimizer(self):
            parameters = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(parameters, lr=self.config['lr'])
            self.stopper = EarlyStopping(os.path.join(self.config['out_dir'], Constants._SAVED_WEIGHTS_FILE), patience=self.config['patience'])
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=self.config['lr_reduce_factor'], \
                patience=self.config['lr_patience'], verbose=True)

        def _build_evaluation(self):
            self.metric = Accuracy(['accuracy'])

        def train(self):
            dur = []
            for epoch in range(self.config['epochs']):
                self.model.train()
                train_loss = []
                train_acc = []
                t0 = time.time()
                for i, data in enumerate(self.train_dataloader):
                    tgt = data['tgt_tensor'].to(self.config['device'])
                    data['graph_data'] = data['graph_data'].to(self.config['device'])
                    logits, loss = self.model(data['graph_data'], tgt, require_loss=True)

                    # add graph regularization loss if available
                    if data['graph_data'].graph_attributes.get('graph_reg', None) is not None:
                        loss = loss + data['graph_data'].graph_attributes['graph_reg']

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss.append(loss.item())

                    pred = torch.max(logits, dim=-1)[1].cpu()
                    train_acc.append(self.metric.calculate_scores(ground_truth=tgt.cpu(), predict=pred.cpu(), zero_division=0)[0])
                    dur.append(time.time() - t0)

                val_acc = self.evaluate(self.val_dataloader)
                self.scheduler.step(val_acc)
                print('Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}'.
                  format(epoch + 1, self.config['epochs'], np.mean(dur), np.mean(train_loss), np.mean(train_acc), val_acc))
                self.logger.write('Epoch: [{} / {}] | Time: {:.2f}s | Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f}'.
                            format(epoch + 1, self.config['epochs'], np.mean(dur), np.mean(train_loss), np.mean(train_acc), val_acc))

                if self.stopper.step(val_acc, self.model):
                    break

            return self.stopper.best_score

        def evaluate(self, dataloader):
            self.model.eval()
            with torch.no_grad():
                pred_collect = []
                gt_collect = []
                for i, data in enumerate(dataloader):
                    tgt = data['tgt_tensor'].to(self.config['device'])
                    data['graph_data'] = data['graph_data'].to(self.config["device"])
                    logits = self.model(data['graph_data'], require_loss=False)
                    pred_collect.append(logits)
                    gt_collect.append(tgt)

                pred_collect = torch.max(torch.cat(pred_collect, 0), dim=-1)[1].cpu()
                gt_collect = torch.cat(gt_collect, 0).cpu()
                score = self.metric.calculate_scores(ground_truth=gt_collect, predict=pred_collect, zero_division=0)[0]

                return score

        def test(self):
            # restored best saved model
            self.model = TextClassifier.load_checkpoint(self.stopper.save_model_path)

            t0 = time.time()
            acc = self.evaluate(self.test_dataloader)
            dur = time.time() - t0
            print('Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}'.
              format(self.num_test, dur, acc))
            self.logger.write('Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f}'.
              format(self.num_test, dur, acc))

            return acc


"""
Run the model
----

.. code-block:: python

    runner = ModelHandler(config)
    val_acc = runner.train()
    test_acc = runner.test()



.. parsed-literal::

    out/trec/graphsage_bi_fuse_dependency_ckpt_1628651059.35833
    Loading pre-built label mappings stored in ../data/trec/processed/dependency_graph/label.pt
    Train size: 5452, Val size: 500, Test size: 500
    [ Fix word embeddings ]
    Epoch: [1 / 500] | Time: 14.28s | Loss: 1.1777 | Train Acc: 0.5249 | Val Acc: 0.7740
    Saved model to out/trec/graphsage_bi_fuse_dependency_ckpt_1628651059.35833/params.saved
    Epoch: [2 / 500] | Time: 13.17s | Loss: 0.6613 | Train Acc: 0.7596 | Val Acc: 0.8280
    Saved model to out/trec/graphsage_bi_fuse_dependency_ckpt_1628651059.35833/params.saved    
    
"""
    
    
    
