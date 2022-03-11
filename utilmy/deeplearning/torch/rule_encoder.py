# -*- coding: utf-8 -*-
MNAME = "utilmy.deeplearning.torch.rule_encoder"
HELP = """ utils for model explanation
"""
import os, random, numpy as np, glob, pandas as pd, matplotlib.pyplot as plt ;from box import Box
from copy import deepcopy

from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

#### Types


#############################################################################################
from utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    ss = HELP + help_create(MNAME)
    log(ss)


#############################################################################################
def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    log(MNAME)
    test()
    # test2()




##############################################################################################
def test():
    """function test
    Args:
    Returns:
        
    """
    model_info = {'dataonly': {'rule': 0.0},
                'ours-beta1.0': {'beta': [1.0], 'scale': 1.0, 'lr': 0.001},
                'ours-beta0.1': {'beta': [0.1], 'scale': 1.0, 'lr': 0.001},
                'ours-beta0.1-scale0.1': {'beta': [0.1], 'scale': 0.1},
                'ours-beta0.1-scale0.01': {'beta': [0.1], 'scale': 0.01},
                'ours-beta0.1-scale0.05': {'beta': [0.1], 'scale': 0.05},
                'ours-beta0.1-pert0.001': {'beta': [0.1], 'pert': 0.001},
                'ours-beta0.1-pert0.01': {'beta': [0.1], 'pert': 0.01},
                'ours-beta0.1-pert0.1': {'beta': [0.1], 'pert': 0.1},
                'ours-beta0.1-pert1.0': {'beta': [0.1], 'pert': 1.0},
                }

    arg = Box({
      "dataurl":  "https://github.com/caravanuden/cardio/raw/master/cardio_train.csv",
      "datapath": './cardio_train.csv',

      ##### Rules
      "rules": {},

      #"rule_threshold": 129.5,
      #"src_ok_ratio": 0.3,
      #"src_unok_ratio": 0.7,
      #"target_rule_ratio": 0.7,
      #"rule_ind": 5,


      #####
      "train_ratio": 0.7,
      "validation_ratio": 0.1,
      "test_ratio": 0.2,

      "model_type": 'dataonly',
      "input_dim_encoder": 16,
      "output_dim_encoder": 16,
      "hidden_dim_encoder": 100,
      "hidden_dim_db": 16,
      "n_layers": 1,


      ##### Training
      "seed": 42,
      "device": 'cpu',  ### 'cuda:0',
      "batch_size": 32,
      "epochs": 1,
      "early_stopping_thld": 10,
      "valid_freq": 1,
      'saved_filename' :'./model.pt',

    })
    arg.model_info = model_info
    arg.merge = 'cat'
    arg.input_dim = 20   ### 20
    arg.output_dim = 1
    log(arg)


    #### Rules setup #############################################################
    arg.rules = {
          "rule_threshold":  129.5,
          "src_ok_ratio":      0.3,
          "src_unok_ratio":    0.7,
          "target_rule_ratio": 0.7,
          "rule_ind": 2,    ### Index of the colum Used for rule:  df.iloc[:, rule_ind ]
    }
    arg.rules.loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.
    arg.rules.loss_rule_calc = loss_rule_calc_cardio


    ### device setup
    device = device_setup(arg)

    #### dataset load
    df = dataset_load_cardio(arg)

    #### dataset preprocess
    train_X, train_y, valid_X,  valid_y, test_X,  test_y  = dataset_preprocess_cardio(df, arg)
    arg.input_dim = train_X.shape[1]



    ### Create dataloader  ############################
    train_loader, valid_loader, test_loader = dataloader_create( train_X, train_y, valid_X, valid_y, test_X, test_y,  arg)

    ### Model Build
    model, losses, arg = model_build(arg=arg)

    ### Model Train
    model_train(model, losses, train_loader, valid_loader, arg=arg, )


    #### Test
    model_eval, losses = model_load(arg)
    model_evaluation(model_eval, losses.loss_task_func , arg=arg, dataset_load1= dataset_load_cardio,  dataset_preprocess1 =  dataset_preprocess_cardio  )


def dataset_load_cardio(arg):
  """function dataset_load_cardio
  Args:
      arg:   
  Returns:
      
  """
  # Load dataset
  #url = "https://github.com/caravanuden/cardio/raw/master/cardio_train.csv"
  import wget, glob
  if len(glob.glob(arg.datapath)) < 1 :
     if 'dataurl' not in arg : raise Exception('no dataurl in arg')
     wget.download(arg.dataurl)

  df = pd.read_csv(arg.datapath,delimiter=';')
  df = df.iloc[:500, :]
  log(df, df.columns, df.shape)

  # y = df[coly]
  # X_raw = df.drop([coly], axis=1)

  return df


def dataset_preprocess_cardio(df, arg):
    """function dataset_preprocess_cardio
    Args:
        df:   
        arg:   
    Returns:
        
    """
    coly = 'cardio'
    y     = df[coly]
    X_raw = df.drop([coly], axis=1)

    log("Target class ratio:")
    log("# of y=1: {}/{} ({:.2f}%)".format(np.sum(y==1), len(y), 100*np.sum(y==1)/len(y)))
    log("# of y=0: {}/{} ({:.2f}%)\n".format(np.sum(y==0), len(y), 100*np.sum(y==0)/len(y)))

    column_trans = ColumnTransformer(
        [('age_norm', StandardScaler(), ['age']),
        ('height_norm', StandardScaler(), ['height']),
        ('weight_norm', StandardScaler(), ['weight']),
        ('gender_cat', OneHotEncoder(), ['gender']),
        ('ap_hi_norm', StandardScaler(), ['ap_hi']),
        ('ap_lo_norm', StandardScaler(), ['ap_lo']),
        ('cholesterol_cat', OneHotEncoder(), ['cholesterol']),
        ('gluc_cat', OneHotEncoder(), ['gluc']),
        ('smoke_cat', OneHotEncoder(), ['smoke']),
        ('alco_cat', OneHotEncoder(), ['alco']),
        ('active_cat', OneHotEncoder(), ['active']),
        ], remainder='passthrough'
    )

    X = column_trans.fit_transform(X_raw)
    nsamples = X.shape[0]
    X_np = X.copy()


    ######## Rule : higher ap -> higher risk   #####################################
    """  Identify Class y=0 /1 from rule 1

    """
    if 'rule1':
        rule_threshold = arg.rules.rule_threshold
        rule_ind       = arg.rules.rule_ind
        rule_feature   = 'ap_hi'
        src_unok_ratio = arg.rules.src_unok_ratio
        src_ok_ratio   = arg.rules.src_ok_ratio

        #### Ok cases: nornal
        low_ap_negative  = (df[rule_feature] <= rule_threshold) & (df[coly] == 0)    # ok
        high_ap_positive = (df[rule_feature] > rule_threshold)  & (df[coly] == 1)    # ok

        ### Outlier cases (from rule)
        low_ap_positive  = (df[rule_feature] <= rule_threshold) & (df[coly] == 1)    # unok
        high_ap_negative = (df[rule_feature] > rule_threshold)  & (df[coly] == 0)    # unok




    #### Merge rules ##############################################
    # Samples in ok group
    idx_ok = low_ap_negative | high_ap_positive


    # Samples in Unok group
    idx_unok = low_ap_negative | high_ap_positive



    ##############################################################################
    # Samples in ok group
    X_ok = X[ idx_ok ]
    y_ok = y[ idx_ok ]
    y_ok = y_ok.to_numpy()
    X_ok, y_ok = shuffle(X_ok, y_ok, random_state=0)
    num_ok_samples = X_ok.shape[0]


    # Samples in Unok group
    X_unok = X[ idx_unok ]
    y_unok = y[ idx_unok ]
    y_unok = y_unok.to_numpy()
    X_unok, y_unok = shuffle(X_unok, y_unok, random_state=0)
    num_unok_samples = X_unok.shape[0]


    ######### Build a source dataset
    n_from_unok = int(src_unok_ratio * num_unok_samples)
    n_from_ok   = int(n_from_unok * src_ok_ratio / (1- src_ok_ratio))

    X_src = np.concatenate((X_ok[:n_from_ok], X_unok[:n_from_unok]), axis=0)
    y_src = np.concatenate((y_ok[:n_from_ok], y_unok[:n_from_unok]), axis=0)

    log("Source dataset statistics:")
    log("# of samples in ok group: {}".format(n_from_ok))
    log("# of samples in Unok group: {}".format(n_from_unok))
    log("ok ratio: {:.2f}%".format(100 * n_from_ok / (X_src.shape[0])))


    ##### Split   #########################################################################
    seed= 42
    train_X, test_X, train_y, test_y = train_test_split(X_src,  y_src,  test_size=1 - arg.train_ratio, random_state=seed)
    valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size= arg.test_ratio / (arg.test_ratio + arg.validation_ratio), random_state=seed)
    return (train_X, train_y, valid_X,  valid_y, test_X,  test_y, )


def loss_rule_calc_cardio(model, batch_train_x, loss_rule_func, output, arg, ):
    """ Calculate loss for constraints rules

    """
    rule_ind = arg.rules.rule_ind
    pert_coeff = arg.rules.pert_coeff
    alpha = arg.alpha

    pert_batch_train_x             = batch_train_x.detach().clone()
    pert_batch_train_x[:,rule_ind] = get_perturbed_input(pert_batch_train_x[:,rule_ind], pert_coeff)
    pert_output = model(pert_batch_train_x, alpha= alpha)
    loss_rule   = loss_rule_func(output.reshape(pert_output.size()), pert_output)    # output should be less than pert_output
    return loss_rule





#####  covtype dataset #########################################################################
def test2():
    """function test2
    Args:
    Returns:
        
    """
    model_info = {'dataonly': {'rule': 0.0},
                'ours-beta0.1': {'beta': [0.1], 'scale': 1.0, 'lr': 0.001},
                'ours-beta0.1-scale0.1': {'beta': [0.1], 'scale': 0.1},
                'ours-beta0.1-scale0.05': {'beta': [0.1], 'scale': 0.05},
                'ours-beta0.1-pert0.001': {'beta': [0.1], 'pert': 0.001},
                'ours-beta0.1-pert0.1': {'beta': [0.1], 'pert': 0.1},
                }

    arg = Box({
      "dataurl":  "https://github.com/caravanuden/cardio/raw/master/cardio_train.csv",
      "datapath": './cardio_train.csv',

      ##### Rules
      "rules" : {},

      #####
      "train_ratio": 0.7,
      "validation_ratio": 0.1,
      "test_ratio": 0.2,

      "model_type": 'dataonly',
      "input_dim_encoder": 16,
      "output_dim_encoder": 16,
      "hidden_dim_encoder": 100,
      "hidden_dim_db": 16,
      "n_layers": 1,


      ##### Training
      "seed": 42,
      "device": 'cpu',  ### 'cuda:0',
      "batch_size": 32,
      "epochs": 1,
      "early_stopping_thld": 10,
      "valid_freq": 1,
      'saved_filename' :'./model.pt',

    })
    arg.model_info = model_info
    arg.merge = 'cat'
    arg.input_dim = 20   ### 20
    arg.output_dim = 1
    log(arg)


    #### Rules setup #############################################################
    arg.rules = {
          "rule_threshold": 129.5,
          "src_ok_ratio": 0.3,
          "src_unok_ratio": 0.7,
          "target_rule_ratio": 0.7,
          "rule_ind": 2,    ### Index of the colum Used for rule:  df.iloc[:, rule_ind ]
    }
    arg.rules.loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.
    arg.rules.loss_rule_calc = loss_rule_calc_covtype



    #### dataset load
    df = dataset_load_covtype(arg)

    #### dataset preprocess
    train_X, train_y, valid_X,  valid_y, test_X,  test_y  = dataset_preprocess_covtype(df, arg)
    arg.input_dim = train_X.shape[1]


    ##### device setup   #############################################################
    device = device_setup(arg)

    ### Create dataloader
    train_loader, valid_loader, test_loader = dataloader_create( train_X, train_y, valid_X, valid_y, test_X, test_y,  arg)

    ### Model Build
    model, losses, arg = model_build(arg=arg)

    ### Model Train
    model_train(model, losses, train_loader, valid_loader, arg=arg, )


    #### Test
    model_eval, losses = model_load(arg)
    model_evaluation(model_eval, losses.loss_task_func , arg=arg, dataset_load1= dataset_load_covtype,  dataset_preprocess1 =  dataset_preprocess_covtype  )


def dataset_load_covtype(arg)->pd.DataFrame:
  """function dataset_load_covtype
  Args:
      arg:   
  Returns:
      
  """
  from sklearn.datasets import fetch_covtype
  df = fetch_covtype(return_X_y=False, as_frame=True)
  df =df.data
  log(df)
  log(df.columns)
  df = df.iloc[:500, :10]
  log(df)
  return df


def dataset_preprocess_covtype(df, arg):
  """function dataset_preprocess_covtype
  Args:
      df:   
      arg:   
  Returns:
      
  """
  coly  = 'Slope'  # df.columns[-1]
  y_raw = df[coly]
  X_raw = df.drop([coly], axis=1)

  X_column_trans = ColumnTransformer(
        [(col, StandardScaler() if not col.startswith('Soil_Type') else Binarizer(), [col]) for col in X_raw.columns],
        remainder='passthrough')

  y_trans = StandardScaler()

  X = X_column_trans.fit_transform(X_raw)
  y = y_trans.fit_transform(y_raw.array.reshape(-1, 1))

  ### Binarize
  y = np.array([  1 if yi >0.5 else 0 for yi in y])

  seed= 42
  train_X, test_X, train_y, test_y = train_test_split(X,  y,  test_size=1 - arg.train_ratio, random_state=seed)
  valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size= arg.test_ratio / (arg.test_ratio + arg.validation_ratio), random_state=seed)
  return (np.float32(train_X), np.float32(train_y), np.float32(valid_X), np.float32(valid_y), np.float32(test_X), np.float32(test_y) )


def loss_rule_calc_covtype(model, batch_train_x, loss_rule_func, output, arg, ):
    """ Calculate loss for constraints rules

    """
    rule_ind   = arg.rules.rule_ind
    pert_coeff = arg.rules.pert_coeff
    alpha      = arg.alpha

    pert_batch_train_x             = batch_train_x.detach().clone()
    pert_batch_train_x[:,rule_ind] = get_perturbed_input(pert_batch_train_x[:,rule_ind], pert_coeff)
    pert_output = model(pert_batch_train_x, alpha= alpha)
    loss_rule   = loss_rule_func(output.reshape(pert_output.size()), pert_output)    # output should be less than pert_output
    return loss_rule




##############################################################################################
###############################################################################################
def device_setup(arg):
    """function device_setup
    Args:
        arg:   
    Returns:
        
    """
    device = arg.device
    seed   = arg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if 'gpu' in device :
        try :
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            log(e)
            device = 'cpu'
    return device


def dataloader_create(train_X=None, train_y=None, valid_X=None, valid_y=None, test_X=None, test_y=None,  arg=None):
    """function dataloader_create
    Args:
        train_X:   
        train_y:   
        valid_X:   
        valid_y:   
        test_X:   
        test_y:   
        arg:   
    Returns:
        
    """
    batch_size = arg.batch_size
    train_loader, valid_loader, test_loader = None, None, None

    if train_X is not None :
        train_X, train_y = torch.tensor(train_X, dtype=torch.float32, device=arg.device), torch.tensor(train_y, dtype=torch.float32, device=arg.device)
        train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
        log("data size", len(train_X) )

    if valid_X is not None :
        valid_X, valid_y = torch.tensor(valid_X, dtype=torch.float32, device=arg.device), torch.tensor(valid_y, dtype=torch.float32, device=arg.device)
        valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=valid_X.shape[0])
        log("data size", len(valid_X)  )

    if test_X  is not None :
        test_X, test_y   = torch.tensor(test_X,  dtype=torch.float32, device=arg.device), torch.tensor(test_y, dtype=torch.float32, device=arg.device)
        test_loader  = DataLoader(TensorDataset(test_X, test_y), batch_size=test_X.shape[0])
        log("data size:", len(test_X) )

    return train_loader, valid_loader, test_loader


def model_load(arg):
    """function model_load
    Args:
        arg:   
    Returns:
        
    """
    model_eval = model_build(arg=arg, mode='test')

    checkpoint = torch.load( arg.saved_filename)
    model_eval.load_state_dict(checkpoint['model_state_dict'])
    log("best model loss: {:.6f}\t at epoch: {}".format(checkpoint['loss'], checkpoint['epoch']))


    ll = Box({})
    ll.loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))
    ll.loss_task_func = nn.BCELoss()
    return model_eval, ll # (loss_task_func, loss_rule_func)
    # model_evaluation(model_eval, loss_task_func, arg=arg)


def model_build(arg:dict, mode='train'):
    """function model_build
    Args:
        arg ( dict ) :   
        mode:   
    Returns:
        
    """
    arg = Box(arg)

    if 'test' in mode :
        rule_encoder = RuleEncoder(arg.input_dim, arg.output_dim_encoder, arg.hidden_dim_encoder)
        data_encoder = DataEncoder(arg.input_dim, arg.output_dim_encoder, arg.hidden_dim_encoder)
        model_eval = Net(arg.input_dim, arg.output_dim, rule_encoder, data_encoder, hidden_dim=arg.hidden_dim_db, n_layers=arg.n_layers, merge=arg.merge).to(arg.device)    # Not residual connection
        return model_eval

    ##### Params  ############################################################################
    model_params = arg.model_info.get( arg.model_type, {} )

    #### Training
    arg.lr      = model_params.get('lr', 0.001)  # if 'lr' in model_params else 0.001

    #### Rules encoding
    from torch.distributions.beta import Beta
    arg.rules.pert_coeff   = model_params.get('pert', 0.1)
    arg.rules.scale        = model_params.get('scale', 1.0)
    beta_param   = model_params.get('beta', [1.0])
    if   len(beta_param) == 1:  arg.rules.alpha_dist = Beta(float(beta_param[0]), float(beta_param[0]))
    elif len(beta_param) == 2:  arg.rules.alpha_dist = Beta(float(beta_param[0]), float(beta_param[1]))
    arg.rules.beta_param = beta_param


    #########################################################################################
    losses    = Box({})

    #### Rule model
    rule_encoder          = RuleEncoder(arg.input_dim, arg.output_dim_encoder, arg.hidden_dim_encoder)
    losses.loss_rule_func = arg.rules.loss_rule_func #lambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.


    #### Data model
    data_encoder = DataEncoder(arg.input_dim, arg.output_dim_encoder, arg.hidden_dim_encoder)
    losses.loss_task_func = nn.BCELoss()    # return scalar (reduction=mean)

    #### Merge Ensembling
    model        = Net(arg.input_dim, arg.output_dim, rule_encoder, data_encoder, hidden_dim=arg.hidden_dim_db,
                        n_layers=arg.n_layers, merge= arg.merge).to(arg.device)    # Not residual connection

    ### Summary
    log('model_type: {}\tscale:\tBeta distribution: Beta()\tlr: \t \tpert_coeff: {}'.format(arg.model_type, arg.rules))
    return model, losses, arg



def loss_rule_calc(model, batch_train_x, loss_rule_func, output, arg:dict):
    """ Calculate loss for constraints rules

    """
    rule_ind   = arg.rules.rule_ind
    pert_coeff = arg.rules.pert_coeff
    alpha      = arg.alpha

    pert_batch_train_x             = batch_train_x.detach().clone()
    pert_batch_train_x[:,rule_ind] = get_perturbed_input(pert_batch_train_x[:,rule_ind], pert_coeff)
    pert_output = model(pert_batch_train_x, alpha= alpha)
    loss_rule   = loss_rule_func(output.reshape(pert_output.size()), pert_output)    # output should be less than pert_output
    return loss_rule



def model_train(model, losses, train_loader, valid_loader, arg:dict=None ):
    """function model_train
    Args:
        model:   
        losses:   
        train_loader:   
        valid_loader:   
        arg ( dict ) :   
    Returns:
        
    """
    arg      = Box(arg)  ### Params
    arghisto = Box({})  ### results


    #### Rules Loss, params  ##################################################
    rule_feature   = arg.rules.get( 'rule_feature',   'ap_hi' )
    loss_rule_func = arg.rules.loss_rule_func
    if 'loss_rule_calc' in arg.rules: loss_rule_calc = arg.rules.loss_rule_calc
    src_ok_ratio   = arg.rules.src_ok_ratio
    src_unok_ratio = arg.rules.src_unok_ratio
    rule_ind       = arg.rules.rule_ind
    pert_coeff     = arg.rules.pert_coeff


    #### Core model params
    model_params   = arg.model_info[ arg.model_type]
    lr             = model_params.get('lr',  0.001)
    optimizer      = torch.optim.Adam(model.parameters(), lr=lr)
    loss_task_func = losses.loss_task_func


    #### Train params
    model_type = arg.model_type
    # epochs     = arg.epochs
    early_stopping_thld    = arg.early_stopping_thld
    counter_early_stopping = 1
    # valid_freq     = arg.valid_freq
    seed=arg.seed
    log('saved_filename: {}\n'.format( arg.saved_filename))
    best_val_loss = float('inf')


    for epoch in range(1, arg.epochs+1):
      model.train()
      for batch_train_x, batch_train_y in train_loader:
        batch_train_y = batch_train_y.unsqueeze(-1)
        optimizer.zero_grad()

        if   model_type.startswith('dataonly'):  alpha = 0.0
        elif model_type.startswith('ruleonly'):  alpha = 1.0
        elif model_type.startswith('ours'):      alpha = arg.rules.alpha_dist.sample().item()
        arg.alpha = alpha

        ###### Base output #########################################
        output    = model(batch_train_x, alpha=alpha).view(batch_train_y.size())
        loss_task = loss_task_func(output, batch_train_y)


        ###### Loss Rule perturbed input and its output  #####################
        loss_rule = loss_rule_calc(model, batch_train_x, loss_rule_func, output, arg )


        #### Total Losses  ##################################################
        scale = 1
        loss  = alpha * loss_rule + scale * (1 - alpha) * loss_task
        loss.backward()
        optimizer.step()


      # Evaluate on validation set
      if epoch % arg.valid_freq == 0:
        model.eval()
        if  model_type.startswith('ruleonly'):  alpha = 1.0
        else:                                   alpha = 0.0

        with torch.no_grad():
          for val_x, val_y in valid_loader:
            val_y = val_y.unsqueeze(-1)

            output = model(val_x, alpha=alpha).reshape(val_y.size())
            val_loss_task = loss_task_func(output, val_y).item()

            # perturbed input and its output
            pert_val_x = val_x.detach().clone()
            pert_val_x[:,rule_ind] = get_perturbed_input(pert_val_x[:,rule_ind], pert_coeff)
            pert_output = model(pert_val_x, alpha=alpha)    # \hat{y}_{p}    predicted sales from perturbed input

            val_loss_rule = loss_rule_func(output.reshape(pert_output.size()), pert_output).item()
            val_ratio = verification(pert_output, output, threshold=0.0).item()

            val_loss = val_loss_task

            y_true = val_y.cpu().numpy()
            y_score = output.cpu().numpy()
            y_pred = np.round(y_score)

            y_true = y_pred.reshape(y_true.shape[:-1])
            y_pred = y_pred.reshape(y_pred.shape[:-1])

            val_acc = mean_squared_error(y_true, y_pred)

          if val_loss < best_val_loss:
            counter_early_stopping = 1
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(model.state_dict())
            log('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f} best model is updated %%%%'
                  .format(epoch, best_val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio))
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, arg.saved_filename)
          else:
            log('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f}({}/{})'
                  .format(epoch, val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio, counter_early_stopping, early_stopping_thld))
            if counter_early_stopping >= early_stopping_thld:
              break
            else:
              counter_early_stopping += 1


def model_evaluation(model_eval, loss_task_func, arg, dataset_load1, dataset_preprocess1 ):
    """function model_evaluation
    Args:
        model_eval:   
        loss_task_func:   
        arg:   
        dataset_load1:   
        dataset_preprocess1:   
    Returns:
        
    """
    ### Create dataloader
    df = dataset_load1(arg)
    train_X, test_X, train_y, test_y, valid_X, valid_y = dataset_preprocess1(df, arg)



    ######
    train_loader, valid_loader, test_loader = dataloader_create( train_X, test_X, train_y, test_y, valid_X, valid_y, arg)
    model_eval.eval()
    with torch.no_grad():
      for te_x, te_y in test_loader:
        te_y = te_y.unsqueeze(-1)

      output         = model_eval(te_x, alpha=0.0)
      test_loss_task = loss_task_func(output, te_y.view(output.size())).item()

    log('\n[Test] Average loss: {:.8f}\n'.format(test_loss_task))

    ########## Pertfubation
    pert_coeff = arg.rules.pert_coeff
    rule_ind   = arg.rules.rule_ind
    model_type = arg.model_type
    alphas     = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]


    model_eval.eval()

    # perturbed input and its output
    pert_test_x = te_x.detach().clone()
    pert_test_x[:,rule_ind] = get_perturbed_input(pert_test_x[:,rule_ind], pert_coeff)
    for alpha in alphas:
      model_eval.eval()
      with torch.no_grad():
        for te_x, te_y in test_loader:
          te_y = te_y.unsqueeze(-1)

        if model_type.startswith('dataonly'):
          output = model_eval(te_x, alpha=0.0)
        elif model_type.startswith('ours'):
          output = model_eval(te_x, alpha=alpha)
        elif model_type.startswith('ruleonly'):
          output = model_eval(te_x, alpha=1.0)

        test_loss_task = loss_task_func(output, te_y.view(output.size())).item()

        if model_type.startswith('dataonly'):
          pert_output = model_eval(pert_test_x, alpha=0.0)
        elif model_type.startswith('ours'):
          pert_output = model_eval(pert_test_x, alpha=alpha)
        elif model_type.startswith('ruleonly'):
          pert_output = model_eval(pert_test_x, alpha=1.0)

        test_ratio = verification(pert_output, output, threshold=0.0).item()

        y_true = te_y.cpu().numpy()
        y_score = output.cpu().numpy()
        y_pred = np.round(y_score)

        test_acc = mean_squared_error(y_true.squeeze(), y_pred.squeeze())

      log('[Test] Average loss: {:.8f} (alpha:{})'.format(test_loss_task, alpha))
      log('[Test] Accuracy: {:.4f} (alpha:{})'.format(test_acc, alpha))
      log("[Test] Ratio of verified predictions: {:.6f} (alpha:{})".format(test_ratio, alpha))
      log()




#############################################################################################
'''model.py '''
class NaiveModel(nn.Module):
  def __init__(self):
    """ NaiveModel:__init__
    Args:
    Returns:
       
    """
    super(NaiveModel, self).__init__()
    self.net = nn.Identity()

  def forward(self, x, alpha=0.0):
    """ Net:forward
    Args:
        x:     
        alpha:     
    Returns:
       
    """
    """ NaiveModel:forward
    Args:
        x:     
        alpha:     
    Returns:
       
    """
    return self.net(x)


class RuleEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=4):
    """ RuleEncoder:__init__
    Args:
        input_dim:     
        output_dim:     
        hidden_dim:     
    Returns:
       
    """
    """ DataEncoder:__init__
    Args:
        input_dim:     
        output_dim:     
        hidden_dim:     
    Returns:
       
    """
    super(RuleEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim))

  def forward(self, x):
    """ RuleEncoder:forward
    Args:
        x:     
    Returns:
       
    """
    """ DataEncoder:forward
    Args:
        x:     
    Returns:
       
    """
    return self.net(x)


class DataEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=4):
    super(DataEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim))

  def forward(self, x):
    return self.net(x)


class Net(nn.Module):
  def __init__(self, input_dim, output_dim, rule_encoder, data_encoder, hidden_dim=4, n_layers=2, merge='cat', skip=False, input_type='state'):
    """ Net:__init__
    Args:
        input_dim:     
        output_dim:     
        rule_encoder:     
        data_encoder:     
        hidden_dim:     
        n_layers:     
        merge:     
        skip:     
        input_type:     
    Returns:
       
    """
    super(Net, self).__init__()
    self.skip = skip
    self.input_type   = input_type
    self.rule_encoder = rule_encoder
    self.data_encoder = data_encoder
    self.n_layers =n_layers
    assert self.rule_encoder.input_dim == self.data_encoder.input_dim
    assert self.rule_encoder.output_dim == self.data_encoder.output_dim
    self.merge = merge
    if merge == 'cat':
      self.input_dim_decision_block = self.rule_encoder.output_dim * 2
    elif merge == 'add':
      self.input_dim_decision_block = self.rule_encoder.output_dim

    self.net = []
    for i in range(n_layers):
      if i == 0:
        in_dim = self.input_dim_decision_block
      else:
        in_dim = hidden_dim

      if i == n_layers-1:
        out_dim = output_dim
      else:
        out_dim = hidden_dim

      self.net.append(nn.Linear(in_dim, out_dim))
      if i != n_layers-1:
        self.net.append(nn.ReLU())

    self.net.append(nn.Sigmoid())

    self.net = nn.Sequential(*self.net)

  def get_z(self, x, alpha=0.0):
    """ Net:get_z
    Args:
        x:     
        alpha:     
    Returns:
       
    """
    rule_z = self.rule_encoder(x)
    data_z = self.data_encoder(x)

    if self.merge == 'add':
      z = alpha*rule_z + (1-alpha)*data_z
    elif self.merge == 'cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)
    elif self.merge == 'equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)

    return z

  def forward(self, x, alpha=0.0):
    # merge: cat or add

    rule_z = self.rule_encoder(x)
    data_z = self.data_encoder(x)

    if self.merge == 'add':
      z = alpha*rule_z + (1-alpha)*data_z
    elif self.merge == 'cat':
      z = torch.cat((alpha*rule_z, (1-alpha)*data_z), dim=-1)
    elif self.merge == 'equal_cat':
      z = torch.cat((rule_z, data_z), dim=-1)

    if self.skip:
      if self.input_type == 'seq':
        return self.net(z) + x[:, -1, :]
      else:
        return self.net(z) + x    # predict delta values
    else:
      return self.net(z)    # predict absolute values



#####################################################################################################################
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc, roc_auc_score, precision_score, recall_score, precision_recall_curve, accuracy_score

def get_metrics(y_true, y_pred, y_score):
    """function get_metrics
    Args:
        y_true:   
        y_pred:   
        y_score:   
    Returns:
        
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return acc, prec, recall, fpr, tpr, roc_auc

def get_correct_results(out, label_Y):
    """function get_correct_results
    Args:
        out:   
        label_Y:   
    Returns:
        
    """
    y_pred_tag = torch.round(out)    # Binary label
    return (y_pred_tag == label_Y).sum().float()

def verification(out, pert_out, threshold=0.0):
    '''
    return the ratio of qualified samples.
    '''
    if isinstance(out, torch.Tensor):
        return 1.0*torch.sum(pert_out.view(out.size())-out < threshold) / out.shape[0]
    else:
        return 1.0*np.sum(pert_out.reshape(out.shape)-out < threshold) / out.shape[0]

def get_perturbed_input(input_tensor, pert_coeff):
    '''
    X = X + pert_coeff*rand*X
    return input_tensor + input_tensor*pert_coeff*torch.rand()
    '''
    device = input_tensor.device
    return input_tensor + torch.abs(input_tensor)*pert_coeff*torch.rand(input_tensor.shape, device=device)




def test_dataset_classification_fake(nrows=500):
    """function test_dataset_classification_fake
    Args:
        nrows:   
    Returns:
        
    """
    from sklearn import datasets as sklearn_datasets
    ndim    =11
    coly    = 'y'
    colnum  = ["colnum_" +str(i) for i in range(0, ndim) ]
    colcat  = ['colcat_1']
    X, y    = sklearn_datasets.make_classification(n_samples=nrows, n_features=ndim, n_classes=1,
                                                   n_informative=ndim-2)
    df         = pd.DataFrame(X,  columns= colnum)
    df[coly]   = y.reshape(-1, 1)

    for ci in colcat :
      df[ci] = np.random.randint(0,1, len(df))

    pars = { 'colnum': colnum, 'colcat': colcat, "coly": coly }
    return df, pars




###################################################################################################
if __name__ == "__main__":
    import fire 
    fire.Fire() 
    # test_all()

