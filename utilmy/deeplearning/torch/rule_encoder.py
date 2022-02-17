# -*- coding: utf-8 -*-
MNAME = "utilmy.deeplearning.torch.rule_encoder"
HELP = """ utils for model explanation
"""
import os, random, numpy as np, glob, pandas as pd, matplotlib.pyplot as plt ;from box import Box
from copy import deepcopy
from argparse import ArgumentParser


from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.beta import Beta

#### Types


#############################################################################################
from utilmy import log, log2

def help():
    from utilmy import help_create
    ss = HELP + help_create(MNAME)
    print(ss)


#############################################################################################
def test_all():
    log(MNAME)
    test()


def test():
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


    args = Box({
      "datapath": './cardio_train.csv',
      "rule_threshold": 129.5,
      "src_usual_ratio": 0.3,
      "src_unusual_ratio": 0.7,
      "target_rule_ratio": 0.7,


      "seed": 42,
      "device": 'cpu',  ### 'cuda:0',



      "batch_size": 32,
      "train_ratio": 0.7,
      "validation_ratio": 0.1,
      "test_ratio": 0.2,
      
      "model_type": 'dataonly',
      "input_dim_encoder": 16,
      "output_dim_encoder": 16,
      "hidden_dim_encoder": 100,
      "hidden_dim_db": 16,
      "n_layers": 1,
      "rule_ind": 5,
      
      "epochs": 2,
      "early_stopping_thld": 10,
      "valid_freq": 1,


      'saved_filename' :'./model.pt',

    })
    print(args)

    #url = "https://github.com/caravanuden/cardio/raw/master/cardio_train.csv"
    #import wget 
    #wget.download(url)
    #datadf = pd.read_csv("./cardio_train.csv",delimiter=';')
    #df = datadf.drop(['id'], axis=1)

    # loss_task_func = nn.BCELoss()

    # device = args.device
    # seed = args.seed
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    # datapath = args.datapath

    args.model_info = model_info
    
    args.merge = 'cat'
    args.input_dim = 20   ### 20
    # args.output_dim_encoder = args.output_dim_encoder
    # args.hidden_dim_encoder = args.hidden_dim_encoder
    # args.hidden_dim_db = args.args.hidden_dim_db
    # args.n_layers = args.args.n_layers
    args.output_dim = 1


    ### device setup
    device = device_setup(args)

    ### dataset load
    df = dataset_load(args)

    ### dataset preprocess
    train_X, test_X, train_y, test_y, valid_X, valid_y = dataset_preprocess(df, args)
           

    ### Create dataloader
    train_loader, valid_loader, test_loader = dataloader_create( train_X, test_X, train_y, test_y, valid_X, valid_y, args)

    ### Model Build
    model, optimizer, (loss_rule_func, loss_task_func) = model_build(args=args)


    ### Model Train
    model_train(model, optimizer, loss_rule_func, loss_task_func, train_loader, valid_loader, args=args )


    #### Test
    model_eval = model_build(args=args, mode='test')

    #rule_encoder = RuleEncoder(args.input_dim, args.output_dim_encoder, args.hidden_dim_encoder)
    #data_encoder = DataEncoder(args.input_dim, args.output_dim_encoder, args.hidden_dim_encoder)
    #model_eval = Net(args.input_dim, args.output_dim, rule_encoder, data_encoder, hidden_dim=args.hidden_dim_db, n_layers=args.n_layers, merge=args.merge).to(args.device)    # Not residual connection

    checkpoint = torch.load( args.saved_filename)
    model_eval.load_state_dict(checkpoint['model_state_dict'])
    print("best model loss: {:.6f}\t at epoch: {}".format(checkpoint['loss'], checkpoint['epoch']))
    

    model_evaluation(model_eval, args=args)
         



#####################################################################################################################
def dataset_load(args):
  # Load dataset
  url = "https://github.com/caravanuden/cardio/raw/master/cardio_train.csv"
  import wget 
  wget.download(url)

  df = pd.read_csv(args.datapath,delimiter=';')
  log(df, df.columns, df.shape)

  # y = df['cardio']
  # X_raw = df.drop(['cardio'], axis=1)

  return df


def dataset_preprocess(df, args):

    y     = df['cardio']
    X_raw = df.drop(['cardio'], axis=1)    

    print("Target class ratio:")
    print("# of cardio=1: {}/{} ({:.2f}%)".format(np.sum(y==1), len(y), 100*np.sum(y==1)/len(y)))
    print("# of cardio=0: {}/{} ({:.2f}%)\n".format(np.sum(y==0), len(y), 100*np.sum(y==0)/len(y)))

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
    num_samples = X.shape[0]
    X_np = X.copy()

    # Rule : higher ap -> higher risk
    rule_threshold = args.rule_threshold
    rule_ind = args.rule_ind
    rule_feature = 'ap_hi'

    low_ap_negative = (df[rule_feature] <= rule_threshold) & (df['cardio'] == 0)    # usual
    high_ap_positive = (df[rule_feature] > rule_threshold) & (df['cardio'] == 1)    # usual
    low_ap_positive = (df[rule_feature] <= rule_threshold) & (df['cardio'] == 1)    # unusual
    high_ap_negative = (df[rule_feature] > rule_threshold) & (df['cardio'] == 0)    # unusual



    # Samples in Usual group
    X_usual = X[low_ap_negative | high_ap_positive]
    y_usual = y[low_ap_negative | high_ap_positive]
    y_usual = y_usual.to_numpy()
    X_usual, y_usual = shuffle(X_usual, y_usual, random_state=0)
    num_usual_samples = X_usual.shape[0]

    # Samples in Unusual group
    X_unusual = X[low_ap_positive | high_ap_negative]
    y_unusual = y[low_ap_positive | high_ap_negative]
    y_unusual = y_unusual.to_numpy()
    X_unusual, y_unusual = shuffle(X_unusual, y_unusual, random_state=0)
    num_unusual_samples = X_unusual.shape[0]

    # Build a source dataset
    src_usual_ratio = args.src_usual_ratio
    src_unusual_ratio = args.src_unusual_ratio
    num_samples_from_unusual = int(src_unusual_ratio * num_unusual_samples)
    num_samples_from_usual = int(num_samples_from_unusual * src_usual_ratio / (1-src_usual_ratio))

    X_src = np.concatenate((X_usual[:num_samples_from_usual], X_unusual[:num_samples_from_unusual]), axis=0)
    y_src = np.concatenate((y_usual[:num_samples_from_usual], y_unusual[:num_samples_from_unusual]), axis=0)
    print()
    print("Source dataset statistics:")
    print("# of samples in Usual group: {}".format(num_samples_from_usual))
    print("# of samples in Unusual group: {}".format(num_samples_from_unusual))
    print("Usual ratio: {:.2f}%".format(100 * num_samples_from_usual / (X_src.shape[0])))
    seed= 42

    train_ratio = args.train_ratio
    validation_ratio = args.validation_ratio
    test_ratio = args.test_ratio
    train_X, test_X, train_y, test_y = train_test_split(X_src, y_src, test_size=1 - train_ratio, random_state=seed)
    valid_X, test_X, valid_y, test_y = train_test_split(test_X, test_y, test_size=test_ratio / (test_ratio + validation_ratio), random_state=seed)
    return (train_X, test_X, train_y, test_y, valid_X,  valid_y)



#####################################################################################################################
def device_setup(args):
    device = args.device
    seed   = args.seed
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
    return device


def dataloader_create(train_X, test_X, train_y, test_y, valid_X, valid_y,  args):
    #device= device_setup(args)
    # train_X, test_X, train_y, test_y, valid_X, test_X, valid_y, test_y = X
    # rain_X, test_X, train_y, test_y, valid_X, test_X, valid_y, test_y=dataset_preprocess(X_raw, y, args)
    train_X, train_y = torch.tensor(train_X, dtype=torch.float32, device=args.device), torch.tensor(train_y, dtype=torch.float32, device=args.device)
    valid_X, valid_y = torch.tensor(valid_X, dtype=torch.float32, device=args.device), torch.tensor(valid_y, dtype=torch.float32, device=args.device)
    test_X, test_y = torch.tensor(test_X,    dtype=torch.float32, device=args.device), torch.tensor(test_y, dtype=torch.float32, device=args.device)

    batch_size = args.batch_size

    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=valid_X.shape[0])
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=test_X.shape[0])
    print("data size: {}/{}/{}".format(len(train_X), len(valid_X), len(test_X)))

    return train_loader, valid_loader, test_loader




def model_build(args, mode='train'):
  # device, seed, datapath, input_dim, args.output_dim,args.output_dim_encoder, args.hidden_dim_encoder, args.hidden_dim_db, args.n_layers,merge= arguments(args)
  # device = device_setup(args)

  if 'test' in mode :
    rule_encoder = RuleEncoder(args.input_dim, args.output_dim_encoder, args.hidden_dim_encoder)
    data_encoder = DataEncoder(args.input_dim, args.output_dim_encoder, args.hidden_dim_encoder)
    model_eval = Net(args.input_dim, args.output_dim, rule_encoder, data_encoder, hidden_dim=args.hidden_dim_db, n_layers=args.n_layers, merge=args.merge).to(args.device)    # Not residual connection
    return model_eval 

  model_info = args.model_info
  model_type = args.model_type
  if model_type not in model_info:
    # default setting
    lr = 0.001
    pert_coeff = 0.1
    scale = 1.0
    beta_param = [1.0]
    alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
    model_params = {}

  else:
    model_params = model_info[model_type]
    lr = model_params['lr'] if 'lr' in model_params else 0.001
    pert_coeff = model_params['pert'] if 'pert' in model_params else 0.1
    scale = model_params['scale'] if 'scale' in model_params else 1.0
    beta_param = model_params['beta'] if 'beta' in model_params else [1.0]

    if len(beta_param) == 1:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
    elif len(beta_param) == 2:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[1]))

    print('model_type: {}\tscale:{}\tBeta distribution: Beta({})\tlr: {}\t \tpert_coeff: {}'.format(model_type, scale, beta_param, lr, pert_coeff))



    rule_encoder = RuleEncoder(args.input_dim, args.output_dim_encoder, args.hidden_dim_encoder)
    data_encoder = DataEncoder(args.input_dim, args.output_dim_encoder, args.hidden_dim_encoder)
    model = Net(args.input_dim, args.output_dim, rule_encoder, data_encoder, hidden_dim=args.hidden_dim_db, n_layers=args.n_layers, merge= args.merge).to(args.device)    # Not residual connection

    optimizer = optim.Adam(model.parameters(), lr=lr)        
    loss_rule_func = lambda x,y: torch.mean(F.relu(x-y))    # if x>y, penalize it.
    loss_task_func = nn.BCELoss()    # return scalar (reduction=mean)

    return model, optimizer, (loss_rule_func, loss_task_func)


def model_train(model, optimizer, loss_rule_func, loss_task_func, train_loader, valid_loader, args ):
    model_type = args.model_type
    epochs     = args.epochs
    early_stopping_thld    = args.early_stopping_thld
    counter_early_stopping = 1
    valid_freq = args.valid_freq 
    src_usual_ratio = args.src_usual_ratio
    src_unusual_ratio = args.src_unusual_ratio
    model_type=args.model_type
    rule_feature = 'ap_hi'
    seed=args.seed
    #saved_filename = 'cardio_{}_rule-{}_src{}-target{}_seed{}.demo.pt'.format(model_type, rule_feature, src_usual_ratio, src_usual_ratio, seed)
    #saved_filename =  os.path.join("/content/drive/MyDrive/", saved_filename)
    print('saved_filename: {}\n'.format( args.saved_filename))
    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
      model.train()
      for batch_train_x, batch_train_y in train_loader:
        batch_train_y = batch_train_y.unsqueeze(-1)

        optimizer.zero_grad()

        if model_type.startswith('dataonly'):
          alpha = 0.0
        elif model_type.startswith('ruleonly'):
          alpha = 1.0
        elif model_type.startswith('ours'):
          alpha = alpha_distribution.sample().item()

        # stable output
        output    = model(batch_train_x, alpha=alpha)
        loss_task = loss_task_func(output, batch_train_y)

        ###### perturbed input and its output  #####################
        pert_batch_train_x = batch_train_x.detach().clone()
        rule_ind = args.rule_ind
        pert_coeff = 0.1
        pert_batch_train_x[:,rule_ind] = get_perturbed_input(pert_batch_train_x[:,rule_ind], pert_coeff)
        pert_output = model(pert_batch_train_x, alpha=alpha)
        scale = 1
        loss_rule = loss_rule_func(output, pert_output)    # output should be less than pert_output
        loss      = alpha * loss_rule + scale * (1 - alpha) * loss_task

        loss.backward()
        optimizer.step()

      # Evaluate on validation set
      if epoch % valid_freq == 0:
        model.eval()
        if  model_type.startswith('ruleonly'):
          alpha = 1.0
        else:
          alpha = 0.0

        with torch.no_grad():
          for val_x, val_y in valid_loader:
            val_y = val_y.unsqueeze(-1)

            output = model(val_x, alpha=alpha)
            val_loss_task = loss_task_func(output, val_y).item()

            # perturbed input and its output
            pert_val_x = val_x.detach().clone()
            pert_val_x[:,rule_ind] = get_perturbed_input(pert_val_x[:,rule_ind], pert_coeff)
            pert_output = model(pert_val_x, alpha=alpha)    # \hat{y}_{p}    predicted sales from perturbed input

            val_loss_rule = loss_rule_func(output, pert_output).item()
            val_ratio = verification(pert_output, output, threshold=0.0).item()

            val_loss = val_loss_task

            y_true = val_y.cpu().numpy()
            y_score = output.cpu().numpy()
            y_pred = np.round(y_score)
            val_acc = 100 * accuracy_score(y_true, y_pred)

          if val_loss < best_val_loss:
            counter_early_stopping = 1
            best_val_loss = val_loss
            best_model_state_dict = deepcopy(model.state_dict())
            print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f} best model is updated %%%%'
                  .format(epoch, best_val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio))
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, args.saved_filename)
          else:
            print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f}({}/{})'
                  .format(epoch, val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio, counter_early_stopping, early_stopping_thld))
            if counter_early_stopping >= early_stopping_thld:
              break
            else:
              counter_early_stopping += 1


def model_evaluation(model_eval, loss_task_func, args):
    X_raw, y = dataset_load(args)
    train_loader, valid_loader, test_loader = dataloader_create(X_raw, y, args)
    model_eval.eval()
    with torch.no_grad():
      for te_x, te_y in test_loader:
        te_y = te_y.unsqueeze(-1)

      output = model_eval(te_x, alpha=0.0)
      test_loss_task = loss_task_func(output, te_y).item()
    print('\n[Test] Average loss: {:.8f}\n'.format(test_loss_task))
    pert_coeff = 0.1
    model_eval.eval()
    rule_ind = args.rule_ind
    model_type=args.model_type
    alphas = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
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

        test_loss_task = loss_task_func(output, te_y).item()

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
        test_acc = accuracy_score(y_true, y_pred)

      print('[Test] Average loss: {:.8f} (alpha:{})'.format(test_loss_task, alpha))
      print('[Test] Accuracy: {:.4f} (alpha:{})'.format(test_acc, alpha))
      print("[Test] Ratio of verified predictions: {:.6f} (alpha:{})".format(test_ratio, alpha))
      print()
 



#############################################################################################
'''model.py '''
class NaiveModel(nn.Module):
  def __init__(self):
    super(NaiveModel, self).__init__()
    self.net = nn.Identity()

  def forward(self, x, alpha=0.0):
    return self.net(x)


class RuleEncoder(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim=4):
    super(RuleEncoder, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Linear(hidden_dim, output_dim))

  def forward(self, x):
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
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, precision_score, recall_score, precision_recall_curve, accuracy_score

def get_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    return acc, prec, recall, fpr, tpr, roc_auc

def get_correct_results(out, label_Y):
    y_pred_tag = torch.round(out)    # Binary label
    return (y_pred_tag == label_Y).sum().float()

def verification(out, pert_out, threshold=0.0):
    '''
    return the ratio of qualified samples.
    '''
    if isinstance(out, torch.Tensor):
        return 1.0*torch.sum(pert_out-out < threshold) / out.shape[0]
    else:
        return 1.0*np.sum(pert_out-out < threshold) / out.shape[0]
      
def get_perturbed_input(input_tensor, pert_coeff):
    '''
    X = X + pert_coeff*rand*X
    return input_tensor + input_tensor*pert_coeff*torch.rand()
    '''
    device = input_tensor.device
    return input_tensor + torch.abs(input_tensor)*pert_coeff*torch.rand(input_tensor.shape, device=device)







###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()

