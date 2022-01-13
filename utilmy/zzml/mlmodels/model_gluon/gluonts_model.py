# -*- coding: utf-8 -*-
"""
Advanded GlutonTS models


"""
import os, copy
import pandas as pd, numpy as np

import importlib
import matplotlib.pyplot as plt
from pathlib import Path
import json


from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
from gluonts.model.seq2seq import Seq2SeqEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.simple_feedforward import  SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator, WaveNetSampler, WaveNet

from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor



from gluonts.dataset.common import ListDataset,load_datasets
from gluonts.dataset.repository.datasets import get_dataset as get_dataset_gluon
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas



from tqdm.autonotebook import tqdm


#### Only for SeqtoSeq
from gluonts.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    RNNCovariateEncoder,
    MLPEncoder,
    Seq2SeqEncoder,  # Buggy, not implemented
)


####################################################################################################
from mlmodels.util import log, path_norm, get_model_uri, json_norm
from mlmodels.util import load_function_uri



VERBOSE = False
MODEL_URI = get_model_uri(__file__)


MODELS_DICT = {
"deepar"         : DeepAREstimator
,"deepstate"     : DeepStateEstimator
,"deepfactor"    : DeepFactorEstimator
,"gp_forecaster" : GaussianProcessEstimator
,"seq2seq"       : Seq2SeqEstimator
,"feedforward"   : SimpleFeedForwardEstimator
,"transformer"   : TransformerEstimator
,"wavenet"       : WaveNetEstimator
}


####################################################################################################
class Model(object):
    def __init__(self, model_pars=None, data_pars=None,  compute_pars=None, **kwargs):
        self.compute_pars = compute_pars
        self.model_pars   = model_pars
        self.data_pars    = data_pars
        

        ##### Empty model for Seiialization
        if model_pars is None :
            self.model = None

        else:
            mpars = json_norm(model_pars['model_pars'] )      #"None" to None
            cpars = json_norm(compute_pars['compute_pars'])
            
            if model_pars["model_name"] == "seq2seq" :
                mpars['encoder'] = MLPEncoder()   #bug in seq2seq


            if model_pars["model_name"] == "deepar" :                    
                # distr_output â€“ Distribution to use to evaluate observations and sample predictions (default: StudentTOutput())
                # mpars['distr_output'] =_load_function(mpars['distr_output'])()  # "gluonts.distribution.neg_binomial:NegativeBinomialOutput"

                if "NegativeBinomialOutput" in  mpars['distr_output'] :  
                   from gluonts.distribution.neg_binomial import NegativeBinomialOutput           
                   mpars['distr_output'] = NegativeBinomialOutput()
                else :
                   del mpars['distr_output']  # = StudentTOutput() default one
                # log(mpars.get('distr_output'))


                ###  Cardinality : Nb of unique values for each category
                # self.train_ds,self.test_ds, self.cardinalities = get_dataset(data_pars)             
                log( 'Cardinality', mpars.get("cardinality" ) )


            ### Setup the compute
            trainer = Trainer( **cpars  )

            ### Setup the model
            self.model = MODELS_DICT[model_pars["model_name"]]( **mpars, trainer=trainer )



def get_params(choice="", data_path="dataset/timeseries/", config_mode="test", **kw):
    if choice == "json":
      data_path = path_norm( data_path )
      config    = json.load(open(data_path))  #, encoding='utf-8'))
      config    = config[config_mode]
      
      return config["model_pars"], config["data_pars"], config["compute_pars"], config["out_pars"]
  
    else :
        raise Exception("Error no JSON FILE") 






def get_dataset2(data_pars):    
    """
        "data_pars": {
            "data_info": {
            },
            "preprocessors": [


                {   "name" :  "get_dataset_process",
                    "uri"  : "mlmodels.model_gluon.gluonts_model:get_dataset_process",
                    "args"  : { 

                        "train": true, 
                        "dt_source": "",

                        "use_feat_dynamic_real":true, 
                        "use_feat_static_cat": true, 
                        "use_feat_static_real":false,
                        
                        "data_type": "gluonts",
                       
                        "submission": false , 
                        
                        "data_path":  "dataset/timeseries/covid19_gluonts/" , 
                       
                        "single_pred_length":28,
                        "freq": "1D",
                        "start" : "2015-02-26 21:42:53",        
                        "startdate" : "2011-01-29",  
                        "col_date"   : "timestamp",                
                        "col_ytarget" : ["value"],  "num_series" : 1,

                        "cols_cat": [],   "cols_num" : []
                    }
                    }
                }
            ]
        },


    """
    from mlmodels.dataloader import DataLoader
    loader = DataLoader(data_pars)
    loader.compute()
    return loader.get_data()




def get_dataset(data_pars):    
    """
      Return   train, test 

    """
    d = data_pars.get("data_type", "single_dataframe")

    if d ==  "gluonts"          : uri = "mlmodels.model_gluonts.gluonts_model:get_dataset_gluonts"
    if d ==  "single_dataframe" : uri = "mlmodels.model_gluonts.gluonts_model:get_dataset_pandas_single"
    if d ==  "multi_dataframe"  : uri = "mlmodels.model_gluonts.gluonts_model:get_dataset_pandas_multi"
    else :  raise Exception("No loader") 

    # uri = "mlmodels.model_gluonts.gluonts_model:get_dataset_gluonts" 
    # loader = load_function_uri(uri)
    # train, test = loader(data_pars)


    if d ==  "gluonts" :
        train, test = get_dataset_gluonts(data_pars

    if d ==  "single_dataframe" :
        train, test = get_dataset_pandas_single(data_pars)

    if d ==  "single_dataframe" :
        # Mutiple dataframe
        train, test = get_dataset_pandas_multi(data_pars)

    else :
         raise Exception("No loader") 

    return train, test



def get_dataset_pandas_multi(data_pars):
    """"
      ### Multiple dataframe, M5 dataset

    """ 
    from mlmodels.preprocess.timeseries import (  gluonts_create_dynamic,  gluonts_create_static,
          gluonts_create_timeseries, create_startdate,
          pandas_to_gluonts_multiseries )

    d = data_pars
    data_path      = d[ "data_path"]
    df_timeseries  = pd.read_csv(data_path+'/df_timeseries.csv')

    #### Optional
    df_static      = pd.read_csv(data_path+'/df_static.csv')  if d.get('use_feat_static_cat', False)  else None
    df_dynamic     = pd.read_csv(data_path+'/df_dynamic.csv')  if d.get('use_feat_dynamic_real', False)  else None
    df_static_real = pd.read_csv(data_path+'/df_static_real.csv')  if d.get('use_feat_static_real', False)  else None


    ###### Set parameters of dataset
    pars                   = {'submission': d['submission'],
                              'single_pred_length'     : d['single_pred_length'],    # 28
                              'submission_pred_length' : d.get('submission_pred_length', d['single_pred_length' * 2]),
                              'n_timeseries'           : d['n_timeseries']   ,
                              'start_date'             : d['startdate'] ,   #  "2011-01-29"
                              'freq'                   : d['freq'] 
                             }
    

    if data_pars['train'] :
       train_ds, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static,pars)       
       return train_ds, test_ds


    else :
       ### Submission mode
       _, test_ds, cardinalities   = pandas_to_gluonts_multiseries(df_timeseries, df_dynamic, df_static,pars) 
       return None, test_ds
    


def get_dataset_gluonts(data_pars):
    """
     Read gluonts data stored in json from disk

    """
    from gluonts.dataset.common import ListDataset, load_datasets
    from pathlib import Path
    dataset_path = Path(data_pars['data_path'])    
    TD           = load_datasets( metadata=dataset_path,
                                  train=dataset_path / "train", test=dataset_path / "test",)
    
    #TD              =get_dataset( data_pars['dataset_name'], Path(data_pars['data_path']))
    
    if data_pars['train'] :     
      return TD.train , TD.test

    else :
      return None, TD.test 
    

    
def get_dataset_pandas_single(data_pars):    
    """
      Using One Single Dataframe as INput
    """
    from mlmodels.preprocess.timeseries import pandas_to_gluonts, pd_clean_v1
    # log(data_pars)
    data_path = data_pars['data_path']

    pars = { "start" : data_pars['start'], 
             "cols_target" : data_pars['col_ytarget'],
             "freq"        : data_pars['freq'],
             "cols_cat"    : data_pars["cols_cat"],
             "cols_num"    : data_pars["cols_num"]
        }   

    ### Train
    if data_pars.get('train', False) :
        df = pd.read_csv(data_path + "/train/data.csv")
        df = df.set_index( data_pars['col_date'] )
        df = pd_clean_v1(df)
        # start_date = pd.Timestamp( data_pars['start'], freq=data_pars['freq'])
 
        gluonts_ds = pandas_to_gluonts(df, pars=pars) 
 

    df = pd.read_csv(data_path + "/test/data.csv")
    df = df.set_index( data_pars['col_date'] )
    df = pd_clean_v1(df)
    # start_date = pd.Timestamp( data_pars['start'], freq=data_pars['freq'])  
    gluonts_ds_test = pandas_to_gluonts(df, pars=pars) 


    if VERBOSE:
        entry        = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig     = data_pars.get('save_fig', "save_fig.png")
        # plt.savefig(save_fig)

    if data_pars['train'] :     
      return gluonts_ds, gluonts_ds_test

    else :
      return None, gluonts_ds_test




def fit(model, sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None, **kwargs):
        """
          Classe Model --> model,   model.model contains thte sub-model
        ### OLD CODE
        log(data_pars,model_pars)
        data_pars['train'] = True
        
        gluont_ds          = get_dataset(data_pars)
        predictor          = model_gluon.train(gluont_ds)

        #### New version
        if data_pars['new'] = True :
        """

        data_pars['train'] = 1        
        train_ds, test_ds  = get_dataset(data_pars)
        
        model_gluon        = model.model
        
        predictor          = model_gluon.train(train_ds)
       
        #predictor          = model_gluon.train(model.train_ds)
        model.model        = predictor
        return model



def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kw):
    """

Converting forecasts back to M5 submission format (if submission is True)
Since GluonTS estimators return a sample-based probabilistic forecasting predictor, we first need to reduce these results to a single pred per time series. This can be done by computing the mean or median over the predicted sample paths.
########################
if submission == True:
    forecasts_acc = np.zeros((len(forecasts), pred_length))
    for i in range(len(forecasts)):
        forecasts_acc[i] = np.mean(forecasts[i].samples, axis=0)


# We then reshape the forecasts into the correct data shape for submission ...
########################
if submission == True:
    forecasts_acc_sub = np.zeros((len(forecasts)*2, single_pred_length))
    forecasts_acc_sub[:len(forecasts)] = forecasts_acc[:,:single_pred_length]
    forecasts_acc_sub[len(forecasts):] = forecasts_acc[:,single_pred_length:]

.. and verfiy that reshaping is consistent.
########################
if submission == True:
    np.all(np.equal(forecasts_acc[0], np.append(forecasts_acc_sub[0], forecasts_acc_sub[30490])))


## Then, we save our submission into a timestamped CSV file which can subsequently be uploaded to Kaggle.
########################
if submission == True:
    import time
    sample_submission            = pd.read_csv(data_folder/sample_submission.csv')
    sample_submission.iloc[:,1:] = forecasts_acc_sub
    submission_id                = 'submission_{}.csv'.format(int(time.time()))
    sample_submission.to_csv(submission_id, index=False)




    """
    
    data_pars['train'] = 0
    _, test_ds         = get_dataset(data_pars)
    model_gluon        = model.model
    

    forecast_it, ts_it = make_evaluation_predictions(
            dataset     = test_ds,      # test dataset
            predictor   = model_gluon,  # predictor
            num_samples = compute_pars['num_samples'],  # number of sample paths we want for evaluation
        )


    tss       = list(tqdm(ts_it, total=len(test_ds)))         ## Actual Values
    forecasts = list(tqdm(forecast_it, total=len(test_ds)))   ## Prediction distirbution

    #forecasts, tss = list(forecast_it), list(ts_it)
    forecast_entry, ts_entry = forecasts[0], tss[0]


    ### External benchmark.py evaluation
    if kw.get("return_ytrue") :

        #### Avg the predictions
        forecasts_acc = np.zeros((len(forecasts), data_pars['single_pred_length']))
        for i in range(len(forecasts)):
          forecasts_acc[i] = np.mean(forecasts[i].samples, axis=0)       

        ypred, ytrue = forecasts_acc, tss
        return ypred, ytrue


    if VERBOSE:
        log(f"Number of sample paths: {forecast_entry.num_samples}")
        log(f"Dimension of samples: {forecast_entry.samples.shape}")
        log(f"Start date of the forecast window: {forecast_entry.start_date}")
        log(f"Frequency of the time series: {forecast_entry.freq}")
        log(f"Mean of the future window:\n {forecast_entry.mean}")
        log(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

    dd = {"forecasts": forecasts, "tss": tss}
    return dd



def evaluate(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kw):
   """
     Actual values tests

   """ 
   pass



def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kw):
        ## load test dataset
       
        data_pars['train']        = 0
        _, test_ds                = get_dataset(data_pars)

        forecasts                 = ypred["forecasts"]
        tss                       = ypred["tss"]

        ## Evaluate
        evaluator                 = Evaluator(quantiles=out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
        metrics_dict              = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics



def evaluate(ypred, data_pars, compute_pars=None, out_pars=None, **kw):
        ### load test dataset
      
        data_pars['train']        = 0
        _, test_ds                = get_dataset(data_pars)
        forecasts                 = ypred["forecasts"]
        tss                       = ypred["tss"]

        ### Evaluate
        evaluator                 = Evaluator(quantiles=out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(tss,forecasts, num_series=len(test_ds))
        metrics_dict              = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics


def save(model, path):
    import pickle
    path = path_norm(path + "/gluonts_model/")
    os.makedirs(path, exist_ok = True)

    model.model.serialize(Path(path) )   
    d = {"model_pars"  :  model.model_pars, 
         "compute_pars":  model.compute_pars,
         "data_pars"   :  model.data_pars
        }
    pickle.dump(d, open(path + "/glutonts_model_pars.pkl", mode="wb"))
    log(os.listdir(path))



def load(path):
    import pickle
    path = path_norm(path  + "/gluonts_model/" )

    predictor_deserialized = Predictor.deserialize(Path(path))
    d = pickle.load( open(path + "/glutonts_model_pars.pkl", mode="rb")  )
    
    ### Setup Model
    model = Model(model_pars= d['model_pars'], compute_pars= d['compute_pars'],
                  data_pars= d['data_pars'])  

    model.model = predictor_deserialized

    return model




def plot_prob_forecasts(ypred, out_pars=None):
    forecast_entry = ypred["forecasts"][0]
    ts_entry = ypred["tss"][0]
   
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()


def plot_predict(item_metrics, out_pars=None):
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath = out_pars['path']
    os.makedirs(outpath, exist_ok=True)
    plt.savefig(outpath)
    plt.clf()
    log('Saved image to {}.'.format(outpath))



####################################################################################################
def test_single(data_path="dataset/", choice="", config_mode="test"):
    model_uri = MODEL_URI
    log("#### Loading params   ##############################################")
    log( model_uri)
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=choice, data_path=data_path, config_mode=config_mode)
    log(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    #gluonts_ds = get_dataset(data_pars)
    
    log("#### Model init     ################################################")
    from mlmodels.models import module_load_full
    module, model = module_load_full(model_uri, model_pars, data_pars, compute_pars)
    log(module, model)


    log("#### Model fit     #################################################")
    #model=Model(model_pars, data_pars, compute_pars)
    model = fit(model, sess=None, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)    
    log(model)


    log("#### Save the trained model  ######################################")
    save(model, out_pars["path"])


    log("#### Load the trained model  ######################################")
    model = load(out_pars["path"])


    log("#### Predict   ####################################################")
    ypred = predict(model, sess=None, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    # log(ypred)


    log("#### metrics   ####################################################")
    metrics_val, item_metrics = metrics(ypred, data_pars, compute_pars, out_pars)
    log(metrics_val)


    log("#### Plot   #######################################################")
    if VERBOSE :
      plot_prob_forecasts(ypred, out_pars)
      plot_predict(item_metrics, out_pars)



def test() :
    ll = [ "deepar" , "deepfactor" , "transformer"  ,"wavenet", "feedforward",
           "gp_forecaster", "deepstate" ]

    ## Not yet  Implemented, error in Glutonts
    ll2 = [   "seq2seq"  ]
    
    for t in ll  :
      test_single(data_path="model_gluon/gluonts_model.json", choice="json", config_mode= t )
      #test_single(data_path="gluonts_model.json", choice="json", config_mode= t )


""" """
if __name__ == '__main__':
    VERBOSE = False

    #test()
