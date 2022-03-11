from jsoncomment import JsonComment ; json = JsonComment()
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor

VERBOSE = False


from mlmodels.util import os_package_root_path, log



####################################################################################################
def _config_process(data_path, config_mode="test"):
    """function _config_process
    Args:
        data_path:   
        config_mode:   
    Returns:
        
    """
    data_path = Path(os.path.realpath(
        __file__)).parent.parent / "model_gluon/gluon_deepar.json" if data_path == "dataset/" else data_path

    with open(data_path, encoding='utf-8') as config_f:
        config = json.load(config_f)
        config = config[config_mode]

    return config["model_pars"], config["data_pars"], config["compute_pars"], config["out_pars"]



    # Model fit
def fit(model, sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None, **kwargs):
        ##loading dataset
        """
          Classe Model --> model,   model.model contains thte sub-model

        """
        model_gluon = model.model
        gluont_ds = get_dataset(data_pars)
        model.model = model_gluon.train(gluont_ds)
        return model


    # Model predict
def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
        """function predict
        Args:
            model:   
            sess:   
            data_pars:   
            compute_pars:   
            out_pars:   
            **kwargs:   
        Returns:
            
        """
        ##  Model is class
        ## load test dataset
        data_pars['train'] = False
        test_ds = get_dataset(data_pars)

        ## predict
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=model.model,  # predictor
            num_samples=compute_pars['num_samples'],  # number of sample paths we want for evaluation
        )

        ##convert generator to list
        forecasts, tss = list(forecast_it), list(ts_it)
        forecast_entry, ts_entry = forecasts[0], tss[0]

        ### output stats for forecast entry
        if VERBOSE:
            print(f"Number of sample paths: {forecast_entry.num_samples}")
            print(f"Dimension of samples: {forecast_entry.samples.shape}")
            print(f"Start date of the forecast window: {forecast_entry.start_date}")
            print(f"Frequency of the time series: {forecast_entry.freq}")
            print(f"Mean of the future window:\n {forecast_entry.mean}")
            print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

        dd = {"forecasts": forecasts, "tss": tss}
        return dd


def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
        """function metrics
        Args:
            ypred:   
            data_pars:   
            compute_pars:   
            out_pars:   
            **kwargs:   
        Returns:
            
        """
        ## load test dataset
        data_pars['train'] = False
        test_ds = get_dataset(data_pars)

        forecasts = ypred["forecasts"]
        tss = ypred["tss"]

        ## evaluate
        evaluator = Evaluator(quantiles=out_pars['quantiles'])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))
        metrics_dict = json.dumps(agg_metrics, indent=4)
        return metrics_dict, item_metrics



###############################################################################################################
# save and load model helper function
class Model_empty(object):
    def __init__(self, model_pars=None, compute_pars=None):
        """ Model_empty:__init__
        Args:
            model_pars:     
            compute_pars:     
        Returns:
           
        """
        ## Empty model for Seaialization
        self.model = None



def save(model, path):
    """function save
    Args:
        model:   
        path:   
    Returns:
        
    """
    if os.path.exists(path):
        model.model.serialize(Path(path))



def load(path):
    """function load
    Args:
        path:   
    Returns:
        
    """
    if os.path.exists(path):
        predictor_deserialized = Predictor.deserialize(Path(path))

    model = Model_empty()
    model.model = predictor_deserialized
    #### Add back the model parameters...

    return model


# Dataaset
def get_dataset(data_pars):
    """function get_dataset
    Args:
        data_pars:   
    Returns:
        
    """
    ##check whether dataset is of kind train or test
    data_path = data_pars['train_data_path'] if data_pars['train'] else data_pars['test_data_path']

    #### read from csv file
    if data_pars.get("uri_type") == "pickle":
        data_set = pd.read_pickle(data_path)
    else:
        data_set = pd.read_csv(data_path)

    ### convert to gluont format
    gluonts_ds = ListDataset([{FieldName.TARGET: data_set.iloc[i].values, FieldName.START: data_pars['start']}
                              for i in range(data_pars['num_series'])], freq=data_pars['freq'])

    if VERBOSE:
        entry = next(iter(gluonts_ds))
        train_series = to_pandas(entry)
        train_series.plot()
        save_fig = data_pars['save_fig']
        # plt.savefig(save_fig)

    return gluonts_ds




###############################################################################################################
### different plots and output metric
def plot_prob_forecasts(ypred, out_pars=None):
    """function plot_prob_forecasts
    Args:
        ypred:   
        out_pars:   
    Returns:
        
    """
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
    """function plot_predict
    Args:
        item_metrics:   
        out_pars:   
    Returns:
        
    """
    item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
    plt.grid(which="both")
    outpath = out_pars['outpath']
    if not os.path.exists(outpath): os.makedirs(outpath, exist_ok=True)
    plt.savefig(outpath)
    plt.clf()
    print('Saved image to {}.'.format(outpath))





