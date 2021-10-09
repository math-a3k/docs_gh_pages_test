# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
Template for tseries type of model:
#### Demand Dataset
https://github.com/arita37/dsa2/tree/multi/data/input/tseries_demand
#### DSA2 model
https://github.com/arita37/dsa2/blob/multi/tseries.py
### Colab example
https://colab.research.google.com/drive/1OZPsaH8ZBk1M5e8X0W2nDQZ5uui8Kvh4
"""
import os, pandas as pd, numpy as np, scipy as sci, sklearn

####################################################################################################
from utilmy import global_verbosity, os_makedirs, pd_read_file
verbosity = global_verbosity(__file__,"/../../config.json", 3 )

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 : print(*s, flush=True)

def log3(*s):
    if verbosity >= 3 : print(*s, flush=True)


####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None

def reset():
    global model, session
    model, session = None, None

####################################################################################################
## Modules && MODELS
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.tree import *
from lightgbm import LGBMModel, LGBMRegressor, LGBMClassifier
from sktime.forecasting.base import ForecastingHorizon
# from sktime.transformers.single_series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.model_selection import temporal_train_test_split

from sktime.utils.plotting import plot_series
from sktime.forecasting.compose import (
       TransformedTargetForecaster,
       ReducedForecaster # ReducedRegressionForecaster
)


class myModel(object):
    pass


def LighGBM_recursive(lightgbm_pars= {'objective':'quantile', 'alpha': 0.5},
                       forecaster_pars = {'window_length': 4}):
    """
    #1.Separate the Seasonal Component.
    #2.Fit a forecaster for the trend.
    #3.Fit a Autoregressor to the resdiual(autoregressing on four historic values).
    """
    #Initialize Light GBM Regressor
    from sktime.forecasting.compose import (
       TransformedTargetForecaster,
       ReducedForecaster #, ReducedRegressionForecaster
    )

    regressor = LGBMRegressor(**lightgbm_pars)
    forecaster = RecursiveRegressionForecaster(regressor=regressor,
                  **forecaster_pars  #hyper-paramter to set recursive strategy
                    )
    return forecaster



####################################################################################################
### Model class && helper function
class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None
        else:
            model_class = globals()[model_pars['model_class'].split(':')[-1]]
            self.model = model_class(**model_pars['model_pars'])
            log(model_class, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute

    # get&split data
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")
    log(Xtrain.shape, model.model)

    if "LGBM" in model.model_pars['model_class']:
        model.model.fit(Xtrain, ytrain, eval_set=[(Xtest, ytest)], **compute_pars.get("compute_pars", {}))
    else:
        model.model.fit(Xtrain, ytrain, **compute_pars.get("compute_pars", {}))



def predict(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    global model, session

    if Xpred is None:
        data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")

    # Xpred_fh = ForecastingHorizon(Xpred.index, is_relative=False)

    ypred = model.model.predict(Xpred)

    ypred_proba = None  ### No proba
    return ypred, ypred_proba



def predict_forward(Xpred=None, data_pars={}, compute_pars={}, out_pars={}, **kw):
    """   Recursive prediction by one step
    https://machinelearningmastery.com/xgboost-for-time-series-forecasting/

    # walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = xgboost_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions


    :param Xpred:
    :param data_pars:
    :param compute_pars:
    :param out_pars:
    :param kw:
    :return:
    """
    global model, session

    ### Auto-regressive Feature calculator
    features_calc =compute_pars.get("_features_calc", None)

    if Xpred is None:
        data_pars['train'] = False
        Xpred = get_dataset(data_pars, task_type="predict")

    coly    = 'sales'
    coldate = 'date'
    ypred_list = []
    for i, Xi in enumerate(Xpred.iterrows()) :
        if Xi[coly] > -999999999.0 : continue  ## Not Missing

        ypred = model.model.predict(Xi)
        Xpred[coly].iloc[i] = ypred
        Xpred.iloc[i,:]     = features_calc(Xpred.iloc[:i+1,:])

    ypred       = Xpred[coly]
    ypred_proba = None
    return ypred, ypred_proba


#########################################################################################################
### helper Function
def save(path=None, info=None):
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )

    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model = model0.model
    model.model_pars = model0.model_pars
    model.compute_pars = model0.compute_pars
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


####################################################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
    return tuple of dataframes OR single dataframe
      "ram"  :
      "file" :
    """
    # log(data_pars)
    data_type  = data_pars.get('type', 'ram')

    ### Sparse columns, Dense Columns
    cols_type  = data_pars.get('cols_model_type2', {})   #### Split input by Sparse, Continous

    log3("Cols Type:", cols_type)

    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]
            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')



####################################################################################################################
def test_dataset_tseries(nrows=10000, coly=None, coldate=None, colcat=None):

    # read data
    train_csv = "data/input/tseries_demand/train/features.zip"
    test_csv = "data/input/tseries_demand/test/features.zip"

    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(test_csv)

    # df = df.groupby([coldate])[coly].sum().reset_index()

    # date as index
    df1 = df1.set_index(coldate)  #### Date as
    df1.index.freq="D"
    df2 = df2.set_index(coldate)  #### Date as
    df2.index.freq="D"

    # df[coldate] = pd.to_datetime(df.index)

    return df1, df2



def time_train_test_split(df, test_size = 0.4, cols=None , coltime ="time_key", sort=True, minsize=5,
                     n_sample=5,
                     verbose=False) :
   cols = list(df.columns) if cols is None else cols
   if sort :
       df   = df.sort_values( coltime, ascending=1 )
   #imax = len(df) - test_period
   colkey = [ t for t in cols if t not in [coltime] ]  #### All time reference be removed
   if verbose : log(colkey)
   imax = test_size * n_sample ## Over sampling
   df1  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[:max(minsize, len(dfi) -imax), :] ).reset_index(colkey, drop=True).reset_index(drop=True)
   df2  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[max(minsize,  len(dfi) -imax):, :] ).reset_index(colkey, drop=True).reset_index(drop=True)
   return df1, df2


def test():
    global model, session

    coly = 'sales'
    coldate = 'date'
    colcat  = ['store', 'item']

    train, valid  = test_dataset_tseries(coldate = coldate,  )

    # Split the df into train/test subsets
    colsX = [i for i in train.columns if i != coly]

    X_train, X_valid, y_train, y_valid = train[colsX], valid[colsX], train[coly], None# valid[coly]
    log('y', np.sum(y_train) )

    # X = df
    # y = df[coly].astype('uint8')
    # log('y', np.sum(y[y==1]) )

    # Split the df into train/test subsets
    # features_cols = [i for i in df.columns if i != coly]

    # train_full, test = time_train_test_split(X, test_size=0.05, coltime = coldate)
    # X_train_full, X_test, y_train_full, y_test = train_full[features_cols], test[features_cols], train_full[coly], test[coly]

    # train, valid         = time_train_test_split(train_full, coltime = coldate)
    # X_train, X_valid, y_train, y_valid = train[features_cols], valid[features_cols], train[coly], valid[coly]


    cols_input_type_1 = {
         "coly"   :   "sales"
        ,"colid"  :   "id_date"   ### used for JOIN tables, duplicate date
        ,"colcat" :   ["store", "item" ]
        ,"colnum" :   []
        ,"coltext" :  []
        ,"coldate" :  []

        ### Specific for time sereis
        ,"col_tseries" :  ['date', 'store', 'item', 'sales']

        ,"colcross" : [ ]
    }

    data_name    = "tseries_demand"         ### in data/input/
    model_class  = "source/models/model_tseries.py:LGBMRegressor"  ### ACTUAL Class name for model_sklearn.py
    n_sample     = 100000

    def post_process_fun(y):   ### After prediction is done
        # ynew = np.exp(y) - 1.0
        ynew = float(y)
        return  ynew

    def pre_process_fun(y):    ### Before the prediction is done
        # ynew = np.log(y+1)
        ynew = float(y)
        return  ynew


    m = {"model_pars": {
         "model_class": model_class
        ,"model_pars" : {"objective": "huber",    ### Regression Type Loss
                           "n_estimators": 100,
                           "learning_rate":0.001,
                           "boosting_type":"gbdt",     ### Model hyperparameters
                           "early_stopping_rounds": 5

                        }

        , "post_process_fun" : post_process_fun   ### After prediction  #######
        , "pre_process_pars" : {"y_norm_fun" :  pre_process_fun ,  ### Before training  ##########################
            ### Pipeline for data processing ##############################
            "pipe_list": [
            #### Example of Custom processor
            {"uri":  "::pd_dsa2_custom",
                "pars"        : {'coldate': 'date'},
                "cols_family" : "col_tseries",
                "cols_out"    : "tseries_feat",  "type": "" },

            ],
                   }
        },

      "compute_pars": { "metric_list": ['root_mean_squared_error', 'mean_absolute_error',
                                       'explained_variance_score', 'r2_score', 'median_absolute_error']
      },

      "data_pars": { "n_sample" : n_sample, "download_pars" : None,
          ### Raw data:  column input
          "cols_input_type" : cols_input_type_1,

          ### Model Input :  Merge family of columns
          "cols_model_group": [  ### cols_out of  pd_dsa2_custom
                               "tseries_feat"
                              ]
          #### Model Input : Separate Category Sparse from Continuous : Aribitrary name is OK (!)
         ,'cols_model_type2': {
             'My123_continuous' : [ 'tseries_feat',   ],
             'my_sparse'        : [ 'colcat',  ],
          }

          ### Filter data rows   ###########################
         ,"filter_pars": { "ymax" : 999999999 ,"ymin" : -1 },


         #########################################################################################
         'train': {'Xtrain': X_train, 'ytrain': y_train,
                   'Xtest': X_valid,  'ytest':  y_valid},
         'eval': {'X': X_valid,  'y': y_valid},
         'predict': {'X': X_valid}



         }
      }


    ll = [
        ('model_tseries.py::LGBMRegressor', # LightGBMregressor
            {
            }
        ),

    ]
    for cfg in ll:
        log(f"******************************************** {cfg[0]} ********************************************")
        reset()
        # Set the ModelConfig
        m['model_pars']['model_class'] = cfg[0]
        m['model_pars']['model_pars']  = {**m['model_pars']['model_pars'] , **cfg[1] }

        log('Setup model..')
        model = Model(model_pars=m['model_pars'], data_pars=m['data_pars'], compute_pars= m['compute_pars'] )

        log('\n\nTraining the model..')
        fit(data_pars=m['data_pars'], compute_pars= m['compute_pars'], out_pars=None)
        log('Training completed!\n\n')

        log('Predict data..')
        ypred, ypred_proba = predict(Xpred=None, data_pars=m['data_pars'], compute_pars=m['compute_pars'])
        log(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')

        log('Model architecture:')
        log(model.model)
        reset()






####################################################################################################################
def test2(nrows=1000, file_path=None, coly=None, coldate=None, colcat=None):
    """
        nrows : take first nrows from dataset
    """
    global model, session
    df, coly, coldate, colcat = test_dataset_tseries(file_path, coly, coldate, colcat)
    X  = df.drop(coly, axis=1)
    y  = df[coly]

    # # Split the df into train/test subsets
    y_train, y_test = temporal_train_test_split(y, test_size=0.2)

    #A 10 percent and 90 percent prediction interval(0.1,0.9 respectively).
    quantiles = 0.5  #Hyper-parameter "alpha" in Light GBM

    #Capture forecasts for 10th/median/90th quantile, respectively.
    forecasts = []
    forecaster = LighGBM_recursive(lightgbm_pars= {'objective':'quantile', 'alpha': quantile} )
    forecaster.fit(y_train)


    #Initialize ForecastingHorizon class to specify the horizon of forecast
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    y_pred = forecaster.predict(fh)

    #List of forecasts made for each quantile.
    y_pred.index.name="date"
    y_pred.name=f"predicted_sales_q_{alpha}"
    forecasts[f"predicted_sales_q_{alpha}"].append(y_pred)

    #Iterate for each quantile.
    for alpha in quantiles:
        forecaster = LighGBM_forecaster(lightgbm_pars= {'objective':'quantile', 'alpha': 0.5} )
        forecaster.fit(y_train)

        #Forecast the values.
        #Initialize ForecastingHorizon class to specify the horizon of forecast
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        y_pred = forecaster.predict(fh)

        #List of forecasts made for each quantile.
        y_pred.index.name="date"
        y_pred.name=f"predicted_sales_q_{alpha}"
        forecasts[f"predicted_sales_q_{alpha}"].append(y_pred)





if __name__ == "__main__":
    import fire
    fire.Fire()

    # test0(nrows=1000, file_path="train.csv", coly="sales", coldate="date", colcat=None)


