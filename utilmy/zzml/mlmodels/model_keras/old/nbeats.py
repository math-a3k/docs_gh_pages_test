# coding: utf-8
"""


"""
import os

import numpy as np


from keras.callbacks import EarlyStopping

####################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri

VERBOSE = False
MODEL_URI = get_model_uri(__file__)
# print( path_norm("dataset") )





#### Import EXISTING model and re-map to mlmodels
from mlmodels.model_keras.raw.nbeats_keras.model import NBeatsNet




def main():
    # https://keras.io/layers/recurrent/
    num_samples, time_steps, input_dim, output_dim = 50_000, 10, 1, 1

    # Definition of the model.
    model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
                      stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                      thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)

    # Definition of the objective function and the optimizer.
    model.compile_model(loss='mae', learning_rate=1e-5)

    # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
    x = np.random.uniform(size=(num_samples, time_steps, input_dim))
    y = np.mean(x, axis=1, keepdims=True)

    # Split data into training and testing datasets.
    c = num_samples // 10
    x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]

    # Train the model.
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128)

    # Save the model for later.
    model.save('n_beats_model.h5')

    # Predict on the testing set.
    predictions = model.predict(x_test)
    print(predictions.shape)

    # Load the model.
    model2 = NBeatsNet.load('n_beats_model.h5')

    predictions2 = model2.predict(x_test)
    np.testing.assert_almost_equal(predictions, predictions2)



####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None  ):
        ### Model Structure        ################################
        if model_pars is None :
            self.model = None
            return None

        m = model_pars
        num_samples = m['num_samples']
        time_steps  = m['time_steps']
        input_dim   = m['input_dim']
        output_dim  = m['output_dim']

        # num_samples, time_steps, input_dim, output_dim = 50_000, 10, 1, 1

        # Definition of the model.
        model = NBeatsNet(backcast_length=time_steps, forecast_length=output_dim,
                          stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2,
                          thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=64)

        loss = compute_pars['loss']
        lr   = compute_pars['learning_rate']
        # Definition of the objective function and the optimizer.
        # model.compile_model(loss='mae', learning_rate=1e-5)
        model.compile_model(loss= loss, learning_rate= learning_rate )



def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """

    batch_size = compute_pars['batch_size']
    epochs = compute_pars['epochs']

    sess = None  #
    Xtrain, Xtest, ytrain, ytest = get_dataset(data_pars)

    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    model.model.fit(Xtrain, ytrain,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=[early_stopping],
                                  validation_data=(Xtest, ytest))

    return model, sess


def evaluate(model, session=None, data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
       Return metrics ofw the model when fitted.
    """
    ddict = {}

    return ddict


def predict(model, session=None, data_pars=None, out_pars=None, compute_pars=None, **kw):
    ##### Get Data ###############################################
    data_pars['train'] = False
    Xpred, ypred = get_dataset(data_pars)

    #### Do prediction
    ypred = model.model.predict(Xpred)

    ### Save Results

    ### Return val
    if compute_pars.get("return_pred_not") is  None:
        return ypred


def reset_model():
    pass


def save(model=None,  save_pars=None, session=None):
    from mlmodels.util import save_keras
    save_keras(model, session, save_pars=save_pars)


def load(load_pars=None):
    from mlmodels.util import load_keras
    model0 = load_keras(load_pars)

    model = Model()
    model.model = model0.model
    session = None
    return model, session


####################################################################################################
def get_dataset(data_pars=None, **kw):
    """
      JSON data_pars to get dataset
      "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
      "size": [0, 0, 6], "output_size": [0, 6] },
    """
    from mlmodels.util import path_norm
    
    if data_pars['train']:

        print('Loading data...')
        train_data = Data(data_source= path_norm( data_pars["train_data_source"]) ,
                             alphabet       = data_pars["alphabet"],
                             input_size     = data_pars["input_size"],
                             num_of_classes = data_pars["num_of_classes"])
        train_data.load_data()
        train_inputs, train_labels = train_data.get_all_data()


        # Load val data
        val_data = Data(data_source = path_norm( data_pars["val_data_source"]) ,
                               alphabet=data_pars["alphabet"],
                               input_size=data_pars["input_size"],
                               num_of_classes=data_pars["num_of_classes"])
        val_data.load_data()
        val_inputs, val_labels = val_data.get_all_data()

        return train_inputs, val_inputs, train_labels, val_labels


    else:
        val_data = Data(data_source = path_norm( data_pars["val_data_source"]) ,
                               alphabet=data_pars["alphabet"],
                               input_size=data_pars["input_size"],
                               num_of_classes=data_pars["num_of_classes"])
        val_data.load_data()
        Xtest, ytest = val_data.get_all_data()
        return Xtest, ytest


def get_params(param_pars={}, **kw):
    from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']


    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        root       = path_norm()
        data_path  = path_norm( "dataset/text/imdb.npz"  )   
        out_path   = path_norm( "ztest/model_keras/nbeats/" )
        model_path = os.path.join(out_path , "model")


        model_pars = {
            "embedding_size": 128,
            "conv_layers": [[256, 10 ], [256, 7 ], [256, 5 ], [256, 3 ] ], 
            "fully_connected_layers": [
                1024,
                1024
            ],
            "threshold": 1e-6,
            "dropout_p": 0.1,
            "optimizer": "adam",
            "loss": "categorical_crossentropy"
        }

        data_pars = {
            "train": True,
            "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
            "alphabet_size": 69,
            "input_size": 1014,
            "num_of_classes": 4,
            "train_data_source": path_norm("dataset/text/ag_news_csv/train.csv") ,
            "val_data_source": path_norm("dataset/text/ag_news_csv/test.csv")
        }


        compute_pars = {
            "epochs": 1,
            "batch_size": 128
        }

        out_pars = {
            "path":  path_norm( "ztest/ml_keras/nbeats/nbeats.h5"),
            "data_type": "pandas",
            "size": [0, 0, 6],
            "output_size": [0, 6]
        }

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")


################################################################################################
########## Tests ###############################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test
    from mlmodels.util import path_norm
    data_path = path_norm(data_path)

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)

    log("#### Loading dataset   #############################################")
    Xtuple = get_dataset(data_pars)
    print(len(Xtuple))

    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    data_pars["train"] = 0
    ypred = predict(model, session, data_pars, compute_pars, out_pars)

    log("#### metrics   #####################################################")
    metrics_val = evaluate(model, session, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = {"path": out_pars['path']}
    save(model, session, save_pars=save_pars)
    model2, session2 = load(save_pars)

    log("#### Save/Load - Predict   #########################################")
    print(model2, session2)
    ypred = predict(model2, session2, data_pars, compute_pars, out_pars)



if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"
    root_path = os_package_root_path()

    ### Local fixed params
    test(pars_choice="test01")

    #### Local json file
    test(pars_choice="json", data_path= f"model_keras/nbeats.json")


    # ####    test_module(model_uri="model_xxxx/yyyy.py", param_pars=None)
    # from mlmodels.models import test_module
    #
    # param_pars = {'choice': "json", 'config_mode': 'test', 'data_path': "model_keras/nbeats.json"}
    # test_module(model_uri=MODEL_URI, param_pars=param_pars)
    #
    # #### get of get_params
    # # choice      = pp['choice']
    # # config_mode = pp['config_mode']
    # # data_path   = pp['data_path']
    #
    # ####    test_api(model_uri="model_xxxx/yyyy.py", param_pars=None)
    # from mlmodels.models import test_api
    #
    # param_pars = {'choice': "json", 'config_mode': 'test', 'data_path': "model_keras/nbeats.json"}
    # test_api(model_uri=MODEL_URI, param_pars=param_pars)



