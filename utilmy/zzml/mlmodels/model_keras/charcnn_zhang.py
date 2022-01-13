# coding: utf-8
"""
Generic template for new model.
Check parameters template in models_config.json

"model_pars":   { "learning_rate": 0.001, "num_layers": 1, "size": 6, "size_layer": 128, "output_size": 6, "timestep": 4, "epoch": 5000 },
"data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] },
"compute_pars": { "distributed": "mpi", "epoch": 10 },
"out_pars":     { "out_path": "dataset/", "data_type": "pandas", "size": [0, 0, 6], "output_size": [0, 6] }



"""
import os
from keras.callbacks import EarlyStopping



######## Logs
from mlmodels.util import os_package_root_path, log, get_model_uri



#### Import EXISTING model and re-map to mlmodels
from mlmodels.model_keras.raw.char_cnn.data_utils import Data
from mlmodels.model_keras.raw.char_cnn.models.char_cnn_zhang import CharCNNZhang


from mlmodels.util import path_norm
print( path_norm("dataset") )


####################################################################################################

VERBOSE = False

MODEL_URI = get_model_uri(__file__)


####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None
                 ):
        ### Model Structure        ################################
        if model_pars is None :
            self.model = None

        else :
            self.model = CharCNNZhang(input_size=data_pars["data_info"]["input_size"],
                                alphabet_size          = data_pars["data_info"]["alphabet_size"],
                                embedding_size         = model_pars["embedding_size"],
                                conv_layers            = model_pars["conv_layers"],
                                fully_connected_layers = model_pars["fully_connected_layers"],
                                num_of_classes         = data_pars["data_info"]["num_of_classes"],
                                threshold              = model_pars["threshold"],
                                dropout_p              = model_pars["dropout_p"],
                                optimizer              = model_pars["optimizer"],
                                loss                   = model_pars["loss"]).model


def fit(model, data_pars={}, compute_pars={}, out_pars={}, **kw):
    """
    """

    batch_size = compute_pars['batch_size']
    epochs = compute_pars['epochs']

    sess = None  #
    dataset, internal_states = get_dataset(data_pars)
    Xtrain, ytrain = dataset
    data_pars["data_info"]["train"] = False
    dataset, internal_states = get_dataset(data_pars)
    Xtest, ytest = dataset

    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    model.model.fit(Xtrain, ytrain,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=[early_stopping],
                                  validation_data=(Xtest, ytest))

    return model, sess


def evaluate(model, data_pars={}, compute_pars={}, out_pars={}, **kw):
    """
       Return metrics ofw the model when fitted.
    """
    ddict = {}

    return ddict


def predict(model, sess=None, data_pars={}, out_pars={}, compute_pars={}, **kw):
    ##### Get Data ###############################################
    data_pars["data_info"]["train"] = False
    dataset, internal_states = get_dataset(data_pars)
    Xpred, ypred = dataset

    #### Do prediction
    ypred = model.model.predict(Xpred)

    ### Save Results

    ### Return val
    if compute_pars.get("return_pred_not") is  None:
        return ypred


def reset_model():
    pass


def save(model=None, session=None, save_pars={}):
    from mlmodels.util import save_keras
    print(save_pars)
    save_keras(model, session, save_pars)


def load(load_pars={}):
    from mlmodels.util import load_keras
    print(load_pars)
    model = load_keras(load_pars)
    session = None
    return model, session


####################################################################################################
def get_dataset(data_pars=None, **kw):
    """
      JSON data_pars to get dataset
      "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
      "size": [0, 0, 6], "output_size": [0, 6] },
    """
    from mlmodels.dataloader import DataLoader
    loader = DataLoader(data_pars)
    loader.compute()
    return loader.get_data()


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
        out_path   = path_norm( "/ztest/model_keras/charcnn_zhang/" )   
        model_path = os.path.join(out_path , "model")


        model_pars = {
            "embedding_size": 128,
            "conv_layers": [[256, 7, 3], [256, 7, 3], [256, 3, -1], [256, 3, 3]], 
            "fully_connected_layers": [
                1024,
                1024,
            ],
            "threshold": 1e-6,
            "dropout_p": 0.1,
            "optimizer": "adam",
            "loss": "categorical_crossentropy"
        }

        data_pars = {
            "train": 'true',
            "alphabet": "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
            "alphabet_size": 69,
            "input_size": 1014,
            "num_of_classes": 4,
            "train_data_source":path_norm("dataset/text/ag_news_csv/train.csv"),
            "val_data_source": path_norm("dataset/text/ag_news_csv/test.csv")
        }


        compute_pars = {
            "epochs": 1,
            "batch_size": 128
        }

        out_pars = {
            "path": "ztest/ml_keras/charcnn_zhang/",
            "data_type": "pandas",
            "size": [
                0,
                0,
                6
            ],
            "output_size": [
                0,
                6
            ]
        }

        return model_pars, data_pars, compute_pars, out_pars

    else:
        raise Exception(f"Not support choice {choice} yet")


################################################################################################
########## Tests are  ##########################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test
    from mlmodels.util import path_norm
    data_path = path_norm(data_path)

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path, "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)

    log("#### Loading daaset   #############################################")
    Xtuple = get_dataset(data_pars)

    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)

 
    log("#### Predict   #####################################################")
    data_pars["train"] = 0
    ypred = predict(model, session, data_pars, compute_pars, out_pars)

    log("#### metrics   #####################################################")
    metrics_val = evaluate(model, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   ########################################################")

    log("#### Save/Load   ###################################################")
    save(model, session, save_pars=out_pars)
    model2 = load(out_pars)
    #     ypred = predict(model2, data_pars, compute_pars, out_pars)
    #     metrics_val = metrics(model2, ypred, data_pars, compute_pars, out_pars)
    print(model2)



if __name__ == '__main__':
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"
    root_path = os_package_root_path()

    ### Local fixed params
    # test(pars_choice="test01")

    ### Local json file
    test(pars_choice="json", data_path=f"dataset/json/refactor/charcnn_zhang.json")
    # test(pars_choice="json", data_path= f"{root_path}/model_keras/charcnn_zhang.json")

    # ####    test_module(model_uri="model_xxxx/yyyy.py", param_pars=None)
    # from mlmodels.models import test_module
    #
    # param_pars = {'choice': "json", 'config_mode': 'test', 'data_path': "model_keras/charcnn_zhang.json"}
    # test_module(model_uri=MODEL_URI, param_pars=param_pars)
    #
    # ##### get of get_params
    # # choice      = pp['choice']
    # # config_mode = pp['config_mode']
    # # data_path   = pp['data_path']
    #
    # ####    test_api(model_uri="model_xxxx/yyyy.py", param_pars=None)
    # from mlmodels.models import test_api
    #
    # param_pars = {'choice': "json", 'config_mode': 'test', 'data_path': "model_keras/charcnn_zhang.json"}
    # test_api(model_uri=MODEL_URI, param_pars=param_pars)
