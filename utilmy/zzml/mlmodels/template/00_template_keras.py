
import os

import numpy as np
import pandas as pd
from keras.models import load_model, save_model


####################################################################################################
# Helper functions
def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)


####################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        # 4.Define Model

        if not model_pars.get('model_type'):
            raise Exception("Missing model type when init model object!")

        else:
            self.model = None





##################################################################################################
def _preprocess_XXXX(df, **kw):
    return df, linear_cols, dnn_cols, train, test, target


def get_dataset(**kw):
    ##check whether dataset is of kind train or test
    data_path = kw['train_data_path']
    data_type = kw['dataset_type']
    test_size = kw['test_size']

    #### read from csv file
    if kw.get("uri_type") == "pickle":
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)

    if data_type == "xxxx":
        df, linear_cols, dnn_cols, train, test, target = _preprocess_criteo(df, **kw)
        
    else:  ## Already define


    return df, linear_cols, dnn_cols, train, test, target


def fit(model, session=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None,
        **kwargs):
    ##loading dataset
    """
          Classe Model --> model,   model.model contains thte sub-model
    """
    _ = get_dataset(**data_pars)
    multiple_value = data_pars.get('multiple_value', None)

    m = compute_pars
   
    model.model.fit(train, data[target].values,
                        batch_size=m['batch_size'], epochs=m['epochs'], verbose=2,
                        validation_split=m['validation_split'], )
    return model


def predict(model, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ##  Model is class
    ## load test dataset
    data, linear_cols, dnn_cols, train, test, target = get_dataset(**data_pars)
    feature_names = get_feature_names(linear_cols + dnn_cols, )
    test_model_input = {name: test[name] for name in feature_names}

    multiple_value = data_pars.get('multiple_value', None)
    ## predict
    if multiple_value is None:
        pred_ans = model.model.predict(test_model_input, batch_size= compute_pars['batch_size'])
    else:
        pred_ans = None

    return pred_ans


def metrics(ypred, data_pars, compute_pars=None, out_pars=None, **kwargs):
    ## load test dataset
    _, linear_cols, dnn_cols, _, test, target = get_dataset(**data_pars)
    metrics_dict = {"LogLoss": log_loss(test[target].values, ypred),
                        "AUC": roc_auc_score(test[target].values, ypred)}
    return metrics_dict


def reset_model():
    pass




########################################################################################################################
class Model_empty(object):
    def __init__(self, model_pars=None, compute_pars=None):
        ## Empty model for Seaialization
        self.model = None


def save(model, path):
    if not os.path.exists(os.path.dirname(path)):
        print("model file path do not exist!")
    else:
        save_model(model.model, path)


def load(path):
    if not os.path.exists(path):
        print("model file do not exist!")
        return None
    else:
        model = Model_empty()
        model_keras = load_model(path, custom_objects)

        #### Add back the model parameters...
        return model


########################################################################################################################
def path_setup(out_folder="", sublevel=0, data_path="dataset/"):
    #### Relative path
    data_path = os_package_root_path(__file__, sublevel=sublevel, path_add=data_path) 
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)
    log(data_path, out_path)
    return data_path, out_path


def get_params(choice=0, data_path="dataset/", **kw):
    if choice == 0:
        log("#### Path params   ###################################################")
        data_path, out_path = path_setup(out_folder="/deepctr_test/", data_path=data_path)

        train_data_path = data_path + "criteo_sample.txt"
        data_pars = {"train_data_path": train_data_path, "dataset_type": "criteo", "test_size": 0.2}

        log("#### Model params   #################################################")
        model_pars = {"model_type": "DeepFM", "optimization": "adam", "cost": "binary_crossentropy"}
        compute_pars = {"task": "binary", "batch_size": 256, "epochs": 10, "validation_split": 0.2}
        out_pars = {"path": out_path}

    return model_pars, data_pars, compute_pars, out_pars


########################################################################################################################
########################################################################################################################
def test(data_path="dataset/", pars_choice=0):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice,
                                                               data_path=data_path)
    print(model_pars, data_pars, compute_pars, out_pars)

    log("#### Loading dataset   #############################################")
    dataset = get_dataset(**data_pars)

    log("#### Model init, fit   #############################################")
    model = Model(model_pars=model_pars, compute_pars=compute_pars, dataset=dataset)
    model = fit(model, data_pars=data_pars, model_pars=model_pars, compute_pars=compute_pars)

    log("#### Predict   ####################################################")
    ypred = predict(model, data_pars, compute_pars, out_pars)

    log("#### metrics   ####################################################")
    metrics_val = metrics(ypred, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   #######################################################")


    log("#### Save/Load   ##################################################")
    save(model, out_pars['path'] + f"/model_{pars_choice}.h5")
    model2 = load(out_pars['path'] + f"/model_{pars_choice}.h5")
    print(model2)



if __name__ == '__main__':
    VERBOSE = True
    test(pars_choice=0)
