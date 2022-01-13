# -*- coding: utf-8 -*-
"""


https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/facebookresearch_pytorch-gan-zoo_pgan.ipynb

https://github.com/pytorch/pytorch/blob/98362d11ffe81ca48748f6b0e1e417cb81ba5998/torch/hub.py#L330


        the following models only: alexnet, densenet121, densenet169, densenet201,\
        densenet161, inception_v3, resnet18, resnet34, resnet50, resnet101, resnet152,\
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2, squeezenet1_0,\
        squeezenet1_1, vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,\
        googlenet, shufflenet_v2_x0_5, shufflenet_v2_x1_0, mobilenet_v2"



"""
import os, json

# import mlmodels.models as M


from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri, path_norm_dict
MODEL_URI = get_model_uri(__file__)



from mlmodels.model_tch.raw import pytorch_vae as md


MODEL_MAP = {
    
 "beta_vae": md.model.beta_vae,

}






###########################################################################################################
###########################################################################################################
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, out_pars=None):
        ### Model Structure        ################################
        if model_pars is None :
            self.model = None
            return self


        m = model_pars 
        _model      = m['model']



        self.model = MODEL_MAP[m['model_name']]()




def get_params(param_pars=None, **kw):
    pp          = param_pars
    choice      = pp['choice']
    config_mode = pp['config_mode']
    data_path   = pp['data_path']

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]

        ####Normalize path  : add /models/dataset/
        cf['data_pars'] = path_norm_dict(cf['data_pars'])
        cf['out_pars']  = path_norm_dict(cf['out_pars'])

        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")




def get_dataset(data_pars=None, **kw):
    data_path        = data_pars['data_path']
    train_batch_size = data_pars['train_batch_size']
    test_batch_size  = data_pars['test_batch_size']

    if data_pars['dataset'] == 'MNIST':
        train_loader, valid_loader  = get_dataset_mnist_torch(data_pars)
        return train_loader, valid_loader  

    else:
        raise Exception("Dataloader not implemented")
        exit





def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model0        = model.model
    lr            = compute_pars['learning_rate']
    epochs        = compute_pars["epochs"]
    criterion     = nn.CrossEntropyLoss()
    device        = _get_device()
    model0.to(device)
    train_loss    = []
    train_acc     = []
    test_loss     = []
    test_acc      = []
    best_test_acc = -1

    optimizer     = optim.Adam(model0.parameters(), lr=lr)
    train_iter, valid_iter = get_dataset(data_pars)

    imax_train = compute_pars.get('max_batch_sample', len(train_iter) )
    imax_valid = compute_pars.get('max_batch_sample', len(valid_iter) )

    os.makedirs(out_pars["checkpointdir"], exist_ok=True)
    

    ####VAE
    from mlmodels.model_tch.raw.pytorch_vae.model import run
    run.train(vae)



    return model, None


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None, imax = 1, return_ytrue=1):
    # get a batch of data
    import numpy as np
    from metrics import metrics_eval
    device = _get_device()
    model = model.model
    _, test_iter = get_dataset(data_pars=data_pars)

    # test_iter = get_dataset(data_pars, out_pars)
    y_pred = []
    y_true = []
    for i,batch in enumerate(test_iter):
        if i >= imax: break
        image, target = batch[0], batch[1]
        image = image.to(device)
        logit = model(image)
        predictions = torch.max(logit,1)[1].cpu().numpy()
        y_pred.append(predictions)
        y_true.append(target)
    y_pred = np.vstack(y_pred)[0]
    y_true = np.vstack(y_true)[0]

    if return_ytrue:
        return y_pred, y_true
    else:
        return y_pred

def evaluate(model, data_pars=None, compute_pars=None, out_pars=None):
    pass


def save(model, session=None, save_pars=None):
    from mlmodels.util import save_tch
    save_tch(model=model, save_pars=save_pars)


def load(load_pars):
    from mlmodels.util import load_tch
    return load_tch(load_pars)



###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test

    log("#### Loading params   ##############################################")
    param_pars = {"choice":pars_choice,  "data_path":data_path,  "config_mode": config_mode}
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    log(  data_pars, out_pars )

    log("#### Loading dataset   #############################################")
    xtuple = get_dataset(data_pars)


    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, data_pars, compute_pars, out_pars)


    log("#### Predict   #####################################################")
    ypred = predict(model, session, data_pars, compute_pars, out_pars)


    log("#### metrics   #####################################################")
    metrics_val = evaluate(model, data_pars, compute_pars, out_pars)
    print(metrics_val)


    log("#### Plot   ########################################################")


    log("#### Save/Load   ###################################################")
    save_pars = { "path": out_pars["path"]  }
    save(model=model, save_pars=save_pars)
    model2 = load( save_pars )
    ypred = predict(model2, data_pars=data_pars, compute_pars=compute_pars, out_pars=out_pars)
    print(model2)



if __name__ == "__main__":
    test(data_path="model_tch/torchhub_cnn.json", pars_choice="json", config_mode="test")









