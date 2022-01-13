

import os


from nltk.tokenize.treebank import TreebankWordDetokenizer



from transformers import GPT2Tokenizer, GPT2LMHeadModel



from mlmodels.model_tch.raw.pplm.pplm_classification_head import ClassificationHead

from mlmodels.model_tch.raw.pplm.run_pplm import run_pplm_example
from mlmodels.model_tch.raw.pplm.run_pplm_discrim_train import train_discriminator as test_case_1





VERBOSE = False

####################################################################################################
from mlmodels.util import os_package_root_path, log, path_norm



def path_setup(out_folder="", sublevel=0, data_path="dataset/"):
    data_path = os_package_root_path( path_add=data_path)
    out_path = os.getcwd() + "/" + out_folder
    os.makedirs(out_path, exist_ok=True)
    model_path = out_path + "/model/"
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path


####################################################################################################




def generate(cond_text,bag_of_words,discrim=None,class_label=-1):
    print(" Generating text ... ")
    unpert_gen_text, pert_gen_text = run_pplm_example(
                        cond_text=cond_text,
                        num_samples=3,
                        bag_of_words=bag_of_words,
                        length=50,
                        discrim=discrim,
                        class_label=class_label,
                        stepsize=0.03,
                        sample=True,
                        num_iterations=3,
                        window_length=5,
                        gamma=1.5,
                        gm_scale=0.95,
                        kl_scale=0.01,
                        verbosity="quiet"
                    )
    print(" Unperturbed generated text :\n")
    print(unpert_gen_text)
    print()
    print(" Perturbed generated text :\n")
    print(pert_gen_text)
    print()
                    

    

####################################################################################################
class Model:
  def __init__(self, model_pars=None, data_pars=None
               ):
    ### Model Structure        ################################
    self.model = None   #ex Keras model
    
    





def fit(model, data_pars=None, compute_pars=None, out_pars=None,   **kw):
  """

  :param model:    Class model
  :param data_pars:  dict of
  :param out_pars:
  :param compute_pars:
  :param kwargs:
  :return:
  """

  sess = None # Session type for compute
  Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
  o = 0
  

  return model, sess



    

def predict(model, sess=None, data_pars=None, compute_pars=None, out_pars=None, **kw):
  ##### Get Data ###############################################
  Xpred, ypred = None, None

  #### Do prediction
  #ypred = model.model.fit(Xpred)
  generate(cond_text="ok", bag_of_words="ok", discrim=None, class_label=-1)

  ### Save Results
  
  
  ### Return val
  if compute_pars.get("return_pred_not") is not None :
    return ypred




####################################################################################################
def get_dataset(data_pars=None, **kw):
  """
    JSON data_pars to get dataset
    "data_pars":    { "data_path": "dataset/GOOG-year.csv", "data_type": "pandas",
    "size": [0, 0, 6], "output_size": [0, 6] },
  """

  if data_pars['train'] :
    Xtrain, Xtest, ytrain, ytest = None, None, None, None  # data for training.
    return Xtrain, Xtest, ytrain, ytest 

  else :
    Xtest, ytest = None, None  # data for training.
    return Xtest, ytest 







def get_params(param_pars=None, **kw):
    from jsoncomment import JsonComment ; json = JsonComment()
    choice      = param_pars['choice']
    config_mode = param_pars['config_mode']
    data_path   = param_pars['data_path']

    if choice == "json":
       data_path = path_norm(data_path)
       cf = json.load(open(data_path, mode='r'))
       cf = cf[config_mode]
       return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']


    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path  = path_norm( "dataset/text/imdb.csv"  )   
        out_path   = path_norm( "ztest/model_tch/pplm/" )
        model_path = os.path.join(out_path , "model")


        data_pars = {"path": data_path, "train": 1, "maxlen": 400, "max_features": 10, }

        model_pars = {"maxlen": 400, "max_features": 10, "embedding_dims": 50,
                      }
        compute_pars = {"engine": "adam", "loss": "binary_crossentropy", "metrics": ["accuracy"],
                        "batch_size": 32, "epochs": 1
                        }

        out_pars = {"path": out_path, "model_path": model_path}

        return model_pars, data_pars, compute_pars, out_pars





#################################################################################        
#################################################################################
if __name__ == '__main__':
    # initializing the model
    # model = Model()
    # generating teh text
    generate(cond_text="The potato",bag_of_words='military')
    # for training classification model give the datset and datset path
    #test_case_1(dataset, dataset_fp=None)

        
    """    
    VERBOSE = True
    test_path = os.getcwd() + "/mytest/"
    
    ### Local
    test(pars_choice="json")
    test(pars_choice="test01")

    ### Global mlmodels
    test_global(pars_choice="json", out_path= test_path,  reset=True)
    """
