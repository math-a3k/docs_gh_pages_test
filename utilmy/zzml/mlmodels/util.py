# -*- coding: utf-8 -*-
import os
import re
import fnmatch

# import toml
from pathlib import Path
from jsoncomment import JsonComment ; json = JsonComment()

import importlib
from inspect import getmembers


####################################################################################################
class to_namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

    def get(self, key):
        return self.__dict__.get(key)


def log(*s, n=0, m=0):
    sspace = "#" * n
    sjump = "\n" * m
    print("")
    print(sjump, sspace, *s, sspace, flush=True)



####################################################################################################
def os_package_root_path(filepath="", sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    import mlmodels, os, inspect 

    path = Path(inspect.getfile(mlmodels)).parent
    # print( path )

    # path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def os_file_current_path():
    import inspect
    val = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # return current_dir + "/"
    # Path of current file
    # from pathlib import Path

    # val = Path().absolute()
    val = str(os.path.join(val, ""))
    # print(val)
    return val



def os_folder_copy(src, dst):
    """Copy a directory structure overwriting existing files"""
    import shutil
    for root, dirs, files in os.walk(src):
        if not os.path.isdir(root):
            os.makedirs(root)

        for file in files:
            rel_path = root.replace(src, '').lstrip(os.sep)
            dest_path = os.path.join(dst, rel_path)

            if not os.path.isdir(dest_path):
                os.makedirs(dest_path)

            shutil.copyfile(os.path.join(root, file), os.path.join(dest_path, file))



def os_get_file(folder=None, block_list=[], pattern=r'*.py'):
    # Get all the model.py into folder
    folder = os_package_root_path() if folder is None else folder
    # print(folder)
    module_names = get_recursive_files3(folder, pattern)
    # print(module_names)


    NO_LIST = []
    NO_LIST = NO_LIST + block_list

    list_select = []
    for t in module_names:
        t = t.replace(folder, "").replace("\\", ".").replace(".py", "").replace("/", ".")

        flag = False
        for x in NO_LIST:
            if x in t: flag = True

        if not flag:
            list_select.append(t)

    return list_select


def model_get_list(folder=None, block_list=[]):
    # Get all the model.py into folder
    folder = os_package_root_path(__file__) if folder is None else folder
    # print(folder)
    module_names = get_recursive_files(folder, r'/*model*/*.py')

    NO_LIST = ["__init__", "util", "preprocess"]
    NO_LIST = NO_LIST + block_list

    list_select = []
    for t in module_names:
        t = t.replace(folder, "").replace("\\", ".").replace(".py", "").replace("/", ".")

        flag = False
        for x in NO_LIST:
            if x in t: flag = True

        if not flag:
            list_select.append(t)

    return list_select



def get_recursive_files(folderPath, ext='/*model*/*.py'):
    import glob
    files = glob.glob(folderPath + ext, recursive=True)
    return files



def get_recursive_files2(folderPath, ext):
    import fnmatch  #Unix type match
    results = os.listdir(folderPath)
    outFiles = []
    # print(results)


    for file in results:
        # print(file)
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)

        elif fnmatch.fnmatch(file, ext):
            outFiles.append( folderPath + "/" + file)

    return outFiles


def get_recursive_files3(folderPath, ext):
    results = os.listdir(folderPath)
    outFiles = []
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            outFiles += get_recursive_files(os.path.join(folderPath, file), ext)
        # elif re.match(ext, file): 
        elif fnmatch.fnmatch(file, ext):
            outFiles.append(file)
    return outFiles



def get_model_uri(file):
  return Path(os.path.abspath(file)).parent.name + "." + os.path.basename(file).replace(".py", "")




def json_norm(ddict):  
  for k,t in ddict.items(): 
     if t == "None" :
         ddict[k] = None
  return ddict    
         


def path_norm(path=""):
    root = os_package_root_path(__file__, 0)

    path = path.strip()
    if len(path) == 0 or path is None:
        path = root

    tag_list = [ "model_", "//model_",  "dataset", "template", "ztest", "example", "config"  ]


    for t in tag_list :
        if path.startswith(t) :
            path = os.path.join(root, path)
            return path
    return path


def path_norm_dict(ddict):
    for k,v in ddict.items():
        if "path" in k :
            ddict[k] = path_norm(v)
    return ddict



####################################################################################################
def test_module(model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", reset=True):
    ###loading the command line arguments
    # model_uri = "model_xxxx/yyyy.py"

    log("#### Module init   #################################################")
    from mlmodels.models import module_load
    module = module_load(model_uri)
    log(module)

    log("#### Loading params   ##############################################")
    param_pars = {"choice": pars_choice, "data_path": data_path}
    model_pars, data_pars, compute_pars, out_pars = module.get_params(param_pars)

    log("#### Run module test   ##############################################")
    from mlmodels.models import test_module as test_module_global
    test_module_global(model_uri, model_pars, data_pars, compute_pars, out_pars)





####################################################################################################
def config_load_root():
    from jsoncomment import JsonComment ; json = JsonComment()
    path_user = os.path.expanduser('~')
    path_config = path_user + "/.mlmodels/config.json"

    ddict = json.load(open(path_config, mode='r'))
    return ddict


def config_path_pretrained():
    ddict = config_load_root()
    return ddict['model_trained']


def config_path_dataset():
    ddict = config_load_root()
    return ddict['dataset']


def config_set(ddict2):
    ddict = config_load_root()

    for k,x in ddict2.items():
      ddict[k] = x

    json.dump(ddict, open(ddict, mode='w'))
   

def params_json_load(path, config_mode="test", 
                     tlist= [ "model_pars", "data_pars", "compute_pars", "out_pars"] ):
    from jsoncomment import JsonComment ; json = JsonComment()
    pars = json.load(open(path, mode="r"))
    pars = pars[config_mode]

    ### HyperParam, model_pars, data_pars,
    list_pars = []
    for t in tlist :
        pdict = pars.get(t)
        if pdict:
            list_pars.append(pdict)
        else:
            log("error in json, cannot load ", t)

    return tuple(list_pars)



def load_config(args, config_file, config_mode, verbose=0):
    ##### Load file dict_pars as dict namespace #############################
    from jsoncomment import JsonComment ; json = JsonComment()
    print(config_file) if verbose else None

    try:
        pars = json.load(open(config_file, mode="r"))
        # print(arg.param_file, model_pars)

        pars = pars[config_mode]  # test / prod
        print(config_file, pars) if verbose else None

        ### Overwrite dict_pars from CLI input and merge with toml file
        for key, x in vars(args).items():
            if x is not None:  # only values NOT set by CLI
                pars[key] = x

        # print(model_pars)
        pars = to_namespace(pars)  # like object/namespace model_pars.instance
        return pars

    except Exception as e:
        print(e)
        return args


def val(x, xdefault):
    try:
        return x if x is not None else xdefault
    except:
        return xdefault


####################################################################################################
####################################################################################################
def env_conda_build(env_pars=None):
    if env_pars is None:
        env_pars = {'name': "test", 'python_version': '3.6.5'}

    p = env_pars
    cmd = f"conda create -n {p['name']}  python={p['python_version']}  -y"
    print(cmd)
    os.system(cmd)


def env_pip_requirement(env_pars=None):
    from time import sleep
    if env_pars is None:
        env_pars = {'name': "test", 'requirement': 'install/requirements.txt'}

    root_path = os_package_root_path(__file__)
    p = env_pars
    # cmd = f"source activate {p['name']}  &&  "
    cmd = ""
    cmd = cmd + f"  pip install -r  {root_path}/{p['requirement']} "

    print("Installing ", cmd)
    os.system(cmd)
    sleep(60)


def env_pip_check(env_pars=None):
    from importlib import import_module

    if env_pars is None:
        env_pars = {'name': "test", 'requirement': 'install/requirements.txt', "import": ['tensorflow', 'sklearn']}

    flag = 0
    try:
        for f in env_pars['import']:
            import_module(f)
    except:
        flag = 1

    if flag:
        env_pip_requirement(env_pars)


def env_build(model_uri, env_pars):
    from time import sleep

    model_uri2 = model_uri.replace("/", ".")
    root = os_package_root_path()
    model_path = os.path.join(root, env_pars["model_path"])

    env_pars['name'] = model_uri2
    env_pars['python_version'] = "3.6.5"
    env_pars['file_requirement'] = model_path + "/install/requirements.txt"

    env_conda_build(env_pars=env_pars)
    sleep(60)

    env_pip_requirement(env_pars=env_pars)


####################################################################################################
########## Specific ################################################################################
def tf_deprecation():
    try:
        from tensorflow.python.util import deprecation
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        print("Deprecaton set to False")
    except:
        pass


def get_device_torch():
    import torch, numpy as np
    if torch.cuda.is_available():
        device = "cuda:{}".format(np.random.randint(torch.cuda.device_count()))
    else:
        device = "cpu"
    print("use device", device)
    return device










####################################################################################################
###########  Utils #################################################################################
class Model_empty(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None
                 ):
        ### Model Structure        ################################
        self.model = None


def os_path_split(path) :
  return str(Path( path ).parent), str(Path( path ).name) # + str(Path( path ).suffix) 



def load(load_pars):
    p = load_pars

    if "model_keras" in p['model_uri']:
        path = os.path.abspath(p['path'] + "/../")
        name = os.path.basename(p['path']) if ".h5" in p['path'] else "model.h5"
        return load_keras(load_pars)


def save(model=None, session=None, save_pars=None):
    p = save_pars
    if "model_keras" in p['model_uri']:
        path = os.path.abspath( p['path'] + "/../")
        name = os.path.basename(p['path']) if ".h5" in p['path'] else "model.h5"
        save_keras( model, session, save_pars)




def load_tf(load_pars=""):
    """
    https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#
    https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#restore

           import tensorflow as tf

        import main
        import Process
        import Input

        eval_dir = "/Users/Zanhuang/Desktop/NNP/model.ckpt-30"
        checkpoint_dir = "/Users/Zanhuang/Desktop/NNP/checkpoint"

        init_op = tf.initialize_all_variables()

        ### Here Comes the fake variable that makes defining a saver object possible.
        _ = tf.Variable(initial_value='fake_variable')

        ###
        saver = tf.train.Saver()

        with tf.Session() as sess:
          ckpt = tf.train.get_checkpoint_state('./'):
          if ckpt: # checkpointがある場合
        last_model = ckpt.model_checkpoint_path # 最後に保存したmodelへのパス
        print "load " + last_model
        saver.restore(sess, last_model) # 変数データの読み込み
        ...


    """
    import tensorflow as tf
    sess =  tf.compat.v1.Session() # tf.Session()
    model_path = os.path.join(load_pars['path'], "model")
    
    full_name  = model_path + "/model.ckpt"
    # saver = tf.train.import_meta_graph(model_path + '/model.ckpt.meta')


    ## Need Fake
    #_ = tf.Variable(initial_value='xxxxxx_fake')
    #saver      = tf.compat.v1.train.Saver()  

    # with  tf.compat.v1.Session() as sess:
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(model_path + '/model.ckpt.meta')
    saver.restore(sess,  full_name)
    #saver.restore(sess, tf.train.latest_checkpoint(model_path+'/'))
    print(f"Loaded saved model from {model_path}")
    return sess




def save_tf(model=None, sess=None, save_pars= None):
    """
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Later, launch the model, initialize the variables, do some work, and save the
        # variables to disk.
    with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    inc_v1.op.run()
    dec_v2.op.run()
    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
    
    
    """
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Saver#restore  
    import tensorflow as tf
    saver = tf.compat.v1.train.Saver()
    model_path = save_pars['path']  + "/model/"
    os.makedirs(model_path, exist_ok=True)
    save_path = saver.save(sess, model_path + "/model.ckpt")
    print("Model saved in path: %s" % save_path)
  




def load_tch(load_pars):
    import torch
    #path, filename = load_pars['path'], load_pars.get('filename', "model.pkl")
    #path = path + "/" + filename if "." not in path else path
    if os.path.isdir(load_pars['path']):
        path, filename = load_pars['path'], "model.pb"
    else:
        path, filename = os_path_split(load_pars['path'])
    model = Model_empty()
    model.model = torch.load(Path(path) / filename)
    return model


def save_tch(model=None, optimizer=None, save_pars=None):
    import torch
    if os.path.isdir(save_pars['path']):
        path, filename = save_pars['path'], "model.pb"
    else:
        path, filename = os_path_split(save_pars['path'])
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)

    if save_pars.get('save_state') is not None:
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{path}/{filename}" )

    else:
        torch.save(model.model, f"{path}/{filename}")


def save_tch_checkpoint(model, optimiser, save_pars):
    import torch
    path = save_pars['checkpoint_name']
    torch.save({
        'grad_step': save_pars["grad_step"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, path )



# def load(model, optimiser, CHECKPOINT_NAME='nbeats-fiting-checkpoint.th'):
def load_tch_checkpoint(model, optimiser, load_pars):
    import torch
    CHECKPOINT_NAME = load_pars['checkpoint_name']
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0



def load_pkl(load_pars):
    import cloudpickle as pickle
    return pickle.load(open( load_pars['path'], mode='rb') )


def save_pkl(model=None, session=None, save_pars=None):
    import cloudpickle as pickle
    if os.path.isdir(save_pars['path']):
        path, filename = save_pars['path'], "model.pkl"
    else:
        path, filename = os_path_split(save_pars['path'])
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)
    return pickle.dump(model, open( f"{path}/{filename}" , mode='wb') )


def load_keras(load_pars, custom_pars=None):
    from tensorflow.keras.models import load_model
    if os.path.isfile(load_pars['path']):
        path, filename = os_path_split(load_pars['path']  )
    else:
        path = load_pars['path']
        filename = "model.h5"

    path_file = path + "/" + filename if ".h5" not in path else path
    model = Model_empty()
    if custom_pars:
        if custom_pars.get("custom_objects"):
            model.model = load_model(path_file, custom_objects=custom_pars["custom_objects"])
        else:
            model.model = load_model(path_file,
                                     custom_objects={"MDN": custom_pars["MDN"],
                                                     "mdn_loss_func": custom_pars["loss"]})
    else:
        model.model = load_model(path_file)
    return model


def save_keras(model=None, session=None, save_pars=None, ):
    if os.path.isdir(save_pars['path']):
        path = save_pars['path']
        filename = "model.h5"

    else:
        path, filename = os_path_split(save_pars['path'])
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)
    model.model.save(str(Path(path) / filename))

def save_gluonts(model=None, session=None, save_pars=None):
    path = save_pars['path']
    if not os.path.exists(path): os.makedirs(path, exist_ok=True)
    model.model.serialize(Path(path))



def load_gluonts(load_pars=None):
    from gluonts.model.predictor import Predictor
    predictor_deserialized = Predictor.deserialize(Path( load_pars['path'] ))

    model = Model_empty()
    model.model = predictor_deserialized
    return model



#########################################################################################
#########################################################################################
def load_function(package="mlmodels.util", name="path_norm"):
  import importlib
  return  getattr(importlib.import_module(package), name)



def load_function_uri(uri_name="path_norm"):
    """
    #load dynamically function from URI

    ###### Pandas CSV case : Custom MLMODELS One
    #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"

    ###### External File processor :
    #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"


    """
    
    import importlib, sys
    from pathlib import Path
    pkg = uri_name.split(":")

    assert len(pkg) > 1, "  Missing :   in  uri_name module_name:function_or_class "
    package, name = pkg[0], pkg[1]
    
    try:
        #### Import from package mlmodels sub-folder
        return  getattr(importlib.import_module(package), name)

    except Exception as e1:
        try:
            ### Add Folder to Path and Load absoluate path module
            path_parent = str(Path(package).parent.parent.absolute())
            sys.path.append(path_parent)
            #log(path_parent)

            #### import Absolute Path model_tf.1_lstm
            model_name   = Path(package).stem  # remove .py
            package_name = str(Path(package).parts[-2]) + "." + str(model_name)
            #log(package_name, model_name)
            return  getattr(importlib.import_module(package_name), name)

        except Exception as e2:
            raise NameError(f"Module {pkg} notfound, {e1}, {e2}")




def load_callable_from_uri(uri):
    assert(len(uri)>0 and ('::' in uri or '.' in uri))
    if '::' in uri:
        module_path, callable_name = uri.split('::')
    else:
        module_path, callable_name = uri.rsplit('.',1)
    if os.path.isfile(module_path):
        module_name = '.'.join(module_path.split('.')[:-1])
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    return dict(getmembers(module))[callable_name]
        

def load_callable_from_dict(function_dict, return_other_keys=False):
    function_dict = function_dict.copy()
    uri = function_dict.pop('uri')
    func = load_callable_from_uri(uri)
    try:
        assert(callable(func))
    except:
        raise TypeError(f'{func} is not callable')
    arg = function_dict.pop('arg', {})
    if not return_other_keys:
        return func, arg
    else:
        return func, arg, function_dict
    











"""
def path_local_setup(current_file=None, out_folder="", sublevel=0, data_path="dataset/"):
    root = os_package_root_path(__file__, sublevel=0, path_add="")

    out_path = path_norm(out_folder)
    data_path = path_norm(data_path)

    model_path = f"{out_path}/model/"
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    log(data_path, out_path, model_path)
    return data_path, out_path, model_path
"""







def os_folder_getfiles(folder, ext, dirlevel = -1, mode="fullpath"):
    """

    :param folder: folder path to be analyzed
    :type folder: string
    :param ext: file extension hint example: "*.json"
    :type ext: string
    :param dirlevel: number of levels to be analyzed
    :type dirlevel: int
    :param mode: either fullpath or filename
    :type mode: string
    :return: list of files paths or names (depending on mode param)
    :rtype: list of str
    """

    files_list = os.listdir(folder)
    if dirlevel==0:
        if (mode=="fullpath"):
            return [os.path.join(folder, p) for p in files_list if fnmatch.fnmatch(p, ext)]
        if (mode=="filename"):
            return [f for f in files_list if fnmatch.fnmatch(f, ext)]
        else:
            print("Error: mode parameter is either fullpath or filename")
    elif (dirlevel==-1 or dirlevel >= 1):
        all_files = []
        for entry in files_list:
            full_path = os.path.join(folder, entry)
            if os.path.isdir(full_path):
                if dirlevel==-1:
                    all_files += os_folder_getfiles(full_path, ext, dirlevel, mode)
                if dirlevel >= 1:
                    all_files += os_folder_getfiles(full_path, ext, dirlevel-1, mode)
            elif fnmatch.fnmatch(entry, ext):
                if (mode=="fullpath"):
                    all_files.append(full_path)
                if (mode=="filename"):
                    all_files.append(entry)
    else:
        print("Error: dirlevel parameter is either -1 or >=1")
    return all_files