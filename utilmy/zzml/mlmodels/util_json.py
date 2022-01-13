"""

Alll related to json dynamic parsing


"""# -*- coding: utf-8 -*-
import os
import re
import fnmatch
import ast
import json
import sys

# import toml
from pathlib import Path
from jsoncomment import JsonComment ; json = JsonComment()
import fire

import importlib
from inspect import getmembers

from mlmodels.util import *
from mlmodels.util import path_norm



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


###################################################################################################
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
    



def test_functions_json(arg=None):
  """
  
       args :[]   , kw_args : {}
  
  """
  from mlmodels.util import load_function_uri

  path = path_norm("dataset/test_json/test_functions.json")
  dd   = json.load(open( path ))['test']
  
  for p in dd  :
     try :
         log("\n\n","#"*20, p)

         myfun = load_function_uri( p['uri'])
         log(myfun)

         w  = p.get('args', []) 
         kw = p.get('kw_args', {} )
         
         if len(kw) == 0 and len(w) == 0   : log( myfun())
         elif  len(kw) > 0 and len(w) > 0  : log( myfun( *w,  ** kw ))
         elif  len(kw) > 0 and len(w) == 0 : log( myfun( ** kw ))
         elif  len(kw) == 0 and len(w) > 0 : log( myfun( *w ))
                                
     except Exception as e:
        log(e, p )    


def json_to_object(ddict):
  """
     Execute a function from json to actual arguments
     {uri:    args :[]   , kw_args : {}   }
      { "uri" : "mlmodels.util.log", "args" : [ "x1" , "passed"] ,     "kw_args" : { }  }
     ,{ "uri" : "mlmodels.util:log", "args" : [ "x1" , "passed"] ,      "kw_args" : { }   }     
     
     , {"uri": "mlmodels.data:download_gogledrive",
        "args": [[{"fileid": "1-K72L8aQPsl2qt_uBF-kzbai3TYG6Qg4", "path_target": "ztest/covid19/test.json"},
                  {"fileid": "1-8Ij1ZXL9YmQRylwRloABdqnxEC1mhP_", "path_target": "ztest/covid19/train.json" }
          ]], 
      "kw_args" : {}} 
     
  """
  from mlmodels.util import load_function_uri
  p = ddict
  try :
      myfun = load_function_uri( p['uri'])
      w  = p.get('args', [])
      kw = p.get('kw_args', {} )
      if len(kw) == 0 and len(w) == 0   : return myfun()
      elif  len(kw) > 0 and len(w) > 0  : return myfun( *w,  ** kw )
      elif  len(kw) > 0 and len(w) == 0 : return myfun( ** kw )
      elif  len(kw) == 0 and len(w) > 0 : return myfun( *w )
  except Exception as e:
      log(e, p )

        
def json_norm_val(x):
    if isinstance(x, list) : return x
    if isinstance(x, dict) : return x
    if x == "none" or x == "None"   : return None
    if x == "" :     return None
    else : return x
    
def json_norm(ddict):
     return { key:json_norm_val(x)     for key,x in ddict.items() }
        
def json_parse(ddict) :
    """
      https://github.com/arita37/mlmodels/blob/dev/mlmodels/dataset/test_json/test_functions.json
      https://github.com/arita37/mlmodels/blob/dev/mlmodels/dataset/json/benchmark_timeseries/gluonts_m5.json
          "deepar": {
         "model_pars": {
             "model_uri"  : "model_gluon.gluonts_model",
             "model_name" : "deepar",
             "model_pars" : {
                 "prediction_length": 12, 
                 "freq": "D",
                 "distr_output" :  {"uri" : "gluonts.distribution.neg_binomial:NegativeBinomialOutput"}, 

                 "distr_output" :  "uri::gluonts.distribution.neg_binomial:NegativeBinomialOutput", 
    """
    import copy
    js = ddict
    js2 = copy.deepcopy(js)
    
    def parse2(d2) :
        if "uri" in d2 :
            return json_to_object(d2)   ### Be careful not to include heavy compute
        else :
            return json_norm(d2)
        
    for k,val in js.items() :
        if isinstance(val, dict):
            js2[k] = parse2(val)
            
        elif  "uri::" in val :   ## Shortcut when nor argument
            js2[k] = json_to_object({ "uri" :  val.split("uri::")[-1] })
        else :
            js2[k] = json_norm_val(val)
    return js2    


def json_codesource_to_json(fpath) :
    """
        read a python file and create json
        Ex:
        def MyClass():
           def __init__(fname, zout=""):
           def method1(x=1, y=2)
       --->
               {"uri": "MyClass",   "arg" : ["fname"] , "kwargs": {"out" : "ztmp"} }       
               
               {"uri": "MyClass.method1",   "arg" : [] , "kwargs": {"x" : 1, "y" : 2, } }       
           

    """
    ff = open(fpath, mode="r")
        
        

###################################################################################################            
import json
import os
import pandas as pd
import time
from mlmodels.util import os_folder_getfiles


def jsons_to_df(json_paths):
    """
    This function takes as a parameter list of json paths to be read and transformes them into a single Dataframe
    the dataframe contain five different columns:
            column1: file_path, original path of the json
            column2: filename, original json file name
            column3: json_name, highest key level that leads to the subkey in question
            column4: fullname, name containing the subkey and its higher levels for example:
                     fullname(F , {A:{B:1,C:{E:4,F:5}},D:3}) = "A.C.F"
            column5: field_value, value of the subkey in question
    :param json_paths: list of json paths
    :type json_paths: list of str
    :return: DataFrame of the jsons
    :rtype: DataFrame
    """
    #Indexed dictionaries is a list of dictionaries containing two keys, one for json path and the other for the dictionary of the json
    indexed_dicts = []
    problem = 0
    for i in range(len(json_paths)):
        #For each json file, read file
        try:
            with open(json_paths[i]) as json_file:
                d = dict()
                d['Path'] = json_paths[i]
                d['Json'] = json.load(json_file)
                indexed_dicts.append(d)
        #If there is a problem reading file, prints its path
        except:
            if problem == 0:
                print("Files That have a structure problem:\n")
            problem += 1
            print('\t', json_paths[i])
            continue
    print("Total flawed jsons:\t", problem)
    all_jsons = []
    for i in range(len(indexed_dicts)):
        all_jsons.append(indexed_dicts[i]['Json'])
    #.json_normalize() help create nested json keys names
    ddf = pd.json_normalize(all_jsons)
    result=[]
    keys=list(ddf.columns)
    #Create list of dictionaries of the given jsons, and add keys that contain extra information
    for i in range(len(all_jsons)):
        for k in keys:
            if(str(ddf[k][i]) != 'nan'):
                d=dict()
                d['file_path'] = indexed_dicts[i]['Path']
                d['filename'] = os.path.basename(indexed_dicts[i]['Path'])
                d['json_name'] = k.split(".")[0]
                d['fullname'] = k
                d['field_value'] = ddf[k][i]
                result.append(d)
    del ddf
    df = pd.DataFrame(result)
    
    def getlevel(x, i) :
        try :    return x.split(".")[i]
        except : return ""
    df['level_1'] = df['fullname'].apply(lambda x :  getlevel(x, 1) )
    df['level_2'] = df['fullname'].apply(lambda x :  getlevel(x, 2) )    
    df['level_3'] = df['fullname'].apply(lambda x :  getlevel(x, 3) )        
    return df


def dict_update(fields_list, d, value):
    """
    :param fields_list: list of hierarchically sorted dictionary fields leading to value to be modified
    :type fields_list: list of str
    :param d: dictionary to be modified
    :type d: dict
    :param value: new value
    :type value: any type
    :return: updated dictionary
    :rtype: dict
    """
    if len(fields_list) > 1:
        l1 = fields_list[1:]
        k = fields_list[0]
        if k not in list(d.keys()):
            d[k] = dict()
        #Recursive call if the level of the key to be updated is > 1
        d[k] = dict_update(l1, d[k], value)
    else:
        k = fields_list[0]
        d[k] = value
        return d
    return d


def json_csv_to_json(file_csv="", out_path="dataset/"):
    """
    This function takes as a parameter csv file of jsons, create a normalized json structure,
    and creates new normalized jsons in out_path
    :param csv: csv file containing jsons to be normalized
    :type csv: str
    :return: list of normalized jsons as dictionaries
    :rtype: list of dicts
    """
    ddf       = pd.read_csv(file_csv)
    paths     = list(ddf['file_path'].unique() )
    fullnames = list(ddf['fullname'].unique() )
    dicts=[]
    for fp in paths:
        dd=dict()
        #Here for each json, normalized json skeleton is created
        for fn in fullnames:
            l = fn.split('.')
            dd = dict_update(l, dd, None)
        json_ddf = ddf[ddf['file_path']==fp]
        filled_values = list(json_ddf['fullname'])
        #Then update normalized json skeleton by entering filled values on the given json
        for fv in filled_values:
            dd.update(dict_update(fv.split('.'), dd, list(json_ddf[json_ddf['fullname'] == fv]['field_value'])[0]))
        dicts.append(dd)
        
    dataset_dir = path_norm(out_path)    
    #dataset_dir = os_package_root_path()+'dataset'
    #os.chdir(dataset_dir)
    paths = [p[len(dataset_dir)+1:] for p in paths]
    #Create new json paths respecting same hierarchical structure of input paths
    new_paths = []
    for i in range(len(paths)):
        lp = paths[i].split('/')
        lp[0]='normalized_jsons'
        dire = '/'.join(''.join(i) for i in lp[:len(lp)-1])
        new_paths.append(dire)
    #Create new folders and subfolders of the new normalized jsons
    for p in list(set(new_paths)):
        if not os.path.exists(p):
            os.makedirs(p)
    #Create the new normalized json files
    for i in range(len(paths)):
        with open(new_paths[i] + '/' + paths[i].split('/')[-1], 'w') as fp:
            json.dump(dicts[i], fp, indent=4)
    print("New normalized jsons created, check mlmodels\\mlmodels\\dataset")
    return dicts


class _EmptyJSONData(Exception):
    """
    The generated json data contains nothing. This may occur if user passes
    a Python file containing no function or class defintion.
    """
    pass


class _ParserBase(ast.NodeVisitor):
    """
    This is an abstract class. It has two subclasses: _ExtractFunc, which
    extracts JSON from function definitions, and _ExtractClass, which
    extracts JSON from class defintions. This class provides common functions
    used by the two subclasses.

    :param srcfile: path of Python source file
    :raises _EmptyJSONData: the generated json data contains nothing
    """
    def __init__(self, srcfile):
        self.srcfile = os.path.abspath(srcfile)
        self.data = []
        with open(self.srcfile) as f:
            self.tree = ast.parse(f.read())
        self.visit(self.tree)
        if not self.data:
            raise _EmptyJSONData()
        self.datafile = os.path.splitext(self.srcfile)[0] + ".json"
        with open(self.datafile, 'w') as f:
            output = json.dumps(self.data, indent=4) + '\n'
            f.write(self.merge_lines(output))

    def get_name(self, node):
        """
        Get name string from an ast.Name or ast.Attribute node.

        :param node: a node of ast.Name or ast.Attribute type
        :returns: name string
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self.get_name(node.value) + "." + node.attr

    def get_value(self, node):
        """
        Get value from an AST node. If the node is of ast.Num, ast.Str,
        ast.NameConstant type, the returned value is of Python int, str,
        or bool type, respectively. If the node is of ast.Name or
        ast.Attribute, the returned value is of Python str type.

        :param node: an AST node
        :returns: value
        """
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
            return self.get_name(node)

    def get_signature(self, node: ast.FunctionDef, parenturi):
        """
        Generate JSON data for a function or method.

        :param node: a node of ast.FunctionDef type
        :param parenturi: URI of the parent node
        :returns: JSON data representing the function or method
        """
        fndata = {"uri": "%s:%s" % (parenturi, node.name), "args": [], "kwargs": {}}
        argumentnode = node.args
        identifiers = [argnode.arg for argnode in argumentnode.args
                       if argnode.arg != "self"]
        values = [self.get_value(valnode) for valnode in argumentnode.defaults]
        size = len(identifiers)
        for index, name in enumerate(identifiers):
            try:
                value = values[index - size]
                fndata["kwargs"][name] = value
            except IndexError:
                fndata["args"].append(name)
        return fndata

    def merge_lines(self, text):
        """
        Merge items of args and kwargs to a single line.

        :param text: JSON data
        :returns: merged JSON data
        """
        def _merge_lines(text, keyword, offset, open_bracket, close_bracket):
            start = 0
            while start <= len(text):
                index_start = text.find(keyword, start)
                if index_start == -1:
                    break
                index_start += offset
                index_end = index_start
                count = 1
                while index_end <= len(text) - 1:
                    if text[index_end] == open_bracket:
                        count += 1
                    if text[index_end] == close_bracket:
                        count -= 1
                    if count == 0:
                        break
                    index_end += 1
                if count != 0:
                    raise Exception("Failed to find close bracket '%s' when processing %s" %
                                    (close_bracket, self.datafile))
                text = text[:index_start] + ' '.join(text[index_start:index_end].split()) + \
                    text[index_end:]
                start = index_end + 1
            return text
        # Merge all items of args list into a single line
        keyword1 = '"args": ['
        open_bracket1 = '['
        close_bracket1 = ']'
        text = _merge_lines(text, keyword1, len(keyword1), open_bracket1, close_bracket1)
        # Merge all items of kwargs dict into a single line
        keyword1 = '"kwargs": {'
        open_bracket1 = '{'
        close_bracket1 = '}'
        text = _merge_lines(text, keyword1, len(keyword1), open_bracket1, close_bracket1)
        return text


class _ExtractFunc(_ParserBase):
    """
    The class generate JSON data for all functions defined in a Python
    source file. It works by looking method defintions in a file and
    generates JSON based on those methods' signatures.

    :param srcfile: Python source file
    """
    def __init__(self, srcfile):
        self.data = []
        super().__init__(srcfile)

    def visit_Module(self, node):
        for stmtnode in node.body:
            if isinstance(stmtnode, ast.FunctionDef):
                self.data.append(self.get_signature(stmtnode, self.srcfile))


class _ExtractClass(_ParserBase):
    """
    The class generate JSON data for all classes defined in a Python
    source file. It works by looking for method defintions in a class
    and generates JSON based on those methods' signatures.

    :param srcfile: Python source file
    """
    def __init__(self, srcfile):
        self.data = []
        super().__init__(srcfile)

    def visit_ClassDef(self, node):
        if node not in self.tree.body:
            return
        clsuri = "%s:%s" % (self.srcfile, node.name)
        methods = []
        for stmtnode in node.body:
            if isinstance(stmtnode, ast.FunctionDef):
                methods.append(self.get_signature(stmtnode, clsuri))
        self.data.append({clsuri: methods})


def json_extract_code(path):
    """
    Convert Python source code to JSON. The path argument can be either
    a Python source file or a diretory. In latter case, all Python source
    files in that directory (including its subdirs) are converted.

    :param path: path for a Python file or a directory
    """
    def _json_extract_code(srcfile):
        try:
            with open(srcfile) as f:
                tree = ast.parse(f.read())
            hasClassDef = False
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    hasClassDef = True
                    break
            if hasClassDef:
                t = _ExtractClass(srcfile)
            else:
                t = _ExtractFunc(srcfile)
            print(t.datafile)
        except FileNotFoundError:
            print("Failed to find %s." % srcfile, file=sys.stderr)
        except _EmptyJSONData:
            print("Skipped %s." % srcfile, file=sys.stderr)
        except Exception as e:
            print("Failed to process %s." % srcfile, file=sys.stderr)
            print(e, file=sys.stderr)
    if os.path.isfile(path):
        _json_extract_code(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                _, suffix = os.path.splitext(filename)
                if suffix != ".py":
                    continue
                filepath = root + "/" + filename
                _json_extract_code(filepath)
    else:
        print("%s is not a regular file or directory." % path, file=sys.stderr)


def test_json_extract_code():
    """
    Function to test extracting json from code.

    Note: since json_extract_code() doesn't return any value. It outputs the
    JSON file(s) it generates on stdout. user is responsible to check the
    contents of the files.
    """
    srcfile1 = os_package_root_path(path_add="model_tf/raw/27_byte_net.py")
    srcfile2 = os_package_root_path(path_add="model_tf/raw/6_encoder_gru.py")
    json_extract_code(srcfile1)
    json_extract_code(srcfile2)

def test_json_conversion():
    """
    Function to test converting jsons in dataset/json to normalized jsons
    :rtype: list of normalized jsons as dictionaries
    """
    json_folder_path = path_norm("dataset\\json")
    jsons_paths      = os_folder_getfiles(json_folder_path,ext = "*.json")
    df               = jsons_to_df(jsons_paths)
    df.to_csv(json_folder_path + '\\table_json.csv')
    print('csv created successfully')
    time.sleep(1)
    dicts2 = json_csv_to_json(json_folder_path + '\\table_json.csv')
    print(dicts2)
    return dicts2


# Testing code
if __name__ == "__main__":
    import fire
    fire.Fire()

    ### python mlmodels/util_json.py  test_json_conversion
