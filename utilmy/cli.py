MNAME="utilmy.cli"
HELP ="""


"""
import fire, argparse, os, sys



#############################################################################################
def log(*s):
    """function log
    Args:
        *s:   
    Returns:
        
    """
    print(*s, flush=True)


#############################################################################################
def run_cli():
    """
        utilmy   deeplearning.keras.util_my




    """
    import argparse
    p   = argparse.ArgumentParser()
    add = p.add_argument

    add('task', metavar='task', type=str, nargs=1, help='colab')

    add("--dirin",    type=str, default=None,     help = "repo_url")
    add("--repo_dir",    type=str, default="./",     help = "repo_dir")
    add("--dirout",     type=str, default="docs/",  help = "doc_dir")
    add("--out_file",     type=str, default="",      help = "out_file")
    add("--exclude_dir", type=str, default="",       help = "path1,path2")
    add("--prefix",      type=str, default=None,     help = "https://github.com/user/repo/tree/a")
  
    args = p.parse_args()


    if args.task == 'colab':
        from utilmy import util_colab as mm
        mm.help()


    if args.task =='module':
        import importlib
        myfun = load_function_uri(uri_name="utilmy.ppandas::test")
        myfun()
        #fire.Fire(myfun)








#############################################################################################

def load_function_uri(uri_name="myfolder/myfile.py::myFunction"):
    """
    #load dynamically function from URI pattern
    #"dataset"        : "mlmodels.preprocess.generic:pandasDataset"
    ###### External File processor :
    #"dataset"        : "MyFolder/preprocess/myfile.py:pandasDataset"
    """

    import importlib, sys
    from pathlib import Path
    pkg = uri_name.split("::")

    assert len(pkg) > 1, "  Missing :   in  uri_name module_name:function_or_class "
    package_path, class_name = pkg[0], pkg[1]

    package = package_path.replace("/", ".").replace(".py", "")

    try:
        #### Import from package mlmodels sub-folder
        return  getattr(importlib.import_module(package), class_name)

    except Exception as e1:
        try:
            ### Add Folder to Path and Load absoluate path module
            path_parent = str(Path(package_path).parent.parent.absolute())
            sys.path.append(path_parent)
            log(path_parent)

            #### import Absolute Path model_tf.1_lstm
            model_name   = Path(package_path).stem  # remove .py
            package_name = str(Path(package_path).parts[-2]) + "." + str(model_name)

            #log(package_name, config_name)
            return  getattr(importlib.import_module(package_name), class_name)

        except Exception as e2:
            raise NameError(  f"Module {pkg} notfound, {e1}, {e2}, os.cwd: {os.getcwd()}")





###################################################################################################
if __name__ == "__main__":

    fire.Fire()


