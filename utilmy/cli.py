
import fire


def log(*s):
    print(*s, flush=True)


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




def run_cli():
    """
        utilmy   deeplearning.keras.util_my




    """
    import importlib
    myfun = load_function_uri(uri_name="utilmy.ppandas::test")

    fire.Fire(myfun)





###################################################################################################
if __name__ == "__main__":

    fire.Fire()


