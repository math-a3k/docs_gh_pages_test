# -*- coding: utf-8 -*-
"""
# python util_config.py test





"""
import importlib, os, yaml
from pathlib import Path
from typing import Union
from box import Box



#########################################################################################################
def log(*s):
    print(*s, flush=True)


def loge(*s):
    print(*s, flush=True)



#########################################################################################################
#########################################################################################################
def test_yamlschema():
    cfg_dict = config_load("config.yaml")
    isok     = config_isvalid_yamlschema(cfg_dict, "config_val.yaml")
    log(isok)


def test_pydanticgenrator():
    from datamodel_code_generator import InputFileType
    # generating from json file
    pydantic_model_generator(
        Path("config.json"), InputFileType.Json, Path("pydantic_model_json.py")
    )
    assert Path("pydantic_model_json.py").exists(), "File does not exist"

    # generating from yaml file
    pydantic_model_generator(
        Path("config.yaml"), InputFileType.Yaml, Path("pydantic_model_yaml.py")
    )
    assert Path("pydantic_model_yaml.py").exists(), "File does not exist"


def test4():
    from pydantic import BaseModel
    cfg_dict = config_load("config.yaml")
    pydantic_model = convert_dict_to_pydantic(cfg_dict, "pydantic_config_val.yaml")
    assert isinstance(pydantic_model, BaseModel)


def test_example():
    ss = """
        string: ok
        regex:  abcd
        number: 6.7
        integer: 3
        boolean: True
        list: [1,2,3]
        enum: 'one'
        map:
        null:
        date:
        nest:
            integer: 7
            nest:
                string: "ok"

    """
    return ss


#########################################################################################################
def config_load(
        config_path:    str  = None,
        path_default:   str  = None,
        config_default: dict = None,
        save_default:   bool = False,
        to_dataclass:   bool = True,
) -> Union[dict, Box]:
    """Load Config file into a dict
    1) load config_path
    2) If not, load in USER/.myconfig/.config.yaml
    3) If not, create default save in USER/.myconfig/.config.yaml
    Args:
        config_path:   path of config or 'default' tag value
        path_default : path of default config
        config_default: dict value of default config
        save_default: save default config on disk
    Returns: dict config
    """
    import json, yaml, pathlib

    #########Default value setup ###########################################
    path_default = (
        pathlib.Path.home() / ".myconfig" if path_default is None else path_default
    )
    config_path_default = path_default / "config.yaml"
    if config_default is None:
        config_default = {"field1": "", "field2": {}}

    #########Config path setup #############################################
    if config_path is None or config_path == "default":
        log(f"Config: Using {config_path_default}")
        config_path = config_path_default
    else:
        config_path = pathlib.Path(config_path)

    ######### Load Config ##################################################
    try:
        log("Config: Loading ", config_path)
        if config_path.suffix == ".yaml":
            cfg = yaml.safe_load(config_path.read_text())

        elif config_path.suffix == ".json":
            cfg = json.loads(config_path.read_text())

        elif config_path.suffix in [".properties", ".ini"]:
            from configparser import SafeConfigParser
            cfg = SafeConfigParser()
            cfg.read(str(config_path))

        elif config_path.suffix == ".toml":
            import toml
            cfg = toml.loads(config_path.read_text())
        else:
            raise Exception(f"not supported file {config_path}")

        if to_dataclass:  ### myconfig.val  , myconfig.val2
            return Box(cfg)
        return cfg

    except Exception as e:
        log(f"Config: Cannot read file {config_path}", e)

    ######################################################################
    log("Config: Using default config")
    log(config_default)
    if save_default:
        log(f"Config: Writing config in {config_path_default}")
        os.makedirs(path_default, exist_ok=True)
        with open(config_path_default, mode="w") as fp:
            yaml.dump(config_default, fp)
    return config_default


def config_isvalid_yamlschema(config_dict: dict, schema_path: str = 'config_val.yaml', silent: bool = False) -> bool:
    """Validate using a  yaml file
    Args:
        config_dict:
        schema_path:
        silent:
    Returns: True/False
    """
    import yamale
    schema = yamale.make_schema(schema_path)

    try:
        result = schema.validate(config_dict, data_name=schema_path, strict=True)
        if not result.isValid():
            raise yamale.YamaleError([result])
        return True

    except yamale.YamaleError as e:
        for result in e.results:
            loge(f"Error validating data '{result.data}' with '{result.schema}'\n\t")
            for error in result.errors:
                loge(f"\t{error}")
        return False


def config_isvalid_pydantic(config_dict: dict,
                            pydanctic_schema: str = 'config_py.yaml', silent: bool = False) -> bool:
    """Validate using a pydantic files
    Args:
        config_dict:
        pydanctic_schema:
        silent:
    Returns: True/False
    """
    import yamale
    try:
        return True

    except yamale.YamaleError as e:
        return False


##################################################################################################
##################################################################################################
def convert_yaml_to_box(yaml_path: str) -> Box:
    with open(yaml_path) as f:
        data = yaml.load(f)
    return Box(data)


def convert_dict_to_pydantic(config_dict: dict, schema_name: str):
    # pip install pydantic-gen
    from pydantic_gen import SchemaGen

    generated = SchemaGen(schema_name)
    generated.to_file(f"{schema_name.split('.')[0]}.py")

    pydantic_module = importlib.import_module(
        f"zz936.configs.{schema_name.split('.')[0]}"
    )
    return pydantic_module.MainSchema(**config_dict)


def pydantic_model_generator(
        input_file: Union[Path, str],
        input_file_type,
        output_file: Path,
        **kwargs,
) -> None:
    """
    Args:
        input_file:
        input_file_type:
        output_file:
        **kwargs:

    Returns:
    # https://github.com/koxudaxi/datamodel-code-generator
    # pip install datamodel-code-generator

    """
    from datamodel_code_generator import Error, generate

    try:
        generate(
            input_file, input_file_type=input_file_type, output=output_file, **kwargs
        )
    except Error as e:
        loge(f"Error occurred while generating pydantic model: `{e.message}`")
    else:
        log(
            f"Successfully generated pydantic model from {input_file} to {output_file}"
        )




#########################################################################################################

def global_verbosity(cur_path, path_relative="/../../config.json",
                   default=5, key='verbosity',):
    """ Get global verbosity
    verbosity = global_verbosity(__file__, "/../../config.json", default=5 )

    verbosity = global_verbosity("repo_root", "config/config.json", default=5 )

    :param cur_path:
    :param path_relative:
    :param key:
    :param default:
    :return:
    """
    import utilmy, json
    try   :
      if 'repo_root' == cur_path  :
          cur_path =  utilmy.git_repo_root()

      if '.json' in path_relative :
         dd = json.load(open(os.path.dirname(os.path.abspath(cur_path)) + path_relative , mode='r'))

      elif '.yaml' in path_relative or '.yml' in path_relative :
         import yaml
         dd = yaml.load(open(os.path.dirname(os.path.abspath(cur_path)) + path_relative , mode='r'))

      else :
          raise Exception( path_relative + " not supported ")
      verbosity = int(dd[key])

    except Exception as e :
      verbosity = default
      #raise Exception(f"{e}")
    return verbosity







if __name__ == "__main__":
    import fire
    fire.Fire()






def zzz_config_load_validate(config_path: str, schema_path: str, silent: bool = False
                             ) -> Union[Box, None]:

    import yamale
    schema = yamale.make_schema(schema_path)
    data = yamale.make_data(config_path)

    try:
        yamale.validate(schema, data)
        return convert_yaml_to_box(config_path)

    except yamale.YamaleError as e:
        print("Validation failed!\n")
        for result in e.results:
            print(f"Error validating data '{result.data}' with '{result.schema}'\n\t")
            for error in result.errors:
                print(f"\t{error}")
        if not silent:
            raise e
