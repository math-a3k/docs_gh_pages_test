# -*- coding: utf-8 -*-
import glob
import json
import os
import pathlib
import shutil
import sys
import tarfile
import zipfile
from typing import Optional, Union
import yaml
from loguru import logger





#####################################################################
def test_all():
    """
    #### python test.py   test_utils
    """
    def test_logs(): 
        from utilmy.utils import log,log2, logw, loge, logger_setup
        print("testing logs utils........")
        logger_setup()
        log("simple log ")
        log2("debug log")
        logw("warning log")
        loge("error log")
    
    def config_load_test():
        from utilmy.utils import config_load
        config_load()
    
    def dataset_download_test():
        from utilmy.utils import dataset_donwload
        dataset_donwload("https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz", './testdata/tmp/test/dataset/')
    
    def os_extract_archive_test():
        from utilmy.utils import os_extract_archive
        os_extract_archive("./testdata/tmp/test/dataset/mnist_png.tar.gz","./testdata/tmp/test/dataset/archive/", archive_format = "auto")
    
    def to_file_test():
        from utilmy.utils import to_file
        to_file("to_file_test_str", "./testdata/tmp/test/to_file.txt")

    test_logs()
    config_load_test()
    dataset_download_test()
    os_extract_archive_test()
    to_file_test()


def test0(): 
    logger_setup()
    log("simple log ")
    log2("debug log")
    logw("warning log")
    loge("error log")
    
def test1():
    config_load()
    dataset_donwload("https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz", './testdata/tmp/test/dataset/')
    os_extract_archive("./testdata/tmp/test/dataset/mnist_png.tar.gz","./testdata/tmp/test/dataset/archive/", archive_format = "auto")
    to_file("to_file_test_str", "./testdata/tmp/test/to_file.txt")

##########################################################################################
################### Logs Wrapper #########################################################
def log(*s):
    logger.info(",".join([str(t) for t in s]))


def log2(*s):
    logger.debug(",".join([str(t) for t in s]))


def logw(*s):
    logger.warning(",".join([str(t) for t in s]))


def loge(*s):
    logger.error(",".join([str(t) for t in s]))


def logger_setup():
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<level>{level: <8}</level>| <level>{message}</level>",
            }
        ]
    }
    logger.configure(**config)


logger_setup()


##########################################################################################
################### donwload  ############################################################
def config_load(config_path: Optional[Union[str, pathlib.Path]] = None):
    """Load Config file into a dict
    1) load config_path
    2) If not, load in HOME USER
    3) If not, create default one
    # config_default = yaml.load(os.path.join(os.path.dirname(__file__), 'config', 'config.yaml'))

    Args:
        config_path: path of config or 'default' tag value
    Returns: dict config
    """
    path_default = pathlib.Path.home() / ".mygenerator"
    config_path_default = path_default / "config.yaml"
    config_default = {
        "current_dataset": "mnist",
        "datasets": {
            "mnist": {
                "url": "https://github.com/arita37/mnist_png/raw/master/mnist_png.tar.gz",
                "path": str(path_default / "mnist_png" / "training"),
            }
        },
    }

    ##################################################################
    if config_path is None or config_path == "default":
        logw(f"Using config: {config_path_default}")
        config_path = config_path_default

    try:
        log2("loading config", config_path)
        return yaml.load(config_path.read_text(), Loader=yaml.Loader)
    except Exception as e:
        logw(f"Cannot read yaml file {config_path}", e)

    logw("#### Using default configuration")
    log2(config_default)
    log(f"Creating default config file in {config_path}")
    os.makedirs(path_default, exist_ok=True)
    with open(config_path, mode="w") as fp:
        json.dump(config_default, fp)
    return config_default





##########################################################################################
################### donwload  ############################################################
def dataset_donwload(url, path_target):
    """Donwload on disk the tar.gz file
    Args:
        url:
        path_target:
    Returns:

    """
    import wget
    log(f"Donwloading mnist dataset in {path_target}")
    os.makedirs(path_target, exist_ok=True)
    wget.download(url, path_target)
    tar_name = url.split("/")[-1]
    os_extract_archive(path_target + "/" + tar_name, path_target)
    log2(path_target)
    return path_target + tar_name

  

def os_extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.
    Args:
        file_path: path to the archive file
        path: path to extract the archive file
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = os.path.abspath(file_path)
    path = os.path.abspath(path)

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def to_file(s, filep):
    with open(filep, mode="a") as fp:
        fp.write(str(s) + "\n")
