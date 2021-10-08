# -*- coding: utf-8 -*-
import pytest

from main import spark_init
from src.utils import config_load, log
from tests.test_common import TEST_CONFIG_FILE


# Arrange
@pytest.fixture(scope="session")
def config():
    """
    This method will prepare configuration dictionary from config file for unit testing..
    """
    #config_name = '../config.yaml'
    config      = config_load(TEST_CONFIG_FILE)
    return config


# Arrange
@pytest.fixture(scope="session")
@pytest.mark.usefixtures("config")
def spark_session(config: dict):
    """ Return spark session for unit testing
    @param config: Configuration dictionary object for unit testing
    """
    log("Create spark session")
    spark = spark_init(config)
    return spark


