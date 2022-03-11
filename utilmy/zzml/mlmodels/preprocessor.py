import os
import sys
import inspect
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import sklearn
import keras
from sklearn.model_selection import train_test_split
from cli_code.cli_download import Downloader
from collections.abc import MutableMapping
from jsoncomment import JsonComment ; json = JsonComment()
import cloudpickle as pickle
from util import load_callable_from_dict


class PreprocessorError(Exception):
    pass


class MissingDataPreprocessorError(PreprocessorError):
    def __init__(self):
        """ PreprocessorNotFittedError:__init__
        Args:
        Returns:
           
        """
        """ MissingDataPreprocessorError:__init__
        Args:
        Returns:
           
        """
        print(f"data_preprocessor is missing in preprocessor.")


class PreprocessorNotFittedError(PreprocessorError):
    def __init__(self):
        print(f"""Preprocessor has not been fitted.""")


class Preprocessor:
    def __init__(self, preprocessor_dict):
        """ Preprocessor:__init__
        Args:
            preprocessor_dict:     
        Returns:
           
        """
        self._preprocessor_specs = None
        self._preprocessor = None
        self._names = None
        self._interpret_preprocessor_dict(preprocessor_dict)

    def _interpret_preprocessor_dict(self, pars):
        """ Preprocessor:_interpret_preprocessor_dict
        Args:
            pars:     
        Returns:
           
        """
        if len(pars.keys()) == 0:
            return None
        try:
            data_preprocessor = pars.get("data_preprocessor")
        except KeyError:
            raise MissingDataPreprocessorError()
        self._preprocessor_specs = load_callable_from_dict(data_preprocessor)
        self._names = pars.get("names", None)

    def _name_outputs(self, names, outputs):
        """ Preprocessor:_name_outputs
        Args:
            names:     
            outputs:     
        Returns:
           
        """
        if hasattr(outputs, "__getitem__") and len(outputs) == len(names):
            return dict(zip(names, outputs))
        else:
            raise Exception("Outputs could not be named")

    def fit_transform(self, data):
        """ Preprocessor:fit_transform
        Args:
            data:     
        Returns:
           
        """
        preprocessor, args = self._preprocessor_specs
        if inspect.isclass(preprocessor):
            preprocessor_instance = preprocessor.fit(**args)
            self._preprocessor = preprocessor_instance.transform
            data = preprocessor_instance.transform(data)
        else:
            transform = (
                (lambda x: preprocessor(x, **args))
                if args is not None
                else lambda x: preprocessor(x)
            )
            self._preprocessor = transform
            data = transform(data)
        if self._names is not None:
            data = self._name_outputs(self._names, data)
        return data

    def transform(self, data):
        """ Preprocessor:transform
        Args:
            data:     
        Returns:
           
        """
        if self._preprocessor is None:
            raise PreprocessorNotFittedError()
        data = self._preprocessor(data)
        if self._names is not None:
            data = self._name_outputs(self.names, data)
        return data
