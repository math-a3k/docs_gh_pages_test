# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

import sys, os

from functools import partial

def createDirIfNeeded(dirname):
    """function createDirIfNeeded
    Args:
        dirname:   
    Returns:
        
    """
    try:
        os.mkdir(dirname);
    except:
        pass


class cMemoize:
    def __init__(self, f):
        """ cMemoize:__init__
        Args:
            f:     
        Returns:
           
        """
        self.mFunction = f
        self.mCache = {}
    def __call__(self, *args):
        """ cMemoize:__call__
        Args:
            *args:     
        Returns:
           
        """
        # print("MEMOIZING" , self.mFunction , args)
        if not args in self.mCache:
            self.mCache[args] = self.mFunction(*args)
        return self.mCache[args]
    
    def __get__(self, obj, objtype):
        """ cMemoize:__get__
        Args:
            obj:     
            objtype:     
        Returns:
           
        """
        # Support instance methods.
        return partial(self.__call__, obj)
  
class PyAF_Error(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        """ PyAF_Error:__init__
        Args:
            reason:     
        Returns:
           
        """
        """ Internal_PyAF_Error:__init__
        Args:
            reason:     
        Returns:
           
        """
        self.mReason = reason

class Internal_PyAF_Error(PyAF_Error):
    """Exception raised for errors in the forecasting process.

    Attributes:
        mReason : explanation of the error
    """

    def __init__(self, reason):
        self.mReason = reason



def get_pyaf_logger():
    """function get_pyaf_logger
    Args:
    Returns:
        
    """
    import logging;
    logger = logging.getLogger('pyaf.std');
    if(logger.handlers == []):
        import logging.config
        logging.basicConfig(level=logging.INFO)        
    return logger;

def get_pyaf_hierarchical_logger():
    """function get_pyaf_hierarchical_logger
    Args:
    Returns:
        
    """
    import logging;
    logger = logging.getLogger('pyaf.hierarchical');
    return logger;
