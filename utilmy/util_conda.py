# -*- coding: utf-8 -*-
MNAME = "utilmy.util_conda"
HELP = """ utils for conda/pip


"""
import os, sys

#############################################################################################
from utilmy import log, log2

def help():
    from utilmy import help_create
    print( HELP + help_create(MNAME) )





#############################################################################################
class PipFinder:
    @classmethod
    def find_spec(cls, name, path, target=None):
        from importlib import util
        import subprocess
        import sys

        print(f"Module {name!r} not installed.  Attempting to pip install")
        cmd = f"{sys.executable} -m pip install {name}"
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError:
            return None

        return util.find_spec(name)


def pip_auto_install():
    """ Auto Install pip package

    """
    sys.meta_path.append(PipFinder)





