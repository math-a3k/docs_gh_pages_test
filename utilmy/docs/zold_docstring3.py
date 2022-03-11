"""Automates Python scripts formatting, linting and Mkdocs documentation."""

import ast
import importlib
import re
from collections import defaultdict
from pathlib import Path
from textwrap import indent
from typing import Union, get_type_hints
import sys

from code_parser import get_list_function_info


def custom_generate_docstring(repo_dir: str, dirout: str, overwrite_script: bool = False, test: bool = True):
    """function custom_generate_docstring
    Args:
        repo_dir (  str ) :   
        dirout (  str ) :   
        overwrite_script (  bool  ) :   
        test (  bool  ) :   
    Returns:
        
    """
    
    p = repo_dir.glob("**/*.py")
    scripts = [x for x in p if x.is_file()]

    # print(scripts)
    for script in scripts:
        list_functions = get_list_function_info(f'{script.parent}/{script.name}')
        for function in list_functions:
            print('--------')
            print(function['name'])
            print(function['arg_name'])
            print(function['arg_type'])
            print(function['arg_value'])
            print(function['docs'])
            print(function['indent'])
            print('--------')

            # auto generate docstring
            if function['docs']:
                # function already have docstring, will not update it
                pass
            else:
                # list of new docstring
                new_docstring = []
                new_docstring.append(f'{function["indent"]}"""This is the docstring for function {function["name"]}')
                # new_docstring.append("")

                # add args
                new_docstring.append(f'{function["indent"]}Args:')
                for i in range(len(function['arg_name'])):
                    arg_type_str = f' (function["arg_type"][i]) ' if function["arg_type"][i] else ""
                    new_docstring.append(f'{function["indent"]}    {function["arg_name"][i]}{arg_type_str}: input variable {function["arg_name"][i]}')

                new_docstring.append(f'{function["indent"]}Example:')
                new_docstring.append(f'{function["indent"]}Returns:')
                new_docstring.append(f'{function["indent"]}"""')

                print(new_docstring)


        with open(script, "r") as file:
            script_lines = file.readlines()

        # TODO: how to write back to the file


##########################################################################################################
def main():
    """Execute when running this script."""
    python_dir = Path.cwd()
    custom_generate_docstring(python_dir, "code_parser.py")




def help_get_docstring(func):
    """ Extract Docstring from func name"""
    import inspect
    try:
        lines = func.__doc__
    except AttributeError:
        lines = ""
    return lines





if __name__ == "__main__":
    # import fire
    # fire.Fire()
    main()


