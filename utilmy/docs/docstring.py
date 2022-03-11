MNAME='utilmy.docs.docstring'
HELP=""" Automates Python scripts formatting, linting and Mkdocs documentation.


cd myutil
pip install -e . ## dev install

python utilmy/docs/docstring.py  test1

python docs/docstring.py  --dirin  uitl   --dirout    --overwrite False --test True





"""
import os, sys, ast,re, importlib
from collections import defaultdict
from pathlib import Path
from typing import Union, get_type_hints
from pprint import pprint


##########################################################################################################
from code_parser import get_list_function_info, get_list_method_info
from utilmy.utilmy import log, log2

def help():
    """function help
    Args:
    Returns:
        
    """
    from utilmy import help_create
    print( HELP + help_create(MNAME))


##########################################################################################################
def test_all():
    """function test_all
    Args:
    Returns:
        
    """
    test1()


def test1(mode='test'):
    """function test1
    Args:
        mode:   
    Returns:
        
    """
    log(""" generate_docstring """)
    # python_tips_dir = Path.cwd().joinpath("utilmy/docs")

    # not use
    # docstring_from_type_hints(python_tips_dir, python_tips_dir, overwrite_script=True, test=True)
    
    python_dir = Path.cwd().joinpath("docs/test_script")
    #python_dir = os.getcwd() + "/docs/test_script/*.py"# Path.cwd().joinpath("docs/test_script")
    
    if 'test' in mode :
       # test custom
       generate_docstring(dirin=python_dir, dirout=python_dir)

    elif 'overwrite' in mode :
       # overwrite scripts
       generate_docstring(dirin=python_dir, dirout=python_dir, overwrite=True, test=False)


def run_all(mode='overwrite'):
    """function run_all
    Args:
        mode:   
    Returns:
        
    """
    log(""" generate_docstring """)
    # python_tips_dir = Path.cwd().joinpath("utilmy/docs")

    # not use
    # docstring_from_type_hints(python_tips_dir, python_tips_dir, overwrite_script=True, test=True)
    
    python_dir = Path.cwd().joinpath("utilmy/")
    if 'overwrite' in mode :
       # overwrite scripts
       generate_docstring(dirin=python_dir, dirout=python_dir, overwrite=True, test=False)




##########################################################################################################
def docstring_from_type_hints(dirin: Union[str, Path], dirout:Union[str,Path], 
                              overwrite: bool = False, test: bool = True) -> str:
    """Automate docstring argument variable-type from type-hints.

    Args:
        dirin (< nothing >): textual directory to search for Python functions in
        overwrite_script (< wrong variable type>): enables automatic overwriting of Python scripts in dirin
        test (Something completely different): whether to write script content to a test_it.py file

    Returns:
        str: feedback message

    """
    p = dirin.glob("**/*.py")
    scripts = [x for x in p if x.is_file()]

    print(scripts)

    functions = defaultdict(list)
    for script in scripts:
        # print(script)

        with open(script, "r") as source:
            tree = ast.parse(source.read())

        function_docs = []
        for child in ast.iter_child_nodes(tree):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if child.name not in ["main"]:

                    docstring_node = child.body[0]

                    try:
                        sys.path.append(script.parent)
                        module = importlib.import_module(script.stem)
                    except Exception as e:
                        return str(e)
                    f_ = getattr(module, child.name)


                    type_hints = get_type_hints(f_)  # the same as f_.__annotations__
                    return_hint = type_hints.pop("return", None)
                    function = f_.__name__
                    functions[script].append(function)
                    
                    print('----------------------')
                    print(type_hints)
                    if type_hints:

                        docstring = f'"""{ast.get_docstring(child, clean=True)}\n"""'
                        docstring_lines = docstring.split("\n")

                        if docstring:

                            args = re.search(
                                r'Args:(.*?)(Example[s]?:|Return[s]?:|""")',
                                docstring,
                                re.DOTALL,
                            )

                            new_arguments = {}
                            if args:

                                arguments = args.group()
                                argument_lines = arguments.split("\n")

                                exclude = [
                                    "Args:",
                                    "Example:",
                                    "Examples:",
                                    "Returns:",
                                    '"""',
                                ]

                                argument_lines = [arg for arg in argument_lines if arg]
                                argument_lines = [arg for arg in argument_lines if not any(x in arg for x in exclude)]

                                for argument in argument_lines:
                                    arg_name = argument.split()[0]
                                    if arg_name in argument:

                                        if argument.split(":"):
                                            if "(" and ")" in argument.split(":")[0]:

                                                variable_type = str(type_hints[arg_name])
                                                class_type = re.search(r"(<class ')(.*)('>)", variable_type)
                                                if class_type:
                                                    variable_type = class_type.group(2)

                                                new_argument_docstring = re.sub(
                                                    r"\(.*?\)",
                                                    f"({variable_type})",
                                                    argument,
                                                )

                                                idx = docstring_lines.index(f"{argument}")
                                                new_arguments[idx] = f"{new_argument_docstring}"

                                            else:
                                                print(f"no variable type in the argument: {arg_name}")
                                        else:
                                            print(f"no 'arg : description'-format for this argument: {arg_name}")
                                    else:
                                        print(f"no docstring for this argument: {arg_name}")
                            else:
                                print(f"there are no arguments in this docstring: {function}")

                            if return_hint:

                                raw_return = re.search(
                                    # r'(?<=Returns:\n).*',
                                    r"Return[s]?:\n(.*)",
                                    docstring,
                                    re.DOTALL,
                                )

                                if raw_return:

                                    return_argument = raw_return.group(1)
                                    return_lines = return_argument.split("\n")

                                    exclude = ["Returns:", '"""']

                                    return_lines = [return_arg for return_arg in return_lines if return_arg]
                                    return_lines = [
                                        return_arg
                                        for return_arg in return_lines
                                        if not any(x in return_arg for x in exclude)
                                    ]

                                    if return_lines and len(return_lines) == 1:

                                        return_arg = return_lines[0]
                                        if return_arg.split(":"):

                                            variable_type = str(return_hint)
                                            class_type = re.search(r"(<class ')(.*)('>)", variable_type)
                                            if class_type:
                                                variable_type = class_type.group(2)

                                            new_return_docstring = re.sub(
                                                r"\S(.*:)",
                                                f"{variable_type}:",
                                                return_arg,
                                            )

                                            idx = docstring_lines.index(f"{return_arg}")
                                            new_arguments[idx] = f"{new_return_docstring}\n"

                                        else:
                                            print(f"no variable-type in return statement docstring: {function}")
                                    else:
                                        print(f"no return statement docstring argument: {function}")
                                else:
                                    print(f"no return argument in docstring for function: {function}")
                            else:
                                print(f"no return type-hint for function: {function}")

                            sorted_arguments = sorted(new_arguments.items(), reverse=True)
                            for (idx, new_arg) in sorted_arguments:
                                docstring_lines[idx] = new_arg

                            docstring_lines = [f"{' '*docstring_node.col_offset}{line}" for line in docstring_lines]
                            new_docstring = "\n".join(docstring_lines)

                            function_docs.append(
                                (
                                    docstring_node.lineno,
                                    {
                                        "function_name": function,
                                        "col_offset": docstring_node.col_offset,
                                        "begin_lineno": docstring_node.lineno,
                                        "end_lineno": docstring_node.end_lineno,
                                        "value": new_docstring,
                                    },
                                )
                            )

                            # print(ast.unparse(child))
                            # you would be able to use ast.unparse(child), however this does not include # comments.
                            # https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c

                        else:
                            print(f"no docstring for function: {function}")
                    else:
                        print(f"no type-hints for function: {function}")

        with open(script, "r") as file:
            script_lines = file.readlines()

        function_docs.sort(key=lambda x: x[0], reverse=True)
        for (idx, docstring_attr) in function_docs:
            script_lines = (
                script_lines[: docstring_attr["begin_lineno"] - 1]
                + [f'{docstring_attr["value"]}\n']
                + script_lines[docstring_attr["end_lineno"] :]
            )

        if overwrite:
            if test:
                script = f"{dirout}/test_{script.stem}.py"
            else:
               script = script.replace( str(dirin), dirout)

            with open(script, "w") as script_file:
                script_file.writelines(script_lines)

            print(f"Automated docstring generation from type hints: {script}")

    return "Docstring generation from type-hints complete!"



def generate_docstring(dirin: Union[str, Path],  dirout: Union[str, Path], overwrite: bool = False, test: bool = True):
    """  Generate docstring
        dirin (< nothing >): textual directory to search for Python functions in
        overwrite_script (< wrong variable type>): enables automatic overwriting of Python scripts in dirin
        test (Something completely different): whether to write script content to a test_it.py file
    """    
    dirin = Path(dirin) if isinstance(dirin, str) else dirin
    p = dirin.glob("**/*.py")

    # exclude = "*zml*"
    # p = glob_glob_python(dirin, suffix ="*.py", nfile=15000, exclude=exclude)
    scripts = [x for x in p if Path(x).is_file()]

    # print(scripts)
    for script in scripts:
        try :
            log("\n", script)
            log2('########## Process functions  #############################') 
            list_functions = get_list_function_info(f'{script.parent}/{script.name}')
            for function in list_functions:
                # print('--------')
                # print(function['name'])
                # print(function['line'])
                # print(function['start_idx'])
                # print(function['arg_name'])
                # print(function['arg_type'])
                # print(function['arg_value'])
                # print(function['docs'])
                # print(function['indent'])
                # print('--------')

                new_docstring = []
                # auto generate docstring
                if function['docs']:
                    # function already have docstring, will not update it
                    pass
                else:
                    # list of new docstring
                    new_docstring.append(f'{function["indent"]}"""function {function["name"]}\n')
                    # new_docstring.append("")

                    # add args
                    new_docstring.append(f'{function["indent"]}Args:\n')
                    for i in range(len(function['arg_name'])):
                        arg_type_str = f' ( {(function["arg_type"][i])} ) ' if function["arg_type"][i] else ""

                        ### TODO argname:   type, default-value,   text explanation
                        new_docstring.append(f'{function["indent"]}    {function["arg_name"][i]}{arg_type_str}:   \n')
                        # new_docstring.append(f'{function["indent"]}    {function["arg_name"][i]}{arg_type_str}: {function["arg_name"][i]}\n')

                    # new_docstring.append(f'{function["indent"]}Example:')
                    new_docstring.append(f'{function["indent"]}Returns:\n')
                    new_docstring.append(f'{function["indent"]}    \n')
                    new_docstring.append(f'{function["indent"]}"""\n')

                # print(new_docstring)
                function["new_docs"] = new_docstring

            # 1. Update the file with new update docstring for functions first
            # print(len(list_functions))
            list_functions.sort(key=lambda x: x['line'], reverse=True)
            for function in list_functions:
                if function['name'] == 'export_stats_perrepo':
                    pprint(function)

            with open(f'{script.parent}/{script.name}', "r") as file:
                script_lines = file.readlines()

            for function in list_functions:
                if function["new_docs"]:
                    script_lines = (
                        script_lines[: function["line"]]
                        # + [f'{function["new_docs"]}\n']
                        + function["new_docs"]
                        + script_lines[function["line"] + function['start_idx'] -1:]
                    )

            file_temp = f"{Path.cwd()}/temp_{script.name}"
            with open(file_temp, "w") as script_file:
                script_file.writelines(script_lines)


            log2('########## Process methods  ###############################') 
            list_methods = get_list_method_info(file_temp)
            for method in list_methods:
                new_docstring = []
                # auto generate docstring
                if method['docs']:
                    # function already have docstring, will not update it
                    pass
                else:
                    # list of new docstring
                    new_docstring.append(f'{method["indent"]}""" {method["name"]}\n')
                    # new_docstring.append("")

                    # add args
                    new_docstring.append(f'{method["indent"]}Args:\n')
                    for i in range(len(method['arg_name'])):
                        arg_type_str = f' (function["arg_type"][i]) ' if method["arg_type"][i] else ""

                        ####3 TODO   argname : type, value
                        if method["arg_name"][i] not in  ['self'] :
                           new_docstring.append(f'{method["indent"]}    {method["arg_name"][i]}{arg_type_str}:     \n')
                           #new_docstring.append(f'{method["indent"]}    {method["arg_name"][i]}{arg_type_str}: {method["arg_name"][i]}\n')


                    # new_docstring.append(f'{function["indent"]}Example:')
                    new_docstring.append(f'{method["indent"]}Returns:\n')
                    new_docstring.append(f'{method["indent"]}   \n')
                    new_docstring.append(f'{method["indent"]}"""\n')

                # print(new_docstring)
                method["new_docs"] = new_docstring


            # 2. Update the file with new update docstring for methods
            # print(len(list_methods))
            list_methods.sort(key=lambda x: x['line'], reverse=True)
            # for method in list_methods:
            #     print(method)
            with open(file_temp, "r") as file:
                script_lines = file.readlines()
            os.remove(file_temp)

            for method in list_methods:
                if method["new_docs"]:
                    script_lines = (
                        script_lines[: method["line"]]
                        # + [f'{function["new_docs"]}\n']
                        + method["new_docs"]
                        + script_lines[method["line"] + method['start_idx'] -1:]
                    )

            log2('########## Write on Disk ################################') 
            if overwrite:
                script_tmp  = f'{script.parent}/ztmp.py'
                script_test = f'{script.parent}/{script.name}'
                with open(script_tmp, "w") as script_file:
                    script_file.writelines(script_lines)
                isok = os_file_compile_check(script_tmp, verbose=0)   
                log('compile', isok)
                if isok :
                    if os.path.exists(script_test): os.remove(script_test)
                    os.rename(script_tmp, script_test)
                else :
                    os.remove(script_tmp)


            elif test:
                script_tmp  = f'{dirout}/ztmp.py'                
                script_test = f"{dirout}/test_{script.name}"
                with open(script_test, "w") as script_file:
                    script_file.writelines(script_lines)

                isok = os_file_compile_check(script_tmp, verbose=0)   
                log('compile', isok)
                if isok :
                    if os.path.exists(script_test): os.remove(script_test)
                    os.rename(script_tmp, script_test)
                else :
                    os.remove(script_tmp)


        except Exception as e :
            log("\n",e, "\n")







if 'utilties':
    def os_path_norm(diroot:str):
        """os_path_norm 
        Args:
            diroot:
        Returns:
            _description_
        """
        diroot = diroot.replace("\\", "/")
        return diroot + "/" if diroot[-1] != "/" else  diroot


    def glob_glob_python(dirin, suffix ="*.py", nfile=7, exclude=""):
        """glob_glob_python 
        Args:
            dirin: _description_
            suffix: _description_. Defaults to "*.py".
            nfile: _description_. Defaults to 7.
            exclude: _description_. Defaults to "".

        Returns:
            _description_
        """
        import glob
        dirin = str(dirin)
        flist = glob.glob(dirin + suffix) 
        flist = flist + glob.glob(dirin + "/**/" + suffix ) 
        elist = []
        
        if exclude != "":    
           for ei in exclude.split(";"):
               elist = glob.glob(ei + "/" + suffix ) 
        flist = [ fi for fi in flist if fi not in elist ]

        #### Unix format 
        flist = [  fi.replace("\\", "/") for fi in flist]

        flist = flist[:nfile]
        log(dirin, flist)
        return flist

    def os_makedirs(filename):
        if isinstance(filename, str):
            filename = [os.path.dirname(filename)]

        if isinstance(filename, list):
            folder_list = filename
            for f in folder_list:
                try:
                    if not os.path.exists(f):
                        os.makedirs(f)
                except Exception as e:
                    print(e)
            return folder_list



    #############################################################################################
    def os_file_compile_check_batch(dirin:str, nfile=10) -> dict:
        """ check if .py can be compiled
        """
        flist   = glob_glob_python( dirin, "*.py",nfile= nfile)
        results = []
        for fi in flist :
            res = os_file_compile_check(fi)
            results.append(res)

        #results = [os.system(f"python -m py_compile {i}") for i in flist]
        results = { flist[i]:  results[i] for i in range(len(flist)) }
        return results


    def os_file_compile_check(filename:str, verbose=1):
        """ check if .py can be compiled

        """
        import ast, traceback
        try : 
            with open(filename, mode='r') as f:
                source = f.read()
            ast.parse(source)
            return True
        except Exception as e:
            if verbose >0 : 
                print(e)
                traceback.print_exc() # Remove to silence any errros
        return False




##########################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()
    #main()


